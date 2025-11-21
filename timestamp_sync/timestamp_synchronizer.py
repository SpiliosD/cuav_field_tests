"""Timestamp Synchronization based on Peak Profile Thresholds.

This module synchronizes timestamps between log files and peak profile files
by detecting timestamps where SNR values exceed a threshold within a specified
distance interval, finding the log file that brackets each detected timestamp,
and replacing the closest timestamp in that log file.

The synchronization process:
1. Loads all output*.txt files (azimuth, elevation, timestamp)
2. Loads all _Peak.txt files from subfolders (timestamp in column 3, SNR in dB from column 4+)
3. Identifies profile timestamps where any SNR value within the specified distance interval exceeds the threshold
4. For each detected timestamp, finds the output*.txt file whose timestamps bracket it
5. Locates within that file the closest timestamp to the detected one
6. Replaces that timestamp with the detected one
7. Writes the result to new files named output*_new.txt without modifying the originals
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import re
from collections import defaultdict

from data_reader.parsing.logs import read_log_files, extract_log_timestamps
from data_reader.reading.readers import read_processed_data_file
from config import Config


class TimestampSynchronizer:
    """
    Synchronizes timestamps between log files and peak profile files.
    
    The synchronization process:
    1. Loads all log files matching output*.txt pattern (3 columns: azimuth, elevation, timestamp)
    2. Loads all _Peak.txt files from subfolders (timestamp in column 3, SNR in dB from column 4+)
    3. Identifies profile timestamps where any SNR value within the specified distance interval exceeds the threshold
    4. For each detected timestamp, finds the output*.txt file whose timestamps bracket it
    5. Locates within that file the closest timestamp to the detected one
    6. Replaces that timestamp with the detected one
    7. Writes the result to new files named output*_new.txt without modifying the originals
    """
    
    def __init__(
        self,
        log_folder: str | Path,
        peak_folder: str | Path,
        snr_threshold: float = 0.0,
        distance_interval: tuple[float, float] | None = None,
        timestamp_tolerance: float = 0.1,
        processed_timestamp_column: int = 2,
        processed_data_start_column: int = 3,
        range_step: float | None = None,
        starting_range_index: int | None = None,
    ):
        """
        Initialize the timestamp synchronizer.
        
        Parameters
        ----------
        log_folder : str | Path
            Folder containing log files matching output*.txt pattern
        peak_folder : str | Path
            Folder containing subfolders with _Peak.txt files
        snr_threshold : float
            SNR threshold in dB. Timestamps where any SNR value exceeds this
            threshold within the specified distance interval are considered (default: 0.0)
        distance_interval : tuple[float, float] | None
            Distance interval in meters (min_distance, max_distance).
            Only SNR values within this distance range are checked.
            If None, checks all ranges (default: None)
        timestamp_tolerance : float
            Maximum time difference for matching (seconds, default: 0.1)
        processed_timestamp_column : int
            Column index for timestamps in _Peak.txt files (default: 2, 0-indexed, so column 3)
        processed_data_start_column : int
            Column index where SNR data starts in _Peak.txt files (default: 3, 0-indexed, so column 4)
        range_step : float | None
            Range step in meters for converting distance to indices (default: from Config)
        starting_range_index : int | None
            Starting range index for converting distance to indices (default: from Config)
        """
        self.log_folder = Path(log_folder)
        self.peak_folder = Path(peak_folder)
        self.snr_threshold = snr_threshold
        self.distance_interval = distance_interval
        self.timestamp_tolerance = timestamp_tolerance
        self.processed_timestamp_column = processed_timestamp_column
        self.processed_data_start_column = processed_data_start_column
        
        # Get range parameters from Config if not provided
        try:
            Config.load_from_file(silent=True)
        except Exception:
            # Config might already be loaded or file might not exist
            pass
        
        self.range_step = range_step if range_step is not None else Config.RANGE_STEP
        self.starting_range_index = starting_range_index if starting_range_index is not None else Config.STARTING_RANGE_INDEX
        
        # Convert distance interval to range indices if provided
        if distance_interval is not None:
            min_distance, max_distance = distance_interval
            # Convert distance to range index using the formula: index = round(distance / range_step + starting_range_index)
            min_index = Config.distance_to_range_index(min_distance)
            max_index = Config.distance_to_range_index(max_distance)
            # Ensure min_index <= max_index
            if min_index > max_index:
                min_index, max_index = max_index, min_index
            # Create list of all indices within the range (inclusive)
            self.range_indices = list(range(min_index, max_index + 1))
        else:
            self.range_indices = None  # Check all ranges
        
        if not self.log_folder.exists():
            raise FileNotFoundError(f"Log folder not found: {self.log_folder}")
        if not self.peak_folder.exists():
            raise FileNotFoundError(f"Peak folder not found: {self.peak_folder}")
    
    def find_log_files(self) -> list[Path]:
        """Find all log files matching output*.txt pattern."""
        log_files = sorted(self.log_folder.glob("output*.txt"))
        print(f"Found {len(log_files)} log files matching output*.txt pattern")
        return log_files
    
    def find_peak_files(self) -> list[Path]:
        """Find all files ending with _Peak.txt."""
        peak_files = sorted(self.peak_folder.rglob("*_Peak.txt"))
        print(f"Found {len(peak_files)} _Peak.txt files")
        return peak_files
    
    def load_peak_file(self, peak_file: Path) -> dict[str, Any]:
        """
        Load a _Peak.txt file and extract timestamps and profiles.
        
        Returns
        -------
        dict with keys:
            - timestamps: array of timestamps
            - profiles: 2D array of profile values (n_timestamps, n_ranges)
            - file_path: Path to the file
        """
        print(f"Loading peak file: {peak_file}")
        
        # Read the processed data file
        # Returns 2D array where each row is a timestamp, columns are: [timestamp, ...data...]
        data = read_processed_data_file(str(peak_file))
        
        # Extract timestamps from the specified column (default: column 2, index 2)
        if data.shape[1] <= self.processed_timestamp_column:
            raise ValueError(
                f"Peak file {peak_file} doesn't have enough columns. "
                f"Expected at least {self.processed_timestamp_column + 1} columns, got {data.shape[1]}"
            )
        
        timestamps = data[:, self.processed_timestamp_column]
        
        # Extract profile data starting from the specified column (default: column 3, index 3)
        if data.shape[1] <= self.processed_data_start_column:
            raise ValueError(
                f"Peak file {peak_file} doesn't have enough columns for data. "
                f"Expected at least {self.processed_data_start_column + 1} columns, got {data.shape[1]}"
            )
        
        profiles = data[:, self.processed_data_start_column:]  # Shape: (n_timestamps, n_ranges)
        
        return {
            "timestamps": timestamps,
            "profiles": profiles,
            "file_path": peak_file,
        }
    
    def identify_range_timestamps(
        self,
        peak_data: dict[str, Any],
    ) -> np.ndarray:
        """
        Identify timestamps where SNR values exceed threshold within the specified distance interval.
        
        Parameters
        ----------
        peak_data : dict
            Dictionary with 'timestamps' and 'profiles' keys
            profiles contains SNR values in dB
            
        Returns
        -------
        np.ndarray
            Array of timestamps where SNR exceeds threshold in the specified distance interval
        """
        timestamps = peak_data["timestamps"]
        profiles = peak_data["profiles"]  # SNR values in dB
        
        # Determine which range indices to check
        if self.range_indices is None:
            # Check all ranges
            range_indices_to_check = list(range(profiles.shape[1]))
        else:
            # Check only specified range indices (converted from distance interval)
            range_indices_to_check = self.range_indices
            # Validate indices
            max_index = profiles.shape[1] - 1
            invalid_indices = [idx for idx in range_indices_to_check if idx < 0 or idx > max_index]
            if invalid_indices:
                print(f"  Warning: Invalid range indices {invalid_indices} (max index: {max_index}). Ignoring.")
                range_indices_to_check = [idx for idx in range_indices_to_check if idx not in invalid_indices]
        
        if not range_indices_to_check:
            print(f"  Warning: No valid range indices to check. Skipping this file.")
            return np.array([])
        
        # Check if SNR exceeds threshold in specified ranges for each timestamp
        exceeds_threshold_mask = np.zeros(len(timestamps), dtype=bool)
        
        for i in range(len(timestamps)):
            snr_values = profiles[i, :]  # All SNR values for this timestamp (in dB)
            # Check only the specified range indices (within distance interval)
            snr_in_distance_interval = snr_values[range_indices_to_check]
            # Check if any SNR value in specified distance interval exceeds threshold
            exceeds = np.any(snr_in_distance_interval > self.snr_threshold)
            exceeds_threshold_mask[i] = exceeds
        
        detected_timestamps = timestamps[exceeds_threshold_mask]
        
        if self.distance_interval:
            min_dist, max_dist = self.distance_interval
            range_str = f"distance {min_dist:.0f}-{max_dist:.0f} m (indices {range_indices_to_check[0]}-{range_indices_to_check[-1]})"
        else:
            range_str = f"all ranges (indices {range_indices_to_check[0]}-{range_indices_to_check[-1]})"
        
        print(
            f"  Found {len(detected_timestamps)} timestamps with SNR > {self.snr_threshold} dB "
            f"in {range_str}"
        )
        
        return detected_timestamps
    
    def load_log_file(self, log_file: Path) -> dict[str, Any]:
        """
        Load a log file and extract data.
        
        Returns
        -------
        dict with keys:
            - data: transposed data array
            - azimuth: array of azimuth values
            - elevation: array of elevation values
            - timestamps: array of timestamp values
            - file_path: Path to the file
        """
        print(f"Loading log file: {log_file}")
        
        # read_log_files returns transposed array: rows are azimuth, elevation, timestamp
        log_data = read_log_files(str(log_file))
        
        if log_data.size == 0 or log_data.shape[0] < 3:
            return {
                "data": None,
                "azimuth": np.array([]),
                "elevation": np.array([]),
                "timestamps": np.array([]),
                "file_path": log_file,
            }
        
        # Log files are transposed, so:
        # Row 0: azimuth (from column 0)
        # Row 1: elevation (from column 1)
        # Row 2: timestamps (from column 2)
        return {
            "data": log_data,
            "azimuth": log_data[0] if log_data.shape[0] > 0 else np.array([]),
            "elevation": log_data[1] if log_data.shape[0] > 1 else np.array([]),
            "timestamps": log_data[2] if log_data.shape[0] > 2 else np.array([]),
            "file_path": log_file,
        }
    
    def find_closest_timestamp(
        self,
        target_timestamp: float,
        reference_timestamps: np.ndarray,
    ) -> tuple[int | None, float | None, float]:
        """
        Find the closest timestamp in reference_timestamps to target_timestamp.
        
        Parameters
        ----------
        target_timestamp : float
            Target timestamp to match
        reference_timestamps : np.ndarray
            Array of reference timestamps
            
        Returns
        -------
        tuple[int | None, float | None, float]
            (closest_index, closest_timestamp, time_difference)
            Returns (None, None, inf) if no timestamp within tolerance
        """
        if len(reference_timestamps) == 0:
            return None, None, np.inf
        
        # Calculate absolute differences
        differences = np.abs(reference_timestamps - target_timestamp)
        min_idx = np.argmin(differences)
        min_difference = differences[min_idx]
        
        if min_difference <= self.timestamp_tolerance:
            return int(min_idx), float(reference_timestamps[min_idx]), float(min_difference)
        else:
            return None, None, float(min_difference)
    
    def find_bracketing_log_file(
        self,
        target_timestamp: float,
        log_files_data: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, bool]:
        """
        Find the log file whose timestamps bracket the target timestamp.
        
        A log file brackets a timestamp if it has at least one timestamp before
        and at least one timestamp after the target timestamp.
        
        If multiple files bracket the timestamp, selects the one with the tightest
        bracket (smallest time span between the closest before and after timestamps).
        
        Parameters
        ----------
        target_timestamp : float
            Target timestamp to bracket
        log_files_data : list[dict]
            List of log file data dictionaries, each with 'timestamps' key
            
        Returns
        -------
        tuple[dict | None, bool]
            (bracketing_log_file, found_exact_bracket)
            Returns (None, False) if no log file brackets the timestamp
            found_exact_bracket is True if timestamps are on both sides, False if only one side
        """
        best_bracket = None
        best_bracket_span = np.inf
        best_is_exact = False
        
        # First pass: find all files with perfect brackets (timestamps on both sides)
        perfect_brackets = []
        for log_data in log_files_data:
            timestamps = log_data["timestamps"]
            if len(timestamps) == 0:
                continue
            
            # Check if any timestamp is before and any is after
            timestamps_before = timestamps < target_timestamp
            timestamps_after = timestamps > target_timestamp
            
            has_before = np.any(timestamps_before)
            has_after = np.any(timestamps_after)
            
            # Perfect bracket: has timestamps on both sides
            if has_before and has_after:
                # Calculate bracket span (distance from closest before to closest after)
                before_timestamps = timestamps[timestamps_before]
                after_timestamps = timestamps[timestamps_after]
                
                # Find closest before and after
                closest_before = np.max(before_timestamps)
                closest_after = np.min(after_timestamps)
                bracket_span = closest_after - closest_before
                
                perfect_brackets.append((log_data, bracket_span))
        
        # Select the tightest perfect bracket
        if perfect_brackets:
            best_bracket, best_bracket_span = min(perfect_brackets, key=lambda x: x[1])
            return best_bracket, True
        
        # If no perfect bracket found, find best partial match (timestamps on one side only)
        # This should rarely happen, but handle it for robustness
        best_partial = None
        best_partial_distance = np.inf
        
        for log_data in log_files_data:
            timestamps = log_data["timestamps"]
            if len(timestamps) == 0:
                continue
            
            # Find closest timestamp (on either side)
            differences = np.abs(timestamps - target_timestamp)
            min_idx = np.argmin(differences)
            min_distance = differences[min_idx]
            
            if min_distance < best_partial_distance:
                best_partial_distance = min_distance
                best_partial = log_data
        
        if best_partial is not None:
            return best_partial, False
        
        return None, False
    
    def synchronize_timestamps_with_bracketing(
        self,
        peak_timestamps: np.ndarray,
        peak_file_map: dict[float, Path],
        log_files_data: list[dict[str, Any]],
    ) -> dict[Path, dict[str, Any]]:
        """
        Synchronize timestamps using bracketing logic.
        
        For each detected peak timestamp:
        1. Find the log file whose timestamps bracket it
        2. Find the closest timestamp within that file
        3. Replace that timestamp with the detected peak timestamp
        
        Parameters
        ----------
        peak_timestamps : np.ndarray
            Array of detected peak timestamps (already filtered by SNR threshold)
        peak_file_map : dict[float, Path]
            Mapping from peak timestamp to the _Peak.txt file it came from
        log_files_data : list[dict]
            List of all log file data dictionaries
            
        Returns
        -------
        dict[Path, dict]
            Mapping from log file path to modified log data dictionary
        """
        # Initialize result: each log file gets a copy of its original data
        synchronized_logs = {}
        for log_data in log_files_data:
            synchronized_logs[log_data["file_path"]] = {
                "data": log_data["data"].copy() if log_data["data"] is not None else None,
                "azimuth": log_data["azimuth"].copy() if len(log_data["azimuth"]) > 0 else np.array([]),
                "elevation": log_data["elevation"].copy() if len(log_data["elevation"]) > 0 else np.array([]),
                "timestamps": log_data["timestamps"].copy(),
                "file_path": log_data["file_path"],
            }
        
        # Track replacements per log file
        replacement_counts = {log_data["file_path"]: 0 for log_data in log_files_data}
        
        # Process each detected peak timestamp
        for peak_ts in peak_timestamps:
            # Find the log file that brackets this timestamp
            bracketing_log, has_exact_bracket = self.find_bracketing_log_file(peak_ts, log_files_data)
            
            if bracketing_log is None:
                continue
            
            log_file_path = bracketing_log["file_path"]
            log_timestamps = bracketing_log["timestamps"]
            
            # Find closest timestamp within this log file
            closest_idx, closest_ts, time_diff = self.find_closest_timestamp(peak_ts, log_timestamps)
            
            if closest_idx is not None:
                # Replace the timestamp in the synchronized copy
                synchronized_logs[log_file_path]["timestamps"][closest_idx] = peak_ts
                replacement_counts[log_file_path] += 1
        
        # Print replacement statistics per file
        for log_file_path, count in replacement_counts.items():
            if count > 0:
                log_data = synchronized_logs[log_file_path]
                total = len(log_data["timestamps"])
                print(f"  {log_file_path.name}: Replaced {count} out of {total} timestamps")
        
        return synchronized_logs
    
    def save_synchronized_log_file(
        self,
        log_data: dict[str, Any],
        output_folder: Path | None = None,
    ) -> Path:
        """
        Save synchronized log file with _new.txt suffix.
        
        Parameters
        ----------
        log_data : dict
            Modified log data dictionary
        output_folder : Path | None
            Output folder (default: same as original log file folder)
            
        Returns
        -------
        Path
            Path to saved file
        """
        original_file = log_data["file_path"]
        
        # Create output filename
        output_filename = original_file.stem + "_new.txt"
        
        if output_folder is None:
            output_folder = original_file.parent
        else:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        
        output_path = output_folder / output_filename
        
        # Reconstruct log file format
        # Original format: columns are azimuth, elevation, timestamp (row-based)
        # read_log_files transposes it, so we need to transpose back
        data = log_data["data"]
        azimuth = log_data["azimuth"]
        elevation = log_data["elevation"]
        timestamps = log_data["timestamps"]
        
        # Read original file to detect number format
        original_format = self._detect_number_format(original_file)
        
        # Write file in original format: rows are entries, columns are azimuth, elevation, timestamp
        with open(output_path, "w") as f:
            n_entries = len(timestamps)
            
            # Write row by row: azimuth, elevation, timestamp
            for i in range(n_entries):
                az = azimuth[i] if i < len(azimuth) else 0.0
                el = elevation[i] if i < len(elevation) else 0.0
                ts = timestamps[i]
                
                # Format numbers to match original format (no scientific notation)
                az_str = self._format_number(az, original_format)
                el_str = self._format_number(el, original_format)
                ts_str = self._format_number(ts, original_format)
                
                f.write(f"{az_str}\t{el_str}\t{ts_str}\n")
        
        print(f"  Saved synchronized log file: {output_path}")
        return output_path
    
    def _detect_number_format(self, file_path: Path) -> dict[str, Any]:
        """
        Detect the number format used in the original file.
        
        Returns
        -------
        dict
            Format information including decimal places, etc.
        """
        try:
            with open(file_path, "r") as f:
                # Read first few lines to detect format
                lines = []
                for i, line in enumerate(f):
                    if i >= 5:  # Read first 5 lines
                        break
                    if line.strip():
                        lines.append(line.strip())
                
                if not lines:
                    return {"decimal_places": 6, "use_scientific": False}
                
                # Analyze first line
                first_line = lines[0]
                parts = first_line.split()
                if len(parts) < 3:
                    return {"decimal_places": 6, "use_scientific": False}
                
                # Check each number in the first line
                decimal_places = []
                has_scientific = False
                
                for part in parts[:3]:  # Check first 3 columns
                    try:
                        num = float(part)
                        # Check if original had scientific notation
                        if 'e' in part.lower() or 'E' in part.lower():
                            has_scientific = True
                        
                        # Count decimal places
                        if '.' in part:
                            decimal_part = part.split('.')[1]
                            # Remove scientific notation if present
                            if 'e' in decimal_part.lower():
                                decimal_part = decimal_part.split('e')[0].split('E')[0]
                            decimal_places.append(len(decimal_part))
                        else:
                            decimal_places.append(0)
                    except ValueError:
                        continue
                
                # Use maximum decimal places found, or default to 6
                max_decimals = max(decimal_places) if decimal_places else 6
                
                return {
                    "decimal_places": max_decimals,
                    "use_scientific": has_scientific,
                }
        except Exception:
            # Default format if detection fails
            return {"decimal_places": 6, "use_scientific": False}
    
    def _format_number(self, value: float, format_info: dict[str, Any]) -> str:
        """
        Format a number to match the original file format.
        
        Parameters
        ----------
        value : float
            Number to format
        format_info : dict
            Format information from _detect_number_format
            
        Returns
        -------
        str
            Formatted number string
        """
        decimal_places = format_info.get("decimal_places", 6)
        use_scientific = format_info.get("use_scientific", False)
        
        if use_scientific:
            # Use scientific notation if original had it
            return f"{value:.{decimal_places}e}"
        else:
            # Use fixed point notation, removing trailing zeros
            formatted = f"{value:.{decimal_places}f}"
            # Remove trailing zeros and decimal point if not needed
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted
    
    def process_all(
        self,
        output_folder: Path | None = None,
    ) -> dict[str, Any]:
        """
        Process all log files and peak files to synchronize timestamps using bracketing logic.
        
        Parameters
        ----------
        output_folder : Path | None
            Output folder for synchronized log files (default: same as log_folder)
            
        Returns
        -------
        dict
            Processing results with statistics
        """
        print("=" * 70)
        print("Timestamp Synchronization Process")
        print("=" * 70)
        print()
        
        # Find all files
        log_files = self.find_log_files()
        peak_files = self.find_peak_files()
        
        if len(log_files) == 0:
            print("No log files found. Exiting.")
            return {"error": "No log files found"}
        
        if len(peak_files) == 0:
            print("No peak files found. Exiting.")
            return {"error": "No peak files found"}
        
        # Load all log files first
        print()
        print("Step 1: Loading log files...")
        print("-" * 70)
        
        log_files_data = []
        for log_file in log_files:
            log_data = self.load_log_file(log_file)
            if len(log_data["timestamps"]) > 0:
                log_files_data.append(log_data)
        
        print(f"\nLoaded {len(log_files_data)} log files with timestamps")
        
        # Load all peak files and identify timestamps with SNR above threshold in distance interval
        print()
        print("Step 2: Loading peak files and identifying timestamps with SNR above threshold...")
        print("-" * 70)
        print(f"SNR threshold: {self.snr_threshold} dB")
        if self.distance_interval:
            min_dist, max_dist = self.distance_interval
            print(f"Distance interval: {min_dist:.0f} to {max_dist:.0f} m")
            if self.range_indices:
                print(f"  (Range indices: {self.range_indices[0]} to {self.range_indices[-1]})")
        else:
            print("Distance interval: All ranges")
            if self.range_indices:
                print(f"Range indices: {self.range_indices[0]} to {self.range_indices[-1]}")
        print()
        
        all_peak_timestamps = []
        peak_file_map = {}  # Map timestamp -> peak file path
        peak_file_info = []
        
        for peak_file in peak_files:
            peak_data = self.load_peak_file(peak_file)
            detected_ts = self.identify_range_timestamps(peak_data)
            
            # Store mapping from timestamp to source file
            for ts in detected_ts:
                all_peak_timestamps.append(ts)
                peak_file_map[float(ts)] = peak_file
            
            peak_file_info.append({
                "file": peak_file,
                "total_timestamps": len(peak_data["timestamps"]),
                "detected_timestamps": len(detected_ts),
            })
        
        all_peak_timestamps = np.unique(np.array(all_peak_timestamps))
        print(f"\nTotal unique timestamps with SNR > {self.snr_threshold} dB in distance interval: {len(all_peak_timestamps)}")
        
        # Synchronize timestamps using bracketing logic
        print()
        print("Step 3: Synchronizing timestamps using bracketing logic...")
        print("-" * 70)
        print("For each detected timestamp:")
        print("  1. Find log file whose timestamps bracket it")
        print("  2. Find closest timestamp within that file")
        print("  3. Replace that timestamp with the detected one")
        print()
        
        synchronized_logs = self.synchronize_timestamps_with_bracketing(
            all_peak_timestamps,
            peak_file_map,
            log_files_data,
        )
        
        # Save synchronized files
        print()
        print("Step 4: Saving synchronized log files...")
        print("-" * 70)
        
        results = []
        total_replaced = 0
        total_entries = 0
        
        for log_file_path, synchronized_data in synchronized_logs.items():
            original_log_data = next((ld for ld in log_files_data if ld["file_path"] == log_file_path), None)
            if original_log_data is None:
                continue
            
            # Count replacements
            original_ts = original_log_data["timestamps"]
            synchronized_ts = synchronized_data["timestamps"]
            replaced_count = np.sum(original_ts != synchronized_ts)
            
            if replaced_count > 0 or len(synchronized_ts) > 0:
                # Save synchronized file
                output_path = self.save_synchronized_log_file(
                    synchronized_data,
                    output_folder=output_folder,
                )
                
                results.append({
                    "original_file": str(log_file_path),
                    "output_file": str(output_path),
                    "total_entries": len(synchronized_ts),
                    "replaced_timestamps": replaced_count,
                })
                
                total_replaced += replaced_count
                total_entries += len(synchronized_ts)
        
        # Summary
        print()
        print("=" * 70)
        print("Synchronization Summary")
        print("=" * 70)
        print(f"Processed {len(log_files_data)} log files")
        print(f"Used {len(peak_files)} peak files")
        print(f"Total unique timestamps with SNR > {self.snr_threshold} dB in distance interval: {len(all_peak_timestamps)}")
        print()
        
        print(f"Total timestamp replacements: {total_replaced} out of {total_entries} entries")
        print(f"Replacement rate: {100.0 * total_replaced / max(total_entries, 1):.1f}%")
        print()
        
        return {
            "log_files_processed": len(log_files_data),
            "peak_files_used": len(peak_files),
            "total_detected_timestamps": len(all_peak_timestamps),
            "snr_threshold": self.snr_threshold,
            "distance_interval": self.distance_interval,
            "range_indices": self.range_indices,
            "total_replacements": total_replaced,
            "total_entries": total_entries,
            "replacement_rate": 100.0 * total_replaced / max(total_entries, 1),
            "results": results,
            "peak_file_info": peak_file_info,
        }


def synchronize_timestamps(
    log_folder: str | Path,
    peak_folder: str | Path,
    snr_threshold: float = 0.0,
    distance_interval: tuple[float, float] | None = None,
    timestamp_tolerance: float = 0.1,
    output_folder: str | Path | None = None,
    processed_timestamp_column: int = 2,
    processed_data_start_column: int = 3,
    range_step: float | None = None,
    starting_range_index: int | None = None,
) -> dict[str, Any]:
    """
    Convenience function to synchronize timestamps between log files and peak files.
    
    This function implements the improved synchronization algorithm:
    1. Identifies profile timestamps where any SNR value within the specified distance interval exceeds the threshold
    2. For each such timestamp, finds the output*.txt file whose timestamps bracket it
    3. Locates within that file the closest timestamp
    4. Replaces that timestamp with the detected one
    5. Writes the result to new files named output*_new.txt without modifying the originals
    
    Parameters
    ----------
    log_folder : str | Path
        Folder containing log files matching output*.txt pattern
    peak_folder : str | Path
        Folder containing subfolders with _Peak.txt files
    snr_threshold : float
        SNR threshold in dB. Timestamps where any SNR value exceeds this
        threshold within the specified distance interval are considered (default: 0.0)
    distance_interval : tuple[float, float] | None
        Distance interval in meters (min_distance, max_distance).
        Only SNR values within this distance range are checked.
        If None, checks all ranges (default: None)
    timestamp_tolerance : float
        Maximum time difference for matching (seconds, default: 0.1)
    output_folder : str | Path | None
        Output folder for synchronized log files (default: same as log_folder)
    processed_timestamp_column : int
        Column index for timestamps in _Peak.txt files (default: 2, 0-indexed, so column 3)
    processed_data_start_column : int
        Column index where SNR data starts in _Peak.txt files (default: 3, 0-indexed, so column 4)
    range_step : float | None
        Range step in meters for converting distance to indices (default: from Config)
    starting_range_index : int | None
        Starting range index for converting distance to indices (default: from Config)
        
    Returns
    -------
    dict
        Processing results with statistics
    """
    synchronizer = TimestampSynchronizer(
        log_folder=log_folder,
        peak_folder=peak_folder,
        snr_threshold=snr_threshold,
        distance_interval=distance_interval,
        timestamp_tolerance=timestamp_tolerance,
        processed_timestamp_column=processed_timestamp_column,
        processed_data_start_column=processed_data_start_column,
        range_step=range_step,
        starting_range_index=starting_range_index,
    )
    
    output_path = Path(output_folder) if output_folder is not None else None
    
    return synchronizer.process_all(output_folder=output_path)


if __name__ == "__main__":
    # Example usage
    from config import Config
    
    Config.load_from_file(silent=False)
    
    # Example: synchronize timestamps
    results = synchronize_timestamps(
        log_folder="G:/Raymetrics_Tests/BOMA2025/20250926",
        peak_folder="G:/Raymetrics_Tests/BOMA2025/20250926/Wind",
        intensity_threshold=10.0,  # Intensity must exceed 10.0
        range_indices=[10, 11, 12, 13, 14],  # Only check these range indices (distances)
        timestamp_tolerance=0.1,
        output_folder="synchronized_logs",
    )
    
    print("\nResults:")
    print(results)

