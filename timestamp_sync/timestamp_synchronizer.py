"""Timestamp Synchronization based on Peak Profile Thresholds.

This module synchronizes timestamps between log files and peak profile files
by detecting timestamps where profile values exceed a threshold and matching
them with corresponding log file entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import re
from collections import defaultdict

from data_reader.parsing.logs import read_log_files
from data_reader.reading.readers import read_processed_data_file


class TimestampSynchronizer:
    """
    Synchronizes timestamps between log files and peak profile files.
    
    The synchronization process:
    1. Loads all log files matching output*.txt pattern
    2. Loads all _Peak.txt files
    3. Identifies timestamps where profile values fall within the specified intensity range
    4. Matches log file entries with peak file timestamps
    5. Replaces log timestamps with closest peak timestamps
    6. Saves modified log files with _new.txt suffix
    """
    
    def __init__(
        self,
        log_folder: str | Path,
        peak_folder: str | Path,
        intensity_threshold: float = 0.0,
        range_indices: list[int] | None = None,
        timestamp_tolerance: float = 0.1,
        processed_timestamp_column: int = 2,
        processed_data_start_column: int = 3,
    ):
        """
        Initialize the timestamp synchronizer.
        
        Parameters
        ----------
        log_folder : str | Path
            Folder containing log files matching output*.txt pattern
        peak_folder : str | Path
            Folder containing _Peak.txt files
        intensity_threshold : float
            Threshold for profile values. Timestamps where intensity exceeds this
            threshold in the specified range indices are considered (default: 0.0)
        range_indices : list[int] | None
            List of range indices (column indices) to check in the profile.
            Only these specific ranges are checked for intensity > threshold.
            If None, checks all ranges (default: None)
        timestamp_tolerance : float
            Maximum time difference for matching (seconds, default: 0.1)
        processed_timestamp_column : int
            Column index for timestamps in _Peak.txt files (default: 2)
        processed_data_start_column : int
            Column index where profile data starts in _Peak.txt files (default: 3)
        """
        self.log_folder = Path(log_folder)
        self.peak_folder = Path(peak_folder)
        self.intensity_threshold = intensity_threshold
        self.range_indices = range_indices  # None means check all ranges
        self.timestamp_tolerance = timestamp_tolerance
        self.processed_timestamp_column = processed_timestamp_column
        self.processed_data_start_column = processed_data_start_column
        
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
        Identify timestamps where profile values exceed threshold in specific range indices.
        
        Parameters
        ----------
        peak_data : dict
            Dictionary with 'timestamps' and 'profiles' keys
            
        Returns
        -------
        np.ndarray
            Array of timestamps where intensity exceeds threshold in the specified range indices
        """
        timestamps = peak_data["timestamps"]
        profiles = peak_data["profiles"]
        
        # Determine which range indices to check
        if self.range_indices is None:
            # Check all ranges
            range_indices_to_check = list(range(profiles.shape[1]))
        else:
            # Check only specified range indices
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
        
        # Check if intensity exceeds threshold in specified ranges for each timestamp
        exceeds_threshold_mask = np.zeros(len(timestamps), dtype=bool)
        
        for i in range(len(timestamps)):
            profile_values = profiles[i, :]  # All range values for this timestamp
            # Check only the specified range indices
            values_in_specified_ranges = profile_values[range_indices_to_check]
            # Check if any value in specified ranges exceeds threshold
            exceeds = np.any(values_in_specified_ranges > self.intensity_threshold)
            exceeds_threshold_mask[i] = exceeds
        
        range_timestamps = timestamps[exceeds_threshold_mask]
        
        range_str = f"indices {range_indices_to_check}" if self.range_indices else "all ranges"
        print(
            f"  Found {len(range_timestamps)} timestamps with intensity > {self.intensity_threshold} "
            f"in {range_str}"
        )
        
        return range_timestamps
    
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
    ) -> tuple[float | None, float]:
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
        tuple[float | None, float]
            (closest_timestamp, time_difference)
            Returns (None, inf) if no timestamp within tolerance
        """
        if len(reference_timestamps) == 0:
            return None, np.inf
        
        # Calculate absolute differences
        differences = np.abs(reference_timestamps - target_timestamp)
        min_idx = np.argmin(differences)
        min_difference = differences[min_idx]
        
        if min_difference <= self.timestamp_tolerance:
            return float(reference_timestamps[min_idx]), float(min_difference)
        else:
            return None, float(min_difference)
    
    def synchronize_log_file(
        self,
        log_data: dict[str, Any],
        peak_timestamps: np.ndarray,
    ) -> dict[str, Any]:
        """
        Synchronize log file timestamps with peak file timestamps.
        
        Parameters
        ----------
        log_data : dict
            Log file data dictionary
        peak_timestamps : np.ndarray
            Array of peak file timestamps to match against
            
        Returns
        -------
        dict
            Modified log data with synchronized timestamps
        """
        log_timestamps = log_data["timestamps"]
        
        if len(log_timestamps) == 0 or len(peak_timestamps) == 0:
            print(f"  No timestamps to synchronize")
            return log_data.copy()
        
        # Create mapping: log_timestamp -> closest_peak_timestamp
        synchronized_timestamps = log_timestamps.copy()
        replacement_count = 0
        
        for i, log_ts in enumerate(log_timestamps):
            closest_ts, time_diff = self.find_closest_timestamp(log_ts, peak_timestamps)
            
            if closest_ts is not None:
                synchronized_timestamps[i] = closest_ts
                replacement_count += 1
        
        print(f"  Replaced {replacement_count} out of {len(log_timestamps)} timestamps")
        
        # Create modified log data
        modified_data = log_data.copy()
        modified_data["timestamps"] = synchronized_timestamps
        
        return modified_data
    
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
        Process all log files and peak files to synchronize timestamps.
        
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
        
        # Load all peak files and identify timestamps with intensity above threshold in specific ranges
        print()
        print("Step 1: Loading peak files and identifying timestamps with intensity above threshold...")
        print("-" * 70)
        print(f"Intensity threshold: {self.intensity_threshold}")
        if self.range_indices:
            print(f"Range indices to check: {self.range_indices}")
        else:
            print("Range indices to check: All ranges")
        print()
        
        all_peak_timestamps = []
        peak_file_info = []
        
        for peak_file in peak_files:
            peak_data = self.load_peak_file(peak_file)
            range_ts = self.identify_range_timestamps(peak_data)
            
            all_peak_timestamps.extend(range_ts.tolist())
            peak_file_info.append({
                "file": peak_file,
                "total_timestamps": len(peak_data["timestamps"]),
                "range_timestamps": len(range_ts),
            })
        
        all_peak_timestamps = np.array(all_peak_timestamps)
        print(f"\nTotal unique timestamps within intensity range: {len(all_peak_timestamps)}")
        
        # Process each log file
        print()
        print("Step 2: Processing log files and synchronizing timestamps...")
        print("-" * 70)
        
        results = []
        
        for log_file in log_files:
            print(f"\nProcessing: {log_file.name}")
            
            # Load log file
            log_data = self.load_log_file(log_file)
            
            if len(log_data["timestamps"]) == 0:
                print(f"  Skipping: No timestamps in log file")
                continue
            
            # Synchronize timestamps
            synchronized_data = self.synchronize_log_file(log_data, all_peak_timestamps)
            
            # Save synchronized file
            output_path = self.save_synchronized_log_file(
                synchronized_data,
                output_folder=output_folder,
            )
            
            results.append({
                "original_file": str(log_file),
                "output_file": str(output_path),
                "total_entries": len(log_data["timestamps"]),
                "replaced_timestamps": np.sum(
                    synchronized_data["timestamps"] != log_data["timestamps"]
                ),
            })
        
        # Summary
        print()
        print("=" * 70)
        print("Synchronization Summary")
        print("=" * 70)
        print(f"Processed {len(log_files)} log files")
        print(f"Used {len(peak_files)} peak files")
        print(f"Total timestamps with intensity > {self.intensity_threshold} in specified ranges: {len(all_peak_timestamps)}")
        print()
        
        total_replaced = sum(r["replaced_timestamps"] for r in results)
        total_entries = sum(r["total_entries"] for r in results)
        
        print(f"Total timestamp replacements: {total_replaced} out of {total_entries} entries")
        print(f"Replacement rate: {100.0 * total_replaced / max(total_entries, 1):.1f}%")
        print()
        
        return {
            "log_files_processed": len(log_files),
            "peak_files_used": len(peak_files),
            "total_range_timestamps": len(all_peak_timestamps),
            "intensity_threshold": self.intensity_threshold,
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
    intensity_threshold: float = 0.0,
    range_indices: list[int] | None = None,
    timestamp_tolerance: float = 0.1,
    output_folder: str | Path | None = None,
    processed_timestamp_column: int = 2,
    processed_data_start_column: int = 3,
) -> dict[str, Any]:
    """
    Convenience function to synchronize timestamps between log files and peak files.
    
    Parameters
    ----------
    log_folder : str | Path
        Folder containing log files matching output*.txt pattern
    peak_folder : str | Path
        Folder containing _Peak.txt files
    intensity_threshold : float
        Threshold for profile values. Timestamps where intensity exceeds this
        threshold in the specified range indices are considered (default: 0.0)
    range_indices : list[int] | None
        List of range indices (column indices) to check in the profile.
        Only these specific ranges are checked for intensity > threshold.
        If None, checks all ranges (default: None)
    timestamp_tolerance : float
        Maximum time difference for matching (seconds, default: 0.1)
    output_folder : str | Path | None
        Output folder for synchronized log files (default: same as log_folder)
    processed_timestamp_column : int
        Column index for timestamps in _Peak.txt files (default: 2)
    processed_data_start_column : int
        Column index where profile data starts in _Peak.txt files (default: 3)
        
    Returns
    -------
    dict
        Processing results with statistics
    """
    synchronizer = TimestampSynchronizer(
        log_folder=log_folder,
        peak_folder=peak_folder,
        intensity_threshold=intensity_threshold,
        range_indices=range_indices,
        timestamp_tolerance=timestamp_tolerance,
        processed_timestamp_column=processed_timestamp_column,
        processed_data_start_column=processed_data_start_column,
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

