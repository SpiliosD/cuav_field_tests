"""Aggregate data from multiple sources into a unified dictionary structure.

This module provides utilities for combining data from processed files (_Peak.txt,
_Spectrum.txt, _Wind.txt), log files (for azimuth/elevation), and raw spectra files
into a single nested dictionary keyed by processed timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data_reader.parsing.logs import read_log_files
from data_reader.reading.readers import read_raw_spectra_file, read_text_data_file

__all__ = ["build_timestamp_data_dict"]


def build_timestamp_data_dict(
    timestamp_path_pairs: list[tuple[str, str | Path]],
    processed_root: str | Path,
    raw_root: str | Path,
    log_file_path: str | Path,
    *,
    atol: float = 0.0001,
) -> dict[str, dict[str, Any]]:
    """
    Build a nested dictionary aggregating data from multiple sources for each timestamp.

    Inputs
    ------
    timestamp_path_pairs : list[tuple[str, str | Path]]
        List of tuples where each tuple contains:
        - processed_timestamp (str): The processed data timestamp
        - raw_data_directory_path (str | Path): Directory path containing the
          corresponding raw data file
    processed_root : str | Path
        Root directory of the processed data tree. Used to locate _Peak.txt,
        _Spectrum.txt, and _Wind.txt files.
    raw_root : str | Path
        Root directory of the raw data tree. Used to compute relative paths
        for finding corresponding processed directories.
    log_file_path : str | Path
        Path to the log file containing azimuth, elevation, and timestamps.
        The log file should have rows: [azimuth, elevation, timestamps, ...]
    atol : float, optional
        Absolute tolerance for matching timestamps (default: 0.0001).

    Outputs
    -------
    dict[str, dict[str, Any]]
        Dictionary where:
        - Keys are processed timestamps (as strings)
        - Values are dictionaries with keys:
          - 'azimuth': float or None
          - 'elevation': float or None
          - 'peak': np.ndarray or None (data from _Peak.txt, columns 4 onwards)
          - 'spectrum': np.ndarray or None (data from _Spectrum.txt, columns 4 onwards)
          - 'wind': np.ndarray or None (data from _Wind.txt, columns 4 onwards)
          - 'power_density_spectrum': np.ndarray or None (raw spectra data, skipping first 13 lines)

    Why we need it
    --------------
    Downstream analyses often require combining data from multiple sources (processed
    files, log files, raw spectra) for each timestamp. This function centralizes the
    logic for reading, matching, and aggregating this data into a unified structure,
    making it easier to work with the complete dataset for each timestamp.

    Notes
    -----
    - Timestamps are matched using tolerance-based comparison (atol parameter)
    - If a timestamp cannot be found in a particular file, the corresponding value
      will be None
    - The raw data directory path is used to locate the specific raw spectra file
      that corresponds to the timestamp
    - Processed files (_Peak.txt, _Spectrum.txt, _Wind.txt) are located by finding
      the corresponding directory in the processed_root tree that matches the structure
      of the raw_data_directory_path
    """
    processed_base = Path(processed_root).expanduser().resolve()
    raw_base = Path(raw_root).expanduser().resolve()
    log_path = Path(log_file_path).expanduser().resolve()

    if not processed_base.exists():
        raise FileNotFoundError(f"Processed root directory does not exist: {processed_base}")
    if not raw_base.exists():
        raise FileNotFoundError(f"Raw root directory does not exist: {raw_base}")
    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    # Load log file data (azimuth, elevation, timestamps)
    log_data = read_log_files(log_path)
    if log_data.shape[0] < 3:
        raise ValueError(f"Log file must contain at least 3 rows (azimuth, elevation, timestamps)")
    
    log_azimuth = log_data[0]
    log_elevation = log_data[1]
    log_timestamps_raw = log_data[2]

    # Correct log timestamps (they have the same 2091->2025 year issue as processed files)
    from data_reader.parsing.timestamp_correction import correct_processed_timestamp
    log_timestamps_float = np.asarray(log_timestamps_raw, dtype=float)
    log_timestamps = np.array([correct_processed_timestamp(ts) for ts in log_timestamps_float])

    # Normalize log timestamps for comparison
    log_timestamps_normalized = np.round(log_timestamps, decimals=6)

    # Dictionary to store results
    result: dict[str, dict[str, Any]] = {}

    # Group pairs by raw directory to minimize file reads
    # Group by directory, but also track the file path for each timestamp
    # timestamp_path_pairs contains (timestamp, raw_file_path) tuples
    dir_to_timestamps: dict[Path, list[tuple[str, Path]]] = {}  # (timestamp, file_path) pairs
    for processed_ts, raw_path in timestamp_path_pairs:
        raw_path_obj = Path(raw_path).expanduser().resolve()
        # Determine if it's a file or directory
        if raw_path_obj.is_file() or raw_path_obj.suffix:  # It's a file
            raw_dir_path = raw_path_obj.parent
            file_path = raw_path_obj
        else:  # It's a directory (backward compatibility)
            raw_dir_path = raw_path_obj
            file_path = None
        
        if raw_dir_path not in dir_to_timestamps:
            dir_to_timestamps[raw_dir_path] = []
        dir_to_timestamps[raw_dir_path].append((processed_ts, file_path))

    # Process each directory
    for raw_dir_path, timestamps in dir_to_timestamps.items():
        if not raw_dir_path.exists():
            print(f"Warning: Raw directory does not exist: {raw_dir_path}")
            continue

        # Find corresponding processed directory
        # Mirror the directory structure: compute relative path from raw_root
        # and apply it to processed_root
        try:
            relative_path = raw_dir_path.relative_to(raw_base)
            processed_dir = processed_base / relative_path
            if not processed_dir.exists():
                processed_dir = None
        except ValueError:
            # raw_dir_path is not under raw_base, try alternative matching
            processed_dir = _find_corresponding_processed_dir(raw_dir_path, processed_base)
        
        if processed_dir is None or not processed_dir.exists():
            print(f"Warning: Could not find corresponding processed directory for {raw_dir_path}")
            # Initialize entries with None values
            # timestamps is now a list of (timestamp, file_path) tuples
            for ts, _ in timestamps:
                result[ts] = {
                    "azimuth": None,
                    "elevation": None,
                    "peak": None,
                    "spectrum": None,
                    "wind": None,
                    "power_density_spectrum": None,
                }
            continue

        # Load processed data files
        peak_file = processed_dir / "_Peak.txt"
        spectrum_file = processed_dir / "_Spectrum.txt"
        wind_file = processed_dir / "_Wind.txt"

        peak_data = _load_processed_file_with_timestamps(peak_file) if peak_file.exists() else None
        spectrum_data = _load_processed_file_with_timestamps(spectrum_file) if spectrum_file.exists() else None
        wind_data = _load_processed_file_with_timestamps(wind_file) if wind_file.exists() else None

        # Read all processed timestamps from _Peak.txt for index-based matching
        from data_reader.parsing.timestamp_correction import correct_processed_timestamp
        
        processed_timestamps_list: list[str] = []
        if peak_file.exists():
            try:
                with peak_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        tokens = stripped.split()
                        if len(tokens) >= 3:
                            # Apply timestamp correction
                            raw_timestamp = tokens[2]
                            corrected_timestamp = correct_processed_timestamp(raw_timestamp)
                            processed_timestamps_list.append(str(corrected_timestamp))
            except Exception as e:
                print(f"Warning: Failed to read timestamps from {peak_file}: {e}")
        
        # Process each timestamp
        # Note: timestamps from timestamp_path_pairs are already corrected
        # timestamps is now a list of (timestamp, file_path) tuples
        for processed_ts, raw_file_path in timestamps:
            processed_ts_float = float(processed_ts)
            processed_ts_normalized = round(processed_ts_float, 6)

            # Initialize entry
            entry: dict[str, Any] = {
                "azimuth": None,
                "elevation": None,
                "peak": None,
                "spectrum": None,
                "wind": None,
                "power_density_spectrum": None,
            }

            # Match with log file for azimuth and elevation
            differences = np.abs(log_timestamps_normalized - processed_ts_normalized)
            match_idx = np.argmin(differences)
            if differences[match_idx] <= atol:
                entry["azimuth"] = float(log_azimuth[match_idx])
                entry["elevation"] = float(log_elevation[match_idx])
            # Debug: Show first few matches for verification (only in debug mode)
            from config import Config
            if Config.is_debug_mode() and len(result) < 3:
                if entry["azimuth"] is not None:
                    print(f"  Debug: Matched log entry {match_idx}: azimuth={entry['azimuth']:.2f}, elevation={entry['elevation']:.2f}, "
                          f"diff={differences[match_idx]:.6f} s")
                else:
                    print(f"  Debug: No log match for processed_ts={processed_ts_normalized}, "
                          f"closest log_ts={log_timestamps_normalized[match_idx]:.6f}, diff={differences[match_idx]:.6f} s")

            # Match with processed files (peak, spectrum, wind) by index
            # Find the index of this timestamp in processed_timestamps_list
            timestamp_idx = None
            for idx, ts_str in enumerate(processed_timestamps_list):
                try:
                    ts_float = float(ts_str)
                    ts_normalized = round(ts_float, 6)
                    if abs(ts_normalized - processed_ts_normalized) <= atol:
                        timestamp_idx = idx
                        break
                except (ValueError, TypeError):
                    continue
            
            # Use index-based matching for processed files (positional matching)
            if timestamp_idx is not None:
                if peak_data is not None and timestamp_idx < len(peak_data):
                    # Data starts from 4th element (index 3) onwards
                    entry["peak"] = peak_data[timestamp_idx][3:] if len(peak_data[timestamp_idx]) > 3 else np.array([])

                if spectrum_data is not None and timestamp_idx < len(spectrum_data):
                    entry["spectrum"] = spectrum_data[timestamp_idx][3:] if len(spectrum_data[timestamp_idx]) > 3 else np.array([])

                if wind_data is not None and timestamp_idx < len(wind_data):
                    entry["wind"] = wind_data[timestamp_idx][3:] if len(wind_data[timestamp_idx]) > 3 else np.array([])

            # Load raw spectra file - use the file path directly if provided, otherwise use positional matching
            raw_file = None
            if raw_file_path is not None and raw_file_path.exists():
                raw_file = raw_file_path
            elif timestamp_idx is not None:
                # Fallback to positional matching: nth timestamp = nth file
                raw_files = sorted(raw_dir_path.glob("spectra_*.txt"))
                if timestamp_idx < len(raw_files):
                    raw_file = raw_files[timestamp_idx]
            
            if raw_file is not None:
                try:
                    # Skip first 13 lines, then read numeric data
                    raw_spectra = read_text_data_file(raw_file, skiprows=13)
                    entry["power_density_spectrum"] = raw_spectra
                except Exception as e:
                    print(f"Warning: Failed to read raw spectra file {raw_file}: {e}")

            result[processed_ts] = entry

    return result


def _find_corresponding_processed_dir(
    raw_dir_path: Path,
    processed_root: Path,
) -> Path | None:
    """
    Find the processed directory that corresponds to a raw directory.

    This function attempts to match directory structures by looking for common
    path components. It searches for directories in processed_root that contain
    the same final path components as raw_dir_path.

    Inputs
    ------
    raw_dir_path : Path
        Path to the raw data directory
    processed_root : Path
        Root of the processed data tree

    Outputs
    -------
    Path | None
        Path to the corresponding processed directory, or None if not found
    """
    # Try to find a directory in processed_root that matches the structure
    # Look for directories with the same name components
    raw_parts = raw_dir_path.parts
    
    # Search for matching directory structure
    # Try to find a directory that ends with the same path components
    for candidate in processed_root.rglob("*"):
        if candidate.is_dir():
            candidate_parts = candidate.parts
            # Check if the last few components match
            if len(candidate_parts) >= len(raw_parts):
                # Try matching from the end
                if candidate_parts[-len(raw_parts):] == raw_parts:
                    return candidate
            # Also try matching by checking if the directory name matches
            if candidate.name == raw_dir_path.name:
                # Check if parent structure is similar
                if len(candidate.parts) == len(raw_parts):
                    return candidate
    
    # Fallback: try to construct path by matching relative structure
    # This assumes the directory structure is mirrored
    # Extract the relative path components and try to find them in processed_root
    for part in reversed(raw_parts):
        matching_dirs = list(processed_root.rglob(f"**/{part}"))
        if matching_dirs:
            # Return the first match (could be improved with better heuristics)
            return matching_dirs[0].parent if matching_dirs[0].is_file() else matching_dirs[0]
    
    return None


def _load_processed_file_with_timestamps(file_path: Path) -> np.ndarray | None:
    """
    Load a processed file and return the data array with corrected timestamps.

    The file should have lines where the third element (index 2) is the timestamp,
    and data values start from the fourth element (index 3) onwards.
    Timestamps are automatically corrected for year (2091->2025) and day offset.

    Inputs
    ------
    file_path : Path
        Path to the processed file

    Outputs
    -------
    np.ndarray | None
        Array where each row corresponds to a line, with corrected timestamps in column 2,
        or None if file cannot be read
    """
    from data_reader.parsing.timestamp_correction import correct_processed_timestamp
    
    try:
        data = read_text_data_file(file_path)
        # Correct timestamps in column 2 (index 2)
        if data.shape[1] > 2:
            data[:, 2] = np.array([correct_processed_timestamp(ts) for ts in data[:, 2]])
        return data
    except Exception:
        return None


def _find_matching_row(
    data: np.ndarray,
    timestamp: float,
    atol: float,
) -> np.ndarray | None:
    """
    Find the row in data where the third column (index 2) matches the timestamp.

    Inputs
    ------
    data : np.ndarray
        Array where column 2 contains timestamps
    timestamp : float
        Timestamp to match
    atol : float
        Absolute tolerance for matching

    Outputs
    -------
    np.ndarray | None
        Matching row, or None if no match found
    """
    if data.shape[1] < 3:
        return None
    
    timestamps_col = data[:, 2]
    timestamps_normalized = np.round(timestamps_col, decimals=6)
    timestamp_normalized = round(timestamp, 6)
    
    differences = np.abs(timestamps_normalized - timestamp_normalized)
    match_idx = np.argmin(differences)
    
    if differences[match_idx] <= atol:
        return data[match_idx]
    
    return None


def _find_raw_file_for_timestamp(
    raw_dir: Path,
    timestamp: float,
    processed_timestamps: list[str],
    atol: float,
) -> Path | None:
    """
    Find the raw spectra file that corresponds to a given timestamp by index.

    Raw files are matched with processed timestamps by index order. This function
    finds the index of the timestamp in the processed timestamps list and returns
    the raw file at the same index.

    Inputs
    ------
    raw_dir : Path
        Directory containing raw spectra files
    timestamp : float
        Processed timestamp to match
    processed_timestamps : list[str]
        List of all processed timestamps from the _Peak.txt file (in order)
    atol : float
        Absolute tolerance for matching timestamps

    Outputs
    -------
    Path | None
        Path to the matching raw file, or None if not found
    """
    # Get sorted raw files
    raw_files = sorted(raw_dir.glob("spectra_*.txt"))
    if not raw_files:
        return None
    
    # Find the index of the matching timestamp
    timestamp_normalized = round(float(timestamp), 6)
    for idx, ts_str in enumerate(processed_timestamps):
        try:
            ts_float = float(ts_str)
            ts_normalized = round(ts_float, 6)
            if abs(ts_normalized - timestamp_normalized) <= atol:
                # Found matching timestamp at index idx
                if idx < len(raw_files):
                    return raw_files[idx]
                break
        except (ValueError, TypeError):
            continue
    
    return None

