"""Filtering helpers for processed arrays.

Processed text files can contain rows whose timestamps no longer align with the
authoritative log (e.g., due to dropped packets). The helper below removes those
rows by comparing against the log timestamps, ensuring downstream analyses only
operate on synchronized data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_reader.parsing.logs import extract_log_timestamps

__all__ = ["filter_processed_array_by_timestamps"]


def filter_processed_array_by_timestamps(
    processed_array: np.ndarray,
    log_file_path: str | Path,
    *,
    timestamp_column: int = 2,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Keep only rows whose timestamp appears in the given log file.

    Parameters
    ----------
    processed_array
        NumPy array containing processed data; must have timestamps in one column.
    log_file_path
        Log file with the authoritative timestamp row (row index 2).
    timestamp_column
        Which column in ``processed_array`` contains the timestamps to compare.
    atol
        Absolute tolerance for floating-point comparisons.

    Returns
    -------
    np.ndarray
        Filtered copy of ``processed_array`` containing only rows whose timestamps
        are present in the log file.

    Why we need it
    --------------
    When aligning processed and raw datasets, we must ensure we only work with rows
    that correspond to known log entries. This helper encapsulates the filtering
    logic so every caller applies the same matching and tolerance rules.
    """

    from data_reader.parsing.timestamp_correction import correct_processed_timestamp
    
    log_timestamps = extract_log_timestamps(log_file_path)
    processed_ts_raw = processed_array[:, timestamp_column].astype(float)
    
    # Correct processed timestamps
    processed_ts = np.array([correct_processed_timestamp(ts) for ts in processed_ts_raw])

    mask = np.isclose(processed_ts[:, None], log_timestamps[None, :], atol=atol).any(axis=1)
    # Update the timestamp column in the returned array with corrected values
    result = processed_array[mask].copy()
    if result.size > 0:
        result[:, timestamp_column] = processed_ts[mask]
    return result

