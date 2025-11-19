"""Utilities for working with log files containing azimuth/elevation/timestamps.

The lidar logs encode pointing information and timestamps in text files that can be
loaded into NumPy arrays. We provide two helpers:

- `read_log_files`: load the entire transposed matrix (azimuth row, elevation row,
  timestamp row, plus optional additional metrics).
- `extract_log_timestamps`: return just the timestamps row, which is often used to
  validate processed datasets or filter mismatched rows.

Centralizing this logic makes it easier to swap out the loader (e.g., to handle
different delimiters) and ensures every consumer gets the same interpretation of
what each row represents.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_log_files", "extract_log_timestamps"]


def read_log_files(file_path: str | Path) -> np.ndarray:
    """
    Load the log file and return a matrix shaped (rows, samples).

    The log file format has columns: [azimuth, elevation, timestamp, ...]
    This function transposes the data so columns become rows for easier access.

    Parameters
    ----------
    file_path
        Path to the text log file.

    Returns
    -------
    np.ndarray
        Two-dimensional array where:
        - Row 0: azimuth angles (from column 0 of original file)
        - Row 1: elevation angles (from column 1 of original file)
        - Row 2: timestamps (from column 2 of original file)
        - Additional rows: extra measurements if present

    Why we need it
    --------------
    Processed and raw datasets both rely on the log files to provide reference
    timestamps and pointing angles. Having a canonical loader prevents subtle
    differences in dtype or orientation (e.g., row-major vs column-major).
    The transpose operation converts the column-based file format into row-based
    arrays for consistent indexing.
    """

    return np.loadtxt(file_path).T


def extract_log_timestamps(file_path: str | Path) -> np.ndarray:
    """
    Return only the timestamps from the log file.

    The log file has timestamps in column 2 (third column). After transpose,
    this becomes row 2.

    Parameters
    ----------
    file_path
        Path to the same log file used by :func:`read_log_files`.

    Returns
    -------
    np.ndarray
        One-dimensional array of timestamps (from column 2 of the original file).

    Why we need it
    --------------
    Many workflows only care about which timestamps exist in the authoritative log.
    Providing a focused helper avoids reloading or slicing the matrix in multiple
    places and keeps the meaning of "column 2" (timestamp column) documented.
    """

    data = read_log_files(file_path)
    if data.shape[0] < 3:
        raise ValueError(
            f"Log file '{file_path}' must contain at least three columns "
            f"(azimuth, elevation, timestamp)."
        )
    return data[2]  # Row 2 after transpose = Column 2 in original file

