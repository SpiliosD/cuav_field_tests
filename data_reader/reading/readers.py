"""Methods for reading processed and raw data files.

This module provides utilities for reading various types of data files from
the CUAV field tests, including:
- Processed data files (_LogData.txt, _Peak.txt, _Spectrum.txt, _Wind.txt)
- Raw spectra files (binary .bin and ASCII .txt)
- Generic text data files with numeric data

All methods handle file validation, error checking, and return structured
data in NumPy arrays for consistent processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "read_processed_data_file",
    "read_raw_spectra_file",
    "read_text_data_file",
]


def read_text_data_file(
    file_path: str | Path,
    *,
    delimiter: str | None = None,
    skiprows: int = 0,
    dtype: type[np.floating] | type[np.integer] = float,
) -> np.ndarray:
    """
    Read a generic text file containing numeric data.

    Parameters
    ----------
    file_path
        Path to the text file to read.
    delimiter
        Delimiter character for splitting columns. If None, uses whitespace.
    skiprows
        Number of rows to skip at the beginning of the file.
    dtype
        Data type for the resulting array (default: float).

    Returns
    -------
    np.ndarray
        Two-dimensional array containing the numeric data from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed into numeric data.

    Why we need it
    --------------
    Provides a consistent interface for reading text-based data files with
    proper error handling and validation. Centralizes loading logic to avoid
    code duplication and ensures consistent dtype and format handling.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        data = np.loadtxt(
            path,
            delimiter=delimiter,
            skiprows=skiprows,
            dtype=dtype,
            ndmin=2,
        )
        return data
    except Exception as e:
        raise ValueError(f"Failed to read data from '{path}': {e}") from e


def read_processed_data_file(
    file_path: str | Path,
    *,
    delimiter: str | None = None,
    skiprows: int = 0,
) -> np.ndarray:
    """
    Read a processed data file (e.g., _LogData.txt, _Peak.txt, _Spectrum.txt, _Wind.txt).

    Processed data files contain lines where the first token is typically a timestamp,
    followed by measurement data columns. The lines are sorted in ascending timestamp order.

    Parameters
    ----------
    file_path
        Path to the processed data file.
    delimiter
        Delimiter character for splitting columns. If None, uses whitespace.
    skiprows
        Number of rows to skip at the beginning of the file.

    Returns
    -------
    np.ndarray
        Two-dimensional array where each row corresponds to a line in the file.
        Column 0 typically contains timestamps, and subsequent columns contain
        measurement data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed or has invalid format.

    Why we need it
    --------------
    Processed data files follow a consistent format across the dataset. This method
    provides a standardized way to read them with proper validation and error handling.
    Ensures all processed data is loaded with consistent dtype and format.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    try:
        data = np.loadtxt(
            path,
            delimiter=delimiter,
            skiprows=skiprows,
            dtype=float,
            ndmin=2,
        )

        if data.ndim != 2:
            raise ValueError(
                f"Processed data file '{path}' must contain at least 2 columns "
                f"(timestamp + at least one measurement column)",
            )

        return data
    except Exception as e:
        raise ValueError(f"Failed to read processed data from '{path}': {e}") from e


def read_raw_spectra_file(
    file_path: str | Path,
    *,
    dtype: type[np.floating] | type[np.integer] = np.float32,
) -> np.ndarray:
    """
    Read a raw spectra file (binary .bin or ASCII .txt).

    Raw spectra files can be either binary (.bin) or ASCII text (.txt) format.
    The method automatically detects the format based on the file extension and
    reads the data accordingly.

    Parameters
    ----------
    file_path
        Path to the raw spectra file (either .bin or .txt).
    dtype
        Data type for reading binary files (default: np.float32).
        For ASCII files, this is used as a hint but may be overridden.

    Returns
    -------
    np.ndarray
        Array containing the spectra data. The shape depends on the file format
        and structure.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported or cannot be read.
    NotImplementedError
        If binary file reading is not yet implemented.

    Why we need it
    --------------
    Raw spectra files come in multiple formats (binary and ASCII). This method
    provides a unified interface for reading them, abstracting away the format
    differences. Centralizes the reading logic to ensure consistent handling
    and makes it easier to add support for additional formats in the future.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw spectra file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        # ASCII text format
        try:
            data = np.loadtxt(path, dtype=dtype)
            return data
        except Exception as e:
            raise ValueError(f"Failed to read ASCII spectra from '{path}': {e}") from e

    elif suffix == ".bin":
        # Binary format
        # Note: Binary file reading may require format-specific knowledge
        # (e.g., header size, data layout, endianness)
        # This is a placeholder for future implementation
        raise NotImplementedError(
            f"Binary spectra file reading not yet implemented for '{path}'. "
            f"Please use ASCII (.txt) format or implement binary reading based on "
            f"the specific file format specification.",
        )

    else:
        raise ValueError(
            f"Unsupported file format for raw spectra file '{path}'. "
            f"Expected .txt or .bin, got '{suffix}'.",
        )


