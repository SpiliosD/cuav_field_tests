"""Helpers for interpreting spectra filenames and timestamps.

This module focuses on utilities shared by multiple readers:

- Extracting datetime objects from spectra filenames, which follow a strict pattern
  (`spectra_YYYY-MM-DD_HH-MM-SS.ff.<ext>`).
- Converting datetime objects to fractional seconds since the Unix epoch.

Why this exists: multiple parts of the pipeline (matching processed/raw data,
reporting timestamp metadata to downstream systems) need consistent interpretation
of the spectra filenames. Centralizing the logic prevents duplication and guarantees
that changes to the naming pattern only need to happen in one place.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

TIMESTAMP_PATTERN = "%Y-%m-%d_%H-%M-%S.%f"

__all__ = [
    "timestamp_from_spectra_filename",
    "datetime_to_epoch_seconds",
]


def timestamp_from_spectra_filename(file_path: str | Path) -> datetime:
    """
    Extract a datetime from filenames like ``spectra_2025-09-22_14-19-27.56.txt``.

    Parameters
    ----------
    file_path
        Full path or filename string. Only the stem is parsed, so either .bin or .txt
        extensions work as long as the stem matches the expected pattern.

    Returns
    -------
    datetime
        Parsed timestamp object ready for comparisons or formatting.

    Why we need it
    --------------
    The raw spectra filenames encode acquisition timestamps. To align raw files with
    processed text, we must recover that timestamp. Duplicating this parsing logic
    across modules would be error-prone; having a shared helper guarantees consistent
    behavior and validation.
    """

    stem = Path(file_path).stem
    if not stem.startswith("spectra_"):
        raise ValueError(f"Filename must start with 'spectra_': {file_path}")

    _, timestamp_str = stem.split("spectra_", maxsplit=1)
    try:
        return datetime.strptime(timestamp_str, TIMESTAMP_PATTERN)
    except ValueError as exc:
        raise ValueError(
            f"Filename '{file_path}' does not match pattern {TIMESTAMP_PATTERN!r}",
        ) from exc


def datetime_to_epoch_seconds(value: datetime, *, precision: int = 6) -> str:
    """
    Convert a datetime into fractional seconds since the Unix epoch.

    Parameters
    ----------
    value
        The datetime to convert.
    precision
        Number of decimal places for the fractional seconds component.

    Returns
    -------
    str
        The formatted seconds-since-epoch string.

    Why we need it
    --------------
    Processed data timestamps are often stored as floating-point seconds. When we
    parse datetimes from filenames we frequently need to express them in the same
    format so that numeric comparisons are straightforward. This helper ensures both
    representations stay in sync.
    """

    format_str = f"{{:.{precision}f}}"
    return format_str.format(value.timestamp())

