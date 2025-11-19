"""Integration functions for building and storing aggregated data.

This module provides high-level functions that combine data aggregation
with database storage, making it easy to process and persist data in one step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_reader.processing.aggregation import build_timestamp_data_dict
from data_reader.storage.database import DataDatabase, init_database, save_timestamp_data

__all__ = [
    "build_and_save_to_database",
    "load_from_database",
]


def build_and_save_to_database(
    timestamp_path_pairs: list[tuple[str, str | Path]],
    processed_root: str | Path,
    raw_root: str | Path,
    log_file_path: str | Path,
    db_path: str | Path,
    *,
    atol: float = 0.0001,
    return_dict: bool = False,
) -> dict[str, dict[str, Any]] | int:
    """
    Build aggregated data dictionary and save to database in one step.

    This is a convenience function that combines data aggregation with
    database persistence. It's equivalent to calling build_timestamp_data_dict
    followed by save_timestamp_data.

    Parameters
    ----------
    timestamp_path_pairs : list[tuple[str, str | Path]]
        List of (processed_timestamp, raw_data_directory_path) tuples
    processed_root : str | Path
        Root directory of processed data
    raw_root : str | Path
        Root directory of raw data
    log_file_path : str | Path
        Path to log file
    db_path : str | Path
        Path to database file
    atol : float
        Timestamp matching tolerance
    return_dict : bool
        If True, return the data dictionary. If False, return count of saved records.

    Returns
    -------
    dict[str, dict[str, Any]] | int
        Data dictionary if return_dict=True, otherwise count of saved records
    """
    # Build the data dictionary
    data_dict = build_timestamp_data_dict(
        timestamp_path_pairs,
        processed_root,
        raw_root,
        log_file_path,
        atol=atol,
    )

    # Save to database
    count = save_timestamp_data(
        data_dict,
        db_path,
        source_processed_dir=str(processed_root),
        source_raw_dir=str(raw_root),
        source_log_file=str(log_file_path),
    )

    if return_dict:
        return data_dict
    return count


def load_from_database(
    db_path: str | Path,
    start_timestamp: str | float | None = None,
    end_timestamp: str | float | None = None,
    limit: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load data from database and return as dictionary format.

    This converts database records back to the dictionary format used by
    build_timestamp_data_dict, making it easy to switch between dictionary
    and database storage.

    Parameters
    ----------
    db_path : str | Path
        Path to database file
    start_timestamp : str | float | None
        Start timestamp (inclusive)
    end_timestamp : str | float | None
        End timestamp (inclusive)
    limit : int | None
        Maximum number of records

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary keyed by timestamps, compatible with build_timestamp_data_dict output
    """
    from data_reader.storage.database import query_timestamp_range

    records = query_timestamp_range(db_path, start_timestamp, end_timestamp, limit)

    # Convert to dictionary format
    result = {}
    for record in records:
        timestamp_str = str(record["timestamp"])
        result[timestamp_str] = {
            "azimuth": record.get("azimuth"),
            "elevation": record.get("elevation"),
            "peak": record.get("peak"),
            "spectrum": record.get("spectrum"),
            "wind": record.get("wind"),
            "power_density_spectrum": record.get("power_density_spectrum"),
        }

    return result

