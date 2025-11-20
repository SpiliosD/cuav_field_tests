"""
High-level helpers for pairing processed and raw spectra data.

This package consolidates four main capabilities:

1. Parsing: understand spectra filenames and log files to recover datetime/timestamp
   metadata (`data_reader.parsing`).
2. Reading: load processed and raw data files from disk into NumPy arrays
   (`data_reader.reading`).
3. Processing: filter processed arrays to keep entries that align with log timestamps
   (`data_reader.processing`).
4. Matching: traverse processed/raw directory trees and generate aligned metadata
   tuples (`data_reader.matching`).

Expose the most commonly used functions at the package root so downstream scripts
can simply `from data_reader import match_processed_and_raw, filter_matches_by_log_timestamps`.
"""

from data_reader.matching.pairs import (
    filter_matches_by_log_timestamps,
    match_processed_and_raw,
)
from data_reader.parsing.spectra import (
    datetime_to_epoch_seconds,
    timestamp_from_spectra_filename,
)
from data_reader.parsing.logs import read_log_files
from data_reader.processing.aggregation import build_timestamp_data_dict
from data_reader.processing.filters import filter_processed_array_by_timestamps
from data_reader.processing.integration import build_and_save_to_database, load_from_database
# Lazy import visualization functions to avoid circular imports
# These will be imported on-demand when needed
from data_reader.storage.database import (
    DataDatabase,
    init_database,
    query_timestamp,
    query_timestamp_range,
    save_timestamp_data,
)
from data_reader.reading.readers import (
    read_processed_data_file,
    read_raw_spectra_file,
    read_text_data_file,
)

__all__ = [
    "match_processed_and_raw",
    "filter_matches_by_log_timestamps",
    "timestamp_from_spectra_filename",
    "datetime_to_epoch_seconds",
    "read_log_files",
    "build_timestamp_data_dict",
    "build_and_save_to_database",
    "load_from_database",
    "filter_processed_array_by_timestamps",
    "read_processed_data_file",
    "read_raw_spectra_file",
    "read_text_data_file",
    "DataDatabase",
    "init_database",
    "save_timestamp_data",
    "query_timestamp",
    "query_timestamp_range",
    # Visualization functions (lazy imported to avoid circular imports)
    "create_heatmaps",
    "extract_range_values",
    "aggregate_azimuth_elevation_data",
]

# Lazy import helpers for visualization functions
def _lazy_import_visualization():
    """Lazy import visualization functions to avoid circular imports."""
    from data_reader.analysis.visualization import (
        aggregate_azimuth_elevation_data,
        create_heatmaps,
        extract_range_values,
    )
    return {
        "aggregate_azimuth_elevation_data": aggregate_azimuth_elevation_data,
        "create_heatmaps": create_heatmaps,
        "extract_range_values": extract_range_values,
    }

# Make visualization functions available via lazy loading
def __getattr__(name: str):
    """Lazy import for visualization functions."""
    if name in ("aggregate_azimuth_elevation_data", "create_heatmaps", "extract_range_values"):
        return _lazy_import_visualization()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

