"""
High-level helpers for pairing processed and raw spectra data.

This package consolidates three main capabilities:

1. Parsing: understand spectra filenames and log files to recover datetime/timestamp
   metadata (`data_reader.parsing`).
2. Processing: filter processed arrays to keep entries that align with log timestamps
   (`data_reader.processing`).
3. Matching: traverse processed/raw directory trees and generate aligned metadata
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
from data_reader.processing.filters import filter_processed_array_by_timestamps

__all__ = [
    "match_processed_and_raw",
    "filter_matches_by_log_timestamps",
    "timestamp_from_spectra_filename",
    "datetime_to_epoch_seconds",
    "read_log_files",
    "filter_processed_array_by_timestamps",
]

