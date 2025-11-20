"""Match processed timestamps with raw spectra files.

These utilities traverse parallel directory trees containing processed text files
and raw spectra files. The goal is to align the first timestamp of each processed
line with the corresponding raw spectra file, producing tuples with:

1. The processed timestamp (as string, typically seconds since epoch)
2. The human-readable datetime parsed from the raw filename
3. The epoch-style timestamp derived from #2
4. The absolute path to the raw spectra file

We also provide a filter that removes matches whose processed timestamps do not
appear in the authoritative log file (row 2). Centralizing this logic guarantees
consistent ordering, validation, and error handling across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from data_reader.parsing.spectra import (
    datetime_to_epoch_seconds,
    timestamp_from_spectra_filename,
)
from data_reader.parsing.logs import read_log_files, extract_log_timestamps

PROCESSED_SUFFIX = "_Peak.txt"
RAW_FILE_PATTERN = "spectra_*.txt"


@dataclass(frozen=True)
class MatchedEntry:
    """
    Container representing a matched processed/raw pair.

    Attributes
    ----------
    processed_timestamp
        Timestamp string from the processed `_Peak.txt` line (corrected).
    raw_datetime
        Formatted datetime string parsed from the raw spectra filename.
    raw_timestamp
        Seconds-since-epoch representation derived from ``raw_datetime``.
    raw_path
        Absolute path to the raw ASCII spectra file.
    original_timestamp
        Original uncorrected timestamp from processed file (for debugging).

    Why we need it
    --------------
    Creating explicit objects (even lightweight ones) improves readability when we
    build the list of tuples and enables us to document each component clearly.
    """

    processed_timestamp: str
    raw_datetime: str
    raw_timestamp: str
    raw_path: str
    original_timestamp: str | None

    def __init__(
        self,
        *,
        processed_timestamp: str,
        raw_datetime: str,
        raw_timestamp: str,
        raw_path: str,
        original_timestamp: str | None = None,
    ):
        self.processed_timestamp = processed_timestamp
        self.raw_datetime = raw_datetime
        self.raw_timestamp = raw_timestamp
        self.raw_path = raw_path
        self.original_timestamp = original_timestamp

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.processed_timestamp,
            self.raw_datetime,
            self.raw_timestamp,
            self.raw_path,
        )
    
    def as_tuple_with_original(self) -> tuple[str, str, str, str, str | None]:
        """Return tuple including original timestamp for database storage."""
        return (
            self.processed_timestamp,
            self.raw_datetime,
            self.raw_timestamp,
            self.raw_path,
            self.original_timestamp,
        )


def _resolve_existing_path(path_like: str | Path) -> Path:
    """Resolve a user-supplied path and ensure it exists."""

    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _iter_processed_log_files(root: Path) -> Iterable[Path]:
    """Yield every processed `_Peak.txt` found under ``root``."""

    yield from root.rglob(f"*{PROCESSED_SUFFIX}")


def _read_timestamps_from_file(file_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Read the third whitespace-delimited token from each line in a processed log.
    
    Timestamps are automatically corrected for year (2091->2025) and day offset.

    Inputs
    ------
    file_path : Path
        Location of the `_Peak.txt` file whose lines contain timestamps.

    Outputs
    -------
    tuple[list[str], dict[str, str]]
        - List of corrected timestamp strings (for matching/operations)
        - Dictionary mapping corrected timestamp -> original uncorrected timestamp (for debugging)

    Why we need it
    --------------
    The processed `_Peak.txt` files store multiple columns per line, with the
    timestamp in the third column (index 2). Isolating this extraction keeps
    the main matching routine tidy and makes it easier to extend (e.g., to capture
    additional metadata from the same lines).
    """
    from data_reader.parsing.timestamp_correction import correct_processed_timestamp

    timestamps: list[str] = []
    original_timestamps: dict[str, str] = {}  # corrected -> original mapping
    with file_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if len(tokens) < 3:
                raise ValueError(
                    f"Line {line_num} in {file_path} has fewer than 3 columns. "
                    f"Found {len(tokens)} column(s). Expected format: <col1> <col2> <timestamp> ..."
                )
            # Read timestamp and apply correction
            raw_timestamp = tokens[2]
            corrected_timestamp = correct_processed_timestamp(raw_timestamp)
            corrected_str = str(corrected_timestamp)
            timestamps.append(corrected_str)
            original_timestamps[corrected_str] = raw_timestamp
    return timestamps, original_timestamps


def _sorted_raw_files(folder: Path) -> list[Path]:
    """
    Return chronologically sorted spectra files for ``folder``.

    Inputs
    ------
    folder : Path
        Folder mirroring a processed directory, expected to contain ASCII spectra.

    Outputs
    -------
    list[Path]
        Sorted list of spectra files matching ``RAW_FILE_PATTERN``.

    Why we need it
    --------------
    Spectra filenames already encode timestamps; sorting ensures we pair them with
    processed timestamps by index, even if the filesystem order is not deterministic.
    """

    files = sorted(folder.glob(RAW_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No raw files matching '{RAW_FILE_PATTERN}' found in {folder}")
    return files


def match_processed_and_raw(
    processed_root: str | Path,
    raw_root: str | Path,
) -> list[tuple[str, str, str, str]]:
    """
    Traverse processed/raw directory trees and align timestamps by index.

    Inputs
    ------
    processed_root : Path-like
        Root of the processed tree (e.g., ``Wind/YYYY-MM-DD``). Each leaf folder
        must contain `_Peak.txt`.
    raw_root : Path-like
        Root of the raw spectra tree that mirrors ``processed_root``.

    Outputs
    -------
    list[tuple[str, str, str, str]]
        Each tuple contains (processed_timestamp, raw_datetime, raw_epoch, raw_path).

    Why we need it
    --------------
    Downstream analyses often require referencing both processed metrics and the
    original raw spectra file for each timestamp. This function automates the
    directory traversal and pairing so higher-level scripts can work with a
    pre-aligned array instead of reimplementing the logic.
    """

    processed_base = _resolve_existing_path(processed_root)
    raw_base = _resolve_existing_path(raw_root)

    combined: list[MatchedEntry] = []

    for log_file in _iter_processed_log_files(processed_base):
        folder = log_file.parent
        relative_folder = folder.relative_to(processed_base)
        raw_folder = raw_base / relative_folder

        if not raw_folder.exists():
            raise FileNotFoundError(
                f"Missing raw folder for processed folder '{folder}': expected '{raw_folder}'",
            )

        processed_timestamps, original_timestamps_map = _read_timestamps_from_file(log_file)
        raw_files = _sorted_raw_files(raw_folder)

        if len(processed_timestamps) != len(raw_files):
            raise ValueError(
                "Mismatch between processed timestamps and raw files in "
                f"folder '{folder}': {len(processed_timestamps)} vs {len(raw_files)}",
            )

        for processed_ts, raw_path in zip(processed_timestamps, raw_files):
            raw_dt = timestamp_from_spectra_filename(raw_path)
            # Store original timestamp for this corrected timestamp
            original_ts = original_timestamps_map.get(processed_ts)
            combined.append(
                MatchedEntry(
                    processed_timestamp=processed_ts,
                    raw_datetime=raw_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    raw_timestamp=datetime_to_epoch_seconds(raw_dt),
                    raw_path=str(raw_path),
                    original_timestamp=original_ts,  # Store original for debugging
                ),
            )

    # Return tuples with original timestamp included
    return [entry.as_tuple_with_original() for entry in combined]


def filter_matches_by_log_timestamps(
    matches: list[tuple[str, str, str, str]],
    log_file_path: str | Path,
    *,
    atol: float = 0.0001,
) -> list[tuple[str, str, str, str]]:
    """
    Drop matched tuples whose processed timestamp is absent from the log file.

    Inputs
    ------
    matches : list of tuples
        Output of :func:`match_processed_and_raw`.
    log_file_path : Path-like
        Log file whose third row contains the canonical timestamps.
    atol : float
        Absolute tolerance for comparing float timestamps. Default is 0.0001.

    Outputs
    -------
    list[tuple[str, str, str, str]]
        Filtered list retaining only entries whose processed timestamps match
        log timestamps within the specified tolerance.

    Why we need it
    --------------
    In some runs, processed files may contain timestamps that were dropped or never
    recorded in the log due to connectivity issues. Filtering keeps subsequent
    analyses consistent with the authoritative log. Uses tolerance-based matching
    (atol=0.00001 by default) to account for small floating-point precision differences.
    """

    log_timestamps = np.asarray(extract_log_timestamps(log_file_path), dtype=float)
    
    # Normalize timestamps to 6 decimal places to match common precision format
    # This helps with floating-point precision issues
    log_timestamps_normalized = np.round(log_timestamps, decimals=6)

    from data_reader.parsing.timestamp_correction import correct_processed_timestamp
    
    filtered: list[tuple[str, str, str, str]] = []
    for entry in matches:
        try:
            # Correct processed timestamp before comparison
            raw_processed_ts = float(entry[0])
            processed_ts = correct_processed_timestamp(raw_processed_ts)
            # Normalize processed timestamp to same precision
            processed_ts_normalized = round(processed_ts, 6)
            
            # Check if processed_ts is within tolerance of any log timestamp
            differences = np.abs(log_timestamps_normalized - processed_ts_normalized)
            if np.any(differences <= atol):
                # Store corrected timestamp in the filtered entry
                filtered.append((str(processed_ts), entry[1], entry[2], entry[3]))
        except (ValueError, TypeError) as e:
            # Skip entries with invalid timestamps
            continue
    
    return filtered


if __name__ == "__main__":
    processed_root = r"G:\Raymetrics_Tests\BOMA2025\20250922\Wind"
    raw_root = r"G:\Raymetrics_Tests\BOMA2025\20250922\Spectra\User\20250922\Wind"
    log_file = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"

    all_matches = match_processed_and_raw(processed_root, raw_root)
    print(f"Total matches found: {len(all_matches)}")
    print("First match:", all_matches[0] if all_matches else "None")

    filtered_matches = filter_matches_by_log_timestamps(all_matches, log_file)
    print(f"Matches after filtering by log timestamps: {len(filtered_matches)}")
    print("First filtered match:", filtered_matches[0] if filtered_matches else "None")

