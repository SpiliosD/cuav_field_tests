"""Main script for testing and running data_reader methods."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_reader import (
    datetime_to_epoch_seconds,
    filter_matches_by_log_timestamps,
    filter_processed_array_by_timestamps,
    match_processed_and_raw,
    read_log_files,
    timestamp_from_spectra_filename,
)


def test_spectra_parsing():
    """Test parsing spectra filenames and datetime conversions."""
    print("=" * 70)
    print("Testing Spectra Parsing")
    print("=" * 70)

    test_files = [
        "spectra_2025-09-22_14-19-27.56.txt",
        "spectra_2025-09-22_14-19-27.56.bin",
    ]

    for test_file in test_files:
        print(f"\nParsing: {test_file}")
        try:
            dt = timestamp_from_spectra_filename(test_file)
            print(f"  Datetime: {dt}")
            epoch_str = datetime_to_epoch_seconds(dt)
            print(f"  Epoch seconds: {epoch_str}")
        except Exception as e:
            print(f"  ERROR: {e}")


def test_log_reading(log_file_path: str | Path | None = None):
    """Test reading log files."""
    print("\n" + "=" * 70)
    print("Testing Log File Reading")
    print("=" * 70)

    if log_file_path is None:
        log_file_path = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"

    log_path = Path(log_file_path)
    if not log_path.exists():
        print(f"\nWARNING: Log file not found: {log_path}")
        print("Skipping log reading test.")
        return

    try:
        print(f"\nReading log file: {log_path}")
        log_data = read_log_files(log_path)
        print(f"  Shape: {log_data.shape}")
        print(f"  First 5 azimuth values: {log_data[0][:5]}")
        print(f"  First 5 elevation values: {log_data[1][:5]}")
        print(f"  First 5 timestamps: {log_data[2][:5]}")
    except Exception as e:
        print(f"  ERROR: {e}")


def test_matching(
    processed_root: str | Path | None = None,
    raw_root: str | Path | None = None,
):
    """Test matching processed and raw data."""
    print("\n" + "=" * 70)
    print("Testing Processed/Raw Matching")
    print("=" * 70)

    if processed_root is None:
        processed_root = r"G:\Raymetrics_Tests\BOMA2025\20250922\Wind"
    if raw_root is None:
        raw_root = r"G:\Raymetrics_Tests\BOMA2025\20250922\Spectra\User\20250922\Wind"

    processed_path = Path(processed_root)
    raw_path = Path(raw_root)

    if not processed_path.exists():
        print(f"\nWARNING: Processed root not found: {processed_path}")
        print("Skipping matching test.")
        return

    if not raw_path.exists():
        print(f"\nWARNING: Raw root not found: {raw_path}")
        print("Skipping matching test.")
        return

    try:
        print(f"\nProcessed root: {processed_path}")
        print(f"Raw root: {raw_path}")
        print("\nMatching processed and raw files...")

        matches = match_processed_and_raw(processed_root, raw_root)

        print(f"\n✓ Total matches found: {len(matches)}")

        if matches:
            print("\nFirst 3 matches:")
            for i, match in enumerate(matches[:3], 1):
                processed_ts, raw_dt, raw_ts, raw_path_str = match
                print(f"  {i}. Processed TS: {processed_ts}")
                print(f"     Raw datetime: {raw_dt}")
                print(f"     Raw timestamp: {raw_ts}")
                print(f"     Raw path: {Path(raw_path_str).name}")

            if len(matches) > 3:
                print(f"  ... and {len(matches) - 3} more matches")
        else:
            print("\nNo matches found.")

        return matches
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_filtering(
    matches: list[tuple[str, str, str, str]] | None,
    log_file_path: str | Path | None = None,
):
    """Test filtering matches by log timestamps."""
    print("\n" + "=" * 70)
    print("Testing Match Filtering by Log Timestamps")
    print("=" * 70)

    if matches is None or len(matches) == 0:
        print("\nWARNING: No matches provided. Skipping filtering test.")
        return

    if log_file_path is None:
        log_file_path = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"

    log_path = Path(log_file_path)
    if not log_path.exists():
        print(f"\nWARNING: Log file not found: {log_path}")
        print("Skipping filtering test.")
        return

    try:
        print(f"\nFiltering {len(matches)} matches using log: {log_path}")

        filtered = filter_matches_by_log_timestamps(matches, log_file_path)

        print(f"\n✓ Matches after filtering: {len(filtered)}")
        print(f"  Removed: {len(matches) - len(filtered)} entries")

        if filtered:
            print("\nFirst 3 filtered matches:")
            for i, match in enumerate(filtered[:3], 1):
                processed_ts, raw_dt, raw_ts, raw_path_str = match
                print(f"  {i}. Processed TS: {processed_ts}")
                print(f"     Raw datetime: {raw_dt}")
                print(f"     Raw timestamp: {raw_ts}")
                print(f"     Raw path: {Path(raw_path_str).name}")
        else:
            print("\nNo matches remained after filtering.")

        return filtered
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main entry point for testing all data_reader methods."""
    print("\n" + "=" * 70)
    print("Data Reader Test Suite")
    print("=" * 70)

    # Configure paths (update these to match your setup)
    PROCESSED_ROOT = r"G:\Raymetrics_Tests\BOMA2025\20250922\Wind"
    RAW_ROOT = r"G:\Raymetrics_Tests\BOMA2025\20250922\Spectra\User\20250922\Wind"
    LOG_FILE = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"

    # Test 1: Spectra parsing
    test_spectra_parsing()

    # Test 2: Log file reading
    test_log_reading(LOG_FILE)

    # Test 3: Matching processed and raw data
    matches = test_matching(PROCESSED_ROOT, RAW_ROOT)

    # Test 4: Filtering matches by log timestamps
    filtered_matches = test_filtering(matches, LOG_FILE)

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total matches found: {len(matches) if matches else 0}")
    print(f"Filtered matches: {len(filtered_matches) if filtered_matches else 0}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

