"""Test script for data_reader methods - testing and debugging."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_reader import (
    build_timestamp_data_dict,
    datetime_to_epoch_seconds,
    filter_matches_by_log_timestamps,
    filter_processed_array_by_timestamps,
    match_processed_and_raw,
    read_log_files,
    timestamp_from_spectra_filename,
)
from data_reader.parsing.logs import extract_log_timestamps
from config import Config


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
        log_file_path = Config.LOG_FILE

    log_path = Path(log_file_path)
    if not log_path.exists():
        print(f"\nWARNING: Log file not found: {log_path}")
        print("Skipping log reading test.")
        return

    try:
        print(f"\nReading log file: {log_path}")
        log_data = read_log_files(log_path)
        log_timestamps = extract_log_timestamps(log_path)
        print(f"  Shape: {log_data.shape}")
        print(f"  Total timestamps in log file: {len(log_timestamps)}")
        print(f"  First 5 azimuth values: {log_data[0][:5]}")
        print(f"  First 5 elevation values: {log_data[1][:5]}")
        print(f"  First 5 timestamps: {log_data[2][:5]}")
        return len(log_timestamps)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def test_matching(
    processed_root: str | Path | None = None,
    raw_root: str | Path | None = None,
):
    """Test matching processed and raw data."""
    print("\n" + "=" * 70)
    print("Testing Processed/Raw Matching")
    print("=" * 70)

    if processed_root is None:
        processed_root = Config.PROCESSED_ROOT
    if raw_root is None:
        raw_root = Config.RAW_ROOT

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
        initial_match_count = len(matches)

        print(f"\n✓ Initial datasets found: {initial_match_count}")

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
        log_file_path = Config.LOG_FILE

    log_path = Path(log_file_path)
    if not log_path.exists():
        print(f"\nWARNING: Log file not found: {log_path}")
        print("Skipping filtering test.")
        return

    try:
        # Get log timestamps count
        try:
            log_timestamps = extract_log_timestamps(log_path)
            log_count = len(log_timestamps)
        except Exception as e:
            print(f"\nWARNING: Could not read log timestamps: {e}")
            log_count = None

        initial_count = len(matches)
        print(f"\nInitial datasets (before filtering): {initial_count}")
        if log_count is not None:
            print(f"Timestamps in log file (output.txt): {log_count}")

        print(f"\nFiltering {initial_count} matches using log: {log_path}")
        
        # Diagnostic: show sample timestamps to understand the matching
        if matches and log_count is not None and log_count > 0:
            import numpy as np
            sample_processed_ts = [float(m[0]) for m in matches[:5]]
            log_ts_array = np.asarray(log_timestamps, dtype=float)
            print(f"\nSample processed timestamps: {sample_processed_ts[:3]}")
            print(f"Sample log timestamps: {log_ts_array[:3]}")
            print(f"Log timestamp range: {log_ts_array.min():.6f} to {log_ts_array.max():.6f}")
            if sample_processed_ts:
                print(f"Processed timestamp range: {min(sample_processed_ts):.6f} to {max([float(m[0]) for m in matches[:100]]):.6f}")

        filtered = filter_matches_by_log_timestamps(matches, log_file_path, atol=Config.TIMESTAMP_TOLERANCE)
        final_count = len(filtered)
        
        # Sanity check: we shouldn't have more filtered matches than log timestamps
        if log_count is not None and final_count > log_count:
            print(f"\n⚠ WARNING: More filtered matches ({final_count}) than log timestamps ({log_count})!")
            print("This suggests the filtering logic may not be working correctly.")

        print(f"\n✓ Datasets remaining after comparison with output: {final_count}")
        print(f"  Removed: {initial_count - final_count} entries")
        if initial_count > 0:
            retention_rate = (final_count / initial_count) * 100
            print(f"  Retention rate: {retention_rate:.1f}%")

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

        return filtered, initial_count, final_count, log_count
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        initial_count = len(matches) if matches else 0
        return None, initial_count, None, None


def extract_and_save_timestamps(
    processed_root: str | Path,
    raw_root: str | Path,
    log_file_path: str | Path,
    output_dir: str | Path | None = None,
):
    """
    Extract timestamps from log file and raw files, then save to CSV files.

    Parameters
    ----------
    processed_root
        Root directory containing processed data files.
    raw_root
        Root directory containing raw spectra files.
    log_file_path
        Path to the log file (output.txt) containing timestamps.
    output_dir
        Directory to save CSV files. If None, saves to current directory.
    """
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Timestamp Extraction and Comparison")
    print("=" * 70)

    # Step 1: Extract timestamps from log file
    print(f"\n1. Reading timestamps from log file: {log_file_path}")
    try:
        log_timestamps = extract_log_timestamps(log_file_path)
        log_timestamps_list = [float(ts) for ts in log_timestamps]
        print(f"   Found {len(log_timestamps_list)} timestamps in log file")
        print(f"   First 5 timestamps: {log_timestamps_list[:5]}")
        print(f"   Last 5 timestamps: {log_timestamps_list[-5:]}")
    except Exception as e:
        print(f"   ERROR: Failed to read log file: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 2: Match processed and raw files
    print(f"\n2. Matching processed and raw files...")
    print(f"   Processed root: {processed_root}")
    print(f"   Raw root: {raw_root}")
    try:
        matches = match_processed_and_raw(processed_root, raw_root)
        print(f"   Found {len(matches)} matched pairs")
    except Exception as e:
        print(f"   ERROR: Failed to match files: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 3: Extract timestamps from matched pairs (processed timestamps)
    print(f"\n3. Extracting processed timestamps from matches...")
    processed_timestamps = []
    for match in matches:
        processed_ts_str = match[0]  # First element is processed timestamp
        try:
            processed_ts = float(processed_ts_str)
            processed_timestamps.append(processed_ts)
        except (ValueError, TypeError) as e:
            print(f"   WARNING: Could not convert timestamp '{processed_ts_str}' to float: {e}")
            continue

    print(f"   Extracted {len(processed_timestamps)} processed timestamps")
    print(f"   First 5 processed timestamps: {processed_timestamps[:5]}")
    print(f"   Last 5 processed timestamps: {processed_timestamps[-5:]}")

    # Step 4: Compare timestamps with detailed diagnostics
    print(f"\n4. Comparing timestamps...")
    import numpy as np
    
    log_ts_array = np.array(log_timestamps_list, dtype=float)
    processed_ts_array = np.array(processed_timestamps, dtype=float)
    
    log_ts_set = set(log_timestamps_list)
    processed_ts_set = set(processed_timestamps)

    exact_matches = processed_ts_set & log_ts_set
    only_in_processed = processed_ts_set - log_ts_set
    only_in_log = log_ts_set - processed_ts_set

    print(f"   Timestamps in log file: {len(log_ts_set)}")
    print(f"   Timestamps in processed files: {len(processed_ts_set)}")
    print(f"   Exact matches (in both): {len(exact_matches)}")
    print(f"   Only in processed files: {len(only_in_processed)}")
    print(f"   Only in log file: {len(only_in_log)}")
    
    # Show detailed sample comparisons
    print(f"\n   Detailed sample comparison:")
    print(f"   Log timestamp range: {log_ts_array.min():.10f} to {log_ts_array.max():.10f}")
    print(f"   Processed timestamp range: {processed_ts_array.min():.10f} to {processed_ts_array.max():.10f}")
    
    # Show first few timestamps with full precision
    print(f"\n   First 3 log timestamps (full precision):")
    for i, ts in enumerate(log_timestamps_list[:3]):
        print(f"     [{i}] {ts:.15f} (raw: {ts})")
    
    print(f"\n   First 3 processed timestamps (full precision):")
    for i, ts in enumerate(processed_timestamps[:3]):
        print(f"     [{i}] {ts:.15f} (raw: {ts})")
    
    # Try tolerance-based matching with different tolerances
    print(f"\n   Tolerance-based matching:")
    for atol in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        matches_with_tol = 0
        for processed_ts in processed_ts_array[:100]:  # Sample first 100
            differences = np.abs(log_ts_array - processed_ts)
            if np.any(differences <= atol):
                matches_with_tol += 1
        if matches_with_tol > 0:
            print(f"     atol={atol:.0e}: {matches_with_tol}/100 sample timestamps matched")
    
    # Find closest matches for first few processed timestamps
    print(f"\n   Closest matches for first 3 processed timestamps:")
    for i, processed_ts in enumerate(processed_timestamps[:3]):
        differences = np.abs(log_ts_array - processed_ts)
        closest_idx = np.argmin(differences)
        closest_diff = differences[closest_idx]
        closest_log_ts = log_ts_array[closest_idx]
        print(f"     Processed[{i}]: {processed_ts:.15f}")
        print(f"       Closest log: {closest_log_ts:.15f}, difference: {closest_diff:.15f} ({closest_diff:.2e})")

    # Step 5: Save to CSV files
    print(f"\n5. Saving timestamps to CSV files...")

    # Save log timestamps
    log_csv_path = output_dir / "log_timestamps.csv"
    print(f"   Saving log timestamps to: {log_csv_path}")
    with log_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "index"])
        for idx, ts in enumerate(log_timestamps_list, 1):
            writer.writerow([ts, idx])
    print(f"   ✓ Saved {len(log_timestamps_list)} log timestamps")

    # Save processed timestamps
    processed_csv_path = output_dir / "processed_timestamps.csv"
    print(f"   Saving processed timestamps to: {processed_csv_path}")
    with processed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "index"])
        for idx, ts in enumerate(processed_timestamps, 1):
            writer.writerow([ts, idx])
    print(f"   ✓ Saved {len(processed_timestamps)} processed timestamps")

    # Save comparison results
    comparison_csv_path = output_dir / "timestamp_comparison.csv"
    print(f"   Saving comparison results to: {comparison_csv_path}")
    with comparison_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "log_timestamps_count",
                "processed_timestamps_count",
                "exact_matches_count",
                "only_in_processed_count",
                "only_in_log_count",
            ],
        )
        writer.writerow(
            [
                len(log_ts_set),
                len(processed_ts_set),
                len(exact_matches),
                len(only_in_processed),
                len(only_in_log),
            ],
        )

    # Save sample of mismatches for debugging
    if only_in_processed:
        mismatch_csv_path = output_dir / "timestamps_only_in_processed.csv"
        print(f"   Saving sample of timestamps only in processed to: {mismatch_csv_path}")
        sample_size = min(100, len(only_in_processed))
        with mismatch_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "is_in_log"])
            for ts in sorted(list(only_in_processed))[:sample_size]:
                writer.writerow([ts, False])
        print(f"   ✓ Saved {sample_size} sample timestamps (only in processed)")

    if only_in_log:
        mismatch_csv_path = output_dir / "timestamps_only_in_log.csv"
        print(f"   Saving sample of timestamps only in log to: {mismatch_csv_path}")
        sample_size = min(100, len(only_in_log))
        with mismatch_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "is_in_processed"])
            for ts in sorted(list(only_in_log))[:sample_size]:
                writer.writerow([ts, False])
        print(f"   ✓ Saved {sample_size} sample timestamps (only in log)")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Log timestamps saved to: {log_csv_path}")
    print(f"Processed timestamps saved to: {processed_csv_path}")
    print(f"Comparison results saved to: {comparison_csv_path}")
    if only_in_processed:
        print(f"Sample mismatches (only in processed) saved to: {output_dir / 'timestamps_only_in_processed.csv'}")
    if only_in_log:
        print(f"Sample mismatches (only in log) saved to: {output_dir / 'timestamps_only_in_log.csv'}")
    print("\n✓ All timestamps extracted and saved successfully!")
    print("=" * 70 + "\n")


def test_aggregation(
    matches: list[tuple[str, str, str, str]] | None,
    processed_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    log_file_path: str | Path | None = None,
):
    """Test building timestamp data dictionary from multiple sources."""
    print("\n" + "=" * 70)
    print("Testing Data Aggregation")
    print("=" * 70)

    if matches is None or len(matches) == 0:
        print("\nWARNING: No matches provided. Skipping aggregation test.")
        return None

    if processed_root is None:
        processed_root = Config.PROCESSED_ROOT
    if raw_root is None:
        raw_root = Config.RAW_ROOT
    if log_file_path is None:
        log_file_path = Config.LOG_FILE

    processed_path = Path(processed_root)
    raw_path = Path(raw_root)
    log_path = Path(log_file_path)

    if not processed_path.exists():
        print(f"\nWARNING: Processed root not found: {processed_path}")
        print("Skipping aggregation test.")
        return None

    if not raw_path.exists():
        print(f"\nWARNING: Raw root not found: {raw_path}")
        print("Skipping aggregation test.")
        return None

    if not log_path.exists():
        print(f"\nWARNING: Log file not found: {log_path}")
        print("Skipping aggregation test.")
        return None

    try:
        print(f"\nProcessed root: {processed_path}")
        print(f"Raw root: {raw_path}")
        print(f"Log file: {log_path}")
        print(f"\nBuilding data dictionary from {len(matches)} matches...")

        # Prepare timestamp-path pairs from matches
        # Each match is (processed_ts, raw_datetime, raw_ts, raw_path)
        # We need (processed_ts, raw_directory_path)
        timestamp_path_pairs = []
        for match in matches:
            processed_ts = match[0]
            raw_file_path = Path(match[3])  # Full path to raw file
            raw_dir_path = raw_file_path.parent  # Directory containing the raw file
            timestamp_path_pairs.append((processed_ts, raw_dir_path))

        print(f"  Prepared {len(timestamp_path_pairs)} timestamp-path pairs")
        if Config.MAX_TEST_ENTRIES is not None and len(timestamp_path_pairs) > Config.MAX_TEST_ENTRIES:
            print(f"  (Testing with first {Config.MAX_TEST_ENTRIES} pairs for demonstration)")
            timestamp_path_pairs = timestamp_path_pairs[:Config.MAX_TEST_ENTRIES]

        # Build the data dictionary
        data_dict = build_timestamp_data_dict(
            timestamp_path_pairs,
            processed_root,
            raw_root,
            log_file_path,
            atol=Config.TIMESTAMP_TOLERANCE,
        )

        print(f"\n✓ Successfully built data dictionary with {len(data_dict)} entries")

        # Display sample entries
        if data_dict:
            print("\nSample entries:")
            for i, (timestamp, entry) in enumerate(list(data_dict.items())[:3], 1):
                print(f"\n  Entry {i} - Timestamp: {timestamp}")
                print(f"    Azimuth: {entry['azimuth']}")
                print(f"    Elevation: {entry['elevation']}")
                
                if entry['peak'] is not None:
                    peak_shape = entry['peak'].shape if hasattr(entry['peak'], 'shape') else len(entry['peak'])
                    print(f"    Peak data: shape={peak_shape}, sample={entry['peak'][:3] if len(entry['peak']) > 0 else 'empty'}")
                else:
                    print(f"    Peak data: None")
                
                if entry['spectrum'] is not None:
                    spectrum_shape = entry['spectrum'].shape if hasattr(entry['spectrum'], 'shape') else len(entry['spectrum'])
                    print(f"    Spectrum data: shape={spectrum_shape}, sample={entry['spectrum'][:3] if len(entry['spectrum']) > 0 else 'empty'}")
                else:
                    print(f"    Spectrum data: None")
                
                if entry['wind'] is not None:
                    wind_shape = entry['wind'].shape if hasattr(entry['wind'], 'shape') else len(entry['wind'])
                    print(f"    Wind data: shape={wind_shape}, sample={entry['wind'][:3] if len(entry['wind']) > 0 else 'empty'}")
                else:
                    print(f"    Wind data: None")
                
                if entry['power_density_spectrum'] is not None:
                    pds_shape = entry['power_density_spectrum'].shape if hasattr(entry['power_density_spectrum'], 'shape') else len(entry['power_density_spectrum'])
                    print(f"    Power density spectrum: shape={pds_shape}")
                else:
                    print(f"    Power density spectrum: None")

            # Statistics
            print(f"\n  Statistics:")
            entries_with_azimuth = sum(1 for e in data_dict.values() if e['azimuth'] is not None)
            entries_with_elevation = sum(1 for e in data_dict.values() if e['elevation'] is not None)
            entries_with_peak = sum(1 for e in data_dict.values() if e['peak'] is not None)
            entries_with_spectrum = sum(1 for e in data_dict.values() if e['spectrum'] is not None)
            entries_with_wind = sum(1 for e in data_dict.values() if e['wind'] is not None)
            entries_with_pds = sum(1 for e in data_dict.values() if e['power_density_spectrum'] is not None)

            print(f"    Entries with azimuth: {entries_with_azimuth}/{len(data_dict)}")
            print(f"    Entries with elevation: {entries_with_elevation}/{len(data_dict)}")
            print(f"    Entries with peak data: {entries_with_peak}/{len(data_dict)}")
            print(f"    Entries with spectrum data: {entries_with_spectrum}/{len(data_dict)}")
            print(f"    Entries with wind data: {entries_with_wind}/{len(data_dict)}")
            print(f"    Entries with power density spectrum: {entries_with_pds}/{len(data_dict)}")
        else:
            print("\n  No entries in data dictionary")

        return data_dict
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point for running all tests."""
    print("\n" + "=" * 70)
    print("Data Reader Test Suite")
    print("=" * 70)

    # Load configuration
    Config.print_config()
    
    # Get paths from config
    PROCESSED_ROOT = Config.PROCESSED_ROOT
    RAW_ROOT = Config.RAW_ROOT
    LOG_FILE = Config.LOG_FILE
    OUTPUT_DIR = Config.get_output_dir_path()

    # Test 1: Spectra parsing
    test_spectra_parsing()

    # Test 2: Log file reading
    log_timestamp_count = test_log_reading(LOG_FILE)

    # Test 3: Matching processed and raw data
    matches = test_matching(PROCESSED_ROOT, RAW_ROOT)
    initial_count = len(matches) if matches else 0

    # Test 4: Filtering matches by log timestamps
    result = test_filtering(matches, LOG_FILE)
    if isinstance(result, tuple):
        filtered_matches, initial_filtered, final_count, log_count = result
    else:
        filtered_matches = result
        initial_filtered = initial_count
        final_count = len(filtered_matches) if filtered_matches else 0
        log_count = log_timestamp_count

    # Test 5: Data aggregation
    data_dict = test_aggregation(
        filtered_matches if filtered_matches else matches,
        PROCESSED_ROOT,
        RAW_ROOT,
        LOG_FILE,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Initial datasets found: {initial_count}")
    if log_count is not None:
        print(f"Timestamps in log file (output.txt): {log_count}")
    print(f"Datasets remaining after comparison with output: {final_count}")
    if initial_count > 0:
        removed_count = initial_count - final_count
        print(f"Datasets removed: {removed_count}")
        retention_rate = (final_count / initial_count) * 100 if initial_count > 0 else 0
        print(f"Retention rate: {retention_rate:.1f}%")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test script for data_reader methods - testing and debugging.",
    )
    parser.add_argument(
        "--debug-timestamps",
        action="store_true",
        help="Extract and save timestamps to CSV files for debugging",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("timestamp_debug_output"),
        help="Output directory for timestamp debug files (default: timestamp_debug_output)",
    )

    args = parser.parse_args()

    if args.debug_timestamps:
        # Get paths from config
        PROCESSED_ROOT = Config.PROCESSED_ROOT
        RAW_ROOT = Config.RAW_ROOT
        LOG_FILE = Config.LOG_FILE

        extract_and_save_timestamps(PROCESSED_ROOT, RAW_ROOT, LOG_FILE, args.output_dir)
    else:
        # Run all tests by default
        main()

