"""Simple script to run timestamp synchronization from IDE.

This is the main entry point for running timestamp synchronization.
Just modify the parameters below and run this script.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from timestamp_sync.timestamp_synchronizer import synchronize_timestamps
from config import Config

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Load configuration from config.txt
Config.load_from_file(silent=False)

# Folder containing log files matching output*.txt pattern
LOG_FOLDER = "G:/Raymetrics_Tests/BOMA2025/20250922"

# Folder containing _Peak.txt files
PEAK_FOLDER = "G:/Raymetrics_Tests/BOMA2025/20250922/Wind"

# SNR threshold in dB for profile values
# Timestamps where any SNR value exceeds this threshold within the specified distance interval are considered
SNR_THRESHOLD = 8.0  # dB

# Distance interval to check in meters (from-to range)
# Only SNR values within this distance range are checked for SNR > threshold
# If None, checks all ranges
# Example: (140, 300) means check all distances from 140m to 300m (inclusive)
DISTANCE_INTERVAL = (140, 300)  # (min_distance, max_distance) in meters, or None for all ranges

# Maximum time difference for matching timestamps (seconds)
TIMESTAMP_TOLERANCE = 0.1

# Output folder for synchronized log files (None = same as log_folder)
OUTPUT_FOLDER = None  # e.g., "synchronized_logs"

# Column indices in _Peak.txt files (usually don't need to change)
PROCESSED_TIMESTAMP_COLUMN = 2
PROCESSED_DATA_START_COLUMN = 3

# ============================================================================
# RUN SYNCHRONIZATION
# ============================================================================

if __name__ == "__main__":
    print("Starting timestamp synchronization...")
    print(f"Log folder: {LOG_FOLDER}")
    print(f"Peak folder: {PEAK_FOLDER}")
    print(f"SNR threshold: {SNR_THRESHOLD} dB")
    
    if DISTANCE_INTERVAL:
        min_distance, max_distance = DISTANCE_INTERVAL
        print(f"Distance interval: {min_distance} to {max_distance} m")
        range_step = Config.RANGE_STEP
        starting_range_index = Config.STARTING_RANGE_INDEX
        min_index = Config.distance_to_range_index(min_distance)
        max_index = Config.distance_to_range_index(max_distance)
        print(f"  (Converted to range indices: {min_index} to {max_index})")
        print(f"  (using range_step={range_step}m, starting_range_index={starting_range_index})")
    else:
        print("Distance interval: All ranges")
    
    print(f"Timestamp tolerance: {TIMESTAMP_TOLERANCE} seconds")
    print()
    
    # Run the synchronization
    results = synchronize_timestamps(
        log_folder=LOG_FOLDER,
        peak_folder=PEAK_FOLDER,
        snr_threshold=SNR_THRESHOLD,
        distance_interval=DISTANCE_INTERVAL,
        timestamp_tolerance=TIMESTAMP_TOLERANCE,
        output_folder=OUTPUT_FOLDER,
        processed_timestamp_column=PROCESSED_TIMESTAMP_COLUMN,
        processed_data_start_column=PROCESSED_DATA_START_COLUMN,
    )
    
    # Print summary
    if "error" not in results:
        print("\n" + "=" * 70)
        print("Synchronization Complete!")
        print("=" * 70)
        print(f"Processed {results['log_files_processed']} log files")
        print(f"Used {results['peak_files_used']} peak files")
        print(f"Total detected timestamps: {results['total_detected_timestamps']}")
        print(f"Total replacements: {results['total_replacements']} out of {results['total_entries']}")
        print(f"Replacement rate: {results['replacement_rate']:.1f}%")
        print()
        print("Output files:")
        for result in results["results"]:
            print(f"  {Path(result['output_file']).name}")
    else:
        print(f"\nError: {results['error']}")

