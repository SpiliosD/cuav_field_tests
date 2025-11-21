# Timestamp Synchronization Module

## Overview

This module synchronizes timestamps between log files and peak profile files based on SNR threshold detection within a specified distance interval. It identifies profile timestamps where any SNR value within the specified distance interval exceeds a threshold, finds the log file that brackets each detected timestamp, and replaces the closest timestamp in that log file with the detected one.

## Process Steps

1. **Find log files**: Locates all text files matching `output*.txt` pattern in the specified folder (3 columns: azimuth, elevation, timestamp)
2. **Find peak files**: Locates all files ending with `_Peak.txt` in subfolders of the specified folder (timestamp in column 3, SNR values in dB from column 4 onward)
3. **Identify detected timestamps**: For each `_Peak.txt` file, finds all timestamps where any SNR value within the specified distance interval exceeds the threshold
4. **Find bracketing log files**: For each detected timestamp, finds the `output*.txt` file whose timestamps bracket it (has timestamps before and after)
5. **Find closest timestamp**: Locates within that file the closest timestamp to the detected one
6. **Replace timestamps**: Replaces that closest timestamp with the detected peak timestamp
7. **Save modified files**: Saves synchronized log files with `_new.txt` suffix (e.g., `output7.txt` → `output7_new.txt`) without modifying the originals

## Usage

### From IDE

```python
from timestamp_sync.timestamp_synchronizer import synchronize_timestamps

results = synchronize_timestamps(
    log_folder="G:/Raymetrics_Tests/BOMA2025/20250926",
    peak_folder="G:/Raymetrics_Tests/BOMA2025/20250926/Wind",
    snr_threshold=8.0,  # SNR threshold in dB
    distance_interval=(140.0, 300.0),  # Distance interval in meters
    timestamp_tolerance=0.1,
    output_folder="synchronized_logs",
)
```

### Using the Script

1. Open `timestamp_sync/run_sync.py`
2. Modify the configuration parameters:
   - `LOG_FOLDER`: Folder containing log files matching `output*.txt`
   - `PEAK_FOLDER`: Folder containing subfolders with `_Peak.txt` files
   - `SNR_THRESHOLD`: SNR threshold in dB for profile values
   - `DISTANCE_INTERVAL`: Distance interval tuple (min_distance, max_distance) in meters
   - `TIMESTAMP_TOLERANCE`: Maximum time difference for matching (seconds)
   - `OUTPUT_FOLDER`: Output folder (None = same as log_folder)
3. Run the script

## Parameters

- **log_folder** (str | Path): Folder containing log files matching `output*.txt` pattern (3 columns: azimuth, elevation, timestamp)
- **peak_folder** (str | Path): Folder containing subfolders with `_Peak.txt` files (searches recursively)
- **snr_threshold** (float): SNR threshold in dB. Timestamps where any SNR value exceeds this threshold within the specified distance interval are considered (default: 0.0)
- **distance_interval** (tuple[float, float] | None): Distance interval in meters (min_distance, max_distance). Only SNR values within this distance range are checked. If None, checks all ranges (default: None)
- **timestamp_tolerance** (float): Maximum time difference for matching timestamps in seconds (default: 0.1)
- **output_folder** (str | Path | None): Output folder for synchronized log files. If None, saves in same folder as original log files
- **processed_timestamp_column** (int): Column index for timestamps in `_Peak.txt` files (default: 2, 0-indexed, so column 3)
- **processed_data_start_column** (int): Column index where SNR data starts in `_Peak.txt` files (default: 3, 0-indexed, so column 4)

## Output

The synchronization process:

1. **Processes all log files** matching `output*.txt` pattern (3 columns: azimuth, elevation, timestamp)
2. **Loads all `_Peak.txt` files** from subfolders in the peak folder (recursively)
3. **Identifies detected timestamps** where any SNR value within the specified distance interval exceeds the threshold
4. **For each detected timestamp**:
   - Finds the log file whose timestamps bracket it (has timestamps before and after)
   - Locates the closest timestamp within that file
   - Replaces that timestamp with the detected peak timestamp
5. **Saves modified log files** with `_new.txt` suffix (e.g., `output7.txt` → `output7_new.txt`)

### Output Files

For each log file `output7.txt`, creates `output7_new.txt` with synchronized timestamps.

### Results Dictionary

Returns a dictionary with:
- `log_files_processed`: Number of log files processed
- `peak_files_used`: Number of peak files used
- `total_detected_timestamps`: Total unique timestamps with SNR above threshold in distance interval
- `snr_threshold`: The SNR threshold used (dB)
- `distance_interval`: The distance interval used (min_distance, max_distance) in meters
- `range_indices`: The range indices that correspond to the distance interval
- `total_replacements`: Total number of timestamp replacements
- `total_entries`: Total number of log entries processed
- `replacement_rate`: Percentage of entries that were replaced
- `results`: List of per-file results
- `peak_file_info`: Information about each peak file processed

## Example

```python
from timestamp_sync.timestamp_synchronizer import synchronize_timestamps

# Synchronize timestamps
results = synchronize_timestamps(
    log_folder="data/logs",
    peak_folder="data/processed",
    snr_threshold=8.0,  # SNR threshold in dB
    distance_interval=(140.0, 300.0),  # Check distances from 140m to 300m
    timestamp_tolerance=0.1,  # Match within 0.1 seconds
    output_folder="data/synchronized_logs",
)

# Check results
print(f"Processed {results['log_files_processed']} log files")
print(f"Detected {results['total_detected_timestamps']} timestamps with SNR > {results['snr_threshold']} dB")
print(f"Replacement rate: {results['replacement_rate']:.1f}%")

# List output files
for result in results["results"]:
    print(f"  {result['output_file']}: {result['replaced_timestamps']} replacements")
```

## Algorithm Details

### SNR Threshold Detection

For each `_Peak.txt` file:
1. Loads all timestamps and SNR profile data (in dB)
2. Converts the distance interval (meters) to range indices using range_step and starting_range_index from config
3. For each timestamp, checks SNR values at the range indices within the distance interval
4. If any SNR value exceeds the threshold, includes timestamp in detected set

### Bracketing and Timestamp Matching

For each detected peak timestamp:
1. Finds all log files whose timestamps bracket it (has at least one timestamp before and one after)
2. If multiple files bracket it, selects the one with the tightest bracket (smallest time span)
3. If only partial matches (one side only), selects the log file with the closest timestamp
4. Locates within that file the closest timestamp to the detected peak timestamp
5. If within tolerance, replaces that timestamp with the detected peak timestamp
6. If not within tolerance or no bracketing file found, keeps original timestamp

### File Format

- **Log files (output*.txt)**: Tab-separated columns: `azimuth`, `elevation`, `timestamp`
  - Each line represents one measurement with azimuth angle, elevation angle, and timestamp
- **Peak files (_Peak.txt)**: Tab-separated with:
  - Columns 1-2: Unused values
  - Column 3 (index 2): Timestamp
  - Columns 4+ (index 3+): SNR values in dB, corresponding to range bins defined by start range and range step from config
  - Each line represents one profile with SNR values at different range bins

### Distance to Range Index Conversion

The distance interval (in meters) is converted to range indices using:
- `range_step`: Spacing between range bins in meters (from config)
- `starting_range_index`: Index of the range bin corresponding to 0 meters (from config)
- Formula: `index = round(distance / range_step + starting_range_index)`

## Notes

- The module searches recursively for `_Peak.txt` files in subfolders of the peak folder
- The `_Peak.txt` files contain more profiles than the `output*.txt` files
- Every timestamp in any `output*.txt` file appears in at most one `_Peak.txt` file
- All detected timestamps from all peak files are combined, but each is matched to only one log file entry
- Original log files are not modified; new files are created with `_new.txt` suffix
- The output format matches the original log file format (tab-separated columns)
- If multiple log files bracket a detected timestamp, the one with the tightest bracket (smallest time span) is selected

