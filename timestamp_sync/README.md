# Timestamp Synchronization Module

## Overview

This module synchronizes timestamps between log files and peak profile files based on intensity range detection. It identifies timestamps where profile values fall within a specified intensity range in `_Peak.txt` files and replaces corresponding log file timestamps with the closest matching peak file timestamps.

## Process Steps

1. **Find log files**: Locates all text files matching `output*.txt` pattern in the specified folder
2. **Find peak files**: Locates all files ending with `_Peak.txt` in the specified folder
3. **Identify range timestamps**: For each `_Peak.txt` file, finds all timestamps where any profile value falls within the specified intensity range
4. **Match log entries**: For each detected timestamp, searches log files for entries whose timestamps fall within the tolerance range
5. **Replace timestamps**: Replaces log file timestamps with the closest matching peak file timestamp
6. **Save modified files**: Saves synchronized log files with `_new.txt` suffix (e.g., `output7.txt` → `output7_new.txt`)

## Usage

### From IDE

```python
from timestamp_sync.timestamp_synchronizer import synchronize_timestamps

results = synchronize_timestamps(
    log_folder="G:/Raymetrics_Tests/BOMA2025/20250926",
    peak_folder="G:/Raymetrics_Tests/BOMA2025/20250926/Wind",
    intensity_range=(5.0, 50.0),  # Values between 5.0 and 50.0
    timestamp_tolerance=0.1,
    output_folder="synchronized_logs",
)
```

### Using the Script

1. Open `timestamp_sync/run_sync.py`
2. Modify the configuration parameters:
   - `LOG_FOLDER`: Folder containing log files matching `output*.txt`
   - `PEAK_FOLDER`: Folder containing `_Peak.txt` files
   - `INTENSITY_RANGE`: Intensity range tuple (min, max) for profile values
   - `TIMESTAMP_TOLERANCE`: Maximum time difference for matching (seconds)
   - `OUTPUT_FOLDER`: Output folder (None = same as log_folder)
3. Run the script

## Parameters

- **log_folder** (str | Path): Folder containing log files matching `output*.txt` pattern
- **peak_folder** (str | Path): Folder containing `_Peak.txt` files (searches recursively)
- **intensity_range** (tuple[float, float] | None): Range for profile values (min, max). Timestamps where any profile value falls within this range are considered. If None, uses (0.0, inf) for all values
- **timestamp_tolerance** (float): Maximum time difference for matching timestamps in seconds (default: 0.1)
- **output_folder** (str | Path | None): Output folder for synchronized log files. If None, saves in same folder as original log files
- **processed_timestamp_column** (int): Column index for timestamps in `_Peak.txt` files (default: 2)
- **processed_data_start_column** (int): Column index where profile data starts in `_Peak.txt` files (default: 3)

## Output

The synchronization process:

1. **Processes all log files** matching `output*.txt` pattern
2. **Loads all `_Peak.txt` files** from the peak folder (recursively)
3. **Identifies high-threshold timestamps** from all peak files
4. **Synchronizes each log file** by replacing timestamps within tolerance
5. **Saves modified log files** with `_new.txt` suffix

### Output Files

For each log file `output7.txt`, creates `output7_new.txt` with synchronized timestamps.

### Results Dictionary

Returns a dictionary with:
- `log_files_processed`: Number of log files processed
- `peak_files_used`: Number of peak files used
- `total_range_timestamps`: Total unique timestamps within intensity range
- `intensity_range`: The intensity range used (min, max)
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
    intensity_range=(5.0, 50.0),  # Only consider timestamps with profile values between 5.0 and 50.0
    timestamp_tolerance=0.05,  # Match within 0.05 seconds
    output_folder="data/synchronized_logs",
)

# Check results
print(f"Processed {results['log_files_processed']} log files")
print(f"Replacement rate: {results['replacement_rate']:.1f}%")

# List output files
for result in results["results"]:
    print(f"  {result['output_file']}: {result['replaced_timestamps']} replacements")
```

## Algorithm Details

### Intensity Range Detection

For each `_Peak.txt` file:
1. Loads all timestamps and profile data
2. For each timestamp, checks all profile values across all ranges
3. If any profile value falls within the specified intensity range [min, max], includes timestamp in range set

### Timestamp Matching

For each log file entry:
1. Finds the closest peak file timestamp within tolerance
2. If found, replaces log timestamp with peak timestamp
3. If not found within tolerance, keeps original timestamp

### File Format

- **Log files**: Tab-separated columns: `azimuth`, `elevation`, `timestamp`
- **Peak files**: Tab-separated with timestamps in column 2 (index 2) and profile data starting at column 3 (index 3)

## Notes

- The module searches recursively for `_Peak.txt` files in the peak folder
- All high-threshold timestamps from all peak files are combined into a single set
- Log file timestamps are matched against this combined set
- Original log files are not modified; new files are created with `_new.txt` suffix
- The output format matches the original log file format (tab-separated columns)

