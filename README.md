## CUAV Field Tests Data Reader

This package provides utilities for working with Raymetrics CUAV field-test data.
It focuses on four main workflows:

1. **Parsing** spectra filenames and log files to extract timestamps.
2. **Reading** processed and raw data files from disk into NumPy arrays.
3. **Processing** processed text arrays so they stay aligned with the log.
4. **Matching** processed timestamps with raw spectra files across mirrored trees.

### Configuration

All paths and parameters are centralized in `config.py`. To customize:

1. Copy `config.example.py` to `config.py`:
   ```bash
   cp config.example.py config.py
   ```

2. Edit `config.py` with your paths:
   ```python
   Config.PROCESSED_ROOT = r"G:\Your\Path\To\Processed\Data"
   Config.RAW_ROOT = r"G:\Your\Path\To\Raw\Data"
   Config.LOG_FILE = r"G:\Your\Path\To\output.txt"
   ```

3. All scripts will automatically use these settings.

You can also validate your configuration:
```python
from config import Config

# Print current configuration
Config.print_config()

# Validate paths
all_valid, errors = Config.validate_paths()
if not all_valid:
    for error in errors:
        print(error)
```

### Directory Structure

```
data_reader/
├── parsing/
│   ├── spectra.py   # filename ↔ datetime helpers
│   └── logs.py      # log loader and timestamp extraction
├── reading/
│   └── readers.py   # read processed and raw data files
├── processing/
│   └── filters.py   # filter processed arrays by log timestamps
├── matching/
│   └── pairs.py     # traverse trees and align processed/raw timestamps
└── __init__.py      # re-exports for convenience
```

### Key Functions

| Function | Description |
| --- | --- |
| `match_processed_and_raw(processed_root, raw_root)` | Walk mirrored directory trees, pairing each processed timestamp with the raw spectra file of the same index. Returns tuples `(processed_ts, raw_datetime, raw_epoch, raw_path)` |
| `filter_matches_by_log_timestamps(matches, log_file_path)` | Remove tuples whose processed timestamps are missing from the log file’s timestamp row |
| `timestamp_from_spectra_filename(path)` | Parse `spectra_YYYY-MM-DD_HH-MM-SS.ff.*` filenames into `datetime` objects |
| `datetime_to_epoch_seconds(dt)` | Convert `datetime` to fractional seconds since epoch (string) |
| `read_log_files(path)` | Load log files (rows: azimuth, elevation, timestamps, …) |
| `extract_log_timestamps(path)` | Return only the timestamp row from the log |
| `read_processed_data_file(path)` | Read processed data files (_LogData.txt, _Peak.txt, _Spectrum.txt, _Wind.txt) |
| `read_raw_spectra_file(path)` | Read raw spectra files (binary .bin or ASCII .txt) |
| `read_text_data_file(path)` | Read generic text files containing numeric data |
| `build_timestamp_data_dict(timestamp_path_pairs, processed_root, raw_root, log_file_path)` | Aggregate data from multiple sources (peak, spectrum, wind, raw spectra, azimuth, elevation) into a nested dictionary keyed by timestamps |
| `filter_processed_array_by_timestamps(array, log_path)` | Keep only processed rows with timestamps present in the log |

### Example Usage

**Matching processed and raw data:**
```python
from data_reader import (
    match_processed_and_raw,
    filter_matches_by_log_timestamps,
)

processed_root = r"G:\Raymetrics_Tests\BOMA2025\20250922\Wind"
raw_root = r"G:\Raymetrics_Tests\BOMA2025\20250922\Spectra\User\20250922\Wind"
log_file = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"

matches = match_processed_and_raw(processed_root, raw_root)
filtered_matches = filter_matches_by_log_timestamps(matches, log_file)
print(f"Total matches: {len(matches)}, after filtering: {len(filtered_matches)}")
print("First filtered match:", filtered_matches[0] if filtered_matches else "None")
```

**Reading data files:**
```python
from data_reader import (
    read_processed_data_file,
    read_raw_spectra_file,
    read_text_data_file,
)

# Read a processed data file (_LogData.txt, _Peak.txt, etc.)
processed_data = read_processed_data_file("path/to/file/_LogData.txt")
timestamps = processed_data[:, 0]  # First column is timestamps
measurements = processed_data[:, 1:]  # Remaining columns are measurements

# Read a raw spectra file (ASCII .txt)
raw_spectra = read_raw_spectra_file("path/to/file/spectra_2025-09-22_14-19-27.56.txt")

# Read a generic text data file
generic_data = read_text_data_file("path/to/file/data.txt", delimiter=" ", skiprows=0)
```

**Aggregating data from multiple sources:**
```python
from data_reader import build_timestamp_data_dict

# Prepare timestamp-path pairs (e.g., from match_processed_and_raw)
timestamp_path_pairs = [
    ("3841395567.560425", r"G:\...\raw_data\Wind\2025-09-22\09-22_14h\09-22_14-01"),
    ("3841395568.123456", r"G:\...\raw_data\Wind\2025-09-22\09-22_14h\09-22_14-01"),
]

processed_root = r"G:\...\processed_data\Wind"
raw_root = r"G:\...\raw_data\Wind"
log_file = r"G:\...\output.txt"

# Build nested dictionary with all data for each timestamp
data_dict = build_timestamp_data_dict(
    timestamp_path_pairs,
    processed_root,
    raw_root,
    log_file,
    atol=0.0001,
)

# Access data for a specific timestamp
ts = "3841395567.560425"
if ts in data_dict:
    entry = data_dict[ts]
    print(f"Azimuth: {entry['azimuth']}")
    print(f"Elevation: {entry['elevation']}")
    print(f"Peak data shape: {entry['peak'].shape if entry['peak'] is not None else None}")
    print(f"Spectrum data shape: {entry['spectrum'].shape if entry['spectrum'] is not None else None}")
    print(f"Wind data shape: {entry['wind'].shape if entry['wind'] is not None else None}")
    print(f"Power density spectrum shape: {entry['power_density_spectrum'].shape if entry['power_density_spectrum'] is not None else None}")
```

### Running the Example Script

`data_reader/matching/pairs.py` contains a `__main__` block that demonstrates the matching and filtering flow. Update the paths to your environment and run:

```
python data_reader/matching/pairs.py
```

### Auto-Commit Scheduler

The project includes an auto-commit scheduler that automatically commits and pushes changes to GitHub every 2 hours.

**Quick Start:**

1. Install the schedule library (optional but recommended):
   ```bash
   pip install schedule
   ```

2. Run the scheduler:
   ```bash
   python scheduler.py
   ```

The scheduler will:
- Check for changes every 2 hours
- Commit all changes with a timestamp
- Push to the remote repository automatically
- Run continuously until stopped (Ctrl+C)

**Windows Task Scheduler Setup:**

To run the scheduler in the background on Windows:

1. Open Task Scheduler (search for it in Start menu)
2. Create Basic Task
3. Set trigger: Daily, repeat every 2 hours
4. Action: Start a program
   - Program: `python` (or full path to Python executable)
   - Arguments: `scheduler.py`
   - Start in: `D:\CursorProjects\cuav_field_tests`
5. Finish the wizard

Alternatively, use the provided `run_scheduler.bat` file:
```bash
# Double-click run_scheduler.bat or schedule it in Task Scheduler
run_scheduler.bat
```

See `SCHEDULER_SETUP.md` for detailed setup instructions.

### Requirements

- Python 3.9+
- NumPy
- schedule (optional, for better scheduling - install with `pip install schedule`)

### Contributing / Notes

- Keep all parsing rules in the `parsing` subpackage to ensure consistency.
- Avoid duplicating filesystem traversal logic; instead, extend `matching/pairs.py`.
- When adding new filters or matchers, prefer small, composable helpers with docstrings following the style in the existing modules.

