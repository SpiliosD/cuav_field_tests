## CUAV Field Tests Data Reader

This package provides utilities for working with Raymetrics CUAV field-test data.
It focuses on three workflows:

1. **Parsing** spectra filenames and log files to extract timestamps.
2. **Processing** processed text arrays so they stay aligned with the log.
3. **Matching** processed timestamps with raw spectra files across mirrored trees.

### Directory Structure

```
data_reader/
├── parsing/
│   ├── spectra.py   # filename ↔ datetime helpers
│   └── logs.py      # log loader and timestamp extraction
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
| `filter_processed_array_by_timestamps(array, log_path)` | Keep only processed rows with timestamps present in the log |

### Example Usage

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

### Running the Example Script

`data_reader/matching/pairs.py` contains a `__main__` block that demonstrates the matching and filtering flow. Update the paths to your environment and run:

```
python data_reader/matching/pairs.py
```

### Requirements

- Python 3.9+
- NumPy

### Contributing / Notes

- Keep all parsing rules in the `parsing` subpackage to ensure consistency.
- Avoid duplicating filesystem traversal logic; instead, extend `matching/pairs.py`.
- When adding new filters or matchers, prefer small, composable helpers with docstrings following the style in the existing modules.

