## CUAV Field Tests Data Reader

This package provides utilities for working with Raymetrics CUAV field-test data.
It focuses on four main workflows:

1. **Parsing** spectra filenames and log files to extract timestamps.
2. **Reading** processed and raw data files from disk into NumPy arrays.
3. **Processing** processed text arrays so they stay aligned with the log.
4. **Matching** processed timestamps with raw spectra files across mirrored trees.

### Configuration

All paths and parameters are centralized in `config.txt` (simple text file with key=value pairs). This makes it easy to configure without editing Python code.

**Setup:**

1. Copy the example config file:
   ```bash
   cp config.txt.example config.txt
   ```

2. Edit `config.txt` with your paths:
   ```txt
   # Directory Paths
   processed_root=G:\Your\Path\To\Processed\Data
   raw_root=G:\Your\Path\To\Raw\Data
   log_file=G:\Your\Path\To\output.txt
   database_path=data/cuav_data.db
   
   # Processing Parameters
   timestamp_tolerance=0.0001
   ```

3. Run your scripts - they will automatically load from `config.txt`:
   ```bash
   # Run all tests to verify project functionality (database created automatically)
   python main.py --test
   
   # Generate heatmaps for specific parameters and ranges (database created automatically)
   python main.py --heatmaps --parameters wind snr --ranges 100 200 300
   
   # Or run the test suite directly
   python total_test.py
   ```

**Note:** The database is automatically created or rebuilt when running `main.py` in either test or heatmaps mode. If a database already exists, it will be replaced with a fresh one containing all filtered matches.

**Configuration File Format:**

The `config.txt` file uses a simple `key=value` format:
- Lines starting with `#` are comments and ignored
- Empty lines are ignored
- Each parameter is on its own line: `key=value`
- Paths can include spaces (no quotes needed, but quotes are stripped if present)

**Available Configuration Parameters:**

- `processed_root`: Root directory containing processed data files
- `raw_root`: Root directory containing raw spectra files
- `log_file`: Path to the log file (output.txt)
- `output_dir`: Output directory for debug files
- `database_path`: Database file path (or `null` to disable)
- `timestamp_tolerance`: Tolerance for timestamp matching (default: 0.0001)
- `timestamp_precision`: Decimal places for timestamp normalization (default: 6)
- `processed_suffix`: Suffix for processed files (default: "_Peak.txt")
- `raw_file_pattern`: Pattern for raw files (default: "spectra_*.txt")
- `raw_spectra_skip_rows`: Lines to skip in raw files (default: 13)
- `processed_timestamp_column`: Column index for timestamps (default: 2)
- `processed_data_start_column`: Column index where data starts (default: 3)
- `max_test_entries`: Max entries for testing (default: 10, or `null` for all)
- `range_step`: Spacing between range bins for visualization (default: 48.0 m)
- `starting_range`: Starting range for range-resolved profiles (default: -1400.0 m)
- `requested_ranges`: Comma-separated list of ranges to visualize (default: "100,200,300")
- `visualization_output_dir`: Output directory for heatmap images (default: "visualization_output")
- `run_mode`: Main script execution mode ('test', 'heatmaps', or 'profiles', default: 'test')
- `heatmap_parameters`: Comma-separated list of parameters for heatmaps (e.g., "wind,snr", default: "wind,peak,spectrum")
- `heatmap_colormap`: Matplotlib colormap for heatmaps (default: "viridis")
- `heatmap_format`: Image format for heatmaps (default: "png")
- `profile_fft_size`: FFT size for frequency computation in Mode 3 (default: 128)
- `profile_sampling_rate`: Sampling rate in Hz for frequency computation in Mode 3 (default: 100000.0)
- `profile_frequency_interval`: Allowable frequency interval for max SNR search in Mode 3, format "min,max" in Hz (default: "0,50000")
- `profile_frequency_shift`: Frequency shift for Doppler lidar equation in Mode 3 (default: 0.0 Hz)
- `profile_laser_wavelength`: Laser wavelength in meters for Doppler lidar equation in Mode 3 (default: 1.55e-6)

**Configuration Validation:**

You can validate your configuration:
```python
from config import Config

# Load from file (if not already loaded)
Config.load_from_file()

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
│   ├── aggregation.py  # aggregate data from multiple sources
│   ├── filters.py      # filter processed arrays by log timestamps
│   └── integration.py  # integration with database storage
├── storage/
│   └── database.py  # SQLite database for persistent storage
├── matching/
│   └── pairs.py     # traverse trees and align processed/raw timestamps
├── analysis/
│   └── visualization.py  # heatmap generation and analysis tools
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
| `build_and_save_to_database(...)` | Build aggregated data and save to database in one step |
| `load_from_database(db_path, ...)` | Load data from database back into dictionary format |
| `query_timestamp(timestamp, db_path)` | Query a single timestamp from database |
| `query_timestamp_range(db_path, start, end, limit)` | Query a range of timestamps from database |
| `init_database(db_path)` | Initialize a new database with tables |
| `DataDatabase` | Database class for advanced operations |
| `filter_processed_array_by_timestamps(array, log_path)` | Keep only processed rows with timestamps present in the log |
| `create_heatmaps(db_path, range_step, starting_range, requested_ranges, ...)` | Generate heatmaps for range-resolved parameters |
| `extract_range_values(profile_array, range_step, starting_range, requested_ranges)` | Extract values from range-resolved profile at specific ranges |
| `aggregate_azimuth_elevation_data(data_records, parameter, ...)` | Aggregate data by azimuth/elevation for visualization |

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

**Storing data in database:**
```python
from data_reader import build_and_save_to_database, query_timestamp, load_from_database
from config import Config

# Build and save to database in one step
count = build_and_save_to_database(
    timestamp_path_pairs,
    processed_root,
    raw_root,
    log_file,
    db_path=Config.get_database_path(),
    atol=0.0001,
)
print(f"Saved {count} entries to database")

# Query a specific timestamp
data = query_timestamp("3841395567.560425", Config.get_database_path())
if data:
    print(f"Azimuth: {data['azimuth']}")
    print(f"Peak data shape: {data['peak'].shape if data['peak'] is not None else None}")

# Load a range of timestamps
data_range = load_from_database(
    Config.get_database_path(),
    start_timestamp="3841395567.0",
    end_timestamp="3841395570.0",
    limit=100,
)
print(f"Loaded {len(data_range)} entries")
```

**Visualizing data with heatmaps:**
```python
from data_reader import create_heatmaps
from config import Config

# Get visualization parameters from config
db_path = Config.get_database_path()
range_step = Config.RANGE_STEP  # 48.0 m
starting_range = Config.STARTING_RANGE  # -1400.0 m
requested_ranges = Config.get_requested_ranges()  # [100, 200, 300] m
output_dir = Config.get_visualization_output_dir_path()

# Generate heatmaps for wind, peak, and spectrum at specified ranges
results = create_heatmaps(
    db_path=db_path,
    range_step=range_step,
    starting_range=starting_range,
    requested_ranges=requested_ranges,
    parameters=["wind", "peak", "spectrum"],
    output_dir=output_dir,
    colormap="viridis",
    save_format="png",
)

# Results contain gridded data and raw scatter points
for key, data in results.items():
    print(f"{data['parameter']} at {data['range']} m: {len(data['azimuth'])} points")
```

### Main Script Usage

The `main.py` script provides three mutually exclusive modes:

**1. Test Mode (Mode 1):**
```bash
python main.py --test
```
This runs the complete test suite (`total_test.py`) to verify project functionality, including:
- Automatic database creation/rebuild (replaces existing database if present)
- Spectra parsing
- Log file reading
- Processed/raw data matching
- Filtering by log timestamps
- Data aggregation
- Database storage
- Visualization

**2. Heatmap Mode (Mode 2):**
```bash
# Generate heatmaps for wind and SNR at specific ranges (database created automatically)
python main.py --heatmaps --parameters wind snr --ranges 100 200 300

# Use default parameters and ranges from config.txt (database created automatically)
python main.py --heatmaps

# Generate heatmaps for wind only at range 150m
python main.py --heatmaps --parameters wind --ranges 150

# Custom output directory and colormap
python main.py --heatmaps --parameters wind peak --ranges 100 200 --output-dir my_heatmaps --colormap plasma
```

**3. Single-Profile Mode (Mode 3):**
```bash
# Generate single-profile visualizations (SNR and wind profiles)
python main.py --profiles
```

Mode 3 performs the following operations:
- Processes range-resolved power density spectra for each timestamp
- Computes frequencies from FFT size and sampling rate (specified in config.txt)
- For each range, finds the frequency at which maximum SNR occurs (within allowable frequency interval)
- Computes wind speed using the coherent Doppler lidar equation: `v = laser_wavelength * (dominant_frequency - frequency_shift) / 2` (result in m/s)
- Stores SNR and wind profiles in database (named after logfile, excluding extension)
- Generates two visualizations:
  - One plot containing all SNR profiles (one line per timestamp)
  - One plot containing all wind profiles (one line per timestamp)

**Output Directory Structure:**
Both Mode 2 (heatmaps) and Mode 3 (profiles) outputs are saved in subdirectories named after the logfile (without extension) inside `visualization_output`:
```
visualization_output/
  <logfile_basename>/
    <heatmap files>  (Mode 2)
    snr_profiles.png  (Mode 3)
    wind_profiles.png (Mode 3)
```

**Running from IDE:**
When running `main.py` directly from an IDE without command-line arguments, set `run_mode` in `config.txt`:
```txt
run_mode=test        # Run test suite
# or
run_mode=heatmaps    # Generate heatmaps (uses heatmap_parameters from config.txt)
# or
run_mode=profiles    # Generate single-profile visualizations (Mode 3)
```

The mode from `config.txt` will be used when no command-line arguments are provided. Command-line arguments (`--test`, `--heatmaps`, or `--profiles`) always take precedence over the config file. **Modes are mutually exclusive: only one mode runs at a time.**

**Parameter Mapping:**
- `snr` or `peak` → SNR data from `_Peak.txt`
- `wind` → Wind data from `_Wind.txt`
- `spectrum` → Spectrum data from `_Spectrum.txt`

**Available Options:**
- `--test`: Run the complete test suite (Mode 1)
- `--heatmaps`: Generate heatmaps (Mode 2)
- `--profiles`: Generate single-profile visualizations (Mode 3)
- `--parameters`, `-p`: Parameters to visualize (snr, peak, wind, spectrum) - Mode 2 only
- `--ranges`, `-r`: Ranges in meters (comma or space separated, e.g., "100,200,300") - Mode 2 only
- `--output-dir`, `-o`: Output directory for visualization images
- `--colormap`, `-c`: Matplotlib colormap (default: viridis) - Mode 2 only
- `--format`, `-f`: Image format (png, pdf, svg, jpg; default: png)
- `--config`: Path to config.txt file

**Help:**
```bash
python main.py --help
```

**Heatmap Output:**
When generating heatmaps, the system:
- Prints dataset information for each heatmap (number of points, first 10 data points)
- Saves heatmap images to the output directory (PNG, PDF, SVG, etc.)
- Displays heatmaps interactively in matplotlib windows (close each window to see the next)

### Database Storage

The project includes SQLite-based persistent storage for aggregated data. This provides a robust, scalable solution for managing time-series data from CUAV field tests.

**Automatic Database Creation:**
The database is automatically created or rebuilt when running `main.py` in either test or heatmaps mode. If an existing database is found, it will be replaced with a fresh one containing all filtered matches from the current data sources. This ensures that the database always reflects the most current data configuration.

#### Benefits

- **Persistent storage**: Data survives between program runs, enabling long-term analysis
- **Efficient queries**: Fast timestamp-based lookups and range queries with indexed access
- **Scalable**: Handles large datasets efficiently (tested with thousands of timestamps)
- **Metadata tracking**: Stores source file paths, import dates, and update timestamps
- **Data integrity**: Foreign key constraints ensure referential integrity
- **Easy export**: Can export to CSV, JSON, or other formats using standard SQL tools
- **Concurrent access**: SQLite supports multiple readers simultaneously
- **No dependencies**: Uses Python's built-in `sqlite3` module

#### Database Schema

The database uses a normalized schema with 5 tables:

**`timestamps` (Primary Table)**
- `timestamp` (REAL, PRIMARY KEY): Processed timestamp
- `azimuth` (REAL): Azimuth angle from log file
- `elevation` (REAL): Elevation angle from log file
- `source_processed_dir` (TEXT): Source processed directory path
- `source_raw_dir` (TEXT): Source raw directory path
- `source_log_file` (TEXT): Source log file path
- `imported_at` (TIMESTAMP): When the record was first imported
- `updated_at` (TIMESTAMP): When the record was last updated

**Array Data Tables** (Foreign key to `timestamps.timestamp`)
- `peak_data`: Stores peak array data as JSON
- `spectrum_data`: Stores spectrum array data as JSON
- `wind_data`: Stores wind array data as JSON
- `power_density_spectrum`: Stores raw spectra data as JSON
- `snr_profile`: Stores computed SNR profiles as JSON (Mode 3)
- `wind_profile`: Stores computed wind profiles as JSON (Mode 3)

All array data tables use `ON DELETE CASCADE` to maintain referential integrity.

#### Configuration

Configure the database path in `config.txt`:
```txt
database_path=data/cuav_data.db  # or null to disable
```

The database file will be created automatically when first used (when running `main.py` in test or heatmaps mode). The parent directory will be created if it doesn't exist.

#### Query Examples

**Get statistics about the database:**
```python
from data_reader import init_database

db = init_database("data/cuav_data.db")
stats = db.get_statistics()
print(f"Total timestamps: {stats['total_timestamps']}")
print(f"With peak data: {stats['count_with_peak']}")
print(f"Timestamp range: {stats['timestamp_range']['min']} to {stats['timestamp_range']['max']}")
db.close()
```

**Query specific timestamp:**
```python
from data_reader import query_timestamp

data = query_timestamp("3841395567.560425", "data/cuav_data.db")
if data:
    print(f"Azimuth: {data['azimuth']}")
    print(f"Peak data: {data['peak']}")
```

**Query timestamp range:**
```python
from data_reader import query_timestamp_range

# Get all data between two timestamps
data = query_timestamp_range(
    "data/cuav_data.db",
    start_timestamp="3841395567.0",
    end_timestamp="3841395570.0",
    limit=100,  # Optional: limit number of results
)
```

**Advanced queries using SQL:**
```python
import sqlite3

conn = sqlite3.connect("data/cuav_data.db")
cursor = conn.cursor()

# Find all timestamps with azimuth > 90 degrees
cursor.execute("""
    SELECT timestamp, azimuth, elevation 
    FROM timestamps 
    WHERE azimuth > 90.0
    ORDER BY timestamp
""")
results = cursor.fetchall()

# Count entries by data availability
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(peak_data.timestamp) as with_peak,
        COUNT(spectrum_data.timestamp) as with_spectrum
    FROM timestamps
    LEFT JOIN peak_data ON timestamps.timestamp = peak_data.timestamp
    LEFT JOIN spectrum_data ON timestamps.timestamp = spectrum_data.timestamp
""")
stats = cursor.fetchone()

conn.close()
```

#### Performance Considerations

- **Indexing**: The `timestamp` column is automatically indexed (PRIMARY KEY)
- **Array storage**: Large arrays are stored as JSON strings, which is efficient for SQLite
- **Batch operations**: Use `build_and_save_to_database()` for bulk inserts (faster than individual inserts)
- **Connection management**: Use context managers (`with` statement) for automatic connection handling

#### Data Migration and Backup

**Backup the database:**
```bash
# Simple file copy (SQLite is a single file)
cp data/cuav_data.db data/cuav_data_backup.db
```

**Export to CSV:**
```python
import sqlite3
import csv

conn = sqlite3.connect("data/cuav_data.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM timestamps")
with open("export.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([desc[0] for desc in cursor.description])
    writer.writerows(cursor.fetchall())

conn.close()
```

#### Best Practices

1. **Regular backups**: SQLite databases are single files - easy to backup
2. **Use transactions**: The database automatically uses transactions for data integrity
3. **Close connections**: Always close database connections when done (or use context managers)
4. **Monitor size**: Large datasets may result in large database files (monitor disk space)
5. **Index optimization**: The primary key on `timestamp` provides fast lookups automatically

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
- NumPy >= 1.20.0
- matplotlib >= 3.5.0 (for visualization)
- schedule >= 1.2.0 (optional, for better scheduling)
- reportlab >= 3.6.0 (optional, for PDF documentation generation)

**Install using conda (recommended):**
```bash
conda env create -f environment.yml
conda activate cuav_field_tests
```

**Or install using pip:**
```bash
pip install -r requirements.txt
```

**Note:** `sqlite3` is included in Python's standard library and does not need to be installed separately.

### PDF Documentation

The project includes a script to generate comprehensive PDF documentation describing the project's architecture, data flow, methods, and usage.

**Generate PDF documentation:**
```bash
python generate_project_documentation.py
```

This creates `project_documentation.pdf` in the project root with:
- Project overview and features
- System architecture and module descriptions
- Data flow description
- Core methods and functions
- Data processing pipeline details
- Database storage schema
- Analysis and visualization capabilities
- Configuration parameters
- Usage examples

**Custom output file:**
```bash
python generate_project_documentation.py my_documentation.pdf
```

**Requirements:** `reportlab` library (included in requirements.txt and environment.yml)

### Contributing / Notes

- Keep all parsing rules in the `parsing` subpackage to ensure consistency.
- Avoid duplicating filesystem traversal logic; instead, extend `matching/pairs.py`.
- When adding new filters or matchers, prefer small, composable helpers with docstrings following the style in the existing modules.

