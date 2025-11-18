"""Example configuration file for CUAV Field Tests Data Reader.

Copy this file to `config.py` and update the paths to match your setup.

Usage:
    1. Copy this file: cp config.example.py config.py
    2. Edit config.py with your paths
    3. Run your scripts - they will automatically use the config
"""

from config import Config

# ============================================================================
# Update these paths to match your setup
# ============================================================================

# Root directory containing processed data files
Config.PROCESSED_ROOT = r"G:\Raymetrics_Tests\BOMA2025\20250922\Wind"

# Root directory containing raw spectra files
Config.RAW_ROOT = r"G:\Raymetrics_Tests\BOMA2025\20250922\Spectra\User\20250922\Wind"

# Path to the log file containing azimuth, elevation, and timestamps
Config.LOG_FILE = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"

# Output directory for debug files, CSV exports, etc.
Config.OUTPUT_DIR = "timestamp_debug_output"

# ============================================================================
# Optional: Adjust processing parameters if needed
# ============================================================================

# Absolute tolerance for timestamp matching (in seconds)
# Default: 0.0001 seconds (0.1 milliseconds)
# Config.TIMESTAMP_TOLERANCE = 0.0001

# Maximum number of entries to process in aggregation test
# Set to None to process all entries
# Config.MAX_TEST_ENTRIES = 10

