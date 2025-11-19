"""Configuration file for CUAV Field Tests Data Reader.

This module centralizes all configuration parameters including file paths,
tolerance values, and other settings used throughout the project.

Configuration is loaded from config.txt file (simple key=value format).
If config.txt doesn't exist, default values are used.

To use this configuration:
    from config import Config
    
    # Access configuration values
    processed_root = Config.PROCESSED_ROOT
    raw_root = Config.RAW_ROOT
    log_file = Config.LOG_FILE
    
    # Or update values programmatically if needed
    Config.PROCESSED_ROOT = r"G:\\Your\\New\\Path"
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

def _load_txt_config(config_path: Path, silent: bool = False) -> dict | None:
    """Load configuration from text file (key=value format)."""
    if not config_path.exists():
        return None
    
    config_dict = {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                # Remove comments and whitespace
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Parse key=value
                if "=" not in line:
                    if not silent:
                        print(f"⚠ Warning: Invalid line {line_num} in {config_path}: {line}")
                    continue
                
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                config_dict[key] = value
        
        return config_dict
    except Exception as e:
        if not silent:
            print(f"⚠ Error loading text config: {e}")
        return None


def _apply_txt_config(config_dict: dict) -> None:
    """Apply text file configuration to Config class."""
    if not config_dict:
        return
    
    # Paths
    if "processed_root" in config_dict:
        Config.PROCESSED_ROOT = config_dict["processed_root"]
    if "raw_root" in config_dict:
        Config.RAW_ROOT = config_dict["raw_root"]
    if "log_file" in config_dict:
        Config.LOG_FILE = config_dict["log_file"]
    if "output_dir" in config_dict:
        Config.OUTPUT_DIR = config_dict["output_dir"]
    if "database_path" in config_dict:
        db_path = config_dict["database_path"]
        Config.DATABASE_PATH = db_path if db_path.lower() not in ("null", "none", "") else None
    
    # Processing parameters
    if "timestamp_tolerance" in config_dict:
        Config.TIMESTAMP_TOLERANCE = float(config_dict["timestamp_tolerance"])
    if "timestamp_precision" in config_dict:
        Config.TIMESTAMP_PRECISION = int(config_dict["timestamp_precision"])
    
    # File patterns
    if "processed_suffix" in config_dict:
        Config.PROCESSED_SUFFIX = config_dict["processed_suffix"]
    if "raw_file_pattern" in config_dict:
        Config.RAW_FILE_PATTERN = config_dict["raw_file_pattern"]
    
    # Raw data parameters
    if "raw_spectra_skip_rows" in config_dict:
        Config.RAW_SPECTRA_SKIP_ROWS = int(config_dict["raw_spectra_skip_rows"])
    if "raw_spectra_dtype" in config_dict:
        Config.RAW_SPECTRA_DTYPE = config_dict["raw_spectra_dtype"]
    
    # Processed data parameters
    if "processed_timestamp_column" in config_dict:
        Config.PROCESSED_TIMESTAMP_COLUMN = int(config_dict["processed_timestamp_column"])
    if "processed_data_start_column" in config_dict:
        Config.PROCESSED_DATA_START_COLUMN = int(config_dict["processed_data_start_column"])
    
    # Log file parameters
    # Note: config file uses column names, but internally we use row indices (after transpose)
    if "log_azimuth_column" in config_dict:
        Config.LOG_AZIMUTH_ROW = int(config_dict["log_azimuth_column"])
    elif "log_azimuth_row" in config_dict:  # Backward compatibility
        Config.LOG_AZIMUTH_ROW = int(config_dict["log_azimuth_row"])
    if "log_elevation_column" in config_dict:
        Config.LOG_ELEVATION_ROW = int(config_dict["log_elevation_column"])
    elif "log_elevation_row" in config_dict:  # Backward compatibility
        Config.LOG_ELEVATION_ROW = int(config_dict["log_elevation_row"])
    if "log_timestamp_column" in config_dict:
        Config.LOG_TIMESTAMP_ROW = int(config_dict["log_timestamp_column"])
    elif "log_timestamp_row" in config_dict:  # Backward compatibility
        Config.LOG_TIMESTAMP_ROW = int(config_dict["log_timestamp_row"])
    
    # Test parameters
    if "max_test_entries" in config_dict:
        max_entries = config_dict["max_test_entries"]
        Config.MAX_TEST_ENTRIES = int(max_entries) if max_entries.lower() not in ("null", "none", "") else None
    
    # Visualization parameters
    if "range_step" in config_dict:
        Config.RANGE_STEP = float(config_dict["range_step"])
    if "starting_range" in config_dict:
        Config.STARTING_RANGE = float(config_dict["starting_range"])
    if "requested_ranges" in config_dict:
        # Parse comma-separated list
        ranges_str = config_dict["requested_ranges"]
        Config.REQUESTED_RANGES = ranges_str
    if "visualization_output_dir" in config_dict:
        Config.VISUALIZATION_OUTPUT_DIR = config_dict["visualization_output_dir"]
    
    # Main script execution mode
    if "run_mode" in config_dict:
        Config.RUN_MODE = config_dict["run_mode"].lower().strip()
    if "heatmap_parameters" in config_dict:
        # Parse comma-separated list
        params_str = config_dict["heatmap_parameters"]
        Config.HEATMAP_PARAMETERS = params_str
    if "heatmap_colormap" in config_dict:
        Config.HEATMAP_COLORMAP = config_dict["heatmap_colormap"]
    if "heatmap_format" in config_dict:
        Config.HEATMAP_FORMAT = config_dict["heatmap_format"]


class Config:
    """Configuration class containing all project settings."""

    # ============================================================================
    # Directory Paths
    # ============================================================================
    
    # Root directory containing processed data files
    # Structure: Wind/YYYY-MM-DD/MM-DD_HHh/MM-DD_HH-##/
    # Each leaf folder contains: _Peak.txt, _Spectrum.txt, _Wind.txt
    PROCESSED_ROOT: ClassVar[str] = r"G:\Raymetrics_Tests\BOMA2025\20250922\Wind"
    
    # Root directory containing raw spectra files
    # Structure: Wind/YYYY-MM-DD/MM-DD_HHh/MM-DD_HH-##/
    # Each leaf folder contains: spectra_*.txt files
    RAW_ROOT: ClassVar[str] = r"G:\Raymetrics_Tests\BOMA2025\20250922\Spectra\User\20250922\Wind"
    
    # Path to the log file containing azimuth, elevation, and timestamps
    # Format: Three rows - [azimuth, elevation, timestamps, ...]
    LOG_FILE: ClassVar[str] = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"
    
    # Output directory for debug files, CSV exports, etc.
    OUTPUT_DIR: ClassVar[str] = "timestamp_debug_output"
    
    # Database file path for storing aggregated data
    # If None, database storage is disabled
    DATABASE_PATH: ClassVar[str | None] = "data/cuav_data.db"
    
    # ============================================================================
    # Processing Parameters
    # ============================================================================
    
    # Absolute tolerance for timestamp matching (in seconds)
    # Used when comparing timestamps from different sources
    # Default: 0.0001 seconds (0.1 milliseconds)
    TIMESTAMP_TOLERANCE: ClassVar[float] = 0.0001
    
    # Number of decimal places to normalize timestamps for comparison
    # Helps with floating-point precision issues
    TIMESTAMP_PRECISION: ClassVar[int] = 6
    
    # ============================================================================
    # File Patterns
    # ============================================================================
    
    # Pattern for processed data files (used in matching)
    PROCESSED_SUFFIX: ClassVar[str] = "_Peak.txt"
    
    # Pattern for raw spectra files
    RAW_FILE_PATTERN: ClassVar[str] = "spectra_*.txt"
    
    # ============================================================================
    # Raw Data Reading Parameters
    # ============================================================================
    
    # Number of lines to skip at the beginning of raw spectra files
    # before reading numeric data
    RAW_SPECTRA_SKIP_ROWS: ClassVar[int] = 13
    
    # Data type for reading raw spectra files
    RAW_SPECTRA_DTYPE: ClassVar[str] = "float32"
    
    # ============================================================================
    # Processed Data Reading Parameters
    # ============================================================================
    
    # Column index containing timestamps in processed files
    # For _Peak.txt, _Spectrum.txt, _Wind.txt: timestamp is in column 2 (0-indexed)
    PROCESSED_TIMESTAMP_COLUMN: ClassVar[int] = 2
    
    # Column index where data values start in processed files
    # Data starts from column 3 (0-indexed) onwards
    PROCESSED_DATA_START_COLUMN: ClassVar[int] = 3
    
    # ============================================================================
    # Log File Parameters
    # ============================================================================
    
    # Row indices in log file (after transpose)
    # Note: In the original log file, these are COLUMNS:
    #   Column 0: azimuth angles
    #   Column 1: elevation angles
    #   Column 2: timestamps
    # After loading with np.loadtxt().T, columns become rows, so:
    #   Row 0: azimuth angles (from column 0)
    #   Row 1: elevation angles (from column 1)
    #   Row 2: timestamps (from column 2)
    LOG_AZIMUTH_ROW: ClassVar[int] = 0
    LOG_ELEVATION_ROW: ClassVar[int] = 1
    LOG_TIMESTAMP_ROW: ClassVar[int] = 2
    
    # ============================================================================
    # Test Parameters
    # ============================================================================
    
    # Maximum number of entries to process in aggregation test
    # Set to None to process all entries
    MAX_TEST_ENTRIES: ClassVar[int | None] = 10
    
    # ============================================================================
    # Visualization Parameters
    # ============================================================================
    
    # Range resolution parameters for heatmap generation
    # Range step: spacing between range bins in meters
    RANGE_STEP: ClassVar[float] = 48.0
    
    # Starting range: range corresponding to first bin in meters
    STARTING_RANGE: ClassVar[float] = -1400.0
    
    # Requested ranges: comma-separated list of ranges to visualize (in meters)
    REQUESTED_RANGES: ClassVar[str] = "100,200,300"
    
    # Output directory for visualization files
    VISUALIZATION_OUTPUT_DIR: ClassVar[str] = "visualization_output"
    
    # ============================================================================
    # Main Script Execution Parameters
    # ============================================================================
    
    # Run mode: 'test' or 'heatmaps'
    # 'test': Run the complete test suite (total_test.py)
    # 'heatmaps': Generate heatmaps using parameters below
    RUN_MODE: ClassVar[str] = "test"
    
    # Heatmap parameters (comma-separated list: 'wind', 'snr', 'peak', 'spectrum')
    # Only used when run_mode='heatmaps'
    # Examples: "wind,snr" or "wind,peak,spectrum"
    HEATMAP_PARAMETERS: ClassVar[str] = "wind,peak,spectrum"
    
    # Matplotlib colormap for heatmaps (default: 'viridis')
    # Options: viridis, plasma, inferno, magma, coolwarm, etc.
    HEATMAP_COLORMAP: ClassVar[str] = "viridis"
    
    # Image format for heatmaps (default: 'png')
    # Options: png, pdf, svg, jpg, jpeg
    HEATMAP_FORMAT: ClassVar[str] = "png"
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    @classmethod
    def get_processed_root_path(cls) -> Path:
        """Get processed root as Path object."""
        return Path(cls.PROCESSED_ROOT).expanduser().resolve()
    
    @classmethod
    def get_raw_root_path(cls) -> Path:
        """Get raw root as Path object."""
        return Path(cls.RAW_ROOT).expanduser().resolve()
    
    @classmethod
    def get_log_file_path(cls) -> Path:
        """Get log file as Path object."""
        return Path(cls.LOG_FILE).expanduser().resolve()
    
    @classmethod
    def get_output_dir_path(cls) -> Path:
        """Get output directory as Path object."""
        return Path(cls.OUTPUT_DIR).expanduser().resolve()
    
    @classmethod
    def get_database_path(cls) -> Path | None:
        """Get database path as Path object, or None if not configured."""
        if cls.DATABASE_PATH is None:
            return None
        return Path(cls.DATABASE_PATH).expanduser().resolve()
    
    @classmethod
    def get_visualization_output_dir_path(cls) -> Path:
        """Get visualization output directory as Path object."""
        return Path(cls.VISUALIZATION_OUTPUT_DIR).expanduser().resolve()
    
    @classmethod
    def get_requested_ranges(cls) -> list[float]:
        """Parse requested_ranges string into list of floats."""
        ranges_str = cls.REQUESTED_RANGES.strip()
        if not ranges_str:
            return []
        try:
            return [float(r.strip()) for r in ranges_str.split(",") if r.strip()]
        except ValueError:
            return []
    
    @classmethod
    def get_heatmap_parameters(cls) -> list[str]:
        """Parse heatmap_parameters string into list of strings."""
        params_str = cls.HEATMAP_PARAMETERS.strip()
        if not params_str:
            return []
        return [p.strip() for p in params_str.split(",") if p.strip()]
    
    @classmethod
    def validate_paths(cls) -> tuple[bool, list[str]]:
        """
        Validate that all configured paths exist.
        
        Returns
        -------
        tuple[bool, list[str]]
            (all_valid, list_of_errors)
        """
        errors = []
        
        processed_path = cls.get_processed_root_path()
        if not processed_path.exists():
            errors.append(f"Processed root does not exist: {processed_path}")
        
        raw_path = cls.get_raw_root_path()
        if not raw_path.exists():
            errors.append(f"Raw root does not exist: {raw_path}")
        
        log_path = cls.get_log_file_path()
        if not log_path.exists():
            errors.append(f"Log file does not exist: {log_path}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def load_from_file(cls, config_file: str | Path | None = None, silent: bool = False) -> None:
        """
        Load configuration from text file (key=value format).
        
        Parameters
        ----------
        config_file : str | Path | None
            Path to config text file. If None, looks for config.txt in project root.
        silent : bool
            If True, suppress print statements (useful for auto-loading on import).
        """
        if config_file is None:
            # Look for config.txt in the same directory as this file
            config_file = Path(__file__).parent / "config.txt"
        else:
            config_file = Path(config_file)
        
        config_dict = _load_txt_config(config_file, silent=silent)
        if config_dict:
            _apply_txt_config(config_dict)
            if not silent:
                print(f"✓ Loaded configuration from {config_file}")
        else:
            if not silent:
                if config_file.exists():
                    print(f"⚠ Could not load configuration from {config_file}")
                else:
                    print(f"⚠ Configuration file not found: {config_file}")
                    print("  Using default configuration values")
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration values."""
        print("=" * 70)
        print("Configuration")
        print("=" * 70)
        print(f"Processed Root: {cls.PROCESSED_ROOT}")
        print(f"Raw Root: {cls.RAW_ROOT}")
        print(f"Log File: {cls.LOG_FILE}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"Database Path: {cls.DATABASE_PATH}")
        print(f"Timestamp Tolerance: {cls.TIMESTAMP_TOLERANCE}")
        print(f"Timestamp Precision: {cls.TIMESTAMP_PRECISION}")
        print(f"Processed Suffix: {cls.PROCESSED_SUFFIX}")
        print(f"Raw File Pattern: {cls.RAW_FILE_PATTERN}")
        print(f"Raw Spectra Skip Rows: {cls.RAW_SPECTRA_SKIP_ROWS}")
        print(f"Processed Timestamp Column: {cls.PROCESSED_TIMESTAMP_COLUMN}")
        print(f"Processed Data Start Column: {cls.PROCESSED_DATA_START_COLUMN}")
        print(f"Max Test Entries: {cls.MAX_TEST_ENTRIES}")
        print(f"Range Step: {cls.RANGE_STEP} m")
        print(f"Starting Range: {cls.STARTING_RANGE} m")
        print(f"Requested Ranges: {cls.REQUESTED_RANGES}")
        print(f"Visualization Output Dir: {cls.VISUALIZATION_OUTPUT_DIR}")
        print(f"Run Mode: {cls.RUN_MODE}")
        if cls.RUN_MODE == "heatmaps":
            print(f"Heatmap Parameters: {cls.HEATMAP_PARAMETERS}")
            print(f"Heatmap Colormap: {cls.HEATMAP_COLORMAP}")
            print(f"Heatmap Format: {cls.HEATMAP_FORMAT}")
        print("=" * 70)
        
        # Validate paths
        all_valid, errors = cls.validate_paths()
        if all_valid:
            print("✓ All paths are valid")
        else:
            print("⚠ Path validation errors:")
            for error in errors:
                print(f"  - {error}")
        print("=" * 70)


# Auto-load configuration from YAML file on import (silently)
Config.load_from_file(silent=True)


# ============================================================================
# Convenience Functions
# ============================================================================

def load_config_from_file(config_file: str | Path) -> None:
    """
    Load configuration from a Python file.
    
    This allows users to create their own config file and load it.
    The config file should define a Config class or update the global Config.
    
    Parameters
    ----------
    config_file : str | Path
        Path to the configuration file to load.
    """
    import importlib.util
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("user_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config file: {config_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # If the module defines a Config class, update our Config
    if hasattr(module, "Config"):
        user_config = module.Config
        for attr in dir(user_config):
            if not attr.startswith("_") and hasattr(Config, attr):
                setattr(Config, attr, getattr(user_config, attr))

