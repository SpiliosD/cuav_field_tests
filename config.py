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
    
    # Paths - support single values or comma-separated lists
    if "processed_root" in config_dict:
        value = config_dict["processed_root"].strip()
        if value:
            # Check if comma-separated (multiple values)
            if "," in value:
                Config.PROCESSED_ROOTS = [p.strip() for p in value.split(",") if p.strip()]
            else:
                Config.PROCESSED_ROOTS = [value]
        else:
            Config.PROCESSED_ROOTS = []
    
    if "raw_root" in config_dict:
        value = config_dict["raw_root"].strip()
        if value:
            if "," in value:
                Config.RAW_ROOTS = [p.strip() for p in value.split(",") if p.strip()]
            else:
                Config.RAW_ROOTS = [value]
        else:
            Config.RAW_ROOTS = []
    
    if "log_file" in config_dict:
        value = config_dict["log_file"].strip()
        if value:
            if "," in value:
                Config.LOG_FILES = [p.strip() for p in value.split(",") if p.strip()]
            else:
                Config.LOG_FILES = [value]
        else:
            Config.LOG_FILES = []
    
    if "output_dir" in config_dict:
        Config.OUTPUT_DIR = config_dict["output_dir"]
    
    if "database_path" in config_dict:
        value = config_dict["database_path"].strip()
        if value:
            if "," in value:
                db_paths = [p.strip() for p in value.split(",") if p.strip()]
                Config.DATABASE_PATHS = [p if p.lower() not in ("null", "none", "") else None for p in db_paths]
            else:
                db_path = value if value.lower() not in ("null", "none", "") else None
                Config.DATABASE_PATHS = [db_path] if db_path else []
        else:
            Config.DATABASE_PATHS = []
    
    # Legacy single-value support (for backward compatibility)
    # Set single values from lists if only one item
    if hasattr(Config, 'PROCESSED_ROOTS') and len(Config.PROCESSED_ROOTS) == 1:
        Config.PROCESSED_ROOT = Config.PROCESSED_ROOTS[0]
    elif hasattr(Config, 'PROCESSED_ROOTS') and len(Config.PROCESSED_ROOTS) > 1:
        Config.PROCESSED_ROOT = Config.PROCESSED_ROOTS[0]  # Default to first for backward compatibility
    
    if hasattr(Config, 'RAW_ROOTS') and len(Config.RAW_ROOTS) == 1:
        Config.RAW_ROOT = Config.RAW_ROOTS[0]
    elif hasattr(Config, 'RAW_ROOTS') and len(Config.RAW_ROOTS) > 1:
        Config.RAW_ROOT = Config.RAW_ROOTS[0]
    
    if hasattr(Config, 'LOG_FILES') and len(Config.LOG_FILES) == 1:
        Config.LOG_FILE = Config.LOG_FILES[0]
    elif hasattr(Config, 'LOG_FILES') and len(Config.LOG_FILES) > 1:
        Config.LOG_FILE = Config.LOG_FILES[0]
    
    if hasattr(Config, 'DATABASE_PATHS') and len(Config.DATABASE_PATHS) == 1:
        Config.DATABASE_PATH = Config.DATABASE_PATHS[0]
    elif hasattr(Config, 'DATABASE_PATHS') and len(Config.DATABASE_PATHS) > 1:
        Config.DATABASE_PATH = Config.DATABASE_PATHS[0]
    
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
    if "starting_range_index" in config_dict:
        Config.STARTING_RANGE_INDEX = int(config_dict["starting_range_index"])
    # Backward compatibility: if old "starting_range" is used, try to infer starting_range_index
    elif "starting_range" in config_dict:
        old_starting_range = float(config_dict["starting_range"])
        # If starting_range is negative, assume it's the distance at index 0
        # Then starting_range_index = -starting_range / range_step
        # For example, if starting_range=-1344 and range_step=48, then starting_range_index=28
        if old_starting_range < 0:
            Config.STARTING_RANGE_INDEX = int(-old_starting_range / Config.RANGE_STEP)
            print(f"⚠ Warning: 'starting_range' is deprecated. Inferred starting_range_index={Config.STARTING_RANGE_INDEX}. Use 'starting_range_index' instead.")
        else:
            Config.STARTING_RANGE_INDEX = 0
            print("⚠ Warning: 'starting_range' is deprecated. Use 'starting_range_index' instead.")
    if "requested_range_indices" in config_dict:
        ranges_str = config_dict["requested_range_indices"].strip()
        # Parse comma-separated list
        if ranges_str:
            Config.REQUESTED_RANGE_INDICES = ranges_str
    # Backward compatibility: if old "requested_ranges" is used, convert to relative indices
    elif "requested_ranges" in config_dict:
        old_ranges_str = config_dict["requested_ranges"].strip()
        if old_ranges_str:
            try:
                # Convert old distance-based ranges to relative indices
                old_ranges = [float(r.strip()) for r in old_ranges_str.split(",") if r.strip()]
                # Compute relative indices: range / range_step
                # These are offsets from starting_range_index
                relative_indices = [int(r / Config.RANGE_STEP) for r in old_ranges]
                Config.REQUESTED_RANGE_INDICES = ",".join(map(str, relative_indices))
                print("⚠ Warning: 'requested_ranges' is deprecated. Converted to relative indices. Use 'requested_range_indices' instead.")
            except (ValueError, ZeroDivisionError):
                Config.REQUESTED_RANGE_INDICES = "0,1,2"
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
    
    # Single-profile mode parameters
    if "profile_fft_size" in config_dict:
        Config.PROFILE_FFT_SIZE = int(config_dict["profile_fft_size"])
    if "profile_sampling_rate" in config_dict:
        Config.PROFILE_SAMPLING_RATE = float(config_dict["profile_sampling_rate"])
    if "profile_frequency_interval" in config_dict:
        Config.PROFILE_FREQUENCY_INTERVAL = config_dict["profile_frequency_interval"]
    if "profile_frequency_shift" in config_dict:
        Config.PROFILE_FREQUENCY_SHIFT = float(config_dict["profile_frequency_shift"])
    if "profile_laser_wavelength" in config_dict:
        Config.PROFILE_LASER_WAVELENGTH = float(config_dict["profile_laser_wavelength"])
    
    # Debug mode
    if "debug_mode" in config_dict:
        debug_str = config_dict["debug_mode"].lower().strip()
        Config.DEBUG_MODE = debug_str in ("true", "1", "yes", "on")


class Config:
    """Configuration class containing all project settings."""

    # ============================================================================
    # Directory Paths
    # ============================================================================
    
    # Root directory containing processed data files (single value, for backward compatibility)
    # Structure: Wind/YYYY-MM-DD/MM-DD_HHh/MM-DD_HH-##/
    # Each leaf folder contains: _Peak.txt, _Spectrum.txt, _Wind.txt
    PROCESSED_ROOT: ClassVar[str] = r"G:\Raymetrics_Tests\BOMA2025\20250922\Wind"
    
    # Multiple processed root directories (comma-separated in config.txt)
    # Each entry corresponds to one dataset configuration
    PROCESSED_ROOTS: ClassVar[list[str]] = []
    
    # Root directory containing raw spectra files (single value, for backward compatibility)
    # Structure: Wind/YYYY-MM-DD/MM-DD_HHh/MM-DD_HH-##/
    # Each leaf folder contains: spectra_*.txt files
    RAW_ROOT: ClassVar[str] = r"G:\Raymetrics_Tests\BOMA2025\20250922\Spectra\User\20250922\Wind"
    
    # Multiple raw root directories (comma-separated in config.txt)
    # Each entry corresponds to one dataset configuration
    RAW_ROOTS: ClassVar[list[str]] = []
    
    # Path to the log file containing azimuth, elevation, and timestamps (single value, for backward compatibility)
    # Format: Three rows - [azimuth, elevation, timestamps, ...]
    LOG_FILE: ClassVar[str] = r"G:\Raymetrics_Tests\BOMA2025\20250922\output.txt"
    
    # Multiple log file paths (comma-separated in config.txt)
    # Each entry corresponds to one dataset configuration
    LOG_FILES: ClassVar[list[str]] = []
    
    # Output directory for debug files, CSV exports, etc.
    OUTPUT_DIR: ClassVar[str] = "timestamp_debug_output"
    
    # Database file path for storing aggregated data (single value, for backward compatibility)
    # If None, database storage is disabled
    DATABASE_PATH: ClassVar[str | None] = "data/cuav_data.db"
    
    # Multiple database file paths (comma-separated in config.txt)
    # Each entry corresponds to one dataset configuration
    DATABASE_PATHS: ClassVar[list[str | None]] = []
    
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
    # Debug Mode
    # ============================================================================
    
    # Enable verbose debug output
    # When True: prints detailed debug information (database content, matching details, etc.)
    # When False: prints only essential information (normal mode)
    DEBUG_MODE: ClassVar[bool] = False
    
    # ============================================================================
    # Visualization Parameters
    # ============================================================================
    
    # Range resolution parameters for heatmap generation
    # Range step: spacing between range bins in meters
    RANGE_STEP: ClassVar[float] = 48.0
    
    # Starting range index: index of the range bin that corresponds to 0 meters
    # Previous bins have negative distances, later bins have positive distances
    # Formula: distance = (index - STARTING_RANGE_INDEX) * RANGE_STEP
    STARTING_RANGE_INDEX: ClassVar[int] = 0
    
    # Requested range indices: comma-separated list of range bin indices relative to starting_range_index
    # These are offsets from the starting_range_index (which is at 0 m)
    # The actual range bin indices are computed as: starting_range_index + requested_range_index
    REQUESTED_RANGE_INDICES: ClassVar[str] = "0,1,2"
    
    # Output directory for visualization files
    VISUALIZATION_OUTPUT_DIR: ClassVar[str] = "visualization_output"
    
    # ============================================================================
    # Main Script Execution Parameters
    # ============================================================================
    
    # Run mode: 'test', 'heatmaps', or 'profiles'
    # 'test': Run the complete test suite (tests/total_test.py)
    # 'heatmaps': Generate heatmaps using parameters below
    # 'profiles': Generate single-profile visualizations (Mode 3)
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
    # Single-Profile Mode Parameters (Mode 3)
    # ============================================================================
    
    # FFT size for frequency computation (e.g., 128 for 64 spectrum values)
    # This is the actual FFT size used, not the number of output bins
    PROFILE_FFT_SIZE: ClassVar[int] = 128
    
    # Sampling rate in Hz for frequency computation
    PROFILE_SAMPLING_RATE: ClassVar[float] = 100000.0  # 100 kHz default
    
    # Allowable frequency interval for searching maximum SNR (Hz)
    # Format: "min_freq,max_freq" (e.g., "0,50000")
    PROFILE_FREQUENCY_INTERVAL: ClassVar[str] = "0,50000"
    
    # Frequency shift for Doppler lidar equation (Hz)
    PROFILE_FREQUENCY_SHIFT: ClassVar[float] = 0.0
    
    # Laser wavelength for Doppler lidar equation (meters)
    PROFILE_LASER_WAVELENGTH: ClassVar[float] = 1.55e-6  # 1.55 micrometers default
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    @classmethod
    def _normalize_windows_path(cls, path_str: str) -> str:
        """
        Normalize Windows absolute paths that might be missing the colon.
        Handles cases like 'D\path\to\file' -> 'D:\path\to\file'
        """
        path_str = path_str.strip()
        # Check if path looks like a Windows drive letter without colon
        # Pattern: single letter followed by backslash or forward slash
        if len(path_str) >= 2 and path_str[0].isalpha() and path_str[1] in ('\\', '/'):
            # Missing colon, add it
            path_str = f"{path_str[0]}:{path_str[1:]}"
        return path_str
    
    @classmethod
    def get_processed_root_path(cls) -> Path:
        """Get processed root as Path object (returns first if multiple)."""
        # Use PROCESSED_ROOTS if available, otherwise fall back to PROCESSED_ROOT
        if hasattr(cls, 'PROCESSED_ROOTS') and len(cls.PROCESSED_ROOTS) > 0:
            normalized = cls._normalize_windows_path(cls.PROCESSED_ROOTS[0])
            return Path(normalized).expanduser().resolve()
        normalized = cls._normalize_windows_path(cls.PROCESSED_ROOT)
        return Path(normalized).expanduser().resolve()
    
    @classmethod
    def get_processed_root_paths(cls) -> list[Path]:
        """Get all processed root directories as Path objects."""
        if hasattr(cls, 'PROCESSED_ROOTS') and len(cls.PROCESSED_ROOTS) > 0:
            return [Path(cls._normalize_windows_path(p)).expanduser().resolve() for p in cls.PROCESSED_ROOTS]
        normalized = cls._normalize_windows_path(cls.PROCESSED_ROOT)
        return [Path(normalized).expanduser().resolve()]
    
    @classmethod
    def get_raw_root_path(cls) -> Path:
        """Get raw root as Path object (returns first if multiple)."""
        # Use RAW_ROOTS if available, otherwise fall back to RAW_ROOT
        if hasattr(cls, 'RAW_ROOTS') and len(cls.RAW_ROOTS) > 0:
            normalized = cls._normalize_windows_path(cls.RAW_ROOTS[0])
            return Path(normalized).expanduser().resolve()
        normalized = cls._normalize_windows_path(cls.RAW_ROOT)
        return Path(normalized).expanduser().resolve()
    
    @classmethod
    def get_raw_root_paths(cls) -> list[Path]:
        """Get all raw root directories as Path objects."""
        if hasattr(cls, 'RAW_ROOTS') and len(cls.RAW_ROOTS) > 0:
            return [Path(cls._normalize_windows_path(p)).expanduser().resolve() for p in cls.RAW_ROOTS]
        normalized = cls._normalize_windows_path(cls.RAW_ROOT)
        return [Path(normalized).expanduser().resolve()]
    
    @classmethod
    def get_log_file_path(cls) -> Path:
        """Get log file as Path object (returns first if multiple)."""
        # Use LOG_FILES if available, otherwise fall back to LOG_FILE
        if hasattr(cls, 'LOG_FILES') and len(cls.LOG_FILES) > 0:
            normalized = cls._normalize_windows_path(cls.LOG_FILES[0])
            return Path(normalized).expanduser().resolve()
        normalized = cls._normalize_windows_path(cls.LOG_FILE)
        return Path(normalized).expanduser().resolve()
    
    @classmethod
    def get_log_file_paths(cls) -> list[Path]:
        """Get all log file paths as Path objects."""
        if hasattr(cls, 'LOG_FILES') and len(cls.LOG_FILES) > 0:
            return [Path(cls._normalize_windows_path(p)).expanduser().resolve() for p in cls.LOG_FILES]
        normalized = cls._normalize_windows_path(cls.LOG_FILE)
        return [Path(normalized).expanduser().resolve()]
    
    @classmethod
    def get_output_dir_path(cls) -> Path:
        """Get output directory as Path object."""
        normalized = cls._normalize_windows_path(cls.OUTPUT_DIR)
        return Path(normalized).expanduser().resolve()
    
    @classmethod
    def get_database_path(cls) -> Path | None:
        """Get database path as Path object, or None if not configured (returns first if multiple)."""
        # Use DATABASE_PATHS if available, otherwise fall back to DATABASE_PATH
        if hasattr(cls, 'DATABASE_PATHS') and len(cls.DATABASE_PATHS) > 0:
            db_path = cls.DATABASE_PATHS[0]
            if db_path is None:
                return None
            return Path(db_path).expanduser().resolve()
        if cls.DATABASE_PATH is None:
            return None
        return Path(cls.DATABASE_PATH).expanduser().resolve()
    
    @classmethod
    def get_database_paths(cls) -> list[Path | None]:
        """Get all database file paths as Path objects."""
        if hasattr(cls, 'DATABASE_PATHS') and len(cls.DATABASE_PATHS) > 0:
            return [Path(p).expanduser().resolve() if p is not None else None for p in cls.DATABASE_PATHS]
        if cls.DATABASE_PATH is None:
            return [None]
        return [Path(cls.DATABASE_PATH).expanduser().resolve()]
    
    @classmethod
    def get_dataset_configs(cls) -> list[tuple[Path, Path, Path, Path | None]]:
        """
        Get all dataset configurations as tuples of (processed_root, raw_root, log_file, database_path).
        
        Returns
        -------
        list[tuple[Path, Path, Path, Path | None]]
            List of configuration tuples, one for each dataset
        """
        processed_roots = cls.get_processed_root_paths()
        raw_roots = cls.get_raw_root_paths()
        log_files = cls.get_log_file_paths()
        db_paths = cls.get_database_paths()
        
        # Determine the number of configurations
        max_count = max(len(processed_roots), len(raw_roots), len(log_files), len(db_paths))
        
        configs = []
        for i in range(max_count):
            processed_root = processed_roots[i] if i < len(processed_roots) else processed_roots[-1]
            raw_root = raw_roots[i] if i < len(raw_roots) else raw_roots[-1]
            log_file = log_files[i] if i < len(log_files) else log_files[-1]
            db_path = db_paths[i] if i < len(db_paths) else db_paths[-1]
            configs.append((processed_root, raw_root, log_file, db_path))
        
        return configs
    
    @classmethod
    def get_visualization_output_dir_path(cls) -> Path:
        """Get visualization output directory as Path object."""
        return Path(cls.VISUALIZATION_OUTPUT_DIR).expanduser().resolve()
    
    @classmethod
    def is_debug_mode(cls) -> bool:
        """
        Check if debug mode is enabled.
        
        Returns
        -------
        bool
            True if debug mode is enabled, False otherwise
        """
        return cls.DEBUG_MODE
    
    @classmethod
    def get_requested_range_indices(cls) -> list[int]:
        """
        Parse requested_range_indices string into a list of relative indices.
        
        Supports:
        - Single values: "1, 2, 3"
        - Ranges: "1-50" (expands to 1, 2, 3, ..., 50)
        - Mixed: "1, 4-10, 8" (expands to 1, 4, 5, 6, 7, 8, 9, 10, 8)
        
        These are offsets from starting_range_index.
        
        Returns
        -------
        list[int]
            List of relative range indices (offsets from starting_range_index)
            Duplicates are preserved (can be removed by caller if needed)
        """
        ranges_str = cls.REQUESTED_RANGE_INDICES.strip()
        if not ranges_str:
            return []
        
        result = []
        # Split by comma to get individual parts
        parts = [p.strip() for p in ranges_str.split(",") if p.strip()]
        
        for part in parts:
            try:
                # Check if part contains a range (dash)
                if "-" in part:
                    # Parse range: "start-end"
                    range_parts = part.split("-", 1)
                    if len(range_parts) != 2:
                        print(f"⚠ Warning: Invalid range format '{part}' in requested_range_indices. Skipping.")
                        continue
                    
                    start_str = range_parts[0].strip()
                    end_str = range_parts[1].strip()
                    
                    if not start_str or not end_str:
                        print(f"⚠ Warning: Invalid range format '{part}' in requested_range_indices. Skipping.")
                        continue
                    
                    start = int(start_str)
                    end = int(end_str)
                    
                    # Handle both ascending and descending ranges
                    if start <= end:
                        # Normal range: 1-5 -> [1, 2, 3, 4, 5]
                        result.extend(range(start, end + 1))
                    else:
                        # Reverse range: 5-1 -> [5, 4, 3, 2, 1]
                        result.extend(range(start, end - 1, -1))
                else:
                    # Single value
                    result.append(int(part))
            except ValueError as e:
                print(f"⚠ Warning: Could not parse '{part}' in requested_range_indices: {e}. Skipping.")
                continue
        
        return result
    
    @classmethod
    def get_actual_range_indices(cls) -> list[int]:
        """
        Get actual range bin indices from relative indices.
        
        Returns
        -------
        list[int]
            List of actual range bin indices: starting_range_index + relative_index
        """
        relative_indices = cls.get_requested_range_indices()
        return [cls.STARTING_RANGE_INDEX + rel_idx for rel_idx in relative_indices]
    
    @classmethod
    def get_requested_ranges(cls) -> list[float]:
        """
        Get requested ranges as actual distances (computed from relative indices).
        
        Returns
        -------
        list[float]
            List of requested ranges in meters (computed from relative indices)
        """
        relative_indices = cls.get_requested_range_indices()
        return [cls.relative_index_to_distance(rel_idx) for rel_idx in relative_indices]
    
    @classmethod
    def range_index_to_distance(cls, index: int) -> float:
        """
        Convert an actual range bin index to distance in meters.
        
        Formula: distance = (index - starting_range_index) * range_step
        
        Parameters
        ----------
        index : int
            Actual range bin index (0-indexed)
        
        Returns
        -------
        float
            Actual distance in meters
        """
        return (index - cls.STARTING_RANGE_INDEX) * cls.RANGE_STEP
    
    @classmethod
    def relative_index_to_distance(cls, relative_index: int) -> float:
        """
        Convert a relative index (offset from starting_range_index) to distance in meters.
        
        Formula: distance = relative_index * range_step
        
        Parameters
        ----------
        relative_index : int
            Relative index (offset from starting_range_index)
        
        Returns
        -------
        float
            Actual distance in meters
        """
        return relative_index * cls.RANGE_STEP
    
    @classmethod
    def distance_to_range_index(cls, distance: float) -> int:
        """
        Convert an actual distance to the nearest range bin index.
        
        Parameters
        ----------
        distance : float
            Actual distance in meters
        
        Returns
        -------
        int
            Nearest range bin index (0-indexed)
        """
        return int(round(distance / cls.RANGE_STEP + cls.STARTING_RANGE_INDEX))
    
    @classmethod
    def get_starting_range(cls) -> float:
        """
        Get the actual starting range distance in meters (distance at index 0).
        
        Returns
        -------
        float
            Starting range distance: (0 - STARTING_RANGE_INDEX) * RANGE_STEP
        """
        return cls.range_index_to_distance(0)
    
    @classmethod
    def get_heatmap_parameters(cls) -> list[str]:
        """Parse heatmap_parameters string into list of strings."""
        params_str = cls.HEATMAP_PARAMETERS.strip()
        if not params_str:
            return []
        return [p.strip() for p in params_str.split(",") if p.strip()]
    
    @classmethod
    def get_profile_frequency_interval(cls) -> tuple[float, float]:
        """Parse profile_frequency_interval string into (min_freq, max_freq) tuple."""
        interval_str = cls.PROFILE_FREQUENCY_INTERVAL.strip()
        if not interval_str:
            return (0.0, float('inf'))
        try:
            parts = interval_str.split(",")
            if len(parts) != 2:
                raise ValueError("Frequency interval must be in format 'min,max'")
            return (float(parts[0].strip()), float(parts[1].strip()))
        except (ValueError, IndexError) as e:
            print(f"⚠ Warning: Invalid frequency interval format '{interval_str}'. Using default (0, inf)")
            return (0.0, float('inf'))
    
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

