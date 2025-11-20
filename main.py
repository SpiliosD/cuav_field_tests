"""Main script entry point for CUAV Field Tests Data Reader.

This script provides two main options:
1. Run the test suite (total_test.py) to verify project functionality
2. Generate heatmaps for specific parameters at specific ranges

Usage:
    # Run all tests
    python main.py --test

    # Generate heatmaps for wind and SNR (peak) at specific ranges
    python main.py --heatmaps --parameters wind snr --ranges 100 200 300

    # Generate heatmaps with default parameters and ranges from config.txt
    python main.py --heatmaps
"""

# Print immediately when module is loaded (before any imports)
from __future__ import annotations
import sys

print(">>> main.py module loading...", flush=True, file=sys.stderr)

import argparse
import os
from pathlib import Path

print(">>> Imports starting...", flush=True, file=sys.stderr)

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load configuration from config.txt
print(">>> Importing config module...", flush=True, file=sys.stderr)
try:
    from config import Config
    print(">>> ✓ Config module imported successfully", flush=True, file=sys.stderr)
except Exception as e:
    print(f">>> ✗ ERROR importing config: {e}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Delay data_reader import until needed (for heatmaps mode only)
# This avoids importing visualization modules when running tests
print(">>> Skipping data_reader import (will import when needed for heatmaps)", flush=True, file=sys.stderr)

print(">>> All module imports completed", flush=True, file=sys.stderr)

# Map user-friendly parameter names to database parameter names
PARAMETER_ALIASES = {
    "snr": "peak",  # SNR comes from _Peak.txt
    "peak": "peak",  # Peak values from _Peak.txt
    "wind": "wind",  # Wind from _Wind.txt
    "spectrum": "spectrum",  # Spectrum from _Spectrum.txt
}


def parse_ranges(ranges_str: str) -> list[float]:
    """
    Parse comma-separated or space-separated ranges string into list of floats.
    
    Examples:
        "100,200,300" -> [100.0, 200.0, 300.0]
        "100 200 300" -> [100.0, 200.0, 300.0]
    """
    # Replace commas with spaces, then split
    ranges_str = ranges_str.replace(",", " ")
    parts = ranges_str.split()
    try:
        return [float(r.strip()) for r in parts if r.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid range format: {ranges_str}. Error: {e}")


def normalize_parameter(param: str) -> str:
    """
    Normalize parameter name (handle aliases).
    
    Maps user-friendly names to database parameter names:
    - 'snr' -> 'peak' (SNR comes from _Peak.txt)
    - 'peak' -> 'peak'
    - 'wind' -> 'wind' (from _Wind.txt)
    - 'spectrum' -> 'spectrum' (from _Spectrum.txt)
    """
    param_lower = param.lower().strip()
    if param_lower in PARAMETER_ALIASES:
        return PARAMETER_ALIASES[param_lower]
    # If not found, return as-is (will be validated by create_heatmaps)
    return param_lower


def _load_visualization_module():
    """
    Load the visualization module directly from file to bypass data_reader.__init__.py.
    
    This avoids importing the entire data_reader package which requires numpy at import time.
    This is especially important when using a venv that might not have all packages installed.
    
    Returns
    -------
    module
        The loaded visualization module
    """
    import importlib.util
    
    project_root = Path(__file__).resolve().parent
    viz_module_path = project_root / "data_reader" / "analysis" / "visualization.py"
    
    if not viz_module_path.exists():
        raise ImportError(f"Visualization module not found at {viz_module_path}")
    
    module_name = "data_reader.analysis.visualization"
    
    # Check if module is already loaded
    if module_name in sys.modules:
        # If it's a proper module (not a broken one), return it
        existing = sys.modules[module_name]
        if hasattr(existing, 'create_heatmaps'):
            return existing
        # Otherwise, remove broken module
        del sys.modules[module_name]
    
    # Load module directly from file
    spec = importlib.util.spec_from_file_location(module_name, viz_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {viz_module_path}")
    
    visualization = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = visualization
    spec.loader.exec_module(visualization)
    
    return visualization


def create_or_rebuild_database() -> bool:
    """
    Create or rebuild the database from processed and raw data.
    
    This function:
    1. Matches processed and raw data files
    2. Filters matches by log timestamps
    3. Deletes existing database if it exists
    4. Creates a new database with all filtered matches
    
    Returns
    -------
    bool
        True if database was created successfully, False otherwise
    """
    print(">>> Creating/rebuilding database...", flush=True, file=sys.stderr)
    
    try:
        # Import required functions
        from data_reader import (
            build_and_save_to_database,
            filter_matches_by_log_timestamps,
            match_processed_and_raw,
        )
        
        # Get paths from config
        processed_root = Config.get_processed_root_path()
        raw_root = Config.get_raw_root_path()
        log_file = Config.get_log_file_path()
        db_path = Config.get_database_path()
        
        if db_path is None:
            print("✗ ERROR: Database path not configured in config.txt", flush=True)
            return False
        
        # Validate paths
        if not processed_root.exists():
            print(f"✗ ERROR: Processed root not found: {processed_root}", flush=True)
            return False
        
        if not raw_root.exists():
            print(f"✗ ERROR: Raw root not found: {raw_root}", flush=True)
            return False
        
        if not log_file.exists():
            print(f"✗ ERROR: Log file not found: {log_file}", flush=True)
            return False
        
        print(f"  Processed root: {processed_root}", flush=True)
        print(f"  Raw root: {raw_root}", flush=True)
        print(f"  Log file: {log_file}", flush=True)
        print(f"  Database: {db_path}", flush=True)
        
        # Match processed and raw data
        print("  Matching processed and raw files...", flush=True)
        matches = match_processed_and_raw(processed_root, raw_root)
        print(f"  ✓ Found {len(matches)} initial matches", flush=True)
        
        if len(matches) == 0:
            print("  ⚠ WARNING: No matches found. Database will be empty.", flush=True)
        
        # Filter matches by log timestamps
        print("  Filtering matches by log timestamps...", flush=True)
        
        # Debug: show sample timestamps before filtering (only in debug mode)
        if Config.is_debug_mode() and len(matches) > 0:
            sample_match_ts = float(matches[0][0])
            from data_reader.parsing.logs import extract_log_timestamps
            from data_reader.parsing.timestamp_correction import correct_processed_timestamp
            import numpy as np
            log_ts_raw = extract_log_timestamps(str(log_file))
            log_ts_sample = float(log_ts_raw[0]) if len(log_ts_raw) > 0 else None
            log_ts_corrected_sample = correct_processed_timestamp(log_ts_sample) if log_ts_sample else None
            print(f"    Debug: Sample processed timestamp (corrected): {sample_match_ts:.6f}", flush=True)
            print(f"    Debug: Sample log timestamp (raw): {log_ts_sample:.6f}" if log_ts_sample else "    Debug: No log timestamps", flush=True)
            print(f"    Debug: Sample log timestamp (corrected): {log_ts_corrected_sample:.6f}" if log_ts_corrected_sample else "", flush=True)
        
        filtered_matches = filter_matches_by_log_timestamps(
            matches, 
            str(log_file), 
            atol=Config.TIMESTAMP_TOLERANCE
        )
        print(f"  ✓ {len(filtered_matches)} matches after filtering", flush=True)
        
        # Delete existing database if it exists
        if db_path.exists():
            print(f"  Removing existing database: {db_path}", flush=True)
            db_path.unlink()
            print(f"  ✓ Existing database removed", flush=True)
        
        # Prepare timestamp-path pairs for database creation
        # Store full file path so we can extract both directory and filename later
        # Also track original uncorrected timestamps for debugging
        timestamp_path_pairs = []
        original_timestamps_map = {}  # corrected -> original mapping
        for match in filtered_matches:
            processed_ts = match[0]  # Corrected timestamp
            raw_file_path = Path(match[3])
            # Store original timestamp if available (match[4] if present)
            if len(match) > 4 and match[4] is not None:
                original_timestamps_map[processed_ts] = match[4]
            # Store full file path so we can extract both directory and filename later
            timestamp_path_pairs.append((processed_ts, str(raw_file_path)))
        
        # Create database
        print(f"  Building database with {len(timestamp_path_pairs)} entries...", flush=True)
        count = build_and_save_to_database(
            timestamp_path_pairs,
            str(processed_root),
            str(raw_root),
            str(log_file),
            str(db_path),
            atol=Config.TIMESTAMP_TOLERANCE,
            original_timestamps_map=original_timestamps_map if original_timestamps_map else None,
        )
        
        print(f"  ✓ Successfully created database with {count} entries", flush=True)
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR creating database: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def run_tests():
    """Run the complete test suite from total_test.py."""
    print(">>> Entering run_tests() function", flush=True, file=sys.stderr)
    
    # Create/rebuild database first
    if not create_or_rebuild_database():
        print("⚠ WARNING: Database creation failed, but continuing with tests...", flush=True)
    
    # Force flush to ensure output appears immediately
    print("=" * 70, flush=True)
    print("Running CUAV Field Tests - Test Suite", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Import and run total_test
    print(">>> Importing total_test module...", flush=True, file=sys.stderr)
    try:
        # Import from tests directory
        tests_path = Path(__file__).resolve().parent / "tests"
        if str(tests_path) not in sys.path:
            sys.path.insert(0, str(tests_path))
        import total_test
        print(">>> ✓ total_test module imported successfully", flush=True, file=sys.stderr)
    except Exception as e:
        print(f">>> ✗ ERROR importing total_test: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return
    
    # Load configuration if not already loaded
    print(">>> Loading configuration...", flush=True, file=sys.stderr)
    Config.load_from_file(silent=False)
    print(">>> Configuration loaded", flush=True, file=sys.stderr)
    
    # Run the main test function
    print(">>> Calling total_test.main()...", flush=True, file=sys.stderr)
    try:
        total_test.main()
        print(">>> total_test.main() completed", flush=True, file=sys.stderr)
    except Exception as e:
        print(f">>> ✗ ERROR in total_test.main(): {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


def get_logfile_basename() -> str:
    """
    Get the basename of the logfile (without extension) for creating output subdirectories.
    
    Returns
    -------
    str
        Basename of logfile without extension, or 'output' if logfile not configured
    """
    log_file = Config.get_log_file_path()
    if log_file is None or not log_file.exists():
        return "output"
    return log_file.stem


def generate_heatmaps(
    parameters: list[str] | None = None,
    ranges: list[float] | None = None,
    output_dir: Path | None = None,
    colormap: str = "viridis",
    save_format: str = "png",
):
    """
    Generate heatmaps for specified parameters at specified ranges.
    
    Parameters
    ----------
    parameters : list[str] | None
        List of parameters to visualize. Options: 'snr' (or 'peak'), 'wind', 'spectrum'.
        If None, uses default from config.txt.
    ranges : list[float] | None
        List of ranges in meters to visualize. If None, uses default from config.txt.
    output_dir : Path | None
        Output directory for heatmap images. If None, uses default from config.txt.
    colormap : str
        Matplotlib colormap name (default: 'viridis').
    save_format : str
        Image format to save (default: 'png').
    """
    print(">>> Importing create_heatmaps (lazy import)...", flush=True, file=sys.stderr)
    sys.stderr.flush()  # Force flush before import
    try:
        visualization = _load_visualization_module()
        create_heatmaps = visualization.create_heatmaps
        print(">>> ✓ create_heatmaps imported successfully", flush=True, file=sys.stderr)
        sys.stderr.flush()
    except ImportError as e:
        print(f">>> ✗ ERROR importing create_heatmaps: {e}", flush=True, file=sys.stderr)
        sys.stderr.flush()
        print(">>>   Hint: Make sure all dependencies are installed (numpy, matplotlib, etc.)", flush=True, file=sys.stderr)
        sys.stderr.flush()
        print(">>>   Run: pip install -r requirements.txt", flush=True, file=sys.stderr)
        sys.stderr.flush()
        print(">>>   Or use the conda Python: C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe main.py", flush=True, file=sys.stderr)
        sys.stderr.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return False
    except Exception as e:
        print(f">>> ✗ ERROR importing create_heatmaps: {e}", flush=True, file=sys.stderr)
        sys.stderr.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return False
    
    print("=" * 70, flush=True)
    print("Generating Heatmaps", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Get database path (should already exist from main() if multimode, or will be created if single mode)
    # Check if database exists, create if needed (for single-mode runs)
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt", flush=True)
        return False
    
    if not db_path.exists():
        print("Database does not exist. Creating database first...", flush=True)
        if not create_or_rebuild_database():
            print("✗ ERROR: Failed to create database. Cannot generate heatmaps.", flush=True)
            return False
    
    # Get database path (should exist now)
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt", flush=True)
        return False
    
    if not db_path.exists():
        print(f"✗ ERROR: Database was not created at {db_path}", flush=True)
        return False
    
    # Get visualization parameters from config (or use provided)
    range_step = Config.RANGE_STEP
    starting_range = Config.get_starting_range()
    
    if ranges is None:
        requested_ranges = Config.get_requested_ranges()
        if not requested_ranges:
            requested_ranges = [100.0, 200.0, 300.0]  # Default
            print(f"⚠ No ranges configured in config.txt, using default: {requested_ranges}")
    else:
        requested_ranges = ranges
    
    if parameters is None:
        # Use all available parameters
        parameters_list = ["wind", "peak", "spectrum"]
        print("⚠ No parameters specified, using all: wind, peak (SNR), spectrum")
    else:
        # Normalize parameter names (handle aliases like 'snr' -> 'peak')
        parameters_list = [normalize_parameter(p) for p in parameters]
        # Remove duplicates while preserving order
        seen = set()
        parameters_list = [p for p in parameters_list if p not in seen and not seen.add(p)]
    
    if output_dir is None:
        base_output_dir = Config.get_visualization_output_dir_path()
        # Create subdirectory named after logfile (without extension)
        logfile_basename = get_logfile_basename()
        output_dir = base_output_dir / logfile_basename
    else:
        # Even if output_dir is provided, append logfile basename subdirectory
        logfile_basename = get_logfile_basename()
        output_dir = Path(output_dir) / logfile_basename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Parameters: {parameters_list}")
    print(f"    - 'peak' = SNR data from _Peak.txt")
    print(f"    - 'wind' = Wind data from _Wind.txt")
    print(f"    - 'spectrum' = Spectrum data from _Spectrum.txt")
    print(f"  Range Step: {range_step} m")
    print(f"  Starting Range: {starting_range} m")
    print(f"  Requested Ranges: {requested_ranges} m")
    print(f"  Output Directory: {output_dir}")
    print(f"  Colormap: {colormap}")
    print(f"  Save Format: {save_format}")
    print()
    
    try:
        print("Generating heatmaps...")
        # Create subfolders for each parameter type
        parameter_subdirs = {}
        for param in parameters_list:
            if param == "peak":
                subfolder_name = "snr_heatmaps"  # Use "snr" instead of "peak" for clarity
            elif param == "wind":
                subfolder_name = "wind_heatmaps"
            elif param == "spectrum":
                subfolder_name = "spectrum_heatmaps"
            else:
                subfolder_name = f"{param}_heatmaps"
            parameter_subdirs[param] = output_dir / subfolder_name
            parameter_subdirs[param].mkdir(parents=True, exist_ok=True)
        
        results = create_heatmaps(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            requested_ranges=requested_ranges,
            parameters=parameters_list,
            output_dir=output_dir,
            parameter_subdirs=parameter_subdirs,  # Pass parameter-specific subdirectories
            colormap=colormap,
            save_format=save_format,
        )
        
        print()
        if len(results) == 0:
            print("⚠ WARNING: No heatmaps were generated!")
            print("   This could mean:")
            print("   - No data exists at the requested ranges")
            print("   - The database doesn't contain the required parameters")
            print("   - Check that azimuth and elevation data is present in the database")
            print()
            return False
        
        print(f"✓ Successfully generated {len(results)} heatmaps:")
        for key, data in results.items():
            param = data["parameter"]
            rng = data["range"]
            azimuth = data["azimuth"]
            elevation = data["elevation"]
            values = data["values"]
            n_points = len(azimuth)
            
            # Display user-friendly parameter name
            param_display = param
            if param == "peak":
                param_display = "SNR (peak)"
            
            print(f"\n  {param_display} at {rng:.0f} m: {n_points} data points")
            print(f"  Dataset (first 10 points):")
            print(f"    Azimuth (°)    Elevation (°)    {param_display} Value")
            print(f"    " + "-" * 50)
            # Print first 10 data points (or all if fewer than 10)
            for i in range(min(10, n_points)):
                print(f"    {azimuth[i]:12.3f}  {elevation[i]:15.3f}  {values[i]:15.6f}")
            if n_points > 10:
                print(f"    ... ({n_points - 10} more points)")
        
        print()
        print(f"✓ Heatmaps saved to: {output_dir}")
        print()
        return True
        
    except ImportError as e:
        print(f"✗ ERROR: {e}")
        print("  Install matplotlib: pip install matplotlib")
        return False
    except ValueError as e:
        print(f"✗ ERROR: {e}")
        print("  Valid parameters are: 'snr' (or 'peak'), 'wind', 'spectrum'")
        return False
    except Exception as e:
        print(f"✗ ERROR: Failed to generate heatmaps: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_differences(
    output_dir: Path | None = None,
    save_format: str = "png",
):
    """
    Generate sequential difference analysis (Mode 4).
    
    Computes SNR and wind speed differences between sequential range pairs
    and creates heatmap visualizations.
    
    Parameters
    ----------
    output_dir : Path | None
        Output directory for difference heatmap images. If None, uses default from config.txt.
    save_format : str
        Image format to save (default: 'png').
    """
    print(">>> Importing difference analysis functions (lazy import)...", flush=True, file=sys.stderr)
    try:
        visualization = _load_visualization_module()
        compute_sequential_differences = visualization.compute_sequential_differences
        create_difference_heatmaps = visualization.create_difference_heatmaps
        print(">>> ✓ Difference functions imported successfully", flush=True, file=sys.stderr)
    except Exception as e:
        print(f">>> ✗ ERROR importing difference functions: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False
    
    print("=" * 70, flush=True)
    print("Generating Sequential Differences (Mode 4)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Get database path
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt", flush=True)
        return False
    
    # Check if database exists (should already exist from main() if multimode)
    if not db_path.exists():
        print("✗ ERROR: Database does not exist. Please run profiles mode first to generate profile data.", flush=True)
        return False
    
    # Get visualization parameters from config
    range_step = Config.RANGE_STEP
    starting_range = Config.get_starting_range()
    requested_ranges = Config.get_requested_ranges()
    
    if not requested_ranges:
        print("✗ ERROR: No requested_ranges specified in config.txt", flush=True)
        return False
    
    if output_dir is None:
        base_output_dir = Config.get_visualization_output_dir_path()
        logfile_basename = get_logfile_basename()
        output_dir = base_output_dir / logfile_basename
    else:
        # Even if output_dir is provided, append logfile basename subdirectory
        logfile_basename = get_logfile_basename()
        output_dir = Path(output_dir) / logfile_basename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Range Step: {range_step} m")
    print(f"  Starting Range: {starting_range} m")
    print(f"  Requested Ranges: {requested_ranges} m")
    print(f"  Output Directory: {output_dir}")
    print(f"  Save Format: {save_format}")
    print()
    
    try:
        # Step 1: Compute differences
        print("Computing sequential differences...")
        processing_stats = compute_sequential_differences(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            requested_ranges=requested_ranges,
        )
        
        print()
        print(f"Processing complete:")
        print(f"  Processed: {processing_stats['processed_count']} timestamps")
        print(f"  Skipped: {processing_stats['skipped_count']} timestamps")
        print(f"  Total: {processing_stats['total_count']} timestamps")
        print()
        
        # Step 2: Generate heatmaps for SNR differences
        # Create subfolder for SNR differences
        snr_diff_subdir = output_dir / "snr_difference"
        snr_diff_subdir.mkdir(parents=True, exist_ok=True)
        
        print("Generating SNR difference heatmaps...")
        snr_results = create_difference_heatmaps(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            requested_ranges=requested_ranges,
            difference_type="snr",
            output_dir=snr_diff_subdir,
            colormap=Config.HEATMAP_COLORMAP,
            save_format=save_format,
        )
        
        # Step 3: Generate heatmaps for wind differences
        # Create subfolder for wind differences
        wind_diff_subdir = output_dir / "wind_difference"
        wind_diff_subdir.mkdir(parents=True, exist_ok=True)
        
        print("Generating wind difference heatmaps...")
        wind_results = create_difference_heatmaps(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            requested_ranges=requested_ranges,
            difference_type="wind",
            output_dir=wind_diff_subdir,
            colormap=Config.HEATMAP_COLORMAP,
            save_format=save_format,
        )
        
        print()
        print(f"✓ All outputs saved to: {output_dir}")
        print()
        return True
        
    except Exception as e:
        print(f"✗ ERROR: Failed to generate differences: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_fwhm(
    output_dir: Path | None = None,
    save_format: str = "png",
):
    """
    Generate FWHM analysis (Mode 5).
    
    Computes Full Width at Half Maximum of dominant frequency peaks
    and creates heatmap visualizations.
    
    Parameters
    ----------
    output_dir : Path | None
        Output directory for FWHM heatmap images. If None, uses default from config.txt.
    save_format : str
        Image format to save (default: 'png').
    """
    print(">>> Importing FWHM analysis functions (lazy import)...", flush=True, file=sys.stderr)
    try:
        visualization = _load_visualization_module()
        compute_fwhm_profile = visualization.compute_fwhm_profile
        create_fwhm_heatmaps = visualization.create_fwhm_heatmaps
        print(">>> ✓ FWHM functions imported successfully", flush=True, file=sys.stderr)
    except Exception as e:
        print(f">>> ✗ ERROR importing FWHM functions: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False
    
    print("=" * 70, flush=True)
    print("Generating FWHM Analysis (Mode 5)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Get database path
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt", flush=True)
        return False
    
    # Check if database exists (should already exist from main() if multimode)
    if not db_path.exists():
        print("✗ ERROR: Database does not exist. Please run profiles mode first to generate profile data.", flush=True)
        return False
    
    # Get visualization parameters from config
    range_step = Config.RANGE_STEP
    starting_range = Config.get_starting_range()
    requested_ranges = Config.get_requested_ranges()
    
    # Get Mode 3 parameters from config (needed for FWHM computation)
    fft_size = Config.PROFILE_FFT_SIZE
    sampling_rate = Config.PROFILE_SAMPLING_RATE
    frequency_interval = Config.get_profile_frequency_interval()
    
    if not requested_ranges:
        print("✗ ERROR: No requested_ranges specified in config.txt", flush=True)
        return False
    
    if output_dir is None:
        base_output_dir = Config.get_visualization_output_dir_path()
        logfile_basename = get_logfile_basename()
        output_dir = base_output_dir / logfile_basename
    else:
        # Even if output_dir is provided, append logfile basename subdirectory
        logfile_basename = get_logfile_basename()
        output_dir = Path(output_dir) / logfile_basename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Range Step: {range_step} m")
    print(f"  Starting Range: {starting_range} m")
    print(f"  Requested Ranges: {requested_ranges} m")
    print(f"  FFT Size: {fft_size}")
    print(f"  Sampling Rate: {sampling_rate} Hz")
    print(f"  Frequency Interval: [{frequency_interval[0]}, {frequency_interval[1]}] Hz")
    print(f"  Output Directory: {output_dir}")
    print(f"  Save Format: {save_format}")
    print()
    
    try:
        # Step 1: Compute FWHM
        print("Computing FWHM profiles...")
        processing_stats = compute_fwhm_profile(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            fft_size=fft_size,
            sampling_rate=sampling_rate,
            frequency_interval=frequency_interval,
        )
        
        print()
        print(f"Processing complete:")
        print(f"  Processed: {processing_stats['processed_count']} timestamps")
        print(f"  Skipped: {processing_stats['skipped_count']} timestamps")
        print(f"  Total: {processing_stats['total_count']} timestamps")
        print()
        
        # Step 2: Generate heatmaps
        # Create subfolder for FWHM heatmaps
        fwhm_subdir = output_dir / "fwhm"
        fwhm_subdir.mkdir(parents=True, exist_ok=True)
        
        print("Generating FWHM heatmaps...")
        results = create_fwhm_heatmaps(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            requested_ranges=requested_ranges,
            output_dir=fwhm_subdir,
            colormap=Config.HEATMAP_COLORMAP,
            save_format=save_format,
        )
        
        print()
        print(f"✓ All outputs saved to: {output_dir}")
        print()
        return True
        
    except Exception as e:
        print(f"✗ ERROR: Failed to generate FWHM: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_profiles(
    output_dir: Path | None = None,
    save_format: str = "png",
):
    """
    Generate single-profile visualizations (Mode 3).
    
    This mode:
    1. Processes range-resolved power density spectra for each timestamp
    2. Computes frequencies from FFT size and sampling rate
    3. Finds frequency at maximum SNR for each range
    4. Computes wind speed using coherent Doppler lidar equation: v = laser_wavelength * (dominant_frequency - frequency_shift) / 2 (result in m/s)
    5. Stores SNR and wind profiles in database
    6. Generates visualizations of all SNR and wind profiles
    
    Parameters
    ----------
    output_dir : Path | None
        Output directory for profile images. If None, uses default from config.txt.
    save_format : str
        Image format to save (default: 'png').
    """
    print(">>> Importing profile processing functions (lazy import)...", flush=True, file=sys.stderr)
    try:
        visualization = _load_visualization_module()
        process_single_profiles = visualization.process_single_profiles
        create_profile_visualizations = visualization.create_profile_visualizations
        print(">>> ✓ Profile functions imported successfully", flush=True, file=sys.stderr)
    except Exception as e:
        print(f">>> ✗ ERROR importing profile functions: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False
    
    print("=" * 70, flush=True)
    print("Generating Single Profiles (Mode 3)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Get database path
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt", flush=True)
        return False
    
    # Check if database exists, create if needed (for single-mode runs)
    if not db_path.exists():
        print("Database does not exist. Creating database first...", flush=True)
        if not create_or_rebuild_database():
            print("✗ ERROR: Failed to create database. Cannot generate profiles.", flush=True)
            return False
    
    # Get visualization parameters from config
    range_step = Config.RANGE_STEP
    starting_range = Config.get_starting_range()
    
    # Get Mode 3 parameters from config
    fft_size = Config.PROFILE_FFT_SIZE
    sampling_rate = Config.PROFILE_SAMPLING_RATE
    frequency_interval = Config.get_profile_frequency_interval()
    frequency_shift = Config.PROFILE_FREQUENCY_SHIFT
    laser_wavelength = Config.PROFILE_LASER_WAVELENGTH
    
    if output_dir is None:
        base_output_dir = Config.get_visualization_output_dir_path()
        # Create subdirectory named after logfile (without extension)
        logfile_basename = get_logfile_basename()
        output_dir = base_output_dir / logfile_basename
    else:
        # Even if output_dir is provided, append logfile basename subdirectory
        logfile_basename = get_logfile_basename()
        output_dir = Path(output_dir) / logfile_basename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Range Step: {range_step} m")
    print(f"  Starting Range: {starting_range} m")
    print(f"  FFT Size: {fft_size}")
    print(f"  Sampling Rate: {sampling_rate} Hz")
    print(f"  Frequency Interval: [{frequency_interval[0]}, {frequency_interval[1]}] Hz")
    print(f"  Frequency Shift: {frequency_shift} Hz")
    print(f"  Laser Wavelength: {laser_wavelength} m")
    print(f"  Output Directory: {output_dir}")
    print(f"  Save Format: {save_format}")
    print()
    
    try:
        # Step 1: Process profiles
        print("Processing range-resolved spectra...")
        processing_stats = process_single_profiles(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            fft_size=fft_size,
            sampling_rate=sampling_rate,
            frequency_interval=frequency_interval,
            frequency_shift=frequency_shift,
            laser_wavelength=laser_wavelength,
        )
        
        print()
        print(f"Processing complete:")
        print(f"  Processed: {processing_stats['processed_count']} timestamps")
        print(f"  Skipped: {processing_stats['skipped_count']} timestamps")
        print(f"  Total: {processing_stats['total_count']} timestamps")
        print()
        
        # Step 2: Generate visualizations
        # Create subfolder for single profiles
        profiles_subdir = output_dir / "single_profile"
        profiles_subdir.mkdir(parents=True, exist_ok=True)
        
        print("Generating profile visualizations...")
        results = create_profile_visualizations(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            output_dir=profiles_subdir,
            save_format=save_format,
        )
        
        print()
        if "snr_plot_path" in results:
            print(f"✓ SNR profiles plot saved: {results['snr_plot_path']}")
        if "wind_plot_path" in results:
            print(f"✓ Wind profiles plot saved: {results['wind_plot_path']}")
        print()
        print(f"✓ All outputs saved to: {output_dir}")
        print()
        return True
        
    except ImportError as e:
        print(f"✗ ERROR: {e}")
        print("  Install matplotlib: pip install matplotlib")
        return False
    except ValueError as e:
        print(f"✗ ERROR: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: Failed to generate profiles: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point with command-line argument parsing.
    
    Mode selection priority:
    1. Command-line arguments (--test or --heatmaps) - highest priority
    2. Config file (run_mode in config.txt) - fallback for IDE execution
    3. Default to 'test' mode if neither specified
    """
    print(">>> Entering main() function", flush=True, file=sys.stderr)
    
    try:
        print(">>> Creating argument parser...", flush=True, file=sys.stderr)
        parser = argparse.ArgumentParser(
            description="CUAV Field Tests Data Reader - Main Entry Point",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run from config.txt (set run_mode in config.txt)
  python main.py

  # Override config with command-line: Run all tests
  python main.py --test

  # Override config with command-line: Generate heatmaps
  python main.py --heatmaps --parameters wind snr --ranges 100 200 300

  # Generate heatmaps using parameters from config.txt
  python main.py --heatmaps

  # Generate single profiles (Mode 3)
  python main.py --profiles

  # Generate both heatmaps and profiles simultaneously
  python main.py --heatmaps --profiles

  # Compute and visualize sequential differences (Mode 4)
  python main.py --differences

  # Compute and visualize FWHM of dominant frequency peaks (Mode 5)
  python main.py --fwhm

  # Run multiple modes simultaneously
  python main.py --heatmaps --profiles --differences --fwhm

Note:
  - 'snr' and 'peak' refer to the same parameter (data from _Peak.txt)
  - 'wind' refers to wind data from _Wind.txt
  - 'spectrum' refers to spectrum data from _Spectrum.txt
  - When running from IDE without arguments, set 'run_mode' in config.txt
  - --test is mutually exclusive with other modes
  - Other modes can be combined to run simultaneously
  - All figures are displayed for 0.5 seconds then automatically closed
  - Mode 4 (differences) and Mode 5 (fwhm) require Mode 3 (profiles) to be run first
        """,
        )
        
        # Test mode is mutually exclusive with other modes
        # But heatmaps and profiles can run together
        parser.add_argument(
            "--test",
            action="store_true",
            help="Run the complete test suite (total_test.py). Overrides config.txt run_mode. Mutually exclusive with --heatmaps and --profiles.",
        )
        
        parser.add_argument(
            "--heatmaps",
            action="store_true",
            help="Generate heatmaps for specified parameters at specified ranges. Can be combined with --profiles.",
        )
        
        parser.add_argument(
            "--profiles",
            action="store_true",
            help="Generate single-profile visualizations (Mode 3). Can be combined with --heatmaps.",
        )
        
        parser.add_argument(
            "--differences",
            action="store_true",
            help="Compute and visualize SNR and wind differences between sequential range pairs (Mode 4).",
        )
        
        parser.add_argument(
            "--fwhm",
            action="store_true",
            help="Compute and visualize FWHM of dominant frequency peaks (Mode 5).",
        )
        
        # Heatmap-specific arguments
        parser.add_argument(
            "--parameters",
            "-p",
            nargs="+",
            help="Parameters to visualize. Options: 'snr' (or 'peak'), 'wind', 'spectrum'. "
                 "Default: from config.txt (heatmap_parameters)",
        )
        
        parser.add_argument(
            "--ranges",
            "-r",
            type=str,
            help="Ranges in meters to visualize. Can be comma-separated or space-separated. "
                 "Example: '100,200,300' or '100 200 300'. "
                 "Default: from config.txt (requested_ranges)",
        )
        
        parser.add_argument(
            "--output-dir",
            "-o",
            type=Path,
            help="Output directory for heatmap images. Default: from config.txt",
        )
        
        parser.add_argument(
            "--colormap",
            "-c",
            type=str,
            help="Matplotlib colormap name. Default: from config.txt (heatmap_colormap). "
                 "Options: viridis, plasma, inferno, magma, coolwarm, etc.",
        )
        
        parser.add_argument(
            "--format",
            "-f",
            type=str,
            choices=["png", "pdf", "svg", "jpg", "jpeg"],
            help="Image format to save. Default: from config.txt (heatmap_format)",
        )
        
        parser.add_argument(
            "--config",
            type=Path,
            help="Path to config.txt file (default: config.txt in project root)",
        )
        
        print("Parsing command-line arguments...", flush=True)
        args = parser.parse_args()
        print(f"Arguments parsed: test={args.test}, heatmaps={args.heatmaps}", flush=True)
        
        # Load configuration from file
        print("Loading configuration from config.txt...", flush=True)
        if args.config:
            Config.load_from_file(args.config, silent=False)
        else:
            Config.load_from_file(silent=False)
        
        # Determine execution mode (command-line takes precedence over config)
        # Test mode is mutually exclusive with others
        if args.test:
            if args.heatmaps or args.profiles or args.differences or args.fwhm:
                print("✗ ERROR: --test cannot be combined with other modes", flush=True)
                sys.exit(1)
            run_test = True
            run_heatmaps = False
            run_profiles = False
            run_differences = False
            run_fwhm = False
            print(">>> Running in 'test' mode (from command-line argument)", flush=True, file=sys.stderr)
            print(flush=True)
        else:
            # Check if modes are requested via command-line
            run_test = False
            run_heatmaps = args.heatmaps
            run_profiles = args.profiles
            run_differences = args.differences
            run_fwhm = args.fwhm
            
            # If none specified, use config file mode
            if not run_heatmaps and not run_profiles and not run_differences and not run_fwhm:
                config_mode = Config.RUN_MODE.lower().strip()
                # Support comma-separated values like "heatmaps,profiles"
                modes_list = [m.strip() for m in config_mode.split(",")]
                
                if "test" in modes_list:
                    if len(modes_list) > 1:
                        print(f">>> ⚠ WARNING: 'test' cannot be combined with other modes in config.txt. Using 'test' only.", flush=True, file=sys.stderr)
                    run_test = True
                else:
                    # Check for all modes
                    if "heatmaps" in modes_list:
                        run_heatmaps = True
                    if "profiles" in modes_list:
                        run_profiles = True
                    if "differences" in modes_list:
                        run_differences = True
                    if "fwhm" in modes_list:
                        run_fwhm = True
                    
                    # Validate modes
                    valid_modes = {"test", "heatmaps", "profiles", "differences", "fwhm"}
                    invalid_modes = [m for m in modes_list if m not in valid_modes]
                    if invalid_modes:
                        print(f">>> ⚠ WARNING: Invalid run_mode values '{invalid_modes}' in config.txt. Valid options: {', '.join(sorted(valid_modes))}", flush=True, file=sys.stderr)
                        if not run_heatmaps and not run_profiles and not run_differences and not run_fwhm:
                            print(f">>>   Defaulting to 'test' mode.", flush=True, file=sys.stderr)
                            run_test = True
                    
                    # Default to test if nothing valid was specified
                    if not run_test and not run_heatmaps and not run_profiles and not run_differences and not run_fwhm:
                        run_test = True
            
            if run_test:
                print(">>> Running in 'test' mode (from config.txt)", flush=True, file=sys.stderr)
                print(flush=True)
            else:
                modes = []
                if run_heatmaps:
                    modes.append("heatmaps")
                if run_profiles:
                    modes.append("profiles")
                if run_differences:
                    modes.append("differences")
                if run_fwhm:
                    modes.append("fwhm")
                if modes:
                    print(f">>> Running in '{' and '.join(modes)}' mode (from command-line/config)", flush=True, file=sys.stderr)
                print(flush=True)
        
        # Execute requested action
        if run_test:
            print(f">>> Executing test mode...", flush=True, file=sys.stderr)
            print(">>> Calling run_tests()...", flush=True, file=sys.stderr)
            run_tests()
        else:
            # Execute heatmaps and/or profiles (can run simultaneously)
            success = True
            
            # Create/rebuild database ONCE before running any modes
            # This avoids rebuilding the database multiple times in multimode runs
            print("=" * 70, flush=True)
            print("Creating/Rebuilding Database (if needed)", flush=True)
            print("=" * 70, flush=True)
            print(flush=True)
            if not create_or_rebuild_database():
                print("✗ ERROR: Failed to create database. Cannot continue.", flush=True)
                sys.exit(1)
            print(flush=True)
            
            # Run profiles FIRST if requested (needed for differences and FWHM)
            if run_profiles:
                print(f">>> Executing profiles mode...", flush=True, file=sys.stderr)
                # Get options from command-line or config
                save_format = args.format if args.format else Config.HEATMAP_FORMAT
                
                profile_success = generate_profiles(
                    output_dir=args.output_dir,
                    save_format=save_format,
                )
                
                if not profile_success:
                    success = False
            
            # Run heatmaps if requested
            if run_heatmaps:
                print(f">>> Executing heatmaps mode...", flush=True, file=sys.stderr)
                # Get parameters from command-line or config
                if args.parameters:
                    parameters = args.parameters
                else:
                    # Parse from config
                    parameters = Config.get_heatmap_parameters()
                    if not parameters:
                        # Default if nothing specified
                        parameters = ["wind", "peak", "spectrum"]
                        print("⚠ No heatmap_parameters in config.txt, using default: wind, peak, spectrum")
                
                # Get ranges from command-line or config
                if args.ranges:
                    try:
                        ranges_list = parse_ranges(args.ranges)
                    except ValueError as e:
                        print(f"✗ ERROR: Invalid ranges format: {e}")
                        print(f"  Example: --ranges '100,200,300' or --ranges '100 200 300'")
                        sys.exit(1)
                else:
                    ranges_list = None  # Will use config default in generate_heatmaps()
                
                # Get other options from command-line or config
                colormap = args.colormap if args.colormap else Config.HEATMAP_COLORMAP
                save_format = args.format if args.format else Config.HEATMAP_FORMAT
                
                heatmap_success = generate_heatmaps(
                    parameters=parameters,
                    ranges=ranges_list,
                    output_dir=args.output_dir,
                    colormap=colormap,
                    save_format=save_format,
                )
                
                if not heatmap_success:
                    success = False
            
            # Run differences if requested (requires profiles to be run first)
            if run_differences:
                print(f">>> Executing differences mode...", flush=True, file=sys.stderr)
                diff_success = generate_differences(
                    output_dir=args.output_dir,
                    save_format=args.format if args.format else Config.HEATMAP_FORMAT,
                )
                
                if not diff_success:
                    success = False
            
            # Run FWHM if requested
            if run_fwhm:
                print(f">>> Executing FWHM mode...", flush=True, file=sys.stderr)
                fwhm_success = generate_fwhm(
                    output_dir=args.output_dir,
                    save_format=args.format if args.format else Config.HEATMAP_FORMAT,
                )
                
                if not fwhm_success:
                    success = False
            
            if not success:
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    
    # Force unbuffered output
    sys.stdout = sys.__stdout__  # Ensure we're using the real stdout
    sys.stderr = sys.__stderr__  # Ensure we're using the real stderr
    
    # Ensure stdout is unbuffered for immediate output
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        # Fallback for older Python versions or if reconfigure fails
        import os
        os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Print startup message to confirm script is running
    print("=" * 70, flush=True)
    print("CUAV Field Tests - Main Script Starting", flush=True)
    print("=" * 70, flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Script path: {__file__}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"sys.argv: {sys.argv}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    try:
        main()
        print("\n" + "=" * 70, flush=True)
        print("✓ Script completed successfully.", flush=True)
        print("=" * 70, flush=True)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70, flush=True)
        print(f"✗ FATAL ERROR: {e}", flush=True, file=sys.stderr)
        print("=" * 70, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
