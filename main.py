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
        timestamp_path_pairs = []
        for match in filtered_matches:
            processed_ts = match[0]
            raw_file_path = Path(match[3])
            raw_dir_path = raw_file_path.parent
            timestamp_path_pairs.append((processed_ts, str(raw_dir_path)))
        
        # Create database
        print(f"  Building database with {len(timestamp_path_pairs)} entries...", flush=True)
        count = build_and_save_to_database(
            timestamp_path_pairs,
            str(processed_root),
            str(raw_root),
            str(log_file),
            str(db_path),
            atol=Config.TIMESTAMP_TOLERANCE,
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
    try:
        from data_reader import create_heatmaps
        print(">>> ✓ create_heatmaps imported successfully", flush=True, file=sys.stderr)
    except Exception as e:
        print(f">>> ✗ ERROR importing create_heatmaps: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False
    
    print("=" * 70, flush=True)
    print("Generating Heatmaps", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Load configuration
    Config.load_from_file(silent=False)
    
    # Create/rebuild database first
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
    starting_range = Config.STARTING_RANGE
    
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
        results = create_heatmaps(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            requested_ranges=requested_ranges,
            parameters=parameters_list,
            output_dir=output_dir,
            colormap=colormap,
            save_format=save_format,
        )
        
        print()
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
        from data_reader.analysis.visualization import (
            process_single_profiles,
            create_profile_visualizations,
        )
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
    
    # Load configuration
    Config.load_from_file(silent=False)
    
    # Get database path
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt", flush=True)
        return False
    
    # Check if database exists, if not create it
    if not db_path.exists():
        print("Database does not exist. Creating database first...", flush=True)
        if not create_or_rebuild_database():
            print("✗ ERROR: Failed to create database. Cannot generate profiles.", flush=True)
            return False
    
    # Get visualization parameters from config
    range_step = Config.RANGE_STEP
    starting_range = Config.STARTING_RANGE
    
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
        print("Generating profile visualizations...")
        results = create_profile_visualizations(
            db_path=db_path,
            range_step=range_step,
            starting_range=starting_range,
            output_dir=output_dir,
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

Note:
  - 'snr' and 'peak' refer to the same parameter (data from _Peak.txt)
  - 'wind' refers to wind data from _Wind.txt
  - 'spectrum' refers to spectrum data from _Spectrum.txt
  - When running from IDE without arguments, set 'run_mode' in config.txt
  - Modes are mutually exclusive: only one mode runs at a time
        """,
        )
        
        # Create mutually exclusive group for main actions (optional now)
        action_group = parser.add_mutually_exclusive_group(required=False)
        
        action_group.add_argument(
            "--test",
            action="store_true",
            help="Run the complete test suite (total_test.py). Overrides config.txt run_mode.",
        )
        
        action_group.add_argument(
            "--heatmaps",
            action="store_true",
            help="Generate heatmaps for specified parameters at specified ranges. Overrides config.txt run_mode.",
        )
        
        action_group.add_argument(
            "--profiles",
            action="store_true",
            help="Generate single-profile visualizations (Mode 3). Overrides config.txt run_mode.",
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
        if args.test:
            run_mode = "test"
            print(">>> Running in 'test' mode (from command-line argument)", flush=True, file=sys.stderr)
            print(flush=True)
        elif args.heatmaps:
            run_mode = "heatmaps"
            print(">>> Running in 'heatmaps' mode (from command-line argument)", flush=True, file=sys.stderr)
            print(flush=True)
        elif args.profiles:
            run_mode = "profiles"
            print(">>> Running in 'profiles' mode (from command-line argument)", flush=True, file=sys.stderr)
            print(flush=True)
        else:
            # Use config file mode
            run_mode = Config.RUN_MODE.lower().strip()
            if run_mode not in ("test", "heatmaps", "profiles"):
                print(f">>> ⚠ WARNING: Invalid run_mode '{run_mode}' in config.txt. Valid options: 'test', 'heatmaps', 'profiles'", flush=True, file=sys.stderr)
                print(f">>>   Defaulting to 'test' mode.", flush=True, file=sys.stderr)
                run_mode = "test"
            
            print(f">>> Running in '{run_mode}' mode (from config.txt: run_mode={Config.RUN_MODE})", flush=True, file=sys.stderr)
            print(flush=True)
        
        # Execute requested action
        print(f">>> Executing {run_mode} mode...", flush=True, file=sys.stderr)
        if run_mode == "test":
            print(">>> Calling run_tests()...", flush=True, file=sys.stderr)
            run_tests()
        elif run_mode == "profiles":
            # Get options from command-line or config
            save_format = args.format if args.format else Config.HEATMAP_FORMAT
            
            success = generate_profiles(
                output_dir=args.output_dir,
                save_format=save_format,
            )
            
            if not success:
                sys.exit(1)
        elif run_mode == "heatmaps":
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
            
            success = generate_heatmaps(
                parameters=parameters,
                ranges=ranges_list,
                output_dir=args.output_dir,
                colormap=colormap,
                save_format=save_format,
            )
            
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
