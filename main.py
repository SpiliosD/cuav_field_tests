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
import sys
print(">>> main.py module loading...", flush=True, file=sys.stderr)

from __future__ import annotations

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

print(">>> Importing data_reader module...", flush=True, file=sys.stderr)
try:
    from data_reader import create_heatmaps
    print(">>> ✓ data_reader module imported successfully", flush=True, file=sys.stderr)
except Exception as e:
    print(f">>> ✗ ERROR importing data_reader: {e}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

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


def run_tests():
    """Run the complete test suite from total_test.py."""
    import sys
    
    # Force flush to ensure output appears immediately
    print("=" * 70, flush=True)
    print("Running CUAV Field Tests - Test Suite", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Import and run total_test
    import total_test
    
    # Load configuration if not already loaded
    Config.load_from_file(silent=False)
    
    # Run the main test function
    total_test.main()


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
    import sys
    
    print("=" * 70, flush=True)
    print("Generating Heatmaps", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    
    # Load configuration
    Config.load_from_file(silent=False)
    
    # Get database path
    db_path = Config.get_database_path()
    if db_path is None:
        print("✗ ERROR: Database path not configured in config.txt")
        print("  Please set 'database_path' in config.txt or create the database first.")
        print("  You can create the database by running: python main.py --test")
        return False
    
    if not db_path.exists():
        print(f"✗ ERROR: Database not found at {db_path}")
        print("  Please create the database first by running: python main.py --test")
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
        output_dir = Config.get_visualization_output_dir_path()
    
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
            n_points = len(data["azimuth"])
            
            # Display user-friendly parameter name
            param_display = param
            if param == "peak":
                param_display = "SNR (peak)"
            
            print(f"  - {param_display} at {rng:.0f} m: {n_points} data points")
        
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

Note:
  - 'snr' and 'peak' refer to the same parameter (data from _Peak.txt)
  - 'wind' refers to wind data from _Wind.txt
  - 'spectrum' refers to spectrum data from _Spectrum.txt
  - When running from IDE without arguments, set 'run_mode' in config.txt
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
        else:
            # Use config file mode
            run_mode = Config.RUN_MODE.lower().strip()
            if run_mode not in ("test", "heatmaps"):
                print(f">>> ⚠ WARNING: Invalid run_mode '{run_mode}' in config.txt. Valid options: 'test', 'heatmaps'", flush=True, file=sys.stderr)
                print(f">>>   Defaulting to 'test' mode.", flush=True, file=sys.stderr)
                run_mode = "test"
            
            print(f">>> Running in '{run_mode}' mode (from config.txt: run_mode={Config.RUN_MODE})", flush=True, file=sys.stderr)
            print(flush=True)
        
        # Execute requested action
        print(f">>> Executing {run_mode} mode...", flush=True, file=sys.stderr)
        if run_mode == "test":
            print(">>> Calling run_tests()...", flush=True, file=sys.stderr)
            run_tests()
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
