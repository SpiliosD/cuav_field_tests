"""Main script for hard target detection analysis.

This script analyzes all visualization outputs to detect hard targets
that could be hit by the lidar beam.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import Config
from hard_target_analysis.target_detection import detect_hard_targets


def main():
    """Main entry point for hard target detection analysis."""
    parser = argparse.ArgumentParser(
        description="Hard Target Detection Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This analysis uses sophisticated multi-criteria detection to identify hard targets
(solid objects) that could be hit by the lidar beam.

Detection Criteria:
1. SNR Anomaly Detection: High SNR values indicate strong reflections
2. Difference Gradient Analysis: Abrupt changes suggest hard boundaries
3. Spatial Consistency: Targets appearing across multiple ranges
4. FWHM Analysis: Narrow FWHM indicates coherent reflections
5. Statistical Outlier Detection: Values significantly above background
6. Spatial Clustering: Grouped detections in azimuth/elevation space

Example:
  python hard_target_analysis/analyze_targets.py
  python hard_target_analysis/analyze_targets.py --output-dir results/targets
        """,
    )
    
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to database file (default: from config.txt)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for detection results (default: hard_target_results)",
    )
    
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=95.0,
        help="SNR threshold percentile (default: 95.0)",
    )
    
    parser.add_argument(
        "--difference-threshold",
        type=float,
        default=90.0,
        help="Difference threshold percentile (default: 90.0)",
    )
    
    parser.add_argument(
        "--consistency-threshold",
        type=int,
        default=3,
        help="Minimum ranges for spatial consistency (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    Config.load_from_file(silent=False)
    
    # Get database path
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = Config.get_database_path()
        if db_path is None:
            print("✗ ERROR: Database path not configured in config.txt", flush=True)
            sys.exit(1)
    
    if not db_path.exists():
        print(f"✗ ERROR: Database does not exist at {db_path}", flush=True)
        sys.exit(1)
    
    # Get ranges from config
    requested_ranges = Config.get_requested_ranges()
    if not requested_ranges:
        print("✗ ERROR: No requested_ranges specified in config.txt", flush=True)
        sys.exit(1)
    
    # Get output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_output = Config.get_visualization_output_dir_path()
        output_dir = base_output.parent / "hard_target_results"
    
    # Get parameters from config
    range_step = Config.RANGE_STEP
    starting_range = Config.get_starting_range()
    
    print("=" * 70)
    print("Hard Target Detection Analysis")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Range Step: {range_step} m")
    print(f"  Starting Range: {starting_range} m")
    print(f"  Requested Ranges: {requested_ranges} m")
    print(f"  Output Directory: {output_dir}")
    print(f"  SNR Threshold: {args.snr_threshold}th percentile")
    print(f"  Difference Threshold: {args.difference_threshold}th percentile")
    print(f"  Consistency Threshold: {args.consistency_threshold} ranges")
    print()
    
    # Run detection
    print("Running hard target detection analysis...")
    results = detect_hard_targets(
        db_path=db_path,
        range_step=range_step,
        starting_range=starting_range,
        requested_ranges=requested_ranges,
        output_dir=output_dir,
        snr_threshold_percentile=args.snr_threshold,
        difference_threshold_percentile=args.difference_threshold,
        spatial_consistency_threshold=args.consistency_threshold,
    )
    
    # Print results
    print()
    print("=" * 70)
    print("Detection Results")
    print("=" * 70)
    print()
    print(f"Total targets detected: {len(results['targets'])}")
    print()
    print("Detection Statistics:")
    for criterion, count in results["scores"].items():
        print(f"  {criterion}: {count}")
    print()
    print(f"Data Points Analyzed: {results['metadata']['total_data_points']}")
    print(f"Ranges Analyzed: {results['metadata']['ranges_analyzed']}")
    print()
    
    if len(results["targets"]) > 0:
        print("Top 10 Detected Targets (by confidence score):")
        print()
        print(f"{'Rank':<6} {'Azimuth (°)':<15} {'Elevation (°)':<15} {'Range (m)':<12} {'Confidence':<12} {'Count':<8}")
        print("-" * 70)
        
        for i, target in enumerate(results["targets"][:10], 1):
            print(
                f"{i:<6} "
                f"{target['azimuth']:<15.2f} "
                f"{target['elevation']:<15.2f} "
                f"{target['range']:<12.1f} "
                f"{target['confidence_score']:<12.3f} "
                f"{target['detection_count']:<8}"
            )
        print()
        
        if len(results["targets"]) > 10:
            print(f"... and {len(results['targets']) - 10} more targets")
            print()
    
    print(f"✓ Results saved to: {output_dir}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

