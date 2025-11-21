"""Simple script to run comprehensive analysis from IDE.

This is the main entry point for running the comprehensive hard target analysis.
Just modify the parameters below and run this script.
"""

from pathlib import Path
from hard_target_analysis.comprehensive_analysis import analyze_results

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Path to the folder containing visualization images
# This is used for reference - the actual analysis uses database data
IMAGE_FOLDER = "visualization_output/output7"  # Change this to your image folder

# Optional: Specify database path explicitly (uses config.txt if None)
DATABASE_PATH = None  # e.g., "data/output7.db"

# Optional: Specify output directory for analysis results
OUTPUT_DIR = None  # e.g., "hard_target_analysis_results"

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("Starting comprehensive hard target analysis...")
    print(f"Image folder: {IMAGE_FOLDER}")
    print()
    
    # Run the analysis
    results = analyze_results(
        db_path=DATABASE_PATH,
        image_folder=IMAGE_FOLDER,
        output_dir=OUTPUT_DIR,
    )
    
    # Access results programmatically if needed
    assessment = results.get("overall_assessment", {})
    print(f"\nAnalysis complete!")
    print(f"Overall Assessment: {assessment.get('assessment')}")
    print(f"Confidence: {assessment.get('confidence')}")
    print(f"Score: {assessment.get('overall_score', 0.0):.3f}")

