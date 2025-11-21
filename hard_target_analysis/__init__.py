"""Hard Target Detection Analysis Module.

This module provides sophisticated methods for detecting hard targets (solid objects)
that could be hit by the lidar beam based on analysis of all visualization outputs.

Hard targets are detected using multiple criteria:
- High SNR values (strong reflections)
- Abrupt changes in sequential differences
- Consistent spatial patterns across ranges
- FWHM characteristics
- Statistical anomaly detection
- Spatial clustering analysis
"""

from pathlib import Path

from hard_target_analysis.target_detection import detect_hard_targets, HardTargetDetector
from hard_target_analysis.comprehensive_analysis import analyze_results, ComprehensiveTargetAnalyzer

__all__ = [
    "detect_hard_targets",
    "HardTargetDetector",
    "analyze_results",
    "ComprehensiveTargetAnalyzer",
]

