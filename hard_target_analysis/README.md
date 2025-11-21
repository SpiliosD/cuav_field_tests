# Hard Target Detection Analysis

## Overview

This module provides sophisticated multi-criteria detection algorithms to identify **hard targets** (solid objects) that could be hit by the lidar beam. The analysis processes all visualization outputs and database data to detect potential obstacles or targets.

## Detection Method

The detection system uses **six complementary criteria** combined with weighted scoring:

### 1. SNR Anomaly Detection
- **Principle**: Hard targets produce strong reflections, resulting in high SNR values
- **Method**: Identifies SNR values above the 95th percentile threshold
- **Weight**: 25%

### 2. Difference Gradient Analysis
- **Principle**: Hard boundaries create abrupt changes in sequential range differences
- **Method**: Detects large differences between adjacent ranges (90th percentile threshold)
- **Weight**: 20%

### 3. FWHM Pattern Analysis
- **Principle**: Coherent reflections from solid surfaces produce narrow FWHM values
- **Method**: Identifies FWHM values below the 20th percentile (narrow peaks)
- **Weight**: 20%

### 4. Statistical Outlier Detection
- **Principle**: Hard targets stand out as statistical anomalies
- **Method**: Z-score analysis to identify values >2.5 standard deviations above mean
- **Weight**: 15%

### 5. Spatial Consistency Analysis
- **Principle**: Real targets appear consistently across multiple ranges
- **Method**: Groups detections by spatial location and requires appearance in ≥3 ranges
- **Weight**: 15%

### 6. Spatial Clustering
- **Principle**: Related detections cluster in azimuth/elevation space
- **Method**: Hierarchical clustering to identify grouped detections
- **Weight**: 5%

## Scoring System

Each detection criterion contributes a score, and final targets are identified by:
1. Aggregating scores in a spatial grid (5° × 5° bins)
2. Identifying peaks above the 75th percentile threshold
3. Ranking by combined confidence score

## Usage

### Basic Usage

```bash
python hard_target_analysis/analyze_targets.py
```

### With Custom Output Directory

```bash
python hard_target_analysis/analyze_targets.py --output-dir results/my_targets
```

### With Custom Thresholds

```bash
python hard_target_analysis/analyze_targets.py \
    --snr-threshold 98.0 \
    --difference-threshold 95.0 \
    --consistency-threshold 5
```

### Programmatic Usage

```python
from hard_target_analysis.target_detection import detect_hard_targets
from config import Config

results = detect_hard_targets(
    db_path="data/output7.db",
    range_step=48.0,
    starting_range=-1344.0,
    requested_ranges=[100, 200, 300, 400, 500],
    output_dir="results/targets",
)

# Access results
targets = results["targets"]
for target in targets[:5]:  # Top 5
    print(f"Target at azimuth={target['azimuth']:.1f}°, "
          f"elevation={target['elevation']:.1f}°, "
          f"confidence={target['confidence_score']:.3f}")
```

## Output Files

The analysis generates:

1. **`hard_target_detections.json`**: JSON file with all detected targets, scores, and metadata
2. **`hard_target_detections.png`**: Polar plot visualization showing target locations with confidence scores

## Output Format

### JSON Structure

```json
{
  "targets": [
    {
      "azimuth": 45.2,
      "elevation": 12.5,
      "range": 250.0,
      "confidence_score": 0.856,
      "detection_count": 8
    },
    ...
  ],
  "scores": {
    "snr_detections": 142,
    "difference_detections": 89,
    "fwhm_detections": 67,
    "outlier_detections": 34,
    "final_targets": 15
  },
  "metadata": {
    "ranges_analyzed": 80,
    "total_data_points": 12500
  }
}
```

## Configuration Parameters

The detector can be customized with the following parameters:

- `snr_threshold_percentile`: SNR threshold (default: 95.0)
- `difference_threshold_percentile`: Difference threshold (default: 90.0)
- `spatial_consistency_threshold`: Min ranges for consistency (default: 3)
- `fwhm_threshold_percentile`: FWHM threshold (default: 20.0)
- `outlier_z_score`: Z-score threshold (default: 2.5)
- `min_cluster_size`: Minimum cluster size (default: 3)
- `range_consistency_window`: Range window for consistency (default: 200.0 m)

## Interpretation

### High Confidence Targets (>0.7)
- Strong indicators of hard targets
- Multiple criteria detected
- Consistent across ranges
- **Action**: Investigate these locations carefully

### Medium Confidence Targets (0.4-0.7)
- Moderate indicators
- Some criteria detected
- **Action**: Review with additional context

### Low Confidence Targets (<0.4)
- Weak indicators
- May be atmospheric features
- **Action**: Use with caution

## Integration with Main Pipeline

To integrate with the main analysis pipeline, add to `config.txt`:

```txt
# Hard Target Detection
run_hard_target_detection=true
hard_target_output_dir=hard_target_results
```

Then modify `main.py` to call the detection after visualization generation.

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Statistical analysis and clustering
- `matplotlib`: Visualization (optional, for plots)

## Algorithm Details

### Spatial Grid Aggregation

Detections are aggregated into a 5° × 5° spatial grid:
- Azimuth: 0-360° in 5° bins (73 bins)
- Elevation: 0-90° in 5° bins (19 bins)

### Clustering Method

Uses hierarchical clustering (Ward linkage) with:
- Distance threshold: 10 degrees
- Minimum cluster size: 3 points

### Score Normalization

All scores are normalized to [0, 1] range:
- Individual criterion scores: Based on percentile distances
- Combined scores: Weighted sum of all criteria
- Final confidence: Normalized by maximum possible score

## Future Enhancements

Potential improvements:
1. Machine learning classification for target type
2. Temporal consistency analysis across timestamps
3. 3D spatial reconstruction of target locations
4. Integration with obstacle avoidance systems
5. Real-time detection during data collection

