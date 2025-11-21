# Quick Start Guide - Comprehensive Hard Target Analysis

## Running from IDE

### Method 1: Use the provided script

1. Open `hard_target_analysis/run_analysis.py`
2. Modify the `IMAGE_FOLDER` variable to point to your image folder:
   ```python
   IMAGE_FOLDER = "visualization_output/output7"
   ```
3. Run the script

### Method 2: Direct function call

```python
from hard_target_analysis.comprehensive_analysis import analyze_results

# Simple call - uses config.txt settings
results = analyze_results(
    image_folder="visualization_output/output7"
)

# Or specify everything explicitly
results = analyze_results(
    db_path="data/output7.db",
    image_folder="visualization_output/output7",
    output_dir="my_analysis_results"
)
```

## What It Analyzes

The comprehensive analysis evaluates **6 criteria** based on your specifications:

### 1. Range-Resolved Signal Intensity Heatmap
- ✓ Contiguous high-intensity blobs
- ✓ Localized energy (not smeared)
- ✓ Smooth range falloff
- ✓ High contrast vs background
- ✓ Temporal consistency
- ✗ Speckling detection

### 2. Dominant Frequency / Wind Speed Heatmap
- ✓ Wind speed plausibility
- ✓ Stability across ranges
- ✓ Smooth transitions

### 3. FWHM of Dominant Peak Heatmap
- ✓ FWHM-intensity correlation
- ✓ Gradual broadening with range
- ✗ Checkerboarding detection

### 4. Sequential Range Differences in Signal Intensity (ΔIntensity)
- ✓ Gradient sharpness
- ✓ Background stability

### 5. Sequential Range Differences in Wind Speed (ΔWind)
- ✓ Background smoothness
- ✓ Localization of anomalies

### 6. Cross-Parameter Consistency
- ✓ Co-location of high intensity + narrow FWHM + stable wind
- ✓ Edge alignment between ΔIntensity and ΔWind

## Output Files

The analysis generates:

1. **`comprehensive_analysis_report.json`** - Complete analysis results
2. **`analysis_summary.txt`** - Human-readable summary
3. **`analysis_visualization.png`** - Visual summary of scores

## Understanding Results

### Overall Assessment Levels

- **PROMISING** (Score > 0.7): Strong indicators of hard targets
- **MODERATE** (Score 0.5-0.7): Some indicators present
- **POOR** (Score < 0.5): Weak or inconsistent indicators

### Key Metrics

- **Promising Indicators**: Positive signs detected
- **Red Flags**: Warning signs or inconsistencies
- **Promising Ratio**: Ratio of promising to total indicators

## Example Output

```
Overall Assessment: PROMISING
Confidence: HIGH
Overall Score: 0.782

Promising Indicators: 12
Red Flags: 3

INTENSITY ANALYSIS:
  ✓ Found 3 contiguous high-intensity region(s)
  ✓ High localization score - energy is well-localized
  ✓ Smooth range falloff detected
  ✓ High contrast vs background (5.23 dB above background)
  ✗ Low temporal consistency - intermittent detections
```

## Tips

1. **Start with default settings** - The analyzer uses sensible defaults
2. **Check the summary file** - `analysis_summary.txt` has detailed explanations
3. **Review red flags** - These indicate potential issues
4. **Compare multiple datasets** - Run analysis on different output folders
5. **Use visualization** - The PNG file shows score breakdowns

## Troubleshooting

**No data found**: Check that database exists and contains profile data
**All scores zero**: Verify that requested_ranges match your data
**Import errors**: Install scipy: `pip install scipy`

