# CUAV Field Tests Data Reader

This project processes and visualizes lidar field test data from processed and raw spectra files.

## Quick Start

### Using Conda Environment

The project uses the conda environment `cuav_field_tests`. 

**Option 1: Run with provided script (recommended)**
```powershell
.\run.ps1
```

**Option 2: Run directly with conda Python**
```powershell
C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe main.py
```

**Option 3: Activate conda and run**
```powershell
conda activate cuav_field_tests
python main.py
```

### IDE Configuration

**For Cursor/VS Code:**
1. Open Command Palette (Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose: `C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe`

**For PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select: `C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe`

## Installation

The conda environment should already have all dependencies. If you need to install them:

```powershell
conda activate cuav_field_tests
pip install -r requirements.txt
```

## Usage

See the main script help:
```powershell
python main.py --help
```

## Configuration

Edit `config.txt` to configure:
- Data paths (processed_root, raw_root, log_file)
- Database path
- Visualization parameters
- Run modes
- Base results folder (optional, for centralized output)
- Plot display settings (show plots or save directly)
- Debug mode (true/false for verbose output)

### Range Notation

The `requested_range_indices` parameter supports flexible range notation:
- Single values: `1, 2, 3`
- Ranges: `1-50` (expands to 1, 2, 3, ..., 50)
- Mixed: `1, 4-10, 8` (expands to 1, 4, 5, 6, 7, 8, 9, 10, 8)

Example: `requested_range_indices=1-80` will process indices 1 through 80.

## Modes

- **Test Mode**: Run test suite (`--test`)
- **Heatmaps Mode**: Generate heatmaps (`--heatmaps`)
- **Profiles Mode**: Generate single-profile visualizations (`--profiles`)
- **Differences Mode**: Compute sequential differences (`--differences`)
- **FWHM Mode**: Compute FWHM of dominant frequency peaks (`--fwhm`)

Modes can be combined (except test mode).

## Output Organization

### Centralized Results Folder

If `base_results_folder` is set in `config.txt`, all outputs are organized under:
- `base_results_folder/databases/` - All database files
- `base_results_folder/images/` - All visualization images

The subfolder structure is preserved within these directories.

### Visualization Output Structure

All visualization outputs are organized into subfolders within `visualization_output/<logfile_basename>/` (or `base_results_folder/images/<logfile_basename>/` if configured):

- `wind_heatmaps/` - Wind speed heatmaps
- `snr_heatmaps/` - SNR (peak) heatmaps
- `spectrum_heatmaps/` - Spectrum heatmaps
- `single_profile/` - SNR, wind, and dominant frequency profile plots
- `snr_difference/` - SNR difference heatmaps between sequential ranges
- `wind_difference/` - Wind difference heatmaps between sequential ranges
- `fwhm/` - FWHM heatmaps of dominant frequency peaks

## Debug Mode

Set `debug_mode=true` in `config.txt` for verbose output including:
- Detailed database content information
- Matching details for timestamps
- Array structure and bounds checking
- Range extraction calculations

Set `debug_mode=false` for normal output (only essential information).

## Hard Target Analysis

The project includes a comprehensive hard target detection analysis module that evaluates lidar data using multiple criteria to identify potential hard targets (solid objects).

### Quick Start

Run the comprehensive analysis from IDE:

```python
from hard_target_analysis.comprehensive_analysis import analyze_results

# Simple call - uses config.txt settings
results = analyze_results(
    image_folder="visualization_output/output7"
)
```

Or use the provided script:

1. Open `hard_target_analysis/run_analysis.py`
2. Modify `IMAGE_FOLDER` to point to your image folder
3. Run the script

### Analysis Criteria

The comprehensive analysis evaluates **6 criteria**:

1. **Range-Resolved Signal Intensity Heatmap**
   - Contiguous high-intensity blob detection
   - Localization check (not smeared)
   - Smooth range falloff analysis
   - Background contrast calculation
   - Temporal consistency
   - Speckling detection

2. **Dominant Frequency / Wind Speed Heatmap**
   - Wind speed plausibility check
   - Stability across ranges
   - Smooth transition analysis

3. **FWHM of Dominant Peak Heatmap**
   - FWHM-intensity correlation
   - Gradual broadening with range
   - Checkerboarding detection

4. **Sequential Range Differences (ΔIntensity)**
   - Gradient sharpness at boundaries
   - Background stability

5. **Sequential Range Differences (ΔWind)**
   - Background smoothness
   - Localization of anomalies

6. **Cross-Parameter Consistency**
   - Co-location of high intensity + narrow FWHM + stable wind
   - Edge alignment between ΔIntensity and ΔWind

### Output

The analysis generates:
- **`comprehensive_analysis_report.json`** - Complete analysis results
- **`analysis_summary.txt`** - Human-readable summary with promising indicators and red flags
- **`analysis_visualization.png`** - Visual summary of scores

### Documentation

See `hard_target_analysis/README.md` and `hard_target_analysis/QUICKSTART.md` for detailed documentation.
