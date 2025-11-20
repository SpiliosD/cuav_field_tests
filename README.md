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

## Modes

- **Test Mode**: Run test suite (`--test`)
- **Heatmaps Mode**: Generate heatmaps (`--heatmaps`)
- **Profiles Mode**: Generate single-profile visualizations (`--profiles`)
- **Differences Mode**: Compute sequential differences (`--differences`)
- **FWHM Mode**: Compute FWHM of dominant frequency peaks (`--fwhm`)

Modes can be combined (except test mode).
