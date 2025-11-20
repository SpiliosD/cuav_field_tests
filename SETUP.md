# Setup Instructions

This project uses the **conda environment `cuav_field_tests`**. No virtual environment is needed.

## Quick Setup

### 1. Verify Conda Environment

Make sure the conda environment exists and has packages installed:

```powershell
conda activate cuav_field_tests
python --version
pip list
```

If packages are missing:
```powershell
pip install -r requirements.txt
```

### 2. Run from Command Line

**Option A: Use the run script (easiest)**
```powershell
.\run.ps1
```

**Option B: Direct Python path**
```powershell
C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe main.py
```

**Option C: Activate conda first**
```powershell
conda activate cuav_field_tests
python main.py
```

### 3. Configure IDE

#### Cursor / VS Code

The `.vscode/settings.json` file is already configured. If it doesn't work:

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Select: `C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe`

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. Click gear icon → Add Interpreter → Existing Environment
3. Browse to: `C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe`
4. Click OK

## Troubleshooting

### "No module named numpy"

**Solution**: Make sure you're using the conda Python interpreter:
- Check IDE interpreter settings
- Or use the direct path: `C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe main.py`

### Script hangs in IDE

**Solution**: 
1. Verify the conda Python has all packages: `pip list` in that environment
2. Try running from command line first to verify it works
3. Check that IDE is using the correct interpreter

### Conda command not found

**Solution**: Use the direct Python path method (Option B above) - no conda activation needed.

## Project Structure

- `main.py` - Main entry point
- `config.txt` - Configuration file
- `run.ps1` - Script to run with conda Python
- `.vscode/settings.json` - IDE configuration for Cursor/VS Code

