# Running the Script

## Quick Start (Recommended)

**Use the provided PowerShell script:**
```powershell
.\run.ps1
```

This automatically uses the conda environment `cuav_field_tests`.

## Alternative Methods

**Option 1: Direct Python path (no conda activation needed)**
```powershell
C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe main.py
```

**Option 2: Activate conda and run**
```powershell
conda activate cuav_field_tests
python main.py
```

## IDE Configuration

### Cursor / VS Code
1. Press `Ctrl+Shift+P` to open Command Palette
2. Type "Python: Select Interpreter"
3. Select: `C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe`

### PyCharm
1. File → Settings → Project → Python Interpreter
2. Click the gear icon → Add Interpreter → Existing Environment
3. Browse to: `C:/ProgramData/anaconda3/envs/cuav_field_tests/python.exe`
4. Click OK

## Troubleshooting

### "No module named numpy" or similar errors

The conda environment should have all packages. If you see this error:
1. Make sure you're using the conda Python interpreter
2. Verify packages are installed:
   ```powershell
   conda activate cuav_field_tests
   pip list
   ```
3. If packages are missing, install them:
   ```powershell
   conda activate cuav_field_tests
   pip install -r requirements.txt
   ```

### Import hangs in IDE

If the script hangs during import:
1. **Check Python interpreter**: Make sure IDE is using the conda Python
2. **Run from command line first**: Verify it works outside the IDE
3. **Check output**: The script prints debug messages to stderr - check the output panel

### Script works in PyCharm but not in Cursor

This is usually because:
- Different Python interpreters are configured
- IDE-specific import mechanisms
- Output buffering differences

**Solution**: Configure Cursor to use the same conda Python interpreter (see IDE Configuration above).

### Conda command not found

If `conda activate` doesn't work:
- Use the direct Python path method (Option 1 above)
- Or initialize conda in PowerShell:
  ```powershell
  C:/ProgramData/anaconda3/Scripts/conda.exe init powershell
  # Then restart PowerShell
  ```

