# Script to run main.py with the conda environment 'cuav_field_tests'
# This script activates the conda environment and runs main.py

$condaEnv = "cuav_field_tests"
$condaBase = "C:/ProgramData/anaconda3"

# Try to activate conda and run
Write-Host "Activating conda environment '$condaEnv'..." -ForegroundColor Green

# Method 1: Try using conda activate (if conda is in PATH)
try {
    conda activate $condaEnv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Conda environment activated successfully" -ForegroundColor Green
        python main.py
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "Conda command not found in PATH, trying direct Python path..." -ForegroundColor Yellow
}

# Method 2: Use conda Python directly (no activation needed)
$pythonPath = "$condaBase/envs/$condaEnv/python.exe"

if (Test-Path $pythonPath) {
    Write-Host "Running main.py with conda Python at: $pythonPath" -ForegroundColor Green
    & $pythonPath main.py
    exit $LASTEXITCODE
} else {
    Write-Host "ERROR: Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  1. Conda is installed at: $condaBase" -ForegroundColor Yellow
    Write-Host "  2. Environment '$condaEnv' exists" -ForegroundColor Yellow
    Write-Host "  3. Or update the path in run.ps1" -ForegroundColor Yellow
    exit 1
}

