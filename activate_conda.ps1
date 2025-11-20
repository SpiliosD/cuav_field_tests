# Script to activate conda environment and set up the shell
# Run this before running main.py if you want to use 'python' directly

$condaEnv = "cuav_field_tests"
$condaBase = "C:/ProgramData/anaconda3"

Write-Host "Initializing conda for PowerShell..." -ForegroundColor Green

# Initialize conda for PowerShell (if not already done)
& "$condaBase/Scripts/conda.exe" init powershell

Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  conda activate $condaEnv" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or use the run.ps1 script which doesn't require activation." -ForegroundColor Yellow

