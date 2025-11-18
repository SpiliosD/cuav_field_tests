@echo off
REM Windows batch file to run the git scheduler
REM This can be used with Task Scheduler or run manually

cd /d "%~dp0"
python scheduler.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Scheduler exited with error code %errorlevel%
    pause
)

