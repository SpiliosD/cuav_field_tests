@echo off
REM Simple batch file to run main.py with conda Python
REM Double-click this file or run: run.bat

set PYTHON_PATH=C:\ProgramData\anaconda3\envs\cuav_field_tests\python.exe

if exist "%PYTHON_PATH%" (
    echo Running main.py with conda Python...
    "%PYTHON_PATH%" main.py
) else (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Please update the path in run.bat
    pause
    exit /b 1
)

pause

