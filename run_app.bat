@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Environment not found. Run install_windows.bat first.
    pause
    exit /b 1
)

".venv\Scripts\pythonw.exe" -m auto_alignment
if errorlevel 1 (
    echo [ERROR] The application could not start.
    ".venv\Scripts\python.exe" -m auto_alignment
    pause
)
