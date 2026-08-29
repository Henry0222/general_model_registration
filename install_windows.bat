@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher was not found. Install 64-bit Python 3.12 first.
    pause
    exit /b 1
)

py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 64-bit Python 3.12 was not found.
    echo Download it from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
if errorlevel 1 (
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

echo Installation completed. Run run_app.bat to start.
pause
