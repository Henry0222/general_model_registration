@echo off
setlocal
cd /d "%~dp0"

set "BUILD_PYTHON=.venv\Scripts\python.exe"
if not exist "%BUILD_PYTHON%" if exist "..\.venv\Scripts\python.exe" set "BUILD_PYTHON=..\.venv\Scripts\python.exe"
if not exist "%BUILD_PYTHON%" (
    echo [ERROR] Development environment not found. Run install_windows.bat first.
    exit /b 1
)

"%BUILD_PYTHON%" -m pip install --no-build-isolation -e ".[dev]"
if errorlevel 1 exit /b 1

"%BUILD_PYTHON%" -m pytest
if errorlevel 1 exit /b 1

if exist "build\GeneralModelRegistration" rmdir /s /q "build\GeneralModelRegistration"
if exist "dist\GeneralModelRegistration-v1.4.1.exe" del /q "dist\GeneralModelRegistration-v1.4.1.exe"
if exist "dist\GeneralModelRegistration-v1.4.1.exe" (
    echo [ERROR] Existing v1.4.1 executable is still in use and cannot be replaced.
    exit /b 1
)
if exist "dist\release\GeneralModelRegistration-v1.4.1-win64" rmdir /s /q "dist\release\GeneralModelRegistration-v1.4.1-win64"
if exist "dist\GeneralModelRegistration-v1.4.1-win64.zip" del /q "dist\GeneralModelRegistration-v1.4.1-win64.zip"
if exist "dist\GeneralModelRegistration-v1.4.1-win64.zip.sha256.txt" del /q "dist\GeneralModelRegistration-v1.4.1-win64.zip.sha256.txt"

"%BUILD_PYTHON%" -m PyInstaller --noconfirm --clean --workpath "build\GeneralModelRegistration" GeneralModelRegistration.spec
if errorlevel 1 exit /b 1

"%BUILD_PYTHON%" "scripts\package_windows_release.py"
if errorlevel 1 exit /b 1

echo Build complete: dist\GeneralModelRegistration-v1.4.1-win64.zip
