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
if exist "dist\GeneralModelRegistration-v1.4.0.exe" del /q "dist\GeneralModelRegistration-v1.4.0.exe"
if exist "dist\release\GeneralModelRegistration-v1.4.0-win64" rmdir /s /q "dist\release\GeneralModelRegistration-v1.4.0-win64"
if exist "dist\GeneralModelRegistration-v1.4.0-win64.zip" del /q "dist\GeneralModelRegistration-v1.4.0-win64.zip"
if exist "dist\GeneralModelRegistration-v1.4.0-win64.zip.sha256.txt" del /q "dist\GeneralModelRegistration-v1.4.0-win64.zip.sha256.txt"

"%BUILD_PYTHON%" -m PyInstaller --noconfirm --clean --workpath "build\GeneralModelRegistration" GeneralModelRegistration.spec
if errorlevel 1 exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$release='dist\release\GeneralModelRegistration-v1.4.0-win64'; [void](New-Item -ItemType Directory -Path $release -Force); Copy-Item -LiteralPath 'dist\GeneralModelRegistration-v1.4.0.exe' -Destination ($release + '\GeneralModelRegistration-v1.4.0.exe') -Force; Copy-Item -LiteralPath 'PORTABLE_README.txt' -Destination ($release + '\使用说明.txt') -Force; Copy-Item -LiteralPath 'LICENSE' -Destination ($release + '\LICENSE') -Force; Copy-Item -LiteralPath 'THIRD_PARTY_NOTICES.md' -Destination ($release + '\THIRD_PARTY_NOTICES.md') -Force"
if errorlevel 1 exit /b 1
"%BUILD_PYTHON%" "scripts\collect_pyside_licenses.py" "dist\release\GeneralModelRegistration-v1.4.0-win64\PySide6-LICENSES"
if errorlevel 1 exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path 'dist\release\GeneralModelRegistration-v1.4.0-win64\*' -DestinationPath 'dist\GeneralModelRegistration-v1.4.0-win64.zip' -CompressionLevel Optimal -Force; $h=(Get-FileHash 'dist\GeneralModelRegistration-v1.4.0-win64.zip' -Algorithm SHA256).Hash; Set-Content -LiteralPath 'dist\GeneralModelRegistration-v1.4.0-win64.zip.sha256.txt' -Value ($h + '  GeneralModelRegistration-v1.4.0-win64.zip') -Encoding ascii"
if errorlevel 1 exit /b 1

echo Build complete: dist\GeneralModelRegistration-v1.4.0-win64.zip
