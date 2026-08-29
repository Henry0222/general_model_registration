@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Development environment not found. Run install_windows.bat first.
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1

python -m pytest
if errorlevel 1 exit /b 1

if exist "build\GeneralModelRegistration" rmdir /s /q "build\GeneralModelRegistration"
if exist "dist\GeneralModelRegistration-v1.3.0.exe" del /q "dist\GeneralModelRegistration-v1.3.0.exe"
if exist "dist\release\GeneralModelRegistration-v1.3.0-win64" rmdir /s /q "dist\release\GeneralModelRegistration-v1.3.0-win64"
if exist "dist\GeneralModelRegistration-v1.3.0-win64.zip" del /q "dist\GeneralModelRegistration-v1.3.0-win64.zip"
if exist "dist\GeneralModelRegistration-v1.3.0-win64.zip.sha256.txt" del /q "dist\GeneralModelRegistration-v1.3.0-win64.zip.sha256.txt"

python -m PyInstaller --noconfirm --clean --workpath "build\GeneralModelRegistration" GeneralModelRegistration.spec
if errorlevel 1 exit /b 1

mkdir "dist\release\GeneralModelRegistration-v1.3.0-win64"
copy /y "dist\GeneralModelRegistration-v1.3.0.exe" "dist\release\GeneralModelRegistration-v1.3.0-win64\" >nul
copy /y "PORTABLE_README.txt" "dist\release\GeneralModelRegistration-v1.3.0-win64\使用说明.txt" >nul
copy /y "LICENSE" "dist\release\GeneralModelRegistration-v1.3.0-win64\" >nul
copy /y "THIRD_PARTY_NOTICES.md" "dist\release\GeneralModelRegistration-v1.3.0-win64\" >nul
python "scripts\collect_pyside_licenses.py" "dist\release\GeneralModelRegistration-v1.3.0-win64\PySide6-LICENSES"
if errorlevel 1 exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path 'dist\release\GeneralModelRegistration-v1.3.0-win64\*' -DestinationPath 'dist\GeneralModelRegistration-v1.3.0-win64.zip' -CompressionLevel Optimal -Force; $h=(Get-FileHash 'dist\GeneralModelRegistration-v1.3.0-win64.zip' -Algorithm SHA256).Hash; ($h + '  GeneralModelRegistration-v1.3.0-win64.zip') | Set-Content 'dist\GeneralModelRegistration-v1.3.0-win64.zip.sha256.txt' -Encoding ascii"
if errorlevel 1 exit /b 1

echo Build complete: dist\GeneralModelRegistration-v1.3.0-win64.zip
