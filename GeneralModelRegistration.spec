# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

open3d_datas, open3d_binaries, open3d_hiddenimports = collect_all("open3d")
project_hiddenimports = collect_submodules("auto_alignment")

analysis = Analysis(
    ["scripts/gui_entry.py"],
    pathex=["src"],
    binaries=open3d_binaries,
    datas=open3d_datas,
    hiddenimports=open3d_hiddenimports + project_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "IPython",
        "ipywidgets",
        "jupyter",
        "matplotlib",
        "pandas",
        "sklearn",
        "torch",
        "tkinter",
    ],
    noarchive=False,
    optimize=1,
)

# Some development environments place Poppler's versioned ICU runtime on
# PATH. PyInstaller may then mistake that DLL for the unversioned Windows ICU
# shim required by Qt6Core, producing WinError 127 at application startup.
# Windows 10/11 already provide the correct icuuc.dll in System32.
incompatible_icu_names = {"icuuc.dll", "icudt78.dll"}
analysis.binaries = [
    entry
    for entry in analysis.binaries
    if Path(entry[0]).name.lower() not in incompatible_icu_names
]
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="GeneralModelRegistration-v1.4.1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
