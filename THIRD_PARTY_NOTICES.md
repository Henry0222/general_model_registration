# Third-party notices

General Model Registration depends on third-party open-source software. The dependency versions actually included in a binary build are resolved from `pyproject.toml`; review their license files again before publishing each release.

## Runtime dependencies

- NumPy — BSD 3-Clause License — https://numpy.org/
- Open3D — MIT License — https://www.open3d.org/
- PySide6 / Qt for Python — LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only, with commercial terms also available from Qt — https://doc.qt.io/qtforpython-6/licenses.html
- CPython — Python Software Foundation License — https://www.python.org/

## Build and development dependencies

- PyInstaller — GPL-2.0-or-later with a special exception for distributing bundled applications — https://pyinstaller.org/
- pytest — MIT License — https://pytest.org/
- setuptools — MIT License — https://setuptools.pypa.io/

The project license applies only to this project's own source code. Each third-party component remains under its respective license. Binary redistributors must preserve all notices required by those licenses.

The Windows build copies the license files shipped in the installed PySide6 package into the release archive. This notice is informational and is not legal advice; redistributors remain responsible for satisfying the applicable Qt/PySide6 terms.
