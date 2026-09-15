"""Measure a real main-window startup in a fresh offscreen Qt process."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child() -> None:
    started = time.perf_counter()
    from PySide6.QtWidgets import QApplication
    qt_import = time.perf_counter()
    app = QApplication([])
    app_created = time.perf_counter()
    gui = importlib.import_module("auto_alignment.gui")
    gui_import = time.perf_counter()
    gui.apply_application_style(app)
    window = gui.AlignmentWindow()
    constructed = time.perf_counter()
    window.show()
    app.processEvents()
    painted = time.perf_counter()
    print(json.dumps({
        "qt_import_s": qt_import-started,
        "application_s": app_created-qt_import,
        "gui_import_s": gui_import-app_created,
        "window_construction_s": constructed-gui_import,
        "first_events_s": painted-constructed,
        "ready_s": painted-started,
        "open3d_loaded_at_ready": "open3d" in sys.modules,
        "source": gui.__file__,
    }), flush=True)
    # Do not close through the application's settings-saving closeEvent.
    window.hide()
    app.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child()
        return
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve())
    rows = []
    for _ in range(args.runs):
        started = time.perf_counter()
        result = subprocess.run([sys.executable, __file__, "--child"], env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr)
        row = json.loads(result.stdout.strip().splitlines()[-1])
        row["process_wall_s"] = time.perf_counter()-started
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
