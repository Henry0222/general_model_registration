from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


def _log_path() -> Path:
    local_app_data = Path.home() / "AppData" / "Local"
    base = local_app_data if local_app_data.is_dir() else Path.home()
    directory = base / "GeneralModelRegistration"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "error.log"


def _install_exception_logging() -> None:
    path = _log_path()
    logging.basicConfig(
        filename=path,
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )

    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logging.error("Unhandled exception\n%s", details)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication([])
            QMessageBox.critical(
                None,
                "程序发生错误",
                f"错误信息已保存到：\n{path}\n\n{exc_value}",
            )
        except Exception:
            pass

    sys.excepthook = handle_exception


if __name__ == "__main__":
    _install_exception_logging()
    from .gui import main

    main()
