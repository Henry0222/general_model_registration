from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import Callable
import uuid

from PySide6.QtCore import QProcess, QSettings, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QFont, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .batch_models import (
    BatchItemResult,
    BatchOutcome,
    RegistrationJob,
)
from .config import AlignmentConfig
from .refinement_modes import REFINEMENT_MODES, GUI_DEFAULT_REFINEMENT_MODE
from .result_details import load_numeric_candidates
from .history import (
    HistoryRecord,
    record_from_results,
    records_from_manifest,
    scan_history,
)
from .version import __version__


def run_batch_analysis(*args, **kwargs):
    """Load the numerical stack in the worker, after the window is visible."""
    from .batch import run_batch_analysis as run

    return run(*args, **kwargs)


APP_TITLE = "通用模型自动配准"
SETTINGS_ORGANIZATION = "GeneralModelRegistration"
SETTINGS_APPLICATION = "GeneralModelRegistration"
SETTINGS_OUTPUT_ROOT = "paths/output_root"


def app_settings() -> QSettings:
    return QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


_WINDOWED_STANDARD_STREAMS: list[object] = []


def ensure_standard_streams() -> None:
    """Give native libraries writable streams in a windowed executable.

    PyInstaller intentionally sets ``sys.stdout`` and ``sys.stderr`` to
    ``None`` for ``console=False`` applications. Open3D occasionally emits a
    RANSAC diagnostic through Python's stream object; without this guard that
    harmless message becomes ``AttributeError: NoneType has no attribute
    write`` and aborts the registration item.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is not None:
            continue
        stream = open(os.devnull, "w", encoding="utf-8", buffering=1)
        setattr(sys, name, stream)
        _WINDOWED_STANDARD_STREAMS.append(stream)

FLIP_CHECKBOX_STYLE = """
QCheckBox {
    spacing: 7px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #2b3440;
    border-radius: 10px;
    background-color: #ffffff;
}
QCheckBox::indicator:hover {
    border-color: #1f6feb;
}
QCheckBox::indicator:checked {
    border-color: #1f6feb;
    background-color: qradialgradient(
        cx: 0.5, cy: 0.5, radius: 0.5,
        fx: 0.5, fy: 0.5,
        stop: 0 #1f6feb,
        stop: 0.45 #1f6feb,
        stop: 0.46 #ffffff,
        stop: 1 #ffffff
    );
}
QCheckBox::indicator:disabled {
    border-color: #9aa4b1;
    background-color: #f3f4f6;
}
"""


APP_STYLE = """
QMainWindow, QDialog {
    background-color: #f3f5f8;
}
QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 10pt;
    color: #1f2933;
}
QLabel#appTitle {
    font-size: 19pt;
    font-weight: 600;
    color: #14213d;
}
QLabel#appSubtitle {
    color: #5b6572;
}
QLabel#sectionHint {
    color: #6b7480;
}
QLabel#versionBadge {
    background-color: #e6eefb;
    color: #2453a3;
    border-radius: 6px;
    padding: 5px 10px;
    font-weight: 600;
}
QComboBox {
    background-color: #ffffff;
    border: 1px solid #c9d1db;
    border-radius: 5px;
    padding: 5px 10px;
    min-height: 24px;
}
QComboBox:focus { border-color: #1f6feb; }
QComboBox:disabled { color: #8a939f; background-color: #f3f4f6; }
QComboBox QAbstractItemView { selection-background-color: #dce7f8; selection-color: #14213d; }
QToolButton#advancedToggle {
    border: none;
    color: #2453a3;
    padding: 6px 8px;
    background: transparent;
}
QToolButton#advancedToggle:hover { background-color: #eef3fb; border-radius: 5px; }
QScrollArea#workspaceScroll { border: none; background: transparent; }
QWidget#workspace { background: transparent; }
QWidget#metricCell { background-color: #f2f6fb; border-radius: 6px; }
QLabel#metricValue { font-size: 16pt; font-weight: 600; color: #183c68; }
QLabel#metricTitle { font-size: 9pt; color: #596e83; }
QLabel#emptyResults { color: #708196; font-size: 11pt; }
QLabel#refinementHelp { color: #5b6572; }
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d9dee5;
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 12px 10px 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 2px;
    padding: 0 6px;
    background-color: #f3f5f8;
    color: #14213d;
    font-weight: 600;
}
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #c9d1db;
    border-radius: 5px;
    min-height: 26px;
    padding: 2px 6px;
    selection-background-color: #1f6feb;
}
QSpinBox, QDoubleSpinBox {
    min-height: 26px;
    padding-left: 4px;
}
QLineEdit:focus {
    border: 1px solid #1f6feb;
}
QLineEdit:disabled {
    background-color: #f3f4f6;
    color: #8a939f;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #c9d1db;
    border-radius: 5px;
    padding: 5px 14px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #eef3fb;
    border-color: #1f6feb;
}
QPushButton:pressed {
    background-color: #dce7f8;
}
QPushButton:disabled {
    color: #9aa4b1;
    background-color: #f3f4f6;
    border-color: #e1e6ec;
}
QPushButton[primary="true"] {
    background-color: #1f6feb;
    border: 1px solid #1a5fcc;
    color: #ffffff;
    font-weight: 600;
    padding: 7px 22px;
}
QPushButton[primary="true"]:hover {
    background-color: #2f7cf0;
}
QPushButton[primary="true"]:pressed {
    background-color: #1a5fcc;
}
QPushButton[primary="true"]:disabled {
    background-color: #a9c4f5;
    border-color: #a9c4f5;
    color: #ffffff;
}
QPushButton[danger="true"]:enabled {
    color: #b42318;
    border-color: #e5b3ae;
}
QPushButton[danger="true"]:enabled:hover {
    background-color: #fdecea;
}
QProgressBar {
    background-color: #e4e8ee;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background-color: #1f6feb;
    border-radius: 4px;
}
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f7f9fc;
    border: 1px solid #d9dee5;
    border-radius: 6px;
    gridline-color: #eceff3;
    selection-background-color: #dce7f8;
    selection-color: #1f2933;
}
QHeaderView::section {
    background-color: #f0f3f7;
    color: #3e4a58;
    border: none;
    border-bottom: 1px solid #d9dee5;
    border-right: 1px solid #e4e8ee;
    padding: 6px 8px;
    font-weight: 600;
}
QScrollArea {
    border: 1px solid #e1e6ec;
    border-radius: 6px;
    background-color: #fbfcfd;
}
QScrollArea > QWidget > QWidget {
    background-color: #fbfcfd;
}
QLabel[badge="true"] {
    background-color: #eef1f5;
    color: #3e4a58;
    border-radius: 9px;
    padding: 2px 8px;
}
QLabel[status="true"] {
    color: #5b6572;
}
QToolTip {
    background-color: #1f2933;
    color: #ffffff;
    border: none;
    padding: 6px 8px;
}
"""


def configure_flip_checkbox(checkbox: QCheckBox) -> QCheckBox:
    """Apply the high-contrast empty/black-dot normal-flip indicator."""
    checkbox.setStyleSheet(FLIP_CHECKBOX_STYLE)
    checkbox.setMinimumHeight(28)
    return checkbox


def default_output_directory() -> Path:
    documents = Path.home() / "Documents"
    base = documents if documents.is_dir() else Path.home()
    return base / "GeneralModelRegistration_Output"


class DropPathEdit(QLineEdit):
    def __init__(self, path_kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path_kind = path_kind
        self.setAcceptDrops(True)
        self.setClearButtonEnabled(True)

    def _accepted_path(self, event) -> Path | None:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        local = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
        if len(local) != 1:
            return None
        path = local[0]
        if self.path_kind == "stl":
            return path if path.is_file() and path.suffix.lower() == ".stl" else None
        return path if path.is_dir() else None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._accepted_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._accepted_path(event)
        if path is None:
            event.ignore()
            return
        self.setText(str(path.resolve()))
        event.acceptProposedAction()


class ModelDropArea(QWidget):
    files_dropped = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    @staticmethod
    def _stl_paths(event) -> list[Path]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [
            Path(url.toLocalFile()).resolve()
            for url in mime.urls()
            if url.isLocalFile()
            and Path(url.toLocalFile()).is_file()
            and Path(url.toLocalFile()).suffix.lower() == ".stl"
        ]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._stl_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = self._stl_paths(event)
        if not paths:
            event.ignore()
            return
        self.files_dropped.emit(paths)
        event.acceptProposedAction()


class ModelRow(QWidget):
    edit_requested = Signal(object)

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self.edit_mesh_path: Path | None = None
        self.edit_state_path: Path | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.number = QLabel(f"{index:02d}")
        self.number.setProperty("badge", True)
        self.number.setMinimumWidth(30)
        self.number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path_edit = DropPathEdit("stl")
        self.path_edit.setPlaceholderText("选择或拖入浮动 STL")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        self.edit_button = QPushButton("3D / 选区")
        self.edit_button.setToolTip("查看当前模型、划定配准主导选区和编辑工作副本。")
        self.edit_button.clicked.connect(lambda: self.edit_requested.emit(self))
        self.edit_badge = QLabel("未编辑")
        self.edit_badge.setProperty("badge", True)
        self.edit_badge.setMinimumWidth(66)
        self.edit_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flip_check = configure_flip_checkbox(QCheckBox("翻转法线"))
        self.flip_check.setToolTip("配准前在内存中翻转三角面绕序，不修改原始 STL。")
        self.status = QLabel("等待")
        self.status.setProperty("status", True)
        path_row.addWidget(self.number)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        controls.addWidget(self.edit_button)
        controls.addWidget(self.edit_badge)
        controls.addWidget(self.flip_check)
        controls.addStretch(1)
        controls.addWidget(self.status)
        layout.addLayout(path_row)
        layout.addLayout(controls)

    @Slot()
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择第 {self.index} 个浮动 STL",
            "",
            "STL 三角网格 (*.stl)",
        )
        if path:
            self.path_edit.setText(path)


@dataclass(frozen=True)
class BatchRequest:
    target: Path
    target_flip_normals: bool
    target_edit_state: Path | None
    jobs: tuple[RegistrationJob, ...]
    output_parent: Path
    config: AlignmentConfig
    minimum_nominal_mm: float
    maximum_nominal_mm: float


class BatchRegistrationWorker(QThread):
    progress_changed = Signal(int, float, str)
    item_finished = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, request: BatchRequest) -> None:
        super().__init__()
        self.request = request
        self._stop = threading.Event()

    def request_safe_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self.progress_changed.emit(0, 0.0, "正在加载配准引擎…")
            outcome = run_batch_analysis(
                self.request.target,
                self.request.target_flip_normals,
                self.request.jobs,
                self.request.output_parent,
                target_edit_state_path=self.request.target_edit_state,
                config=self.request.config,
                minimum_nominal_mm=self.request.minimum_nominal_mm,
                maximum_nominal_mm=self.request.maximum_nominal_mm,
                progress=lambda index, fraction, message: self.progress_changed.emit(
                    index, float(fraction), str(message)
                ),
                stop_requested=self._stop.is_set,
                item_finished=self.item_finished.emit,
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}", traceback.format_exc())
            return
        self.succeeded.emit(outcome)


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _selection_lane_text(enabled: bool, lane: str | None) -> str:
    if not enabled:
        return "全模型"
    return {
        "full_surface_baseline": "选区决策·基线最优",
        "selection_weighted_global": "选区主导·加权全局",
        "selection_strict_global": "选区主导·严格全局",
        "selection_refined_baseline": "选区主导·局部精配准",
        "selection_refined_weighted": "选区主导·加权+精配准",
        "selection_refined_strict": "选区主导·严格+精配准",
    }.get(str(lane), "选区主导")


def _open_path(path: Path) -> None:
    if path.exists():
        os.startfile(path)  # type: ignore[attr-defined]


def _viewer_command(
    results_path: Path,
    *,
    target_override: Path | None = None,
) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--viewer", str(results_path)]
    else:
        command = [sys.executable, "-m", "auto_alignment", "--viewer", str(results_path)]
    if target_override is not None:
        command.extend(("--target-override", str(target_override)))
    return command


def _launch_viewer(parent: QWidget, results_path: Path) -> None:
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        candidates = payload.get("candidate_results") or {}
        if candidates:
            selected = ((payload.get("registration") or {}).get("metrics") or {}).get("refinement") or {}
            labels = {"initial": "原版", "A": "A · 局部共同表面", "B": "B · 连续表面偏差"}
            choices = {f"主结果 · {labels.get(selected.get('selected'), '原版')}": results_path}
            for name, relative in candidates.items():
                path = (results_path.parent / str(relative)).resolve()
                if path.is_relative_to(results_path.parent.resolve()) and path.is_file():
                    choices[labels.get(name, str(name))] = path
            choice, accepted = QInputDialog.getItem(
                parent, "查看对比候选", "选择结果，可分别打开多个窗口对比：", list(choices), 0, False,
            )
            if not accepted:
                return
            results_path = choices[choice]
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        target = payload["target_mesh"]
        archived = target.get("archived_path")
        target_path = (
            (results_path.parent / str(archived)).resolve()
            if archived
            else Path(str(target.get("path", "")))
        )
        if not target_path.is_file():
            replacement, _ = QFileDialog.getOpenFileName(
                parent,
                "历史记录中的固定 STL 已移动，请重新定位",
                "",
                "STL 三角网格 (*.stl)",
            )
            if not replacement:
                return
            target_path = Path(replacement).resolve()
        output_name = payload["outputs"]["aligned_current_stl"]
        aligned_path = (results_path.parent / str(output_name)).resolve()
        if not aligned_path.is_file():
            raise FileNotFoundError(f"找不到已配准 STL：{aligned_path}")
    except Exception as error:
        QMessageBox.critical(parent, "无法打开历史结果", str(error))
        return
    override = target_path if not archived or not (results_path.parent / str(archived)).is_file() else None
    subprocess.Popen(
        _viewer_command(results_path, target_override=override),
        cwd=str(results_path.parent),
        creationflags=0x08000000 if os.name == "nt" else 0,
    )


class HistoryScanWorker(QThread):
    scanned = Signal(object)

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def run(self) -> None:
        try:
            records = scan_history(self.root)
        except Exception:
            records = []
        self.scanned.emit(records)


class HistoryDialog(QDialog):
    def __init__(self, initial_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("既往配准记录")
        self.resize(1120, 620)
        self.records: list[HistoryRecord] = []
        self.root = initial_root
        self._scan_worker: HistoryScanWorker | None = None
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.root_label = QLabel(str(initial_root))
        choose_root = QPushButton("选择记录根目录…")
        choose_root.clicked.connect(self._choose_root)
        import_button = QPushButton("导入结果文件…")
        import_button.clicked.connect(self._import_result)
        manual_button = QPushButton("手动查看固定 STL + 已配准 STL…")
        manual_button.clicked.connect(self._manual_pair)
        top.addWidget(self.root_label, 1)
        top.addWidget(choose_root)
        top.addWidget(import_button)
        top.addWidget(manual_button)
        layout.addLayout(top)
        self.scan_status = QLabel("")
        layout.addWidget(self.scan_status)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ("时间", "浮动模型", "状态", "可信度", "RMS", "P90", "HD95", "翻转法线")
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        view = QPushButton("打开 3D 结果")
        view.setProperty("primary", True)
        view.clicked.connect(self._view)
        log = QPushButton("查看日志")
        log.clicked.connect(self._log)
        folder = QPushButton("打开结果目录")
        folder.clicked.connect(self._folder)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        actions.addWidget(view)
        actions.addWidget(log)
        actions.addWidget(folder)
        actions.addStretch(1)
        actions.addWidget(close)
        layout.addLayout(actions)
        self._scan()

    def _selected(self) -> HistoryRecord | None:
        rows = self.table.selectionModel().selectedRows()
        return self.records[rows[0].row()] if rows else None

    def _populate(self) -> None:
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            values = (
                record.timestamp[:19].replace("T", " "),
                record.source_name,
                record.status,
                record.confidence,
                _format_metric(record.symmetric_rms_mm),
                _format_metric(record.p90_mm),
                _format_metric(record.hd95_mm),
                "是" if record.flip_normals else "否",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        if self.records:
            self.table.selectRow(0)
        self.scan_status.setText(f"共 {len(self.records)} 条记录。")

    def _scan(self) -> None:
        self.root_label.setText(str(self.root))
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.scanned.disconnect(self._on_scanned)
        self.scan_status.setText("正在扫描既往记录…")
        self.table.setRowCount(0)
        self._scan_worker = HistoryScanWorker(self.root)
        self._scan_worker.scanned.connect(self._on_scanned)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_worker.start()

    @Slot(object)
    def _on_scanned(self, value: object) -> None:
        self.records = list(value)
        self._populate()

    def closeEvent(self, event: QCloseEvent) -> None:
        worker = self._scan_worker
        if worker is not None and worker.isRunning():
            worker.scanned.disconnect(self._on_scanned)
            worker.wait(5000)
        super().closeEvent(event)

    @Slot()
    def _choose_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择既往结果根目录", str(self.root))
        if path:
            self.root = Path(path).resolve()
            self._scan()

    @Slot()
    def _import_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 batch_results.json 或 results.json",
            str(self.root),
            "配准结果 (batch_results.json results.json);;JSON (*.json)",
        )
        if not path:
            return
        try:
            selected = Path(path)
            imported = (
                records_from_manifest(selected)
                if selected.name == "batch_results.json"
                else [record_from_results(selected)]
            )
        except Exception as error:
            QMessageBox.critical(self, "导入失败", str(error))
            return
        existing = {(record.results_path, record.directory) for record in self.records}
        self.records = imported + [
            record
            for record in self.records
            if (record.results_path, record.directory) not in existing.intersection(
                {(item.results_path, item.directory) for item in imported}
            )
        ]
        self._populate()

    @Slot()
    def _manual_pair(self) -> None:
        target, _ = QFileDialog.getOpenFileName(self, "选择固定 STL", "", "STL (*.stl)")
        if not target:
            return
        aligned, _ = QFileDialog.getOpenFileName(self, "选择已配准浮动 STL", "", "STL (*.stl)")
        if not aligned:
            return
        command = (
            [sys.executable, "--viewer-pair", target, aligned]
            if getattr(sys, "frozen", False)
            else [sys.executable, "-m", "auto_alignment", "--viewer-pair", target, aligned]
        )
        subprocess.Popen(command, creationflags=0x08000000 if os.name == "nt" else 0)

    @Slot()
    def _view(self) -> None:
        record = self._selected()
        if record is None or record.results_path is None or not record.results_path.is_file():
            QMessageBox.information(self, "没有 3D 结果", "该记录没有可查看的配准结果或失败候选。")
            return
        _launch_viewer(self, record.results_path)

    @Slot()
    def _log(self) -> None:
        record = self._selected()
        if record and record.log_path and record.log_path.is_file():
            _open_path(record.log_path)
        else:
            QMessageBox.information(self, "没有日志", "该记录没有可用日志文件。")

    @Slot()
    def _folder(self) -> None:
        record = self._selected()
        if record:
            _open_path(record.directory)


class AlignmentWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} v{__version__}")
        self.resize(1240, 820)
        self.setMinimumSize(920, 720)
        self._worker: BatchRegistrationWorker | None = None
        self._outcome: BatchOutcome | None = None
        self._items: dict[int, BatchItemResult] = {}
        self._editor_processes: dict[Path, QProcess] = {}
        self._session_edit_state_paths: set[Path] = set()
        self._target_edit_mesh_path: Path | None = None
        self._target_edit_state_path: Path | None = None
        self.model_rows: list[ModelRow] = []
        self._build_ui()
        self._restore_settings()

    def _restore_settings(self) -> None:
        stored = app_settings().value(SETTINGS_OUTPUT_ROOT, "", type=str)
        if stored and Path(stored).expanduser().is_dir():
            self.output_edit.setText(stored)

    def _save_settings(self) -> None:
        text = self.output_edit.text().strip()
        if text:
            app_settings().setValue(SETTINGS_OUTPUT_ROOT, text)

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(12)
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel(APP_TITLE)
        title.setObjectName("appTitle")
        subtitle = QLabel(
            "导入模型  /  选择配准路线  /  复核数值与三维结果"
        )
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)
        version = QLabel(f"v{__version__}")
        version.setObjectName("versionBadge")
        header.addWidget(version, 0, Qt.AlignmentFlag.AlignVCenter)
        history_button = QPushButton("历史记录")
        history_button.clicked.connect(self._history)
        header.addWidget(history_button)
        outer.addLayout(header)

        workspace = QWidget()
        workspace.setObjectName("workspace")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 4, 0)
        workspace_layout.setSpacing(8)
        body = QHBoxLayout()
        body.setSpacing(14)
        self.workspace_scroll = QScrollArea()
        self.workspace_scroll.setObjectName("workspaceScroll")
        self.workspace_scroll.setWidgetResizable(True)
        self.workspace_scroll.setWidget(workspace)
        self.workspace_scroll.setMinimumHeight(260)

        self.files_group = QGroupBox("01  模型与输出")
        files = QVBoxLayout(self.files_group)
        files.setSpacing(8)
        fixed_row = QHBoxLayout()
        fixed_row.setSpacing(8)
        fixed_label = QLabel("参考模型 · 固定不动")
        fixed_label.setObjectName("sectionHint")
        files.addWidget(fixed_label)
        self.target_edit = DropPathEdit("stl")
        self.target_edit.setPlaceholderText("选择或拖入固定 STL")
        fixed_row.addWidget(self.target_edit, 1)
        fixed_browse = QPushButton("浏览…")
        fixed_browse.clicked.connect(self._choose_target)
        fixed_row.addWidget(fixed_browse)
        files.addLayout(fixed_row)
        fixed_controls = QHBoxLayout()
        fixed_controls.setSpacing(6)
        self.target_edit_button = QPushButton("3D / 选区")
        self.target_edit_button.setToolTip("查看固定模型、划定配准主导选区和编辑工作副本。")
        self.target_edit_button.clicked.connect(self._edit_target_model)
        fixed_controls.addWidget(self.target_edit_button)
        self.target_edit_badge = QLabel("未编辑")
        self.target_edit_badge.setProperty("badge", True)
        self.target_edit_badge.setMinimumWidth(66)
        self.target_edit_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fixed_controls.addWidget(self.target_edit_badge)
        self.target_edit.textChanged.connect(
            self._target_model_path_changed
        )
        self.target_flip = configure_flip_checkbox(QCheckBox("翻转法线"))
        self.target_flip.setToolTip("固定模型法线将成为彩虹图正负偏差的唯一方向基准。")
        fixed_controls.addWidget(self.target_flip)
        fixed_controls.addStretch(1)
        files.addLayout(fixed_controls)

        count_row = QHBoxLayout()
        count_row.setSpacing(8)
        count_label = QLabel("待配准模型")
        count_label.setMinimumWidth(76)
        count_row.addWidget(count_label)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(1)
        self.count_spin.setMinimumWidth(80)
        self.count_spin.valueChanged.connect(self._set_model_count)
        count_row.addWidget(self.count_spin)
        count_hint = QLabel("支持批量拖入 STL")
        count_hint.setObjectName("sectionHint")
        count_row.addWidget(count_hint)
        count_row.addStretch(1)
        files.addLayout(count_row)

        self.model_area = ModelDropArea()
        self.model_layout = QVBoxLayout(self.model_area)
        self.model_layout.setContentsMargins(6, 6, 6, 6)
        self.model_layout.setSpacing(4)
        self.model_layout.addStretch(1)
        self.model_area.files_dropped.connect(self._fill_dropped_models)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(92)
        scroll.setWidget(self.model_area)
        files.addWidget(scroll)

        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        output_label = QLabel("保存位置")
        output_label.setObjectName("sectionHint")
        files.addWidget(output_label)
        self.output_edit = DropPathEdit("directory")
        self.output_edit.setText(str(default_output_directory()))
        output_row.addWidget(self.output_edit, 1)
        output_browse = QPushButton("浏览…")
        output_browse.clicked.connect(self._choose_output)
        output_row.addWidget(output_browse)
        files.addLayout(output_row)
        workspace_layout.addWidget(self.files_group)
        self._set_model_count(1)

        self.params_group = QGroupBox("02  配准路线")
        params = QVBoxLayout(self.params_group)
        params.setSpacing(8)
        route_row = QHBoxLayout()
        route_row.setSpacing(16)
        self.refinement_combo = QComboBox()
        self.refinement_combo.setMinimumWidth(216)
        for mode in ("auto", "A", "B", "compare", "baseline"):
            self.refinement_combo.addItem(REFINEMENT_MODES[mode][0], mode)
        self.refinement_combo.setCurrentIndex(
            self.refinement_combo.findData(GUI_DEFAULT_REFINEMENT_MODE)
        )
        self.refinement_help = QLabel()
        self.refinement_help.setObjectName("refinementHelp")
        self.refinement_help.setWordWrap(True)
        self.refinement_help.setMinimumHeight(48)
        self.refinement_combo.currentIndexChanged.connect(self._update_refinement_help)
        self._update_refinement_help()
        route_row.addWidget(self.refinement_combo)
        route_row.addStretch(1)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setText("高级参数")
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        route_row.addWidget(self.advanced_toggle)
        params.addLayout(route_row)
        params.addWidget(self.refinement_help)
        self.advanced_panel = QWidget()
        params_columns = QVBoxLayout(self.advanced_panel)
        params_columns.setContentsMargins(0, 4, 0, 0)
        params_columns.setSpacing(14)
        registration_form = QFormLayout()
        registration_form.setHorizontalSpacing(12)
        registration_form.setVerticalSpacing(8)
        registration_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        deviation_form = QFormLayout()
        deviation_form.setHorizontalSpacing(12)
        deviation_form.setVerticalSpacing(8)
        deviation_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(5.0, 100.0)
        self.overlap_spin.setValue(20.0)
        self.overlap_spin.setSuffix(" %")
        self.coverage_spin = QDoubleSpinBox()
        self.coverage_spin.setRange(0.01, 5.0)
        self.coverage_spin.setDecimals(2)
        self.coverage_spin.setValue(0.60)
        self.coverage_spin.setSuffix(" mm")
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(5_000, 500_000)
        self.samples_spin.setSingleStep(5_000)
        self.samples_spin.setValue(30_000)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(5_000, 500_000)
        self.iterations_spin.setSingleStep(5_000)
        self.iterations_spin.setValue(80_000)
        self.exhaustive_orientation_check = configure_flip_checkbox(
            QCheckBox("启用（较慢）")
        )
        self.exhaustive_orientation_check.setToolTip(
            "在粗配准阶段枚举 PCA 三轴的全部合法排列和正负号，"
            "并对近似轴对称模型补充绕轴姿态。不会生成镜像或反射。"
        )
        self.exhaustive_orientation_step = QSpinBox()
        self.exhaustive_orientation_step.setRange(5, 90)
        self.exhaustive_orientation_step.setSingleStep(5)
        self.exhaustive_orientation_step.setValue(30)
        self.exhaustive_orientation_step.setSuffix("°")
        self.exhaustive_orientation_step.setEnabled(False)
        self.exhaustive_orientation_check.toggled.connect(
            self.exhaustive_orientation_step.setEnabled
        )
        self.minimum_nominal_spin = QDoubleSpinBox()
        self.minimum_nominal_spin.setRange(-100.0, 0.0)
        self.minimum_nominal_spin.setDecimals(3)
        self.minimum_nominal_spin.setValue(-0.05)
        self.minimum_nominal_spin.setSuffix(" mm")
        self.maximum_nominal_spin = QDoubleSpinBox()
        self.maximum_nominal_spin.setRange(0.0, 100.0)
        self.maximum_nominal_spin.setDecimals(3)
        self.maximum_nominal_spin.setValue(0.05)
        self.maximum_nominal_spin.setSuffix(" mm")
        registration_form.addRow("最小相似表面覆盖率", self.overlap_spin)
        registration_form.addRow("覆盖距离", self.coverage_spin)
        registration_form.addRow("表面采样点", self.samples_spin)
        registration_form.addRow("RANSAC 最大迭代", self.iterations_spin)
        deviation_form.addRow("彻底检查可能朝向", self.exhaustive_orientation_check)
        deviation_form.addRow("轴对称角度步长", self.exhaustive_orientation_step)
        deviation_form.addRow("最小名义偏差", self.minimum_nominal_spin)
        deviation_form.addRow("最大名义偏差", self.maximum_nominal_spin)
        params_columns.addLayout(registration_form, 1)
        params_columns.addLayout(deviation_form, 1)
        params.addWidget(self.advanced_panel)
        self.advanced_panel.hide()
        workspace_layout.addWidget(self.params_group)
        workspace_layout.addStretch(1)
        body.addWidget(self.workspace_scroll, 5)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.start_button = QPushButton("开始配准")
        self.start_button.setProperty("primary", True)
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("取消并停止")
        self.stop_button.setProperty("danger", True)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.view_button = QPushButton("查看 3D / 候选")
        self.view_button.clicked.connect(self._view_selected)
        self.log_button = QPushButton("日志")
        self.log_button.clicked.connect(self._log_selected)
        self.folder_button = QPushButton("结果文件夹")
        self.folder_button.clicked.connect(self._open_batch_folder)
        for button in (self.view_button, self.log_button, self.folder_button):
            button.setEnabled(False)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        run_hint = QLabel("多个模型按顺序独立配准 · 人工选区优先")
        run_hint.setObjectName("sectionHint")
        actions.addWidget(run_hint)
        outer.addLayout(actions)

        status_block = QVBoxLayout()
        status_block.setSpacing(4)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.status = QLabel("等待选择模型。")
        self.status.setProperty("status", True)
        self.status.setWordWrap(True)
        status_block.addWidget(self.progress)
        status_block.addWidget(self.status)
        outer.addLayout(status_block)

        results_group = QGroupBox("03  结果与数值")
        result_layout = QVBoxLayout(results_group)
        result_layout.setSpacing(8)
        result_actions = QHBoxLayout()
        result_hint = QLabel("选择模型查看明细；切换候选可比较具体偏差。")
        result_hint.setObjectName("sectionHint")
        result_hint.setWordWrap(True)
        result_layout.addWidget(result_hint)
        result_actions.addWidget(self.view_button)
        result_actions.addWidget(self.log_button)
        result_actions.addWidget(self.folder_button)
        result_actions.addStretch(1)
        result_layout.addLayout(result_actions)
        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels(("序号", "浮动模型", "状态", "可信度", "配准依据", "RMS (mm)", "P95 (mm)", "用时 (s)"))
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setShowGrid(False)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.verticalHeader().setDefaultSectionSize(30)
        header_view = self.results_table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.results_table.setColumnWidth(1, 160)
        header_view.setHighlightSections(False)
        header_view.setStretchLastSection(False)
        self.results_table.itemSelectionChanged.connect(self._selection_changed)
        self.results_stack = QStackedWidget()
        self.results_stack.setMinimumHeight(82)
        self.empty_results = QLabel("尚无配准结果\n添加参考模型与待配准模型后，点击「开始配准」。")
        self.empty_results.setObjectName("emptyResults")
        self.empty_results.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_stack.addWidget(self.empty_results)
        self.results_stack.addWidget(self.results_table)
        result_layout.addWidget(self.results_stack, 1)

        detail_row = QHBoxLayout()
        detail_label = QLabel("数值明细")
        detail_row.addWidget(detail_label)
        self.numeric_candidate_combo = QComboBox()
        self.numeric_candidate_combo.setMinimumWidth(210)
        self.numeric_candidate_combo.setEnabled(False)
        self.numeric_candidate_combo.currentIndexChanged.connect(self._show_numeric_candidate)
        detail_row.addWidget(self.numeric_candidate_combo)
        self.numeric_notice = QLabel("配准后显示所选模型的表面偏差。")
        self.numeric_notice.setObjectName("sectionHint")
        self.numeric_notice.setWordWrap(True)
        detail_row.addWidget(self.numeric_notice, 1)
        result_layout.addLayout(detail_row)
        self.numeric_values: dict[str, QLabel] = {}
        self._numeric_candidates: list[tuple[str, dict]] = []
        metric_row = QGridLayout()
        for index, (key, label) in enumerate((("symmetric_rms_mm", "RMS"), ("mean_mm", "平均距离"),
                           ("median_mm", "中位数"), ("hd95_mm", "P95 / HD95"),
                           ("maximum_mm", "最大距离"))):
            cell = QWidget()
            cell.setObjectName("metricCell")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(12, 7, 12, 7)
            cell_layout.setSpacing(1)
            caption = QLabel(label + " · mm")
            caption.setObjectName("metricTitle")
            value = QLabel("—")
            value.setObjectName("metricValue")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.numeric_values[key] = value
            cell_layout.addWidget(caption)
            cell_layout.addWidget(value)
            metric_row.addWidget(cell, index // 2, index % 2)
        time_cell = QWidget()
        time_cell.setObjectName("metricCell")
        time_layout = QVBoxLayout(time_cell)
        time_layout.setContentsMargins(12, 7, 12, 7)
        time_layout.setSpacing(1)
        time_title = QLabel("本模型用时 · s")
        time_title.setObjectName("metricTitle")
        self.model_elapsed_value = QLabel("—")
        self.model_elapsed_value.setObjectName("metricValue")
        self.model_elapsed_value.setToolTip("本模型的批次处理耗时；切换候选不会重新计算。")
        time_layout.addWidget(time_title)
        time_layout.addWidget(self.model_elapsed_value)
        metric_row.addWidget(time_cell, 2, 1)
        metric_row.setColumnStretch(0, 1)
        metric_row.setColumnStretch(1, 1)
        result_layout.addLayout(metric_row)
        metric_hint = QLabel("双向采样的最近表面距离，数值越小表示表面越贴合；不等同于有真值参照的中心 / 角度误差。")
        metric_hint.setObjectName("sectionHint")
        metric_hint.setWordWrap(True)
        result_layout.addWidget(metric_hint)
        body.addWidget(results_group, 6)
        outer.insertLayout(1, body, 1)
        self.setCentralWidget(central)

    @Slot()
    def _update_refinement_help(self) -> None:
        mode = self.refinement_combo.currentData()
        self.refinement_help.setText(REFINEMENT_MODES[mode][1])

    @Slot(bool)
    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _load_numeric_details(self, item: BatchItemResult | None) -> None:
        self.model_elapsed_value.setText(f"{item.elapsed_seconds:.2f}" if self._outcome and item else "—")
        self.numeric_candidate_combo.blockSignals(True)
        self.numeric_candidate_combo.clear()
        self._numeric_candidates = []
        self.numeric_notice.setText("配准后显示所选模型的表面偏差。")
        if self._outcome and item and item.results_json:
            try:
                root = self._outcome.batch_directory.resolve()
                path = (root / item.results_json).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("结果路径超出本批次目录。")
                self._numeric_candidates = load_numeric_candidates(path)
                for name, payload in self._numeric_candidates:
                    registration = payload.get("registration") or {}
                    selected = ((registration.get("metrics") or {}).get("refinement") or {}).get("selected")
                    method = {"initial": "原版", "A": "A", "B": "B"}.get(selected, "原版" if name == "主结果" else name)
                    candidate_name = "原版" if name == "initial" else name
                    self.numeric_candidate_combo.addItem(f"{name} · {method}" if name == "主结果" else f"候选 · {candidate_name}")
            except (OSError, ValueError, TypeError, AttributeError) as error:
                self._numeric_candidates = []
                self.numeric_candidate_combo.clear()
                self.numeric_notice.setText(f"数值读取失败：{error}")
        elif item and item.error:
            self.numeric_notice.setText(f"无数值结果：{item.error}")
        self.numeric_candidate_combo.setEnabled(bool(self._numeric_candidates))
        self.numeric_candidate_combo.blockSignals(False)
        self._show_numeric_candidate()

    @Slot()
    def _show_numeric_candidate(self) -> None:
        index = self.numeric_candidate_combo.currentIndex()
        payload = self._numeric_candidates[index][1] if 0 <= index < len(self._numeric_candidates) else {}
        stats = payload.get("distance_statistics") or {}
        stats = stats if isinstance(stats, dict) else {}
        for key, label in self.numeric_values.items():
            try:
                value = float(stats.get(key))
                if not math.isfinite(value):
                    raise ValueError("nonfinite metric")
                label.setText(f"{value:.6f}" if value == 0 or abs(value) >= .000001 else f"{value:.3e}")
                label.setToolTip(f"{value:.12g} mm")
            except (TypeError, ValueError):
                label.setText("—")
                label.setToolTip("此结果未提供该指标。")
        if payload:
            registration = payload.get("registration") or {}
            if not isinstance(registration, dict):
                registration = {}
            notice = payload.get("read_error")
            status = {"success": "已完成", "warning": "有警告，请复核", "failed": "失败候选，仅供检查"}.get(registration.get("status"), "状态未知")
            self.numeric_notice.setText(f"无法读取候选：{notice}" if notice else status)

    @Slot(int)
    def _set_model_count(self, count: int) -> None:
        while len(self.model_rows) < count:
            row = ModelRow(len(self.model_rows) + 1)
            row.edit_requested.connect(self._edit_source_model)
            row.path_edit.textChanged.connect(
                lambda text, model_row=row: self._source_model_path_changed(
                    model_row, text
                )
            )
            self.model_rows.append(row)
            self.model_layout.insertWidget(self.model_layout.count() - 1, row)
        while len(self.model_rows) > count:
            row = self.model_rows.pop()
            row.setParent(None)
            row.deleteLater()

    @Slot(object)
    def _fill_dropped_models(self, paths: object) -> None:
        dropped = list(paths)
        items = dropped[:50]
        if len(dropped) > 50:
            QMessageBox.warning(self, "文件过多", "单批次最多处理 50 个浮动 STL。")
        self.count_spin.setValue(max(1, len(items)))
        for row, path in zip(self.model_rows, items):
            row.path_edit.setText(str(path))

    @Slot()
    def _choose_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择固定/参考 STL", "", "STL (*.stl)")
        if path:
            self.target_edit.setText(path)

    @Slot()
    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择结果根目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    @staticmethod
    def _edit_badge_text(state_path: Path) -> str:
        if not state_path.is_file():
            return "未编辑"
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            selected = int(payload.get("selected_count", 0))
            deleted = int(payload.get("deleted_count", 0))
            if selected or deleted:
                return f"选 {selected:,} / 删 {deleted:,}"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return "状态无效"
        return "无选区"

    @staticmethod
    def _valid_stl_path(value: str) -> Path | None:
        text = value.strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_file() or path.suffix.lower() != ".stl":
            return None
        return path.resolve()

    def _new_session_edit_state_path(self) -> Path:
        path = (
            Path(tempfile.gettempdir())
            / "GeneralModelRegistration"
            / "session_edits"
            / f"{uuid.uuid4().hex}.json"
        )
        self._session_edit_state_paths.add(path)
        return path

    def _discard_session_edit_states(self) -> None:
        for path in self._session_edit_state_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._session_edit_state_paths.clear()

    def _target_model_path_changed(self, value: str) -> None:
        mesh_path = self._valid_stl_path(value)
        if mesh_path == self._target_edit_mesh_path:
            return
        self._target_edit_mesh_path = mesh_path
        self._target_edit_state_path = (
            self._new_session_edit_state_path() if mesh_path is not None else None
        )
        self.target_edit_badge.setText("未编辑")
        self.target_edit_badge.setToolTip("")

    def _source_model_path_changed(self, row: ModelRow, value: str) -> None:
        mesh_path = self._valid_stl_path(value)
        if mesh_path == row.edit_mesh_path:
            return
        row.edit_mesh_path = mesh_path
        row.edit_state_path = (
            self._new_session_edit_state_path() if mesh_path is not None else None
        )
        row.edit_badge.setText("未编辑")
        row.edit_badge.setToolTip("")

    @staticmethod
    def _model_editor_command(mesh_path: Path, state_path: Path) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--model-editor", str(mesh_path), str(state_path)]
        return [
            sys.executable,
            "-m",
            "auto_alignment",
            "--model-editor",
            str(mesh_path),
            str(state_path),
        ]

    def _launch_model_editor(
        self,
        mesh_path: Path,
        state_path: Path | None,
        badge: QLabel,
        button: QPushButton,
        current_path: Callable[[], str],
        current_state_path: Callable[[], Path | None],
    ) -> None:
        if not mesh_path.is_file() or mesh_path.suffix.lower() != ".stl":
            QMessageBox.warning(self, "无法查看模型", "请先选择有效的 STL 文件。")
            return
        mesh_path = mesh_path.resolve()
        if state_path is None:
            QMessageBox.warning(self, "无法查看模型", "模型状态尚未初始化，请重新选择 STL。")
            return
        running = self._editor_processes.get(state_path)
        if running is not None and running.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "编辑器已打开", "该模型的选区编辑器正在运行，请先关闭它。")
            return
        command = self._model_editor_command(mesh_path, state_path)
        # Parented to the application rather than the window so that closing
        # the main window does not kill editors the operator is still using.
        process = QProcess(QApplication.instance())
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(mesh_path.parent))
        self._editor_processes[state_path] = process
        badge.setText("编辑中…")
        button.setEnabled(False)

        def release() -> None:
            if self._editor_processes.get(state_path) is process:
                del self._editor_processes[state_path]
            button.setEnabled(True)
            process.deleteLater()

        def finished(*_args) -> None:
            if (
                Path(current_path().strip()).expanduser().resolve() == mesh_path
                and current_state_path() == state_path
            ):
                badge.setText(self._edit_badge_text(state_path))
                badge.setToolTip(str(state_path))
            release()

        def error_occurred(error) -> None:
            if error == QProcess.ProcessError.FailedToStart:
                badge.setText("启动失败")
                release()

        process.finished.connect(finished)
        process.errorOccurred.connect(error_occurred)
        process.start()

    @Slot()
    def _edit_target_model(self) -> None:
        self._launch_model_editor(
            Path(self.target_edit.text().strip()).expanduser(),
            self._target_edit_state_path,
            self.target_edit_badge,
            self.target_edit_button,
            self.target_edit.text,
            lambda: self._target_edit_state_path,
        )

    @Slot(object)
    def _edit_source_model(self, value: object) -> None:
        row = value
        assert isinstance(row, ModelRow)
        self._launch_model_editor(
            Path(row.path_edit.text().strip()).expanduser(),
            row.edit_state_path,
            row.edit_badge,
            row.edit_button,
            row.path_edit.text,
            lambda: row.edit_state_path,
        )

    def _request(self) -> BatchRequest:
        target = Path(self.target_edit.text().strip()).expanduser().resolve()
        if not target.is_file() or target.suffix.lower() != ".stl":
            raise ValueError("请选择有效的固定 STL。")
        output = Path(self.output_edit.text().strip()).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        jobs: list[RegistrationJob] = []
        for row in self.model_rows:
            source = Path(row.path_edit.text().strip()).expanduser().resolve()
            if not source.is_file() or source.suffix.lower() != ".stl":
                raise ValueError(f"第 {row.index} 个浮动模型不是有效 STL。")
            if source == target:
                raise ValueError(f"第 {row.index} 个浮动模型与固定模型相同。")
            state_path = row.edit_state_path
            jobs.append(
                RegistrationJob(
                    row.index,
                    source,
                    row.flip_check.isChecked(),
                    state_path if state_path is not None and state_path.is_file() else None,
                )
            )
        if self.minimum_nominal_spin.value() >= self.maximum_nominal_spin.value():
            raise ValueError("最小名义偏差必须小于最大名义偏差。")
        config = AlignmentConfig(
            refinement_mode=str(self.refinement_combo.currentData()),
            global_sample_points=self.samples_spin.value(),
            metric_sample_points=self.samples_spin.value(),
            ransac_max_iterations=self.iterations_spin.value(),
            partial_overlap_threshold=self.overlap_spin.value() / 100.0,
            coverage_distance_mm=self.coverage_spin.value(),
            exhaustive_orientation_search=self.exhaustive_orientation_check.isChecked(),
            exhaustive_orientation_angle_step_degrees=float(
                self.exhaustive_orientation_step.value()
            ),
        )
        return BatchRequest(
            target,
            self.target_flip.isChecked(),
            (
                self._target_edit_state_path
                if self._target_edit_state_path is not None
                and self._target_edit_state_path.is_file()
                else None
            ),
            tuple(jobs),
            output,
            config,
            self.minimum_nominal_spin.value(),
            self.maximum_nominal_spin.value(),
        )

    @Slot()
    def _start(self) -> None:
        try:
            request = self._request()
        except Exception as error:
            QMessageBox.warning(self, "输入不完整", str(error))
            return
        self._save_settings()
        self._outcome = None
        self._items.clear()
        self._load_numeric_details(None)
        self.results_stack.setCurrentWidget(self.results_table)
        self.results_table.setRowCount(len(request.jobs))
        for row_index, job in enumerate(request.jobs):
            values = (f"{job.index:02d}", job.source_path.name, "等待", "—", "—", "—", "—", "—")
            for column, value in enumerate(values):
                self.results_table.setItem(row_index, column, QTableWidgetItem(value))
            self.model_rows[row_index].status.setText("等待")
        self.progress.setValue(0)
        self._set_busy(True)
        self._worker = BatchRegistrationWorker(request)
        self._worker.progress_changed.connect(self._on_progress)
        self._worker.item_finished.connect(self._on_item_finished)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _set_busy(self, busy: bool) -> None:
        self.files_group.setEnabled(not busy)
        self.params_group.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        if busy:
            for button in (self.view_button, self.log_button, self.folder_button):
                button.setEnabled(False)

    @Slot()
    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_safe_stop()
            self.stop_button.setEnabled(False)
            self.status.setText("已请求取消，将在当前计算阶段结束后停止…")

    @Slot(int, float, str)
    def _on_progress(self, index: int, fraction: float, message: str) -> None:
        self.progress.setValue(round(max(0.0, min(1.0, fraction)) * 100))
        self.status.setText(message)
        if 1 <= index <= len(self.model_rows):
            self.model_rows[index - 1].status.setText("处理中")

    @Slot(object)
    def _on_item_finished(self, value: object) -> None:
        item = value
        assert isinstance(item, BatchItemResult)
        self._items[item.index] = item
        row = item.index - 1
        displayed_status = "失败（可查看）" if item.review_only else {
            "success": "完成", "warning": "警告", "failed": "失败",
            "cancelled": "取消", "skipped": "跳过",
        }.get(item.status, item.status)
        self.model_rows[row].status.setText(displayed_status)
        values = (
            displayed_status,
            item.confidence,
            (_selection_lane_text(item.selection_enabled, item.selection_lane)
             if item.refinement_mode == "baseline" or item.selection_enabled
             else {"initial": "原版", "A": "A 精修", "B": "B 精修"}.get(item.refinement_selected, item.refinement_selected)),
            _format_metric(item.symmetric_rms_mm),
            _format_metric(item.hd95_mm),
            f"{item.elapsed_seconds:.2f}",
        )
        for column, text in zip((2, 3, 4, 5, 6, 7), values):
            self.results_table.setItem(row, column, QTableWidgetItem(text))

    @Slot(object)
    def _on_succeeded(self, value: object) -> None:
        outcome = value
        assert isinstance(outcome, BatchOutcome)
        self._outcome = outcome
        self.progress.setValue(100)
        successful = sum(item.status in {"success", "warning"} for item in outcome.items)
        failed = sum(item.status == "failed" for item in outcome.items)
        cancelled = sum(item.status in {"cancelled", "skipped"} for item in outcome.items)
        self.status.setText(
            f"批次{'已停止' if outcome.stopped else '完成'}：成功/警告 {successful}，失败 {failed}，取消/跳过 {cancelled}，"
            f"耗时 {outcome.total_elapsed_seconds:.1f} 秒。"
        )
        self.folder_button.setEnabled(True)
        if self.results_table.rowCount() and not self.results_table.selectionModel().selectedRows():
            self.results_table.selectRow(0)
        self._selection_changed()

    @Slot(str, str)
    def _on_failed(self, summary: str, details: str) -> None:
        self.status.setText("批次启动或固定模型处理失败。")
        logging.getLogger(__name__).error("批次失败：%s\n%s", summary, details)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("批次处理失败")
        box.setText(summary)
        box.setInformativeText("展开“详细信息”可查看完整技术堆栈。")
        box.setDetailedText(details)
        box.exec()

    @Slot()
    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()
        self._selection_changed()

    def _selected_item(self) -> BatchItemResult | None:
        rows = self.results_table.selectionModel().selectedRows()
        return self._items.get(rows[0].row() + 1) if rows else None

    @Slot()
    def _selection_changed(self) -> None:
        item = self._selected_item()
        ready = self._outcome is not None and item is not None
        self.view_button.setEnabled(bool(ready and item and item.results_json))
        self.log_button.setEnabled(bool(ready and item and item.log_file))
        self._load_numeric_details(item)

    @Slot()
    def _view_selected(self) -> None:
        item = self._selected_item()
        if self._outcome and item and item.results_json:
            _launch_viewer(self, self._outcome.batch_directory / item.results_json)

    @Slot()
    def _log_selected(self) -> None:
        item = self._selected_item()
        if self._outcome and item and item.log_file:
            _open_path(self._outcome.batch_directory / item.log_file)

    @Slot()
    def _open_batch_folder(self) -> None:
        if self._outcome:
            _open_path(self._outcome.batch_directory)

    @Slot()
    def _history(self) -> None:
        root = Path(self.output_edit.text().strip() or default_output_directory())
        HistoryDialog(root, self).exec()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "配准正在运行", "请先停止批次并等待当前模型完成。")
            event.ignore()
            return
        self._save_settings()
        self._discard_session_edit_states()
        super().closeEvent(event)


def apply_application_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette(app.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#f3f5f8"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f7f9fc"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#f7f9fc"))
    palette.setColor(QPalette.ColorRole.Light, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Midlight, QColor("#eef1f5"))
    palette.setColor(QPalette.ColorRole.Mid, QColor("#d9dee5"))
    palette.setColor(QPalette.ColorRole.Dark, QColor("#c9d1db"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1f6feb"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#1f2933"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#1f2933"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#1f2933"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLE)


def main(argv: list[str] | None = None) -> int:
    ensure_standard_streams()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--model-editor":
        if len(arguments) != 3:
            raise SystemExit("--model-editor 需要 STL 路径和选区状态路径。")
        from .model_viewer import run_model_selection_viewer

        run_model_selection_viewer(arguments[1], arguments[2])
        return 0
    if arguments and arguments[0] == "--viewer":
        if len(arguments) < 2:
            raise SystemExit("--viewer 需要 results.json 路径。")
        target_override: str | None = None
        if "--target-override" in arguments:
            position = arguments.index("--target-override")
            if position + 1 >= len(arguments):
                raise SystemExit("--target-override 需要固定 STL 路径。")
            target_override = arguments[position + 1]
        from .result_viewer import run_general_result_viewer

        run_general_result_viewer(arguments[1], target_override=target_override)
        return 0
    if arguments and arguments[0] == "--viewer-pair":
        if len(arguments) != 3:
            raise SystemExit("--viewer-pair 需要固定 STL 和已配准 STL。")
        from .result_viewer import run_pair_result_viewer

        run_pair_result_viewer(arguments[1], arguments[2])
        return 0
    app = QApplication.instance() or QApplication([sys.argv[0], *arguments])
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    apply_application_style(app)
    window = AlignmentWindow()
    window.show()
    return app.exec()
