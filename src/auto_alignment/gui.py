from __future__ import annotations

import json
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

from PySide6.QtCore import QProcess, QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .batch import (
    BatchItemResult,
    BatchOutcome,
    RegistrationJob,
    run_batch_analysis,
)
from .config import AlignmentConfig
from .history import (
    HistoryRecord,
    record_from_results,
    records_from_manifest,
    scan_history,
)
from .version import __version__


APP_TITLE = "通用模型自动配准"


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
    width: 20px;
    height: 20px;
    border: 2px solid #202020;
    border-radius: 11px;
    background-color: #ffffff;
}
QCheckBox::indicator:hover {
    border-color: #000000;
}
QCheckBox::indicator:checked {
    background-color: qradialgradient(
        cx: 0.5, cy: 0.5, radius: 0.5,
        fx: 0.5, fy: 0.5,
        stop: 0 #101010,
        stop: 0.43 #101010,
        stop: 0.44 #ffffff,
        stop: 1 #ffffff
    );
}
QCheckBox::indicator:disabled {
    border-color: #8a8a8a;
    background-color: #ffffff;
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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self.number = QLabel(f"{index:02d}")
        self.number.setMinimumWidth(28)
        self.path_edit = DropPathEdit("stl")
        self.path_edit.setPlaceholderText("可选择或从资源管理器拖入浮动 STL")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        self.edit_button = QPushButton("3D / 选区")
        self.edit_button.setToolTip("查看当前模型、编辑套索选区和工作副本。")
        self.edit_button.clicked.connect(lambda: self.edit_requested.emit(self))
        self.edit_badge = QLabel("未编辑")
        self.edit_badge.setMinimumWidth(86)
        self.flip_check = configure_flip_checkbox(QCheckBox("翻转面朝向/法线"))
        self.flip_check.setToolTip("配准前在内存中翻转三角面绕序，不修改原始 STL。")
        self.status = QLabel("等待")
        self.status.setMinimumWidth(70)
        layout.addWidget(self.number)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(browse)
        layout.addWidget(self.edit_button)
        layout.addWidget(self.edit_badge)
        layout.addWidget(self.flip_check)
        layout.addWidget(self.status)

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


class HistoryDialog(QDialog):
    def __init__(self, initial_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("既往配准记录")
        self.resize(1120, 620)
        self.records: list[HistoryRecord] = []
        self.root = initial_root
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
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ("时间", "浮动模型", "状态", "可信度", "RMS", "P90", "HD95", "翻转法线")
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        view = QPushButton("打开 3D 结果")
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

    def _scan(self) -> None:
        self.records = scan_history(self.root)
        self.root_label.setText(str(self.root))
        self._populate()

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
        self.resize(1080, 800)
        self.setMinimumSize(900, 700)
        self._worker: BatchRegistrationWorker | None = None
        self._outcome: BatchOutcome | None = None
        self._items: dict[int, BatchItemResult] = {}
        self._editor_processes: set[QProcess] = set()
        self._target_edit_mesh_path: Path | None = None
        self._target_edit_state_path: Path | None = None
        self.model_rows: list[ModelRow] = []
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(12)
        title = QLabel(f"{APP_TITLE} v{__version__}")
        title.setFont(QFont("Microsoft YaHei UI", 18, QFont.Weight.Bold))
        subtitle = QLabel(
            "一个固定 STL 与多个浮动 STL 按顺序独立配准。"
            "可从资源管理器拖入文件；法线翻转仅在内存中完成。"
        )
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        self.files_group = QGroupBox("固定模型、浮动模型与输出")
        files = QVBoxLayout(self.files_group)
        fixed_row = QHBoxLayout()
        fixed_row.addWidget(QLabel("固定/参考 STL："))
        self.target_edit = DropPathEdit("stl")
        self.target_edit.setPlaceholderText("可选择或从资源管理器拖入固定 STL")
        fixed_row.addWidget(self.target_edit, 1)
        fixed_browse = QPushButton("浏览…")
        fixed_browse.clicked.connect(self._choose_target)
        fixed_row.addWidget(fixed_browse)
        self.target_edit_button = QPushButton("3D / 选区")
        self.target_edit_button.setToolTip("查看固定模型、编辑套索选区和工作副本。")
        self.target_edit_button.clicked.connect(self._edit_target_model)
        fixed_row.addWidget(self.target_edit_button)
        self.target_edit_badge = QLabel("未编辑")
        self.target_edit_badge.setMinimumWidth(86)
        fixed_row.addWidget(self.target_edit_badge)
        self.target_edit.textChanged.connect(
            self._target_model_path_changed
        )
        self.target_flip = configure_flip_checkbox(QCheckBox("翻转面朝向/法线"))
        self.target_flip.setToolTip("固定模型法线将成为彩虹图正负偏差的唯一方向基准。")
        fixed_row.addWidget(self.target_flip)
        files.addLayout(fixed_row)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("浮动模型数量："))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 50)
        self.count_spin.setValue(1)
        self.count_spin.valueChanged.connect(self._set_model_count)
        count_row.addWidget(self.count_spin)
        count_row.addWidget(QLabel("也可将多个 STL 一次拖入下方列表"))
        count_row.addStretch(1)
        files.addLayout(count_row)

        self.model_area = ModelDropArea()
        self.model_layout = QVBoxLayout(self.model_area)
        self.model_layout.setContentsMargins(0, 0, 0, 0)
        self.model_layout.setSpacing(2)
        self.model_layout.addStretch(1)
        self.model_area.files_dropped.connect(self._fill_dropped_models)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(150)
        scroll.setWidget(self.model_area)
        files.addWidget(scroll)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("结果根目录："))
        self.output_edit = DropPathEdit("directory")
        self.output_edit.setText(str(default_output_directory()))
        output_row.addWidget(self.output_edit, 1)
        output_browse = QPushButton("浏览…")
        output_browse.clicked.connect(self._choose_output)
        output_row.addWidget(output_browse)
        files.addLayout(output_row)
        outer.addWidget(self.files_group)
        self._set_model_count(1)

        self.params_group = QGroupBox("配准与偏差参数")
        params = QFormLayout(self.params_group)
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
        params.addRow("最小相似表面覆盖率：", self.overlap_spin)
        params.addRow("覆盖距离：", self.coverage_spin)
        params.addRow("表面采样点：", self.samples_spin)
        params.addRow("RANSAC 最大迭代：", self.iterations_spin)
        params.addRow("彻底检查可能朝向：", self.exhaustive_orientation_check)
        params.addRow("轴对称角度步长：", self.exhaustive_orientation_step)
        params.addRow("最小名义偏差：", self.minimum_nominal_spin)
        params.addRow("最大名义偏差：", self.maximum_nominal_spin)
        outer.addWidget(self.params_group)

        actions = QHBoxLayout()
        self.start_button = QPushButton("开始顺序配准")
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("停止批次")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        history_button = QPushButton("查看既往配准记录")
        history_button.clicked.connect(self._history)
        self.view_button = QPushButton("查看选中 3D 结果")
        self.view_button.clicked.connect(self._view_selected)
        self.log_button = QPushButton("查看选中日志")
        self.log_button.clicked.connect(self._log_selected)
        self.folder_button = QPushButton("打开批次目录")
        self.folder_button.clicked.connect(self._open_batch_folder)
        for button in (self.view_button, self.log_button, self.folder_button):
            button.setEnabled(False)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(history_button)
        actions.addStretch(1)
        actions.addWidget(self.view_button)
        actions.addWidget(self.log_button)
        actions.addWidget(self.folder_button)
        outer.addLayout(actions)
        self.progress = QProgressBar()
        self.status = QLabel("等待选择模型。")
        outer.addWidget(self.progress)
        outer.addWidget(self.status)
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(("序号", "浮动模型", "状态", "可信度", "对称 RMS (mm)"))
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_table.itemSelectionChanged.connect(self._selection_changed)
        outer.addWidget(self.results_table, 1)
        self.setCentralWidget(central)

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

    @staticmethod
    def _new_session_edit_state_path() -> Path:
        return (
            Path(tempfile.gettempdir())
            / "GeneralModelRegistration"
            / "session_edits"
            / f"{uuid.uuid4().hex}.json"
        )

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
        command = self._model_editor_command(mesh_path, state_path)
        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(mesh_path.parent))
        self._editor_processes.add(process)
        badge.setText("编辑中…")

        def finished(*_args) -> None:
            self._editor_processes.discard(process)
            if (
                Path(current_path().strip()).expanduser().resolve() == mesh_path
                and current_state_path() == state_path
            ):
                badge.setText(self._edit_badge_text(state_path))
                badge.setToolTip(str(state_path))
            process.deleteLater()

        process.finished.connect(finished)
        process.errorOccurred.connect(
            lambda _error: badge.setText("启动失败")
        )
        process.start()

    @Slot()
    def _edit_target_model(self) -> None:
        self._launch_model_editor(
            Path(self.target_edit.text().strip()).expanduser(),
            self._target_edit_state_path,
            self.target_edit_badge,
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
        self._outcome = None
        self._items.clear()
        self.results_table.setRowCount(len(request.jobs))
        for row_index, job in enumerate(request.jobs):
            values = (f"{job.index:02d}", job.source_path.name, "等待", "—", "—")
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
            self.status.setText("将在当前模型完成后停止批次…")

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
        displayed_status = "失败（可查看）" if item.review_only else item.status
        self.model_rows[row].status.setText(displayed_status)
        values = (displayed_status, item.confidence, _format_metric(item.symmetric_rms_mm))
        for column, text in zip((2, 3, 4), values):
            self.results_table.setItem(row, column, QTableWidgetItem(text))

    @Slot(object)
    def _on_succeeded(self, value: object) -> None:
        outcome = value
        assert isinstance(outcome, BatchOutcome)
        self._outcome = outcome
        self.progress.setValue(100)
        successful = sum(item.status in {"success", "warning"} for item in outcome.items)
        failed = sum(item.status == "failed" for item in outcome.items)
        self.status.setText(
            f"批次完成：成功/警告 {successful}，失败 {failed}，"
            f"耗时 {outcome.total_elapsed_seconds:.1f} 秒。"
        )
        self.folder_button.setEnabled(True)
        self._selection_changed()

    @Slot(str, str)
    def _on_failed(self, summary: str, details: str) -> None:
        self.status.setText("批次启动或固定模型处理失败。")
        QMessageBox.critical(self, "批次处理失败", f"{summary}\n\n详细信息已写入批次日志。")

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
        super().closeEvent(event)


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
    app.setStyle("Fusion")
    window = AlignmentWindow()
    window.show()
    return app.exec()
