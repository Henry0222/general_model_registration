from __future__ import annotations

import os
import threading
from pathlib import Path

import open3d as o3d
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .config import AlignmentConfig
from .mesh_io import read_mesh
from .pipeline import AnalysisOutcome, run_analysis


APP_TITLE = "通用模型自动配准"
APP_VERSION = "1.3.0"


def default_output_directory() -> Path:
    documents = Path.home() / "Documents"
    base = documents if documents.is_dir() else Path.home()
    return base / "GeneralModelRegistration_Output"


class RegistrationWorker(QThread):
    progress_changed = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        target: Path,
        current: Path,
        output: Path,
        color_max: float,
        reverse_direction: bool,
    ) -> None:
        super().__init__()
        self.target = target
        self.current = current
        self.output = output
        self.color_max = color_max
        self.reverse_direction = reverse_direction

    def run(self) -> None:
        try:
            outcome = run_analysis(
                self.target,
                self.current,
                self.output,
                self.color_max,
                AlignmentConfig(),
                self._progress,
                green_tolerance_mm=0.05,
                reverse_direction=self.reverse_direction,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(outcome)

    def _progress(self, fraction: float, message: str) -> None:
        self.progress_changed.emit(float(fraction), str(message))


class AlignmentWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} v{APP_VERSION}")
        self.resize(820, 520)
        self.setMinimumSize(720, 470)
        self._outcome: AnalysisOutcome | None = None
        self._worker: RegistrationWorker | None = None
        self._viewer_thread: threading.Thread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)

        title = QLabel(f"{APP_TITLE} v{APP_VERSION}")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)
        hint = QLabel("纯 CPU · 自动粗配准 · 多尺度 ICP · 高精度门控 · 有符号偏差图")
        hint.setStyleSheet("color: #555;")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.target_edit = QLineEdit()
        self.current_edit = QLineEdit()
        self.output_edit = QLineEdit(str(default_output_directory()))
        form.addRow("目标/参考 STL（固定）", self._path_row(self.target_edit, self._choose_target))
        form.addRow("待配准 STL（移动）", self._path_row(self.current_edit, self._choose_current))
        form.addRow("结果输出目录", self._path_row(self.output_edit, self._choose_output))

        self.color_max = QDoubleSpinBox()
        self.color_max.setRange(0.051, 20.0)
        self.color_max.setDecimals(3)
        self.color_max.setSingleStep(0.05)
        self.color_max.setValue(1.0)
        self.color_max.setSuffix(" mm")
        form.addRow("双向色标上限", self.color_max)
        layout.addLayout(form)

        self.reverse_check = QCheckBox("红蓝方向互换（目标 STL 法向相反时使用）")
        layout.addWidget(self.reverse_check)
        warning = QLabel(
            "两份模型必须具有足够的共同稳定表面。首次处理新的 STL 来源时，请用已知区域确认红蓝方向。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #8a4b08;")
        layout.addWidget(warning)

        self.run_button = QPushButton("一键配准并生成方向偏差图")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self._start)
        layout.addWidget(self.run_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel("请选择两份包含足够共同稳定表面的 STL 模型。")
        layout.addWidget(self.status)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        actions = QHBoxLayout()
        self.view_button = QPushButton("打开 3D 方向偏差图")
        self.view_button.setEnabled(False)
        self.view_button.clicked.connect(self._view)
        self.folder_button = QPushButton("打开输出目录")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(self._open_folder)
        actions.addWidget(self.view_button)
        actions.addWidget(self.folder_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.setCentralWidget(central)

    def _path_row(self, edit: QLineEdit, callback) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton("选择")
        button.clicked.connect(callback)
        row.addWidget(button)
        return container

    def _choose_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择目标/参考 STL", "", "STL 网格 (*.stl)")
        if path:
            self.target_edit.setText(path)

    def _choose_current(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择待配准 STL", "", "STL 网格 (*.stl)")
        if path:
            self.current_edit.setText(path)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择结果输出目录")
        if path:
            self.output_edit.setText(path)

    def _start(self) -> None:
        target = Path(self.target_edit.text().strip())
        current = Path(self.current_edit.text().strip())
        output_text = self.output_edit.text().strip()
        if not target.is_file() or not current.is_file():
            QMessageBox.critical(self, "输入不完整", "请选择有效的目标/参考 STL 和待配准 STL。")
            return
        if target.resolve() == current.resolve():
            QMessageBox.critical(self, "输入错误", "目标 STL 和待配准 STL 不能是同一个文件。")
            return
        if not output_text:
            QMessageBox.critical(self, "输入不完整", "请选择结果输出目录。")
            return

        self._outcome = None
        self.progress.setValue(0)
        self.summary.clear()
        self.run_button.setEnabled(False)
        self.view_button.setEnabled(False)
        self.folder_button.setEnabled(False)
        self._worker = RegistrationWorker(
            target,
            current,
            Path(output_text),
            float(self.color_max.value()),
            self.reverse_check.isChecked(),
        )
        self._worker.progress_changed.connect(self._set_progress)
        self._worker.succeeded.connect(self._completed)
        self._worker.failed.connect(self._failed)
        self._worker.start()

    def _set_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(round(max(0.0, min(1.0, fraction)) * 100))
        self.status.setText(message)

    def _failed(self, error: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText("处理失败")
        QMessageBox.critical(self, "自动配准失败", error)

    def _completed(self, outcome: AnalysisOutcome) -> None:
        self._outcome = outcome
        stats = outcome.comparison.statistics
        registration = outcome.registration
        self.run_button.setEnabled(True)
        self.view_button.setEnabled(True)
        self.folder_button.setEnabled(True)
        self.progress.setValue(100)
        self.status.setText(
            f"完成｜配准可信度：{registration.confidence}｜耗时：{outcome.total_elapsed_seconds:.1f} 秒"
        )
        decision = registration.metrics.high_precision_decision or {}
        stage = "高精度 ICP" if decision.get("accepted") else "多尺度 ICP"
        self.summary.setText(
            f"对称 RMS {stats.symmetric_rms_mm:.3f} mm；平均距离 {stats.mean_mm:.3f} mm；"
            f"HD95 {stats.hd95_mm:.3f} mm；最终阶段：{stage}。"
        )
        if registration.status == "warning":
            QMessageBox.warning(self, "结果需要检查", "\n".join(registration.warnings))

    def _view(self) -> None:
        if self._outcome is None:
            return
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            QMessageBox.information(self, "3D 查看器", "3D 方向偏差图窗口已经打开。")
            return
        colored_path = self._outcome.output_files["colored_ply"]
        target_path = Path(self._outcome.target_facts.path)
        max_mm = self._outcome.comparison.statistics.color_max_mm
        self._viewer_thread = threading.Thread(
            target=_show_result_window,
            args=(target_path, colored_path, max_mm),
            daemon=True,
        )
        self._viewer_thread.start()

    def _open_folder(self) -> None:
        path = Path(self.output_edit.text())
        if path.exists():
            os.startfile(path)  # type: ignore[attr-defined]


def _show_result_window(target_path: Path, colored_path: Path, max_mm: float) -> None:
    target = read_mesh(target_path)
    colored = read_mesh(colored_path)
    target.compute_vertex_normals()
    colored.compute_vertex_normals()
    target.paint_uniform_color([0.82, 0.82, 0.82])
    o3d.visualization.draw_geometries(
        [target, colored],
        window_name=f"3D方向偏差｜蓝=负偏差｜绿=目标±0.05｜红=正偏差｜色标±{max_mm:.2f} mm",
        width=1100,
        height=800,
    )


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = AlignmentWindow()
    window.show()
    app.exec()
