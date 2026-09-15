import json
from pathlib import Path

import pytest

from auto_alignment.batch_models import BatchItemResult, BatchOutcome
from auto_alignment.result_details import load_numeric_candidates


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from auto_alignment import gui
    monkeypatch.setattr(gui, "app_settings", lambda: QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    app = QApplication.instance() or QApplication([])
    instance = gui.AlignmentWindow()
    yield instance
    instance.close()
    app.processEvents()


def test_desktop_request_defaults_auto_and_respects_operator_choice(window, tmp_path):
    target, source = tmp_path / "target.stl", tmp_path / "source.stl"
    for path in (target, source):
        path.write_text("solid test\nendsolid test\n")
    window.target_edit.setText(str(target))
    window.model_rows[0].path_edit.setText(str(source))
    window.output_edit.setText(str(tmp_path / "outputs"))
    assert window._request().config.refinement_mode == "auto"
    window.advanced_toggle.setChecked(True)
    window.samples_spin.setValue(45000)
    for mode in ("baseline", "A", "B", "compare", "auto"):
        window.refinement_combo.setCurrentIndex(window.refinement_combo.findData(mode))
        assert window._request().config.refinement_mode == mode
        assert window._request().config.global_sample_points == 45000
    window._set_busy(True)
    assert not window.refinement_combo.isEnabled()
    assert not window.samples_spin.isEnabled()
    window._set_busy(False)
    assert window.refinement_combo.isEnabled()


def test_selected_candidate_uses_own_numbers_and_missing_is_not_zero(window, tmp_path):
    root = tmp_path / "batch"
    candidate_dir = root / "01_model/candidates/B"
    candidate_dir.mkdir(parents=True)
    main = candidate_dir.parents[1] / "results.json"
    main.write_text(json.dumps({"registration": {"status": "success"},
        "distance_statistics": {"mean_mm": .12456789, "median_mm": .002, "hd95_mm": .35},
        "candidate_results": {"B": "candidates/B/results.json"}}), encoding="utf-8")
    (candidate_dir / "results.json").write_text(json.dumps({"registration": {"status": "failed"},
        "distance_statistics": {"mean_mm": .00987654, "median_mm": 1e-9, "maximum_mm": "nan"}}), encoding="utf-8")
    item = BatchItemResult(1, "source.stl", "source.stl", False, "success", "高", 3.25,
        "01_model", "01_model/results.json", None, symmetric_rms_mm=.14, hd95_mm=.35)
    window.results_table.setRowCount(1)
    window._on_item_finished(item)
    assert window.results_table.item(0, 6).text() == "0.350000"
    assert window.results_table.item(0, 7).text() == "3.25"
    window._on_succeeded(BatchOutcome(root, root / "batch_results.json", root / "batch.log", (item,), False, 3.25))
    assert window.numeric_values["mean_mm"].text() == "0.124568"
    window.numeric_candidate_combo.setCurrentIndex(1)
    assert window.numeric_values["mean_mm"].text() == "0.009877"
    assert window.numeric_values["median_mm"].text() == "1.000e-09"
    assert window.numeric_values["maximum_mm"].text() == "—"
    assert window.numeric_values["hd95_mm"].text() == "—"
    assert "失败候选" in window.numeric_notice.text()
    window._outcome = None
    window._selection_changed()
    assert not window.numeric_candidate_combo.isEnabled()
    assert all(label.text() == "—" for label in window.numeric_values.values())


def test_unreadable_and_outside_candidates_remain_visible(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"distance_statistics": {"mean_mm": 123}}))
    main = root / "results.json"
    main.write_text(json.dumps({"candidate_results": {
        "outside": "../outside.json", "missing": "missing.json"}}))
    values = load_numeric_candidates(main)
    assert [name for name, _ in values] == ["主结果", "outside", "missing"]
    assert all("read_error" in payload for _, payload in values[1:])
    assert "超出" in values[1][1]["read_error"]
