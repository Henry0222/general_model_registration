from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from auto_alignment import registration as reg
from auto_alignment.config import AlignmentConfig
from auto_alignment.refinement_modes import RegistrationCancelled, REFINEMENT_MODES
from auto_alignment.refinement import core as geometry
from auto_alignment.mesh_io import load_mesh


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    mesh = o3d.geometry.TriangleMesh.create_box(8, 13, 21)
    mesh.compute_vertex_normals()
    path = tmp_path / "测试模型.stl"
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    mesh, facts = load_mesh(path)
    original = reg.RegistrationResult(np.eye(4), "success", "高",
        reg.RegistrationMetrics(1., .02, 1000, 1., 0., 0.), (), 0.1)
    monkeypatch.setattr(reg, "_register_meshes_v2", lambda *a, **kw: original)
    return mesh, facts, original, path


def invoke(inputs, mode="baseline", **kwargs):
    mesh, facts, _, _ = inputs
    return reg.register_meshes(mesh, mesh, facts, facts, AlignmentConfig(refinement_mode=mode), **kwargs)


def backend(monkeypatch, selected="A"):
    matrices = {name: np.eye(4) for name in ("initial", "A", "B")}
    matrices["A"][0, 3] = .002
    matrices["B"][1, 3] = .003
    def refine(source, target, initial, mode, checkpoint=None):
        if checkpoint:
            checkpoint(.2, "refine")
            checkpoint(.1, "lower-level restart")
        return matrices, dict(mode=mode, selected="initial" if mode == "compare" else selected,
                              reason="test", elapsed_seconds=.1)
    monkeypatch.setattr(geometry, "refine_meshes", refine)
    def reassess(target, source, tf, sf, matrix, template, config, diagnostic_name):
        return replace(template, transformation=matrix.copy(), metrics=replace(template.metrics,
            inlier_rmse_mm=.01 if "A" in diagnostic_name else .03))
    monkeypatch.setattr(reg, "_reassess_full_model_transform", reassess)
    return matrices


def test_default_is_exact_original_and_never_calls_refinement(inputs, monkeypatch):
    monkeypatch.setattr(geometry, "refine_meshes", lambda *a, **kw: pytest.fail("default ran refinement"))
    assert invoke(inputs) is inputs[2]
    assert AlignmentConfig().refinement_mode == "baseline"


def test_invalid_mode_and_early_cancel_do_not_run_front(inputs, monkeypatch):
    monkeypatch.setattr(reg, "_register_meshes_v2", lambda *a, **kw: pytest.fail("front executed"))
    with pytest.raises(ValueError, match="mode"):
        invoke(inputs, "unknown")
    with pytest.raises(RegistrationCancelled):
        invoke(inputs, cancel=lambda: True)


@pytest.mark.parametrize("condition", ["failed", "target_selection", "source_selection"])
def test_front_failure_and_operator_selection_take_priority(inputs, monkeypatch, condition):
    monkeypatch.setattr(geometry, "refine_meshes", lambda *a, **kw: pytest.fail("unsafe refinement"))
    kwargs = {}
    if condition == "failed":
        original = replace(inputs[2], status="failed", confidence="失败")
        monkeypatch.setattr(reg, "_register_meshes_v2", lambda *a, **kw: original)
    else:
        kwargs[condition.replace("selection", "priority_faces")] = np.ones(len(inputs[0].triangles), bool)
    result = invoke(inputs, "auto", **kwargs)
    np.testing.assert_array_equal(result.transformation, inputs[2].transformation)
    assert result.metrics.refinement["state"] == "skipped"
    assert result.metrics.refinement["reason"] == ("front_end_failed" if condition == "failed" else "operator_selection")


@pytest.mark.parametrize("mode,selected", [("A", "A"), ("B", "B"), ("compare", "initial"), ("auto", "B")])
def test_selection_retains_original_and_reassesses_each_candidate(inputs, monkeypatch, mode, selected):
    matrices = backend(monkeypatch, selected=selected)
    fractions = []
    result = invoke(inputs, mode, progress=lambda f, m: fractions.append(f))
    np.testing.assert_array_equal(result.transformation, matrices[selected])
    alternatives = dict(result.alternatives)
    assert set(alternatives) == {"initial", "A", "B"} - {selected}
    assert fractions == sorted(fractions)
    assert result.metrics.refinement["selected"] == selected
    if selected != "initial":
        assert result.status == "warning" and result.confidence == "低"
        assert alternatives["initial"].metrics.inlier_rmse_mm == .02
        assert result.metrics.inlier_rmse_mm == (.01 if selected == "A" else .03)


def test_refinement_error_falls_back_with_structured_reason(inputs, monkeypatch):
    def fail(*a, **kw):
        raise ValueError("ambiguous frame")
    monkeypatch.setattr(geometry, "refine_meshes", fail)
    result = invoke(inputs, "B")
    assert result.metrics.refinement["state"] == "failed"
    assert "ambiguous frame" in result.metrics.refinement["error"]
    np.testing.assert_array_equal(result.transformation, np.eye(4))


def test_quality_failure_keeps_failed_candidate_and_falls_back(inputs, monkeypatch):
    backend(monkeypatch)
    monkeypatch.setattr(reg, "_reassess_full_model_transform", lambda *a: replace(
        inputs[2], transformation=a[4], status="failed", confidence="失败"))
    result = invoke(inputs, "A")
    assert result.metrics.refinement["state"] == "fallback"
    assert result.metrics.refinement["proposed"] == "A"
    assert result.metrics.refinement["selected"] == "initial"
    assert dict(result.alternatives)["A"].status == "failed"


def test_callback_errors_and_mid_refinement_cancel_propagate(inputs, monkeypatch):
    backend(monkeypatch)
    error = RuntimeError("callback failed")
    def progress(*a):
        raise error
    with pytest.raises(RuntimeError) as caught:
        invoke(inputs, "A", progress=progress)
    assert caught.value is error
    states = iter((False, False, True))
    with pytest.raises(RegistrationCancelled):
        invoke(inputs, "A", cancel=lambda: next(states))


def test_actual_adapter_rejects_nonrigid_initial_before_computing(inputs):
    matrix = np.eye(4); matrix[0, 0] = 1.01
    with pytest.raises(ValueError, match="rigid"):
        geometry.refine_meshes(inputs[0], inputs[0], matrix, "A")


def test_pipeline_exports_independent_candidates_and_relocatable_archives(inputs, tmp_path, monkeypatch):
    from auto_alignment import pipeline
    from auto_alignment.integration.review import load_viewer_data
    backend(monkeypatch)
    mesh, facts, _, source = inputs
    batch_dir = tmp_path / "批次"
    batch_dir.mkdir()
    archive = batch_dir / "fixed_target_used.stl"
    archive.write_bytes(source.read_bytes())
    outcome = pipeline.run_analysis_with_target(mesh, facts, source, batch_dir / "01_source",
        config=AlignmentConfig(refinement_mode="compare", metric_sample_points=1000),
        target_archived_path=Path("../fixed_target_used.stl"))
    root = json.loads(outcome.output_files["results_json"].read_text(encoding="utf-8"))
    assert set(root["candidate_results"]) == {"A", "B"}
    assert root["registration"]["metrics"]["refinement"]["selected"] == "initial"
    for name, relative in root["candidate_results"].items():
        path = outcome.output_files["results_json"].parent / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert (path.parent / payload["target_mesh"]["archived_path"]).resolve() == archive
        data = load_viewer_data(path)
        assert data.target_path == archive
        assert data.registration_status == "warning"
        transform = json.loads((path.parent / payload["outputs"]["transform_json"]).read_text(encoding="utf-8"))
        assert transform["transformation_current_to_target"][0 if name == "A" else 1][3] == (.002 if name == "A" else .003)


@pytest.mark.parametrize("persistent_request", [True, False])
def test_batch_cancellation_is_not_a_failure(inputs, tmp_path, monkeypatch, persistent_request):
    from auto_alignment import batch
    from auto_alignment.batch_models import RegistrationJob
    stopped = [False]
    def cancel(*a, **kw):
        assert callable(kw["cancel"])
        stopped[0] = persistent_request
        raise RegistrationCancelled("test cancellation")
    monkeypatch.setattr(batch, "run_analysis_with_target", cancel)
    outcome = batch.run_batch_analysis(inputs[3], False,
        [RegistrationJob(1, inputs[3]), RegistrationJob(2, inputs[3])], tmp_path / "out",
        stop_requested=lambda: stopped[0])
    assert outcome.stopped
    assert [item.status for item in outcome.items] == ["cancelled", "skipped"]
    assert not list(outcome.batch_directory.rglob("failure.json"))
    assert len(list(outcome.batch_directory.rglob("cancelled.json"))) == 1


def test_gui_modes_default_and_candidate_opening(inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from auto_alignment import gui
    app = QApplication.instance() or QApplication([])
    window = gui.AlignmentWindow()
    assert window.refinement_combo.currentData() == "auto"
    assert window.refinement_combo.count() == len(REFINEMENT_MODES)
    assert "正确位姿退化" in window.refinement_help.text()
    window.refinement_combo.setCurrentIndex(window.refinement_combo.findData("baseline"))
    assert "沿用 2.0" in window.refinement_help.text()
    window.close()
    app.processEvents()


def test_gui_opens_chosen_candidate_from_history(inputs, tmp_path, monkeypatch):
    from auto_alignment import gui
    source = inputs[3]
    root = tmp_path / "history"; candidate_dir = root / "candidates" / "A"
    candidate_dir.mkdir(parents=True)
    payload = dict(target_mesh={"path": str(source)}, outputs={"aligned_current_stl": str(source)},
                   registration={"metrics": {"refinement": {"selected": "initial"}}})
    candidate = candidate_dir / "results.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    payload["candidate_results"] = {"A": "candidates/A/results.json"}
    main = root / "results.json"; main.write_text(json.dumps(payload), encoding="utf-8")
    opened = []
    monkeypatch.setattr(gui.QInputDialog, "getItem", lambda *a, **kw: ("A · 局部共同表面", True))
    monkeypatch.setattr(gui.subprocess, "Popen", lambda command, **kw: opened.append(command))
    monkeypatch.setattr(gui.QMessageBox, "critical", lambda *a: pytest.fail(str(a)))
    gui._launch_viewer(None, main)
    assert str(candidate.resolve()) in opened[0]
