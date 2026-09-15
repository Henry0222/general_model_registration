from __future__ import annotations
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import open3d as o3d
import pytest

from auto_alignment.integration import INTEGRATION_API_VERSION, GENERAL_MODEL_REGISTRATION_VERSION
from auto_alignment.integration import core, review, selection


@pytest.fixture
def mesh_file(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box(8, 13, 21)
    mesh.compute_vertex_normals()
    path = tmp_path / "基准模型.stl"
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    return path


def regions():
    return [selection.SelectionRegionSpec("left", "左侧", (1., .2, .1)),
            selection.SelectionRegionSpec("right", "右侧", (.1, .3, 1.))]


def test_public_exports_and_lightweight_version_import():
    assert INTEGRATION_API_VERSION == 1
    assert GENERAL_MODEL_REGISTRATION_VERSION == "3.0.0"
    for module in (core, review, selection):
        assert all(getattr(module, name) is not None for name in module.__all__)
    source = Path(__file__).resolve().parents[1] / "src"
    code = f"import sys; sys.path.insert(0,{str(source)!r}); import auto_alignment.integration; assert 'open3d' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_imports_do_not_initialize_or_create_gui():
    source = Path(__file__).resolve().parents[1] / "src"
    code = f"""
import sys
sys.path.insert(0, {str(source)!r})
from open3d.visualization import gui
class Trap:
    def initialize(self, *a): raise AssertionError('GUI initialized on import')
    def create_window(self, *a): raise AssertionError('Window created on import')
class Application:
    instance = Trap()
gui.Application = Application
import auto_alignment.integration.core as core
import auto_alignment.integration.review as review
import auto_alignment.integration.selection as selection
for module in (core, review, selection):
    for name in module.__all__: getattr(module, name)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_replace_types_status_and_optional_quality(mesh_file):
    mesh, facts = core.load_mesh(mesh_file)
    assert isinstance(mesh, o3d.geometry.TriangleMesh)
    assert isinstance(facts, core.MeshFacts)
    assert replace(facts, path="elsewhere").path == "elsewhere"
    assert replace(core.AlignmentConfig(), random_seed=99, partial_overlap_threshold=.3).random_seed == 99
    metrics = core.RegistrationMetrics(1., 0., 12, 1., 0., 0.)
    for status in ("success", "warning", "failed"):
        result = core.RegistrationResult(np.eye(4), status, "界面文字", metrics, (), 0.)
        assert result.quality is None
        assert result.succeeded == (status != "failed")


def test_core_validates_masks_and_keeps_direction(mesh_file, monkeypatch):
    mesh, facts = core.load_mesh(mesh_file)
    matrix = np.eye(4)
    matrix[:3, 3] = [2, -3, 4]
    metrics = core.RegistrationMetrics(1., 0., 12, 1., 0., 0.)
    result = core.RegistrationResult(matrix, "warning", "中等", metrics, (), 0.)
    calls = []
    def solver(target, source, tf, sf, config, progress, **kwargs):
        assert target is mesh and source is source_mesh
        calls.append(kwargs)
        progress(-.1, "start")
        progress(1.1, "end")
        return result
    monkeypatch.setattr(core, "_register_meshes", solver)
    source_mesh = o3d.geometry.TriangleMesh(mesh)
    seen = []
    obtained = core.register_meshes(mesh, source_mesh, facts, facts, core.AlignmentConfig(), lambda f, m: seen.append(f))
    assert obtained is result and seen == [0., 1.]
    for invalid in (np.zeros(len(mesh.triangles)-1, bool), np.zeros((len(mesh.triangles), 1), bool), np.zeros(len(mesh.triangles), int)):
        with pytest.raises(ValueError):
            core.register_meshes(mesh, source_mesh, facts, facts, core.AlignmentConfig(), target_priority_faces=invalid)
    assert len(calls) == 1


def test_actual_solver_direction_and_wrapper_equivalence():
    from test_registration import asymmetric_mesh, facts
    from auto_alignment.registration import register_meshes as original
    target = asymmetric_mesh()
    source = o3d.geometry.TriangleMesh(target)
    imposed = np.eye(4)
    imposed[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((.3, -.2, .4))
    imposed[:3, 3] = [6., -3., 4.]
    source.transform(imposed)
    config = core.AlignmentConfig(global_sample_points=8000, ransac_max_iterations=15000)
    args = (target, source, facts(target, "t"), facts(source, "s"), config)
    raw, api = original(*args), core.register_meshes(*args)
    np.testing.assert_allclose(api.transformation, raw.transformation, atol=1e-9)
    assert api.status == raw.status
    for key, value in raw.quality.as_dict().items():
        obtained = api.quality.as_dict()[key]
        if isinstance(value, (float, int)) or (isinstance(value, tuple) and all(isinstance(x, (float, int)) for x in value)):
            np.testing.assert_allclose(obtained, value, atol=1e-10, rtol=1e-10)
        else:
            assert obtained == value
    delta = api.transformation @ imposed
    assert np.linalg.norm(delta[:3, 3]) < .05
    assert np.linalg.norm(delta[:3, :3]-np.eye(3)) < .01


def test_manifest_roundtrip_chinese_paths_hashes_and_annotations(mesh_file, tmp_path):
    sha = hashlib.sha256(mesh_file.read_bytes()).hexdigest()
    spec = review.RegistrationReviewSpec(mesh_file.name, mesh_file.name, status="warning",
        position_confidence="medium", confidence_display="需要复核", warnings=("检查",), review_only=True,
        direction_reversed=True, target_sha256=sha, aligned_sha256=sha)
    path = review.write_review_manifest(tmp_path/"中文清单.json", spec)
    restored = review.load_review_manifest(path)
    assert restored.target_path == mesh_file and restored.warnings == spec.warnings
    assert restored.direction_reversed and restored.review_only and restored.status == "warning"
    assert restored.target_sha256 == sha and restored.position_confidence == "medium"
    data = review.load_viewer_data(path)
    assert data.annotations_path == tmp_path/"viewer_annotations.json"
    assert data.registration_status == "warning" and data.registration_warnings == ("检查",)
    assert data.review_only and data.direction_reversed
    assert review.load_viewer_data(path, target_override=mesh_file, aligned_override=mesh_file).target_path == mesh_file
    mesh_file.write_bytes(mesh_file.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        review.load_review_manifest(path)


@pytest.mark.parametrize("schema", [1, "1.4.0", "1.4.1", "1.4.2"])
def test_legacy_v1_and_future_schema(mesh_file, tmp_path, schema):
    payload = dict(schema_version=schema, target_mesh={"path": mesh_file.name},
                   outputs={"aligned_current_stl": mesh_file.name}, registration={"status": "failed"})
    path = tmp_path/"results.json"
    path.write_text(json.dumps(payload))
    loaded = review.load_viewer_data(path)
    assert loaded.review_only and loaded.registration_status == "failed"
    assert review.load_pair_viewer_data(mesh_file, mesh_file).annotations_path is None
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        review.load_viewer_data(path)


def test_region_masks_atomic_overlap_copies_and_ranges(mesh_file):
    session = selection.RegionSelectionSession(mesh_file, 12, regions())
    left = np.zeros(12, bool); left[1:4] = True
    session.set_mask("left", left)
    left[:] = False
    assert session.snapshot().masks["left"].sum() == 3
    snapshot = session.snapshot()
    snapshot.masks["left"][:] = False
    assert session.snapshot().masks["left"].sum() == 3
    right = np.zeros(12, bool); right[[3, 8]] = True
    with pytest.raises(ValueError, match="重叠"): session.set_mask("right", right)
    assert not session.snapshot().masks["right"].any()
    encoded = selection.encode_face_ranges(np.flatnonzero(session.snapshot().masks["left"]))
    np.testing.assert_array_equal(selection.decode_face_ranges(encoded, 12), session.snapshot().masks["left"])
    with pytest.raises(ValueError): session.set_mask("left", np.zeros(11, bool))
    with pytest.raises(ValueError): selection.decode_face_ranges([[1, 99]], 12)
    overlap = selection.RegionSelectionSession(mesh_file, 12, regions(), initial_masks={"left": right, "right": right}, allow_overlap=True)
    assert overlap.snapshot().masks["right"].sum() == 2


def test_single_region_session_is_supported_for_simple_priority_selection(mesh_file):
    region = selection.SelectionRegionSpec("priority", "配准区", (.1, .7, .3))
    session = selection.RegionSelectionSession(mesh_file, 12, [region])
    mask = np.zeros(12, bool)
    mask[[1, 2, 8]] = True
    session.set_mask("priority", mask)
    assert session.snapshot().active_region == "priority"
    np.testing.assert_array_equal(session.snapshot().masks["priority"], mask)


def test_region_switch_undo_redo_callback_order_save_failure(mesh_file, tmp_path, monkeypatch):
    events = []
    def changed(s): events.append(("change", s.active_region, int(s.masks["left"].sum())))
    def saved(s): events.append(("save", s.active_region, int(s.masks["left"].sum())))
    session = selection.RegionSelectionSession(mesh_file, 12, regions(), on_change=changed, on_save=saved)
    mask = np.zeros(12, bool); mask[0] = True
    session.set_mask("left", mask)
    session.select_region("right")
    session.undo()
    session.redo()
    path = tmp_path/"regions.json"
    session.save(path)
    assert events == [("change", "left", 1), ("change", "right", 1), ("change", "left", 1), ("change", "right", 1), ("save", "right", 1)]
    def fail(*a, **kw): raise OSError("disk full")
    monkeypatch.setattr(selection, "atomic_write_text", fail)
    with pytest.raises(OSError): session.save(path)
    assert session.dirty and session.snapshot().masks["left"].sum() == 1
    assert len(events) == 5
    other = selection.RegionSelectionSession(mesh_file, 12, regions())
    other.load(path)
    assert other.snapshot().active_region == "right"
    assert other.snapshot().masks["left"][0]


def test_callback_reentry_and_failure_preserve_committed_state(mesh_file):
    session = selection.RegionSelectionSession(mesh_file, 12, regions())
    session._on_change = lambda snap: session.select_region("right")
    mask = np.zeros(12, bool); mask[0] = True
    with pytest.raises(RuntimeError, match="reenter"):
        session.set_mask("left", mask)
    assert session.snapshot().masks["left"][0] and session.dirty


def test_gui_save_failure_keeps_state_and_window_open(mesh_file, tmp_path):
    from auto_alignment.integration.selection import MultiRegionSelectionViewer
    viewer = object.__new__(MultiRegionSelectionViewer)
    viewer._closing = False
    viewer._selection_busy = False
    viewer.status_label = type("Label", (), {"text": ""})()
    viewer.window = type("Window", (), {"close": lambda self: None})()
    def fail(): raise OSError("disk full")
    viewer.save = fail
    viewer._save_and_close()
    assert viewer._closing is False
    assert "disk full" in viewer.status_label.text
    viewer.save = lambda: tmp_path/"state.json"
    viewer._save_and_close()
    assert viewer._closing is True
