from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from PySide6.QtWidgets import QApplication

from auto_alignment.config import AlignmentConfig
from auto_alignment.exporters import _write_triangle_mesh
from auto_alignment.mesh_io import MeshFacts, load_mesh, sample_registration_cloud
from auto_alignment.gui import AlignmentWindow, ensure_standard_streams
from auto_alignment.mesh_selection import (
    ModelEditState,
    apply_edit_state,
    bounded_component_faces,
    clone_state_with_masks,
    floating_component_faces,
    load_edit_state,
    save_edit_state,
)
from auto_alignment.model_viewer import (
    LASSO_LINE_WIDTH_PX,
    _UI_TEXT as MODEL_VIEWER_UI_TEXT,
    _frontmost_candidate_faces,
    _projected_triangles_intersect_polygon,
    _selection_render_meshes,
    _stroke_ribbon_screen_mesh,
)
from auto_alignment.pipeline import run_analysis_with_target
from auto_alignment.registration import (
    RegistrationMetrics,
    RegistrationResult,
    _exhaustive_principal_axis_candidates,
    _fast_global_registration,
    _proper_signed_axis_permutations,
    register_meshes,
)
from auto_alignment.result_viewer import _font_code_points


def _two_component_mesh() -> o3d.geometry.TriangleMesh:
    main = o3d.geometry.TriangleMesh.create_box(2.0, 2.0, 2.0)
    floating = o3d.geometry.TriangleMesh.create_box(0.2, 0.2, 0.2)
    floating.translate((5.0, 0.0, 0.0))
    mesh = main + floating
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def _facts(mesh: o3d.geometry.TriangleMesh, path: str = "synthetic.stl") -> MeshFacts:
    bounds = mesh.get_axis_aligned_bounding_box()
    return MeshFacts(
        path=path,
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        diagonal_mm=float(np.linalg.norm(bounds.get_extent())),
        bounds_min=tuple(float(value) for value in bounds.min_bound),
        bounds_max=tuple(float(value) for value in bounds.max_bound),
        warnings=(),
    )


def test_edit_state_round_trip_is_bound_to_mesh_hash(tmp_path: Path) -> None:
    mesh_path = tmp_path / "模型.stl"
    assert _write_triangle_mesh(mesh_path, _two_component_mesh())
    mesh, _ = load_mesh(mesh_path)
    state = ModelEditState.empty(mesh_path, len(mesh.triangles))
    selected = state.selected.copy()
    deleted = state.deleted.copy()
    selected[[0, 1, 4]] = True
    deleted[2] = True
    state = clone_state_with_masks(state, selected, deleted)
    state_path = tmp_path / "selection.json"
    save_edit_state(state_path, state)

    loaded = load_edit_state(state_path, mesh_path, len(mesh.triangles))
    assert np.array_equal(loaded.selected, state.selected)
    assert np.array_equal(loaded.deleted, state.deleted)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.4.1"

    changed = o3d.geometry.TriangleMesh.create_box(3.0, 2.0, 2.0)
    changed.compute_triangle_normals()
    assert _write_triangle_mesh(mesh_path, changed)
    changed_mesh, _ = load_mesh(mesh_path)
    invalidated = load_edit_state(state_path, mesh_path, len(changed_mesh.triangles))
    assert not np.any(invalidated.selected)
    assert not np.any(invalidated.deleted)


def test_bounded_component_and_floating_cleanup_use_surface_components() -> None:
    mesh = _two_component_mesh()
    triangle_count = len(mesh.triangles)
    deleted = np.zeros(triangle_count, dtype=bool)
    selected = np.zeros(triangle_count, dtype=bool)
    selected[0] = True

    expanded = bounded_component_faces(mesh, selected, deleted)
    floating = floating_component_faces(mesh, deleted)

    assert np.count_nonzero(expanded) == 12
    assert np.count_nonzero(floating) == 12
    assert not np.any(expanded & floating)


def test_deleting_faces_only_changes_the_working_copy(tmp_path: Path) -> None:
    mesh_path = tmp_path / "input.stl"
    assert _write_triangle_mesh(mesh_path, _two_component_mesh())
    mesh, _ = load_mesh(mesh_path)
    original_triangles = np.asarray(mesh.triangles).copy()
    state = ModelEditState.empty(mesh_path, len(mesh.triangles))
    deleted = floating_component_faces(mesh, state.deleted)
    state = clone_state_with_masks(state, state.selected, deleted)

    applied = apply_edit_state(mesh, state)

    assert len(applied.mesh.triangles) == 12
    assert len(mesh.triangles) == 24
    assert np.array_equal(np.asarray(mesh.triangles), original_triangles)


def test_priority_sampling_allocates_requested_surface_mass() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)),
    )
    mesh.compute_triangle_normals()
    priority = np.array([True, False])
    cloud = sample_registration_cloud(
        mesh,
        10_000,
        seed=42,
        priority_faces=priority,
        priority_fraction=0.70,
    )
    ratio = float(np.mean(np.asarray(cloud.points)[:, 0] < 5.0))
    assert 0.68 <= ratio <= 0.72


def test_no_priority_sampling_matches_v140_reference() -> None:
    mesh = _two_component_mesh()
    count = 2_000
    seed = 20260807
    cloud = sample_registration_cloud(mesh, count, seed=seed)

    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    triangle_vertices = vertices[triangles]
    crosses = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    double_areas = np.linalg.norm(crosses, axis=1)
    valid_ids = np.flatnonzero(np.isfinite(double_areas) & (double_areas > 1e-12))
    weights = double_areas[valid_ids]
    weights = weights / float(np.sum(weights))
    generator = np.random.default_rng(seed)
    sampled_ids = generator.choice(valid_ids, size=count, replace=True, p=weights)
    sampled_triangles = triangle_vertices[sampled_ids]
    first = generator.random(count)
    second = generator.random(count)
    root = np.sqrt(first)
    expected = (
        (1.0 - root)[:, None] * sampled_triangles[:, 0]
        + (root * (1.0 - second))[:, None] * sampled_triangles[:, 1]
        + (root * second)[:, None] * sampled_triangles[:, 2]
    )

    assert np.array_equal(np.asarray(cloud.points), expected)


def test_fgr_resets_open3d_rng_to_v140_baseline(monkeypatch) -> None:
    seeds: list[int] = []
    sentinel = object()
    monkeypatch.setattr(o3d.utility.random, "seed", seeds.append)
    monkeypatch.setattr(
        o3d.pipelines.registration,
        "registration_fgr_based_on_feature_matching",
        lambda *_args, **_kwargs: sentinel,
    )

    cloud = o3d.geometry.PointCloud()
    feature = o3d.pipelines.registration.Feature()
    result = _fast_global_registration(cloud, cloud, feature, feature, 0.45)

    assert result is sentinel
    assert seeds == [0]


def test_windowed_executable_installs_writable_standard_streams(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    ensure_standard_streams()

    assert sys.stdout is not None
    assert sys.stderr is not None
    assert sys.stdout.write("Open3D diagnostic\n") > 0
    assert sys.stderr.write("Open3D warning\n") > 0
    sys.stdout.flush()
    sys.stderr.flush()


def test_model_viewer_font_mapping_contains_selection_ui_characters() -> None:
    code_points = set(_font_code_points(MODEL_VIEWER_UI_TEXT))
    assert all(ord(character) in code_points for character in "自由套索透选删除浮动面片撤销")


def test_changing_model_path_immediately_clears_stale_edit_badge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    mesh_path = tmp_path / "replacement.stl"
    assert _write_triangle_mesh(mesh_path, _two_component_mesh())
    second_path = tmp_path / "replacement-2.stl"
    assert _write_triangle_mesh(second_path, _two_component_mesh())
    app = QApplication.instance() or QApplication([])
    window = AlignmentWindow()
    row = window.model_rows[0]
    row.edit_badge.setText("选 176,122 / 删 0")
    row.path_edit.setText(str(mesh_path))
    app.processEvents()

    assert row.edit_badge.text() == "未编辑"
    first_state_path = row.edit_state_path
    assert first_state_path is not None
    assert not first_state_path.is_file()

    row.edit_badge.setText("选 43,613 / 删 0")
    row.path_edit.setText(str(second_path))
    app.processEvents()

    assert row.edit_badge.text() == "未编辑"
    assert row.edit_state_path is not None
    assert row.edit_state_path != first_state_path
    assert not row.edit_state_path.is_file()
    window.close()


def test_through_lasso_selects_every_projected_triangle_overlap() -> None:
    polygon = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        dtype=float,
    )
    triangles = np.array(
        [
            [[2.0, 2.0], [4.0, 2.0], [3.0, 4.0]],  # wholly inside
            [[-20.0, -20.0], [30.0, -20.0], [5.0, 30.0]],  # contains lasso
            [[-5.0, 4.0], [15.0, 4.0], [15.0, 5.0]],  # crossing only
            [[12.0, 12.0], [14.0, 12.0], [13.0, 14.0]],  # outside
            [[np.nan, 0.0], [1.0, 1.0], [2.0, 0.0]],  # invalid projection
        ],
        dtype=float,
    )

    selected = _projected_triangles_intersect_polygon(triangles, polygon)

    assert np.array_equal(selected, np.array([True, True, True, False, False]))


def test_selection_render_meshes_do_not_duplicate_selected_faces() -> None:
    mesh = _two_component_mesh()
    selected = np.zeros(len(mesh.triangles), dtype=bool)
    selected[::3] = True

    base, highlight = _selection_render_meshes(mesh, selected)

    assert len(base.triangles) == int(np.count_nonzero(~selected))
    assert len(highlight.triangles) == int(np.count_nonzero(selected))
    assert len(base.triangles) + len(highlight.triangles) == len(mesh.triangles)


def test_non_through_visibility_uses_each_face_centroid_ray() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(
            np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        ),
    )
    candidates = np.ones(2, dtype=bool)

    from_below = _frontmost_candidate_faces(mesh, candidates, np.array([0.0, 0.0, 1.0]))
    from_above = _frontmost_candidate_faces(mesh, candidates, np.array([0.0, 0.0, -1.0]))

    assert np.array_equal(from_below, np.array([True, False]))
    assert np.array_equal(from_above, np.array([False, True]))


def test_lasso_line_width_is_three_pixels() -> None:
    assert LASSO_LINE_WIDTH_PX == 3.0
    vertices, triangles = _stroke_ribbon_screen_mesh(
        np.array([[10.0, 20.0], [110.0, 20.0]]),
        LASSO_LINE_WIDTH_PX,
    )

    assert np.isclose(np.ptp(vertices[:, 1]), 3.0)
    assert triangles.shape == (4, 3)


def test_exhaustive_orientation_candidates_cover_all_proper_axis_mappings() -> None:
    orientations = _proper_signed_axis_permutations()

    assert len(orientations) == 24
    assert all(np.isclose(np.linalg.det(item), 1.0) for item in orientations)
    assert len({tuple(item.reshape(-1)) for item in orientations}) == 24


def test_exhaustive_orientation_candidates_include_known_cuboid_pose() -> None:
    target_points = np.array(
        list(itertools.product((-25.0, 25.0), (-35.0, 35.0), (-45.0, 45.0))),
        dtype=float,
    )
    true_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    true_translation = np.array([13.0, -7.0, 4.0], dtype=float)
    source_points = (target_points - true_translation) @ true_rotation
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))

    candidates = _exhaustive_principal_axis_candidates(source, target)

    assert len(candidates) == 24
    errors = []
    for transform in candidates:
        moved = source_points @ transform[:3, :3].T + transform[:3, 3]
        errors.append(
            max(
                min(np.linalg.norm(point - target_points, axis=1))
                for point in moved
            )
        )
        assert np.linalg.det(transform[:3, :3]) > 0.999999
    assert min(errors) < 1e-8


def test_selection_lane_can_override_full_surface_failure_with_warning(monkeypatch) -> None:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=5)
    mesh.compute_triangle_normals()
    facts = _facts(mesh)

    def fake_once(*_args, **kwargs) -> RegistrationResult:
        priority = kwargs.get("source_priority_faces") is not None
        transform = np.eye(4)
        transform[0, 3] = 0.1 if priority else 0.0
        return RegistrationResult(
            transformation=transform,
            status="failed" if priority else "success",
            confidence="失败" if priority else "高",
            metrics=RegistrationMetrics(
                fitness=0.8,
                inlier_rmse_mm=0.1,
                correspondence_count=1000,
                overlap_ratio=0.8,
                rotation_degrees=0.0,
                translation_mm=float(abs(transform[0, 3])),
            ),
            warnings=("完整表面覆盖不足。",) if priority else (),
            elapsed_seconds=1.0,
        )

    def fake_roi(*args, **_kwargs):
        transformation = args[4]
        is_priority = bool(np.asarray(transformation)[0, 3] > 0.05)
        return {
            "directions": {},
            "coverage_ratio": 0.85 if is_priority else 0.70,
            "median_mm": 0.10 if is_priority else 0.20,
            "p90_mm": 0.20 if is_priority else 0.30,
            "rms_mm": 0.12 if is_priority else 0.22,
            "normal_diversity": 0.01,
        }

    monkeypatch.setattr("auto_alignment.registration._register_meshes_once", fake_once)
    monkeypatch.setattr("auto_alignment.registration._selection_candidate_metrics", fake_roi)
    mask = np.ones(len(mesh.triangles), dtype=bool)
    result = register_meshes(
        mesh,
        mesh,
        facts,
        facts,
        AlignmentConfig(selection_min_faces=1),
        source_priority_faces=mask,
    )

    assert result.status == "warning"
    assert result.metrics.selection_decision["selected_lane"] == "selection_priority"
    assert np.isclose(result.transformation[0, 3], 0.1)


def test_quality_gate_failure_still_exports_review_only_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mesh = o3d.geometry.TriangleMesh.create_box(2.0, 2.0, 2.0)
    mesh.compute_triangle_normals()
    target_path = tmp_path / "target.stl"
    source_path = tmp_path / "source.stl"
    assert _write_triangle_mesh(target_path, mesh)
    assert _write_triangle_mesh(source_path, mesh)
    target_mesh, target_facts = load_mesh(target_path)
    failed = RegistrationResult(
        transformation=np.eye(4),
        status="failed",
        confidence="失败",
        metrics=RegistrationMetrics(0.2, 0.4, 200, 0.2, 0.0, 0.0),
        warnings=("共同表面不足。",),
        elapsed_seconds=0.1,
    )
    monkeypatch.setattr("auto_alignment.pipeline.register_meshes", lambda *_a, **_k: failed)

    source_mesh, _ = load_mesh(source_path)
    edit_state = ModelEditState.empty(source_path, len(source_mesh.triangles))
    selected = edit_state.selected.copy()
    selected[0] = True
    edit_state = clone_state_with_masks(edit_state, selected, edit_state.deleted)
    edit_state_path = tmp_path / "source-edit-state.json"
    save_edit_state(edit_state_path, edit_state)

    outcome = run_analysis_with_target(
        target_mesh,
        target_facts,
        source_path,
        tmp_path / "out",
        config=AlignmentConfig(metric_sample_points=1_000),
        current_edit_state_path=edit_state_path,
    )

    payload = json.loads(outcome.output_files["results_json"].read_text(encoding="utf-8"))
    assert payload["review_only"] is True
    assert "FAILED_PREVIEW_ONLY" in outcome.output_files["aligned_stl"].name
    assert outcome.output_files["aligned_stl"].is_file()
    assert outcome.output_files["source_edit_state_json"].is_file()
    assert payload["outputs"]["moving_model_edit_state_json"] == "moving_model_edit_state.json"
