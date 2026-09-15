from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import open3d as o3d
import pytest

from auto_alignment.comparison import compare_meshes
from auto_alignment.exporters import _write_triangle_mesh, export_results
from auto_alignment.mesh_io import clone_mesh, flip_mesh_orientation, load_mesh, read_mesh
from auto_alignment.registration import RegistrationMetrics, RegistrationResult


def test_exported_files_can_be_reloaded(tmp_path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=4.0, resolution=15)
    mesh.compute_vertex_normals()
    input_path = tmp_path / "input.stl"
    assert o3d.io.write_triangle_mesh(str(input_path), mesh)
    loaded, facts = load_mesh(input_path)
    comparison = compare_meshes(loaded, loaded, np.eye(4), 1_000, 1.0)
    registration = RegistrationResult(
        transformation=np.eye(4),
        status="success",
        confidence="高",
        metrics=RegistrationMetrics(1.0, 0.0, 1000, 1.0, 0.0, 0.0),
        warnings=(),
        elapsed_seconds=0.1,
    )

    files = export_results(tmp_path / "out", facts, facts, registration, comparison, 0.2)
    assert all(path.is_file() for path in files.values())
    aligned = o3d.io.read_triangle_mesh(str(files["aligned_stl"]))
    colored = o3d.io.read_triangle_mesh(str(files["colored_ply"]))
    assert not aligned.is_empty()
    assert colored.has_vertex_colors()

    transform = json.loads(files["transform_json"].read_text(encoding="utf-8"))
    assert np.allclose(transform["transformation_current_to_target"], np.eye(4))
    results = json.loads(files["results_json"].read_text(encoding="utf-8"))
    assert results["registration"]["status"] == "success"
    assert "symmetric_rms_mm" in results["distance_statistics"]
    assert results["color_mapping"]["green_rgb"] == [64, 255, 64]
    assert results["color_mapping"]["positive_rgb"] == [255, 64, 64]
    assert results["color_mapping"]["negative_rgb"] == [64, 64, 255]
    assert results["color_mapping"]["saturation"] == 0.75
    assert results["color_mapping"]["green_range_mm"] == [-0.05, 0.05]

    uncertain = replace(registration, status="warning", confidence="低")
    caution_files = export_results(tmp_path / "caution", facts, facts, uncertain, comparison, 0.2)
    caution_payload = json.loads(caution_files["results_json"].read_text(encoding="utf-8"))
    assert "质量警告" in caution_payload["result_notice"]
    assert "已通过" not in caution_payload["result_notice"]
    caution_transform = json.loads(caution_files["transform_json"].read_text(encoding="utf-8"))
    assert "unresolved quality warnings" in caution_transform["notice"]


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("status", ["success", "warning", "failed"])
def test_stl_export_preserves_input_orientation_and_registered_geometry(tmp_path, flipped, status):
    original = o3d.geometry.TriangleMesh.create_box(2.0, 3.0, 5.0)
    original.compute_vertex_normals()
    input_path = tmp_path / "input.stl"
    assert _write_triangle_mesh(input_path, original)
    original, facts = load_mesh(input_path)
    # Simulate the edited working mesh: export must not reload deleted faces.
    original.remove_triangles_by_index([0])
    original.compute_vertex_normals()
    working = clone_mesh(original)
    if flipped:
        flip_mesh_orientation(working)
    source_facts = replace(facts, normals_flipped=flipped, triangles=len(working.triangles))
    matrix = np.eye(4)
    matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.2, -0.3, 0.4))
    matrix[:3, 3] = [1.5, -2.0, 4.0]
    comparison = compare_meshes(original, working, matrix, 1_000, 1.0)
    before = clone_mesh(comparison.aligned_source)
    colored_before = clone_mesh(comparison.colored_source)
    registration = RegistrationResult(matrix, status, "高", RegistrationMetrics(1.0, 0.0, 1000, 1.0, 0.0, 0.0), (), 0.1)
    expected = clone_mesh(original)
    expected.transform(matrix)
    files = export_results(tmp_path / "out", facts, source_facts, registration, comparison, 0.2)
    loaded = read_mesh(files["aligned_stl"])
    expected_corners = np.asarray(expected.vertices)[np.asarray(expected.triangles)]
    loaded_corners = np.asarray(loaded.vertices)[np.asarray(loaded.triangles)]
    # Winding and coordinates must both survive STL serialization.
    np.testing.assert_allclose(loaded_corners, expected_corners, atol=1e-6)
    loaded.compute_triangle_normals()
    expected.compute_triangle_normals()
    np.testing.assert_allclose(loaded.triangle_normals, expected.triangle_normals, atol=1e-6)
    np.testing.assert_array_equal(comparison.aligned_source.vertices, before.vertices)
    np.testing.assert_array_equal(comparison.aligned_source.triangles, before.triangles)
    np.testing.assert_array_equal(comparison.aligned_source.vertex_normals, before.vertex_normals)
    colored = read_mesh(files["colored_ply"])
    np.testing.assert_array_equal(colored.triangles, colored_before.triangles)
    np.testing.assert_allclose(colored.vertex_colors, colored_before.vertex_colors, atol=1 / 255)
    payload = json.loads(files["results_json"].read_text(encoding="utf-8"))
    assert payload["current_mesh"]["normals_flipped"] is flipped
    assert payload["mesh_orientation"]["aligned_stl_flip_reverted"] is flipped
    transform = json.loads(files["transform_json"].read_text(encoding="utf-8"))
    np.testing.assert_array_equal(transform["transformation_current_to_target"], matrix)


def test_result_viewer_reapplies_processing_flip_only_for_new_exports(tmp_path):
    from auto_alignment.result_viewer import load_viewer_data
    original = o3d.geometry.TriangleMesh.create_box(2.0, 3.0, 5.0)
    original.compute_vertex_normals()
    input_path = tmp_path / "input.stl"
    assert _write_triangle_mesh(input_path, original)
    original, facts = load_mesh(input_path)
    working = flip_mesh_orientation(clone_mesh(original))
    comparison = compare_meshes(original, working, np.eye(4), 1_000, 1.0)
    registration = RegistrationResult(np.eye(4), "success", "高", RegistrationMetrics(1.0, 0.0, 1000, 1.0, 0.0, 0.0), (), 0.1)
    files = export_results(tmp_path / "out", facts, replace(facts, normals_flipped=True), registration, comparison, 0.2)
    output_bytes = files["aligned_stl"].read_bytes()
    viewed = load_viewer_data(files["results_json"])
    np.testing.assert_allclose(viewed.aligned.triangle_normals, working.triangle_normals)
    assert files["aligned_stl"].read_bytes() == output_bytes
    overridden = load_viewer_data(files["results_json"], aligned_override=input_path)
    np.testing.assert_allclose(overridden.aligned.triangle_normals, original.triangle_normals)
    # Legacy exports already had processing orientation and must not flip twice.
    assert _write_triangle_mesh(files["aligned_stl"], working)
    payload = json.loads(files["results_json"].read_text(encoding="utf-8"))
    del payload["mesh_orientation"]
    files["results_json"].write_text(json.dumps(payload), encoding="utf-8")
    legacy = load_viewer_data(files["results_json"])
    np.testing.assert_allclose(legacy.aligned.triangle_normals, working.triangle_normals)
