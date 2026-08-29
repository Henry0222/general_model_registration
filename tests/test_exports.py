from __future__ import annotations

import json

import numpy as np
import open3d as o3d

from auto_alignment.comparison import compare_meshes
from auto_alignment.exporters import export_results
from auto_alignment.mesh_io import load_mesh
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
