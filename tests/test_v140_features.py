from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d

from auto_alignment.batch import (
    RegistrationJob,
    create_batch_directory,
    run_batch_analysis,
)
from auto_alignment.config import AlignmentConfig
from auto_alignment.deviation_scale import DeviationScale
from auto_alignment.exporters import _write_triangle_mesh
from auto_alignment.history import scan_history
from auto_alignment.mesh_io import load_mesh
from auto_alignment.result_viewer import load_viewer_data


def _asymmetric_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    for center, radius in (
        ((-3.0, 0.0, 0.0), 2.0),
        ((1.0, 1.0, 0.5), 2.7),
        ((4.0, -1.0, 1.0), 1.4),
    ):
        part = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=12)
        part.translate(center)
        mesh += part
    mesh.compute_vertex_normals()
    return mesh


def test_continuous_deviation_scale_keeps_green_band_and_smooths_outer_colors() -> None:
    scale = DeviationScale(
        minimum_critical_mm=-1.0,
        minimum_nominal_mm=-0.05,
        maximum_nominal_mm=0.05,
        maximum_critical_mm=1.0,
        configured_minimum_nominal_mm=-0.05,
        configured_maximum_nominal_mm=0.05,
    )
    values = np.array([-0.0501, -0.05, 0.0, 0.05, 0.0501, 0.20, 0.2001])
    colors = scale.map_colors(values)
    assert np.allclose(colors[1], colors[2])
    assert np.allclose(colors[2], colors[3])
    assert not np.allclose(colors[5], colors[6])
    assert len(scale.legend_ticks_mm) == 14
    assert scale.legend_ticks_mm[0] == 1.0
    assert scale.legend_ticks_mm[-1] == -1.0


def test_flip_normals_reverses_triangle_winding_without_changing_vertices(tmp_path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_box(2.0, 3.0, 4.0)
    mesh.compute_vertex_normals()
    path = tmp_path / "input.stl"
    assert _write_triangle_mesh(path, mesh)
    normal, normal_facts = load_mesh(path)
    flipped, flipped_facts = load_mesh(path, flip_normals=True)
    assert np.allclose(np.asarray(normal.vertices), np.asarray(flipped.vertices))
    assert np.allclose(
        np.asarray(normal.triangle_normals),
        -np.asarray(flipped.triangle_normals),
    )
    assert normal_facts.normals_flipped is False
    assert flipped_facts.normals_flipped is True


def test_align_directory_uses_minute_timestamp_and_collision_suffix(tmp_path) -> None:
    first = create_batch_directory(tmp_path)
    second = create_batch_directory(tmp_path)
    assert re.fullmatch(r"align_\d{8}", first.name)
    assert second.name == f"{first.name}_02"


def test_batch_continues_after_failure_and_history_reads_manifest(tmp_path) -> None:
    target = _asymmetric_mesh()
    target_path = tmp_path / "target.stl"
    source_path = tmp_path / "source.stl"
    assert _write_triangle_mesh(target_path, target)
    source = o3d.geometry.TriangleMesh(target)
    transform = np.eye(4)
    transform[:3, 3] = (0.8, -0.4, 0.3)
    source.transform(transform)
    assert _write_triangle_mesh(source_path, source)

    outcome = run_batch_analysis(
        target_path,
        False,
        (
            RegistrationJob(1, tmp_path / "missing.stl"),
            RegistrationJob(2, source_path, True),
        ),
        tmp_path / "output",
        config=AlignmentConfig(
            global_sample_points=6_000,
            metric_sample_points=6_000,
            ransac_max_iterations=10_000,
            global_registration_restarts=2,
            final_candidate_count=2,
            icp_iterations=(20, 15, 10),
            high_precision_iterations=8,
            high_precision_max_vertices=30_000,
            min_fitness=0.10,
        ),
    )
    assert [item.status for item in outcome.items][0] == "failed"
    assert outcome.items[1].status in {"success", "warning"}
    assert (outcome.batch_directory / "fixed_target_used.stl").is_file()
    assert (outcome.batch_directory / outcome.items[0].log_file).is_file()
    assert (outcome.batch_directory / outcome.items[1].results_json).is_file()
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.4"
    assert manifest["fixed_model"]["normals_flipped"] is False
    records = scan_history(tmp_path / "output")
    assert len(records) == 2
    assert {record.status for record in records} >= {"failed"}
    target_path.rename(tmp_path / "target_moved_after_registration.stl")
    viewer_data = load_viewer_data(
        outcome.batch_directory / outcome.items[1].results_json
    )
    assert not viewer_data.target.is_empty()
    assert not viewer_data.aligned.is_empty()
