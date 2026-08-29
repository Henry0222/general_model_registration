from __future__ import annotations

import numpy as np
import open3d as o3d

from auto_alignment.config import AlignmentConfig
from auto_alignment.registration import _evaluate_high_precision_candidate


def _asymmetric_local_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    for center, radius in (
        ((-4.0, 0.0, 0.0), 2.2),
        ((1.0, 1.0, 0.6), 3.0),
        ((5.0, -1.5, 1.2), 1.6),
    ):
        part = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=16)
        part.translate(center)
        mesh += part
    mesh.compute_vertex_normals()
    return mesh


def test_local_to_whole_gate_uses_roi_denominator() -> None:
    source = _asymmetric_local_mesh()
    target = o3d.geometry.TriangleMesh(source)
    for center in ((25.0, 0.0, 0.0), (-25.0, 4.0, 2.0), (0.0, 25.0, -3.0)):
        extra = o3d.geometry.TriangleMesh.create_sphere(radius=5.0, resolution=18)
        extra.translate(center)
        target += extra
    target.compute_vertex_normals()

    before = np.eye(4)
    before[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(
        np.deg2rad((0.03, -0.02, 0.01))
    )
    before[:3, 3] = (0.015, -0.010, 0.008)

    decision = _evaluate_high_precision_candidate(
        target,
        source,
        before,
        np.eye(4),
        0.0,
        AlignmentConfig(high_precision_max_vertices=100_000),
    )

    assert decision["mode"] == "local_to_whole"
    assert decision["source_stable_coverage_ratio"] > 0.90
    assert decision["target_stable_coverage_ratio"] < 0.70
    assert decision["local_stable_coverage_ratio"] > 0.90
    assert decision["target_roi_stable_coverage_ratio"] > 0.90
    assert decision["accepted"] is True, decision
