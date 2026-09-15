from __future__ import annotations

import numpy as np
import open3d as o3d

from auto_alignment.config import AlignmentConfig
from auto_alignment.mesh_io import MeshFacts
from auto_alignment.registration import register_meshes


def asymmetric_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    for center, radius in [((-5, 0, 0), 3.1), ((1, 1, 0.5), 4.0), ((6, -1.5, 1.2), 2.5)]:
        part = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        part.translate(center)
        mesh += part
    mesh.compute_vertex_normals()
    return mesh


def facts(mesh: o3d.geometry.TriangleMesh, name: str) -> MeshFacts:
    box = mesh.get_axis_aligned_bounding_box()
    return MeshFacts(
        path=name,
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        diagonal_mm=float(np.linalg.norm(box.get_extent())),
        bounds_min=tuple(box.min_bound),
        bounds_max=tuple(box.max_bound),
        warnings=(),
    )


def test_registration_recovers_rigid_transform() -> None:
    target = asymmetric_mesh()
    source = o3d.geometry.TriangleMesh(target)
    angle = np.deg2rad(18.0)
    applied = np.eye(4)
    applied[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    applied[:3, 3] = [4.0, -3.0, 2.0]
    source.transform(applied)

    config = AlignmentConfig(
        global_sample_points=12_000,
        ransac_max_iterations=35_000,
        icp_iterations=(40, 30, 20),
        min_fitness=0.25,
    )
    result = register_meshes(target, source, facts(target, "target"), facts(source, "source"), config)
    assert result.succeeded
    expected = np.linalg.inv(applied)
    translation_error = np.linalg.norm(result.transformation[:3, 3] - expected[:3, 3])
    rotation_delta = result.transformation[:3, :3] @ expected[:3, :3].T
    rotation_error = np.rad2deg(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1)))
    assert translation_error < 0.35
    assert rotation_error < 2.0
    assert result.metrics.fitness > 0.40
    assert result.metrics.inlier_rmse_mm < 0.15


def test_unmarked_local_deformation_with_large_rigid_motion() -> None:
    target = asymmetric_mesh()
    source = o3d.geometry.TriangleMesh(target)
    vertices = np.asarray(source.vertices)
    changed = vertices[:, 0] > 6.0
    vertices[changed, 0] += 0.45
    applied = np.eye(4)
    applied[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([1.0, -.4, 2.0])
    applied[:3, 3] = [107., -34., 82.]
    source.transform(applied)
    source.compute_vertex_normals()
    config = AlignmentConfig(global_sample_points=12000, ransac_max_iterations=35000,
                             icp_iterations=(40, 30, 20))
    result = register_meshes(target, source, facts(target, "fixed"), facts(source, "moving"), config)
    # Only the test sees the imposed motion; geometry and default rules alone
    # must identify the distributed unchanged part of this non-cube surface.
    delta = result.transformation @ applied
    points = np.asarray(target.vertices)
    displacement = np.linalg.norm(points @ delta[:3, :3].T + delta[:3, 3] - points, axis=1)
    assert result.succeeded
    assert np.quantile(displacement, .95) < .03
    assert result.metrics.candidate_selection["selected_mode"] == "common_surface"
