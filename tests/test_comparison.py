from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from auto_alignment.comparison import (
    compare_meshes,
    directional_colormap,
    point_to_mesh_distances,
    signed_point_to_mesh_distances,
)


def sphere(radius: float = 5.0) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=18)
    mesh.compute_vertex_normals()
    return mesh


def test_identical_mesh_has_near_zero_distance() -> None:
    mesh = sphere()
    result = compare_meshes(mesh, mesh, np.eye(4), sample_count=2_000, color_max_mm=1.0)
    assert result.statistics.symmetric_rms_mm < 1e-4
    assert result.vertex_distances_mm.max() < 1e-4
    assert np.asarray(result.colored_source.vertex_colors).shape == (len(mesh.vertices), 3)


def test_known_radial_offset_distance() -> None:
    target = sphere(5.0)
    points = np.array([[6.0, 0.0, 0.0], [0.0, -6.0, 0.0]], dtype=float)
    distances = point_to_mesh_distances(points, target)
    assert distances == pytest.approx([1.0, 1.0], abs=0.08)


def test_signed_distance_uses_target_outward_normal() -> None:
    target = sphere(5.0)
    points = np.array([[6.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    signed = signed_point_to_mesh_distances(points, target)
    assert signed == pytest.approx([1.0, -1.0], abs=0.08)
    reversed_signed = signed_point_to_mesh_distances(points, target, reverse_direction=True)
    assert reversed_signed == pytest.approx([-1.0, 1.0], abs=0.08)


def test_directional_colormap_rgb_endpoints_and_tolerance() -> None:
    values = np.array([-2.0, -1.0, -0.05, 0.0, 0.05, 1.0, 2.0])
    colors = directional_colormap(values, color_max_mm=1.0, green_tolerance_mm=0.05)
    assert np.allclose(colors[0], [0.25, 0.25, 1.0])
    assert np.allclose(colors[1], [0.25, 0.25, 1.0])
    assert np.allclose(colors[2:5], [0.25, 1.0, 0.25])
    assert np.allclose(colors[5], [1.0, 0.25, 0.25])
    assert np.allclose(colors[6], [1.0, 0.25, 0.25])
    assert np.all((colors >= 0.0) & (colors <= 1.0))


def test_invalid_query_points_are_rejected() -> None:
    with pytest.raises(ValueError):
        point_to_mesh_distances(np.array([[np.nan, 0.0, 0.0]]), sphere())
