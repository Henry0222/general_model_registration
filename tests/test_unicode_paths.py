from __future__ import annotations

from pathlib import Path

import open3d as o3d

from auto_alignment.exporters import _write_triangle_mesh
from auto_alignment.mesh_io import load_mesh, read_mesh


def test_chinese_stl_path_can_be_read(tmp_path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=12)
    mesh.compute_vertex_normals()
    path = tmp_path / "理想位置.stl"
    assert _write_triangle_mesh(path, mesh)
    loaded, facts = load_mesh(path)
    assert not loaded.is_empty()
    assert Path(facts.path).name == path.name


def test_chinese_ply_path_can_be_read(tmp_path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=12)
    mesh.paint_uniform_color([0.2, 0.4, 0.8])
    path = tmp_path / "彩虹比较.ply"
    assert _write_triangle_mesh(path, mesh)
    loaded = read_mesh(path)
    assert loaded.has_vertex_colors()
