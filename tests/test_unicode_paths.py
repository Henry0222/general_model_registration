from __future__ import annotations

from pathlib import Path

import open3d as o3d
import numpy as np
import pytest

from auto_alignment.exporters import _write_triangle_mesh
from auto_alignment.mesh_io import MeshValidationError, load_mesh, load_viewer_mesh, read_mesh


def test_chinese_stl_path_can_be_read(tmp_path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=12)
    mesh.compute_vertex_normals()
    path = tmp_path / "理想位置.stl"
    assert _write_triangle_mesh(path, mesh)
    loaded, facts = load_mesh(path)
    assert not loaded.is_empty()
    assert Path(facts.path).name == path.name
    viewed = load_viewer_mesh(path)
    assert len(viewed.triangles) == len(loaded.triangles)
    assert viewed.has_vertex_normals()


def test_chinese_ply_path_can_be_read(tmp_path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=12)
    mesh.paint_uniform_color([0.2, 0.4, 0.8])
    path = tmp_path / "彩虹比较.ply"
    assert _write_triangle_mesh(path, mesh)
    loaded = read_mesh(path)
    assert loaded.has_vertex_colors()


@pytest.mark.parametrize("failure", ["silent", "exception"])
def test_unicode_write_fallback_preserves_ply_colors(tmp_path, monkeypatch, failure):
    original_write = o3d.io.write_triangle_mesh
    attempted = []

    def ascii_only_write(path, mesh, **kwargs):
        attempted.append(Path(path))
        if not path.isascii():
            Path(path).write_bytes(b"partial failed output")
            if failure == "exception":
                raise UnicodeError("simulated path encoding failure")
            return False
        return original_write(path, mesh, **kwargs)

    monkeypatch.setattr(o3d.io, "write_triangle_mesh", ascii_only_write)
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.paint_uniform_color([0.2, 0.4, 0.8])
    destination = tmp_path / "颜色.ply"
    assert _write_triangle_mesh(destination, mesh)
    loaded = read_mesh(destination)
    np.testing.assert_allclose(np.asarray(loaded.vertex_colors), mesh.vertex_colors)
    assert len(loaded.triangles) == len(mesh.triangles)
    assert len(attempted) == 2
    assert attempted[1].suffix == ".ply"
    assert all(not path.exists() for path in attempted)
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["silent", "exception"])
def test_unicode_read_fallback_preserves_ply_format(tmp_path, monkeypatch, failure):
    original_read = o3d.io.read_triangle_mesh
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.paint_uniform_color([0.2, 0.4, 0.8])
    path = tmp_path / "颜色.ply"
    assert _write_triangle_mesh(path, mesh)
    attempted = []

    def ascii_only_read(filename, **kwargs):
        attempted.append(Path(filename))
        if not filename.isascii():
            if failure == "exception":
                raise UnicodeError("simulated path encoding failure")
            return o3d.geometry.TriangleMesh()
        return original_read(filename, **kwargs)

    monkeypatch.setattr(o3d.io, "read_triangle_mesh", ascii_only_read)
    loaded = read_mesh(path)
    np.testing.assert_allclose(np.asarray(loaded.vertex_colors), mesh.vertex_colors)
    assert len(loaded.triangles) == len(mesh.triangles)
    assert len(attempted) == 2
    assert attempted[1].suffix == ".ply"
    assert not attempted[1].exists()


@pytest.mark.parametrize("failure", ["silent", "exception"])
def test_failed_write_preserves_destination_and_removes_scratch(tmp_path, monkeypatch, failure):
    destination = tmp_path / "已有模型.stl"
    destination.write_bytes(b"original destination")

    def failed_write(filename, mesh, **kwargs):
        Path(filename).write_bytes(b"partial failed output")
        if failure == "exception":
            raise RuntimeError("simulated write failure")
        return False

    monkeypatch.setattr(o3d.io, "write_triangle_mesh", failed_write)
    mesh = o3d.geometry.TriangleMesh.create_box()
    if failure == "exception":
        with pytest.raises(RuntimeError, match="simulated write failure"):
            _write_triangle_mesh(destination, mesh)
    else:
        assert not _write_triangle_mesh(destination, mesh)
    assert destination.read_bytes() == b"original destination"
    assert list(tmp_path.iterdir()) == [destination]


def test_empty_mesh_remains_invalid_after_read_fallback(tmp_path, monkeypatch):
    path = tmp_path / "空模型.stl"
    path.write_bytes(b"invalid mesh")
    monkeypatch.setattr(o3d.io, "read_triangle_mesh", lambda *a, **kw: o3d.geometry.TriangleMesh())
    with pytest.raises(MeshValidationError, match="不包含有效三角网格"):
        load_mesh(path)
