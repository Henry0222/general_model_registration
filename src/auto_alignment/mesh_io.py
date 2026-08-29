from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np
import open3d as o3d


class MeshValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MeshFacts:
    path: str
    vertices: int
    triangles: int
    diagonal_mm: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "vertices": self.vertices,
            "triangles": self.triangles,
            "diagonal_mm": self.diagonal_mm,
            "bounds_min": self.bounds_min,
            "bounds_max": self.bounds_max,
            "warnings": list(self.warnings),
        }


def _read_triangle_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """Work around Open3D's Windows failure on non-ASCII STL paths."""
    try:
        return o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    except UnicodeError:
        with tempfile.TemporaryDirectory(prefix="dental_stl_") as temporary:
            safe_path = Path(temporary) / "input.stl"
            shutil.copyfile(mesh_path, safe_path)
            return o3d.io.read_triangle_mesh(str(safe_path), enable_post_processing=False)


def read_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    """Read a triangle mesh from a path that may contain Chinese characters."""
    return _read_triangle_mesh(Path(path))


def load_mesh(path: str | Path) -> tuple[o3d.geometry.TriangleMesh, MeshFacts]:
    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise MeshValidationError(f"找不到 STL 文件：{mesh_path}")

    mesh = _read_triangle_mesh(mesh_path)
    if mesh.is_empty() or not mesh.has_vertices() or not mesh.has_triangles():
        raise MeshValidationError(f"STL 不包含有效三角网格：{mesh_path.name}")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not np.isfinite(vertices).all():
        raise MeshValidationError(f"STL 含有 NaN 或无穷坐标：{mesh_path.name}")
    if triangles.min(initial=0) < 0 or triangles.max(initial=0) >= len(vertices):
        raise MeshValidationError(f"STL 三角面索引无效：{mesh_path.name}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise MeshValidationError(f"清理后 STL 没有有效三角面：{mesh_path.name}")

    mesh.compute_vertex_normals()
    bounds = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent(), dtype=float)
    diagonal = float(np.linalg.norm(extent))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise MeshValidationError(f"STL 尺寸无效：{mesh_path.name}")

    warnings: list[str] = []
    if diagonal < 5:
        warnings.append("模型包围盒小于 5 mm，请确认 STL 坐标单位为毫米。")
    elif diagonal > 300:
        warnings.append("模型包围盒大于 300 mm，请确认 STL 坐标单位为毫米。")

    facts = MeshFacts(
        path=str(mesh_path.resolve()),
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        diagonal_mm=diagonal,
        bounds_min=tuple(float(v) for v in bounds.min_bound),
        bounds_max=tuple(float(v) for v in bounds.max_bound),
        warnings=tuple(warnings),
    )
    return mesh, facts


def sample_registration_cloud(
    mesh: o3d.geometry.TriangleMesh,
    count: int,
    *,
    seed: int = 20260807,
) -> o3d.geometry.PointCloud:
    """Sample a triangle mesh deterministically and proportionally to area.

    Open3D's uniform sampler uses a process-global parallel random generator.
    Re-seeding it is not enough to guarantee identical point clouds between
    repeated runs. Registration candidates could therefore change for exactly
    the same STL pair. A local NumPy generator makes the sampled geometry fully
    reproducible and avoids cross-thread random-state interference.
    """
    # Do not cap samples by triangle count. A large, coarsely triangulated fixed
    # model still needs dense samples for a small component to be represented.
    count = max(1_000, int(count))
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise MeshValidationError("无法从 STL 表面采样配准点。")

    triangle_vertices = vertices[triangles]
    crosses = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    double_areas = np.linalg.norm(crosses, axis=1)
    valid = np.isfinite(double_areas) & (double_areas > 1e-12)
    valid_ids = np.flatnonzero(valid)
    if len(valid_ids) == 0:
        raise MeshValidationError("STL 没有可采样的有效三角面。")

    weights = double_areas[valid_ids]
    weights = weights / float(np.sum(weights))
    generator = np.random.default_rng(int(seed))
    sampled_ids = generator.choice(valid_ids, size=count, replace=True, p=weights)
    sampled_triangles = triangle_vertices[sampled_ids]
    first = generator.random(count)
    second = generator.random(count)
    root = np.sqrt(first)
    points = (
        (1.0 - root)[:, None] * sampled_triangles[:, 0]
        + (root * (1.0 - second))[:, None] * sampled_triangles[:, 1]
        + (root * second)[:, None] * sampled_triangles[:, 2]
    )
    normals = crosses[sampled_ids] / double_areas[sampled_ids, None]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    if cloud.is_empty():
        raise MeshValidationError("无法从 STL 表面采样配准点。")
    return cloud


def prepare_cloud(
    cloud: o3d.geometry.PointCloud,
    voxel_mm: float,
    normal_radius_multiplier: float,
) -> o3d.geometry.PointCloud:
    down = cloud.voxel_down_sample(voxel_mm)
    if len(down.points) < 100:
        raise MeshValidationError("降采样后有效点过少，无法配准。")
    radius = voxel_mm * normal_radius_multiplier
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
    )
    down.normalize_normals()
    return down


def clone_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh(mesh)
