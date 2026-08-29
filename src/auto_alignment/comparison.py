from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from auto_alignment.mesh_io import clone_mesh, sample_registration_cloud
from auto_alignment.deviation_scale import DeviationScale

@dataclass(frozen=True)
class DistanceStatistics:
    symmetric_rms_mm: float
    mean_mm: float
    median_mm: float
    hd95_mm: float
    maximum_mm: float
    vertex_mean_mm: float
    color_max_mm: float
    green_tolerance_mm: float
    direction_reversed: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {'symmetric_rms_mm': self.symmetric_rms_mm, 'mean_mm': self.mean_mm, 'median_mm': self.median_mm, 'hd95_mm': self.hd95_mm, 'maximum_mm': self.maximum_mm, 'vertex_mean_mm': self.vertex_mean_mm, 'color_max_mm': self.color_max_mm, 'green_tolerance_mm': self.green_tolerance_mm, 'direction_reversed': self.direction_reversed}

@dataclass(frozen=True)
class ComparisonResult:
    aligned_source: o3d.geometry.TriangleMesh
    colored_source: o3d.geometry.TriangleMesh
    vertex_distances_mm: np.ndarray
    signed_vertex_distances_mm: np.ndarray
    statistics: DistanceStatistics
    deviation_scale: DeviationScale | None = None

def _tensor_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.TriangleMesh:
    return o3d.t.geometry.TriangleMesh.from_legacy(mesh)

def _distance_scene(target_mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(_tensor_mesh(target_mesh))
    return scene

def point_to_mesh_distances(points: np.ndarray, target_mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    points = _validated_points(points)
    distances = _distance_scene(target_mesh).compute_distance(o3d.core.Tensor(points)).numpy()
    return np.asarray(distances, dtype=float)

def _validated_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or (not np.isfinite(points).all()):
        raise ValueError('距离查询点必须是有限的 N×3 数组。')
    return points

def signed_point_to_mesh_distances(points: np.ndarray, target_mesh: o3d.geometry.TriangleMesh, reverse_direction: bool=False) -> np.ndarray:
    """Signed closest-surface distance based on target triangle normals."""
    points = _validated_points(points)
    scene = _distance_scene(target_mesh)
    closest = scene.compute_closest_points(o3d.core.Tensor(points))
    closest_points = closest['points'].numpy()
    primitive_normals = closest['primitive_normals'].numpy()
    signed = np.einsum('ij,ij->i', points - closest_points, primitive_normals)
    if reverse_direction:
        signed = -signed
    return np.asarray(signed, dtype=float)

def directional_colormap(signed_distances_mm: np.ndarray, color_max_mm: float, green_tolerance_mm: float=0.05, saturation: float=0.75) -> np.ndarray:
    """Map direction to a softened red-green-blue scale."""
    values = np.asarray(signed_distances_mm, dtype=float)
    if color_max_mm <= green_tolerance_mm or green_tolerance_mm < 0:
        raise ValueError('双向色标上限必须大于绿色容差。')
    if not 0.0 <= saturation <= 1.0:
        raise ValueError('颜色饱和度必须在 0–1 之间。')
    if not np.isfinite(values).all():
        raise ValueError('有符号距离包含非有限数值。')
    colors = np.zeros((len(values), 3), dtype=float)
    colors[:, 1] = 1.0
    denominator = color_max_mm - green_tolerance_mm
    positive = values > green_tolerance_mm
    positive_strength = np.clip((values[positive] - green_tolerance_mm) / denominator, 0.0, 1.0)
    colors[positive, 0] = positive_strength
    colors[positive, 1] = 1.0 - positive_strength
    negative = values < -green_tolerance_mm
    negative_strength = np.clip((-values[negative] - green_tolerance_mm) / denominator, 0.0, 1.0)
    colors[negative, 1] = 1.0 - negative_strength
    colors[negative, 2] = negative_strength
    return colors * saturation + (1.0 - saturation)

def symmetric_surface_statistics(target_mesh: o3d.geometry.TriangleMesh, source_mesh: o3d.geometry.TriangleMesh, sample_count: int, color_max_mm: float, green_tolerance_mm: float, direction_reversed: bool) -> DistanceStatistics:
    target_points = np.asarray(sample_registration_cloud(target_mesh, sample_count, seed=20260821).points)
    source_points = np.asarray(sample_registration_cloud(source_mesh, sample_count, seed=20260822).points)
    target_to_source = point_to_mesh_distances(target_points, source_mesh)
    source_to_target = point_to_mesh_distances(source_points, target_mesh)
    combined = np.concatenate([target_to_source, source_to_target])
    return DistanceStatistics(symmetric_rms_mm=float(np.sqrt(np.mean(np.square(combined)))), mean_mm=float(np.mean(combined)), median_mm=float(np.median(combined)), hd95_mm=float(np.quantile(combined, 0.95)), maximum_mm=float(np.max(combined)), vertex_mean_mm=0.0, color_max_mm=float(color_max_mm), green_tolerance_mm=float(green_tolerance_mm), direction_reversed=direction_reversed)

def compare_meshes(target_mesh: o3d.geometry.TriangleMesh, source_mesh: o3d.geometry.TriangleMesh, transformation: np.ndarray, sample_count: int, color_max_mm: float, green_tolerance_mm: float=0.05, reverse_direction: bool=False, *, minimum_nominal_mm: float | None=None, maximum_nominal_mm: float | None=None) -> ComparisonResult:
    if color_max_mm <= green_tolerance_mm or not np.isfinite(color_max_mm):
        raise ValueError('双向色标上限必须大于绿色容差。')
    transform = np.asarray(transformation, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('配准变换必须是有限的 4×4 矩阵。')
    aligned = clone_mesh(source_mesh)
    aligned.transform(transform)
    aligned.compute_vertex_normals()
    vertices = np.asarray(aligned.vertices)
    vertex_distances = point_to_mesh_distances(vertices, target_mesh)
    signed_distances = signed_point_to_mesh_distances(vertices, target_mesh, reverse_direction)
    minimum_nominal = -float(green_tolerance_mm) if minimum_nominal_mm is None else float(minimum_nominal_mm)
    maximum_nominal = float(green_tolerance_mm) if maximum_nominal_mm is None else float(maximum_nominal_mm)
    scale = DeviationScale.from_signed_distances(
        signed_distances,
        minimum_nominal_mm=minimum_nominal,
        maximum_nominal_mm=maximum_nominal,
        fallback_limit_mm=float(color_max_mm),
    )
    colored = clone_mesh(aligned)
    colored.vertex_colors = o3d.utility.Vector3dVector(scale.map_colors(signed_distances))
    statistics = symmetric_surface_statistics(target_mesh, aligned, sample_count, color_max_mm, green_tolerance_mm, reverse_direction)
    statistics = DistanceStatistics(symmetric_rms_mm=statistics.symmetric_rms_mm, mean_mm=statistics.mean_mm, median_mm=statistics.median_mm, hd95_mm=statistics.hd95_mm, maximum_mm=statistics.maximum_mm, vertex_mean_mm=float(np.mean(vertex_distances)), color_max_mm=statistics.color_max_mm, green_tolerance_mm=statistics.green_tolerance_mm, direction_reversed=statistics.direction_reversed)
    return ComparisonResult(aligned_source=aligned, colored_source=colored, vertex_distances_mm=vertex_distances, signed_vertex_distances_mm=signed_distances, statistics=statistics, deviation_scale=scale)
