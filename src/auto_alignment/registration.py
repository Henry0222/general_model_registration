from __future__ import annotations
import itertools
import math
import time
from dataclasses import dataclass, replace
from typing import Callable
import numpy as np
import open3d as o3d
from auto_alignment.config import AlignmentConfig
from auto_alignment.mesh_io import MeshFacts, clone_mesh, prepare_cloud, sample_registration_cloud
from auto_alignment.quality import CandidateDiagnostic, PositionConfidence, RegistrationQualityReport, assess_registration_quality
ProgressCallback = Callable[[float, str], None]

@dataclass(frozen=True)
class RegistrationMetrics:
    fitness: float
    inlier_rmse_mm: float
    correspondence_count: int
    overlap_ratio: float
    rotation_degrees: float
    translation_mm: float
    candidate_diagnostics: tuple[CandidateDiagnostic, ...] = ()
    high_precision_decision: dict[str, object] | None = None
    selection_decision: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            'fitness': self.fitness,
            'inlier_rmse_mm': self.inlier_rmse_mm,
            'correspondence_count': self.correspondence_count,
            'overlap_ratio': self.overlap_ratio,
            'rotation_degrees': self.rotation_degrees,
            'translation_mm': self.translation_mm,
            'candidate_diagnostics': [diagnostic.as_dict() for diagnostic in self.candidate_diagnostics],
            'high_precision_decision': self.high_precision_decision,
            'selection_decision': self.selection_decision,
        }

@dataclass(frozen=True)
class RegistrationResult:
    transformation: np.ndarray
    status: str
    confidence: str
    metrics: RegistrationMetrics
    warnings: tuple[str, ...]
    elapsed_seconds: float
    quality: RegistrationQualityReport | None = None

    @property
    def succeeded(self) -> bool:
        return self.status != 'failed'

@dataclass(frozen=True)
class RegistrationRefinement:
    transformation: np.ndarray
    fitness: float
    inlier_rmse_mm: float
    correspondence_count: int
    elapsed_seconds: float

@dataclass(frozen=True)
class HighPrecisionRefinement:
    transformation: np.ndarray
    point_to_surface_rmse_mm: float
    correspondence_count: int
    iterations: int

def _notify(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(fraction, message)

def _features(cloud: o3d.geometry.PointCloud, voxel: float, config: AlignmentConfig) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * config.feature_radius_multiplier, max_nn=100))

def _global_registration(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, source_feature: o3d.pipelines.registration.Feature, target_feature: o3d.pipelines.registration.Feature, voxel: float, config: AlignmentConfig, *, mutual_filter: bool=True, distance_multiplier: float | None=None) -> o3d.pipelines.registration.RegistrationResult:
    distance = voxel * (config.ransac_distance_multiplier if distance_multiplier is None else distance_multiplier)
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source, target, source_feature, target_feature, mutual_filter=mutual_filter, max_correspondence_distance=distance, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4, checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance), o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(math.radians(45.0))], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(config.ransac_max_iterations, config.ransac_confidence))

def _fast_global_registration(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, source_feature: o3d.pipelines.registration.Feature, target_feature: o3d.pipelines.registration.Feature, voxel: float) -> o3d.pipelines.registration.RegistrationResult:
    """Return a deterministic feature-based candidate before random restarts."""
    # Open3D FGR samples feature tuples from its process-global RNG.  Reset it
    # here so imports, GUI startup, or a preceding task cannot change the
    # no-selection baseline. Seed 0 reproduces the stable v1.4.0 full pipeline.
    o3d.utility.random.seed(0)
    options = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=voxel * 2.0, iteration_number=128, maximum_tuple_count=2000, tuple_scale=0.95, decrease_mu=True, division_factor=1.4, use_absolute_scale=False)
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source, target, source_feature, target_feature, options)

def _surface_area(mesh: o3d.geometry.TriangleMesh) -> float:
    area = float(mesh.get_surface_area())
    return area if np.isfinite(area) and area > 0 else 1.0

def _registration_sample_counts(target_mesh: o3d.geometry.TriangleMesh, source_mesh: o3d.geometry.TriangleMesh, config: AlignmentConfig) -> tuple[int, int, bool]:
    """Keep small components visible when one mesh is much larger.

    Equal point counts give the larger mesh a much lower sampling density. We
    scale counts by the square root of surface-area ratio, which is enough to
    preserve local features without making cost grow linearly with model area.
    """
    base = max(1000, int(config.global_sample_points))
    target_area = _surface_area(target_mesh)
    source_area = _surface_area(source_mesh)
    smaller_area = min(target_area, source_area)
    area_ratio = max(target_area, source_area) / smaller_area
    area_mismatch = config.partial_registration_enabled and area_ratio >= config.partial_area_ratio_threshold
    if not area_mismatch:
        return (base, base, False)
    cap = max(1.0, float(config.max_global_sample_multiplier))

    def count_for(area: float) -> int:
        multiplier = min(cap, math.sqrt(area / smaller_area))
        return max(base, int(round(base * multiplier)))
    return (count_for(target_area), count_for(source_area), True)

def _principal_axis_candidates(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> list[np.ndarray]:
    """Generate rigid centroid/PCA initializations for FPFH fallback."""
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    _, source_axes = np.linalg.eigh(np.cov((source_points - source_center).T))
    _, target_axes = np.linalg.eigh(np.cov((target_points - target_center).T))
    candidates: list[np.ndarray] = []
    for signs in (np.diag([1.0, 1.0, 1.0]), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])):
        rotation = target_axes @ signs @ source_axes.T
        if np.linalg.det(rotation) < 0:
            continue
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = target_center - rotation @ source_center
        candidates.append(transform)
    return candidates


def _right_handed_principal_frame(
    cloud: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(cloud.points, dtype=float)
    center = points.mean(axis=0)
    eigenvalues, axes = np.linalg.eigh(np.cov((points - center).T))
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return center, eigenvalues, axes


def _proper_signed_axis_permutations() -> tuple[np.ndarray, ...]:
    """Return the 24 orientation-preserving signed axis permutations."""
    orientations: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            orientation = np.zeros((3, 3), dtype=float)
            for source_axis, target_axis in enumerate(permutation):
                orientation[target_axis, source_axis] = signs[source_axis]
            if np.linalg.det(orientation) > 0.5:
                orientations.append(orientation)
    return tuple(orientations)


def _coordinate_axis_rotation(axis: int, angle_radians: float) -> np.ndarray:
    cosine = math.cos(float(angle_radians))
    sine = math.sin(float(angle_radians))
    rotation = np.eye(3, dtype=float)
    first, second = tuple(index for index in range(3) if index != int(axis))
    rotation[first, first] = cosine
    rotation[first, second] = -sine
    rotation[second, first] = sine
    rotation[second, second] = cosine
    return rotation


def _near_symmetric_principal_axes(
    eigenvalues: np.ndarray,
    tolerance_ratio: float,
) -> tuple[int, ...]:
    values = np.asarray(eigenvalues, dtype=float).reshape(3)
    tolerance = max(0.0, float(tolerance_ratio))
    axes: list[int] = []
    for axis in range(3):
        perpendicular = [index for index in range(3) if index != axis]
        first, second = values[perpendicular]
        scale = max(abs(float(first)), abs(float(second)), 1e-12)
        if abs(float(first - second)) / scale <= tolerance:
            axes.append(axis)
    return tuple(axes)


def _exhaustive_principal_axis_candidates(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    angle_step_degrees: float = 30.0,
    eigen_tolerance_ratio: float = 0.08,
    max_candidates: int = 96,
) -> list[np.ndarray]:
    """Enumerate plausible proper rotations for symmetric coarse alignment.

    The first 24 transforms cover every permutation/sign assignment of the
    three PCA axes.  If both clouds have an approximately rotationally
    symmetric PCA pair, additional spins are generated around the matching
    symmetry axis.  Reflections are never produced.
    """
    source_center, source_values, source_axes = _right_handed_principal_frame(source)
    target_center, target_values, target_axes = _right_handed_principal_frame(target)
    source_symmetric = set(
        _near_symmetric_principal_axes(source_values, eigen_tolerance_ratio)
    )
    target_symmetric = set(
        _near_symmetric_principal_axes(target_values, eigen_tolerance_ratio)
    )
    base_orientations = _proper_signed_axis_permutations()
    orientations: list[np.ndarray] = list(base_orientations)
    step = float(np.clip(angle_step_degrees, 5.0, 180.0))
    angles = np.deg2rad(np.arange(step, 360.0 - step * 0.25, step))
    for orientation in base_orientations:
        for source_axis in source_symmetric:
            mapped = np.flatnonzero(np.abs(orientation[:, source_axis]) > 0.5)
            if len(mapped) != 1 or int(mapped[0]) not in target_symmetric:
                continue
            for angle in angles:
                orientations.append(
                    orientation
                    @ _coordinate_axis_rotation(source_axis, float(angle))
                )

    candidates: list[np.ndarray] = []
    limit = max(24, int(max_candidates))
    for orientation in orientations:
        rotation = target_axes @ orientation @ source_axes.T
        if np.linalg.det(rotation) < 0.999999:
            continue
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = target_center - rotation @ source_center
        if any(np.allclose(transform, item, atol=1e-8, rtol=0.0) for item in candidates):
            continue
        candidates.append(transform)
        if len(candidates) >= limit:
            break
    return candidates

def _distance_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene

def _surface_support(source: o3d.geometry.PointCloud, target_scene: o3d.t.geometry.RaycastingScene, transform: np.ndarray, threshold_mm: float) -> tuple[float, float, float, float, float, float]:
    """Score tight and tolerant support of the smaller surface on the target."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return (0.0, 0.0, 0.0, float('inf'), float('inf'), float('inf'))
    points = np.asarray(source.points, dtype=float)
    normals = np.asarray(source.normals, dtype=float)
    transformed = points @ matrix[:3, :3].T + matrix[:3, 3]
    closest = target_scene.compute_closest_points(o3d.core.Tensor(transformed.astype(np.float32)))
    target_points = closest['points'].numpy().astype(float)
    target_normals = closest['primitive_normals'].numpy().astype(float)
    distances = np.linalg.norm(transformed - target_points, axis=1)
    transformed_normals = normals @ matrix[:3, :3].T
    normal_agreement = np.abs(np.einsum('ij,ij->i', transformed_normals, target_normals))
    finite_mask = np.isfinite(distances) & np.isfinite(normal_agreement)
    finite = distances[finite_mask]
    agreement = normal_agreement[finite_mask]
    if len(finite) == 0:
        return (0.0, 0.0, 0.0, float('inf'), float('inf'), float('inf'))
    threshold = max(1e-06, float(threshold_mm))
    tight_threshold = min(threshold, 0.15)
    medium_threshold = min(threshold, 0.3)
    normal_match = agreement >= math.cos(math.radians(45.0))
    return (float(np.mean((finite <= tight_threshold) & normal_match)), float(np.mean((finite <= medium_threshold) & normal_match)), float(np.mean((finite <= threshold) & normal_match)), float(np.mean(np.minimum(finite, threshold))), float(np.quantile(finite, 0.9)), float(np.quantile(finite, 0.95)))

def _transforms_near(first: np.ndarray, second: np.ndarray, translation_tolerance_mm: float) -> bool:
    relative = np.asarray(first, dtype=float) @ np.linalg.inv(np.asarray(second, dtype=float))
    cosine = float(np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    translation = float(np.linalg.norm(relative[:3, 3]))
    return angle <= 2.0 and translation <= translation_tolerance_mm

def _symmetric_support(
    forward: tuple[float, float, float, float, float, float],
    backward: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Combine directed support conservatively for exhaustive pose ranking."""
    return (
        min(forward[0], backward[0]),
        min(forward[1], backward[1]),
        min(forward[2], backward[2]),
        max(forward[3], backward[3]),
        max(forward[4], backward[4]),
        max(forward[5], backward[5]),
    )


def _select_initial_transforms(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_mesh: o3d.geometry.TriangleMesh,
    target_mesh: o3d.geometry.TriangleMesh,
    coarse_candidates: list[tuple[str, np.ndarray]],
    voxel: float,
    config: AlignmentConfig,
    *,
    include_pca: bool = True,
) -> tuple[list[tuple[str, np.ndarray]], tuple[CandidateDiagnostic, ...]]:
    """Refine global candidates and rank them by smaller-surface support."""
    candidates = list(coarse_candidates)
    if include_pca:
        candidates.extend(((f'pca_{index + 1}', transform) for index, transform in enumerate(_principal_axis_candidates(source, target))))
        if config.exhaustive_orientation_search:
            candidates.extend(
                (
                    f'exhaustive_pose_{index + 1}',
                    transform,
                )
                for index, transform in enumerate(
                    _exhaustive_principal_axis_candidates(
                        source,
                        target,
                        angle_step_degrees=config.exhaustive_orientation_angle_step_degrees,
                        eigen_tolerance_ratio=config.exhaustive_orientation_eigen_tolerance_ratio,
                        max_candidates=config.exhaustive_orientation_max_candidates,
                    )
                )
            )
    if config.exhaustive_orientation_search:
        unique: list[tuple[str, np.ndarray]] = []
        for name, transform in candidates:
            matrix = np.asarray(transform, dtype=float)
            if any(
                np.allclose(matrix, existing, atol=1e-8, rtol=0.0)
                for _, existing in unique
            ):
                continue
            unique.append((name, matrix))
        candidates = unique
    distance = voxel * 4.0
    target_scene = _distance_scene(target_mesh)
    source_scene = (
        _distance_scene(source_mesh)
        if config.exhaustive_orientation_search
        else None
    )
    ranked: list[tuple[tuple[float, float, float, float, float, float], str, np.ndarray]] = []
    diagnostics: list[CandidateDiagnostic] = []
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    for name, candidate in candidates:
        refined = o3d.pipelines.registration.registration_icp(source, target, distance, candidate, estimator, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
        refined_transform = np.asarray(refined.transformation, dtype=float)
        diagnostics.append(CandidateDiagnostic(name=name, transformation=refined_transform, fitness=float(refined.fitness), inlier_rmse_mm=float(refined.inlier_rmse)))
        support_values = _surface_support(
            source,
            target_scene,
            refined_transform,
            config.coverage_distance_mm,
        )
        if source_scene is not None:
            support_values = _symmetric_support(
                support_values,
                _surface_support(
                    target,
                    source_scene,
                    np.linalg.inv(refined_transform),
                    config.coverage_distance_mm,
                ),
            )
        tight, medium, support, clipped_mean, p90, p95 = support_values
        score = (tight, medium, support, -clipped_mean, -p95, float(refined.fitness) - p90 * 0.001)
        ranked.append((score, name, refined_transform))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, np.ndarray]] = []
    tolerance = max(0.15, voxel * 0.75)
    for _, name, transform in ranked:
        if any((_transforms_near(transform, existing, tolerance) for _, existing in selected)):
            continue
        selected.append((name, transform))
        keep_count = (
            max(config.final_candidate_count, config.exhaustive_orientation_finalist_count)
            if config.exhaustive_orientation_search
            else config.final_candidate_count
        )
        if len(selected) >= max(1, int(keep_count)):
            break
    if not selected and candidates:
        selected.append(candidates[0])
    return (selected, tuple(diagnostics))

def _robust_estimator(config: AlignmentConfig):
    try:
        kernel = o3d.pipelines.registration.TukeyLoss(k=config.robust_kernel_scale_mm)
        return o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel)
    except (AttributeError, TypeError):
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()

def _vertex_area_weights(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Return one-third incident triangle area for every mesh vertex."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    weights = np.zeros(len(vertices), dtype=float)
    if len(vertices) == 0 or len(triangles) == 0:
        return weights
    corners = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    shares = np.where(np.isfinite(areas), areas / 3.0, 0.0)
    for corner in range(3):
        np.add.at(weights, triangles[:, corner], shares)
    positive = weights > 0.0
    if np.any(positive):
        weights[positive] /= float(np.mean(weights[positive]))
    return weights

def _high_precision_source(
    mesh: o3d.geometry.TriangleMesh,
    max_vertices: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use all area-weighted vertices, falling back to dense area sampling."""
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) <= max(1_000, int(max_vertices)):
        return (
            vertices.copy(),
            np.asarray(mesh.vertex_normals, dtype=float).copy(),
            _vertex_area_weights(mesh),
        )
    cloud = sample_registration_cloud(
        mesh,
        max(1_000, int(max_vertices)),
        seed=seed,
    )
    count = len(cloud.points)
    return (
        np.asarray(cloud.points, dtype=float),
        np.asarray(cloud.normals, dtype=float),
        np.ones(count, dtype=float),
    )

def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-15:
        return np.eye(3, dtype=float)
    axis = np.asarray(vector, dtype=float) / angle
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3, dtype=float) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)

def _mesh_correspondences(
    points: np.ndarray,
    normals: np.ndarray,
    transform: np.ndarray,
    scene: o3d.t.geometry.RaycastingScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    transformed_normals = normals @ transform[:3, :3].T
    closest = scene.compute_closest_points(
        o3d.core.Tensor(transformed.astype(np.float32))
    )
    target_points = closest['points'].numpy().astype(float)
    target_normals = closest['primitive_normals'].numpy().astype(float)
    deltas = transformed - target_points
    distances = np.linalg.norm(deltas, axis=1)
    normal_agreement = np.abs(
        np.einsum('ij,ij->i', transformed_normals, target_normals)
    )
    residuals = np.einsum('ij,ij->i', deltas, target_normals)
    return transformed, target_points, target_normals, distances, np.column_stack((residuals, normal_agreement))


def _weighted_ratio(mask: np.ndarray, weights: np.ndarray, eligible: np.ndarray | None = None) -> float:
    positive = np.isfinite(weights) & (weights > 0.0)
    denominator_mask = positive if eligible is None else (positive & eligible)
    denominator = float(np.sum(weights[denominator_mask]))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(weights[denominator_mask & mask]))
    return numerator / denominator


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(finite) == 0:
        return float('inf')
    selected_values = np.asarray(values[finite], dtype=float)
    selected_weights = np.asarray(weights[finite], dtype=float)
    order = np.argsort(selected_values)
    selected_values = selected_values[order]
    selected_weights = selected_weights[order]
    cumulative = np.cumsum(selected_weights)
    position = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, position)), len(selected_values) - 1)
    return float(selected_values[index])


def _bidirectional_stable_metrics(
    direct_distances: np.ndarray,
    reverse_distances: np.ndarray,
    direct_weights: np.ndarray,
    reverse_weights: np.ndarray,
    direct_mask: np.ndarray,
    reverse_mask: np.ndarray,
    robust_cap_mm: float,
) -> dict[str, float]:
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for distances, area_weights, mask in (
        (direct_distances, direct_weights, direct_mask),
        (reverse_distances, reverse_weights, reverse_mask),
    ):
        selected = mask & np.isfinite(distances) & np.isfinite(area_weights) & (area_weights > 0.0)
        if np.count_nonzero(selected) == 0:
            continue
        direction_weights = np.asarray(area_weights[selected], dtype=float)
        direction_weights /= float(np.sum(direction_weights))
        values.append(np.asarray(distances[selected], dtype=float))
        weights.append(direction_weights)
    if not values:
        return {
            'symmetric_rms_mm': float('inf'),
            'robust_rms_mm': float('inf'),
            'median_mm': float('inf'),
            'p90_mm': float('inf'),
        }
    direction_scale = 1.0 / len(values)
    combined_values = np.concatenate(values)
    combined_weights = np.concatenate([item * direction_scale for item in weights])
    cap = max(1e-9, float(robust_cap_mm))
    return {
        'symmetric_rms_mm': float(np.sqrt(np.sum(combined_weights * np.square(combined_values)))),
        'robust_rms_mm': float(
            np.sqrt(np.sum(combined_weights * np.square(np.minimum(combined_values, cap))))
        ),
        'median_mm': _weighted_quantile(combined_values, combined_weights, 0.50),
        'p90_mm': _weighted_quantile(combined_values, combined_weights, 0.90),
    }


def _spatial_support_ratio(
    points: np.ndarray,
    weights: np.ndarray,
    stable_mask: np.ndarray,
    eligible_mask: np.ndarray | None = None,
) -> float:
    positive = np.isfinite(weights) & (weights > 0.0) & np.isfinite(points).all(axis=1)
    eligible = positive if eligible_mask is None else (positive & eligible_mask)
    if np.count_nonzero(eligible) < 8:
        return 0.0
    selected_points = np.asarray(points[eligible], dtype=float)
    selected_weights = np.asarray(weights[eligible], dtype=float)
    selected_weights /= float(np.sum(selected_weights))
    center = np.sum(selected_points * selected_weights[:, None], axis=0)
    centered = selected_points - center
    covariance = (centered * selected_weights[:, None]).T @ centered
    _, basis = np.linalg.eigh(covariance)
    coordinates = centered @ basis
    cell_ids = (
        (coordinates[:, 0] >= 0.0).astype(np.int8)
        + 2 * (coordinates[:, 1] >= 0.0).astype(np.int8)
        + 4 * (coordinates[:, 2] >= 0.0).astype(np.int8)
    )
    eligible_indices = np.flatnonzero(eligible)
    occupied = 0
    supported = 0
    for cell_id in range(8):
        local = cell_ids == cell_id
        cell_weight = float(np.sum(selected_weights[local]))
        if cell_weight < 0.01:
            continue
        occupied += 1
        original_indices = eligible_indices[local]
        stable_weight = float(np.sum(weights[original_indices[stable_mask[original_indices]]]))
        total_weight = float(np.sum(weights[original_indices]))
        if total_weight > 0.0 and stable_weight / total_weight >= 0.50:
            supported += 1
    return supported / occupied if occupied > 0 else 0.0


def _observability_metrics(
    transformed_points: np.ndarray,
    target_normals: np.ndarray,
    weights: np.ndarray,
    stable_mask: np.ndarray,
) -> dict[str, float | int]:
    valid = (
        stable_mask
        & np.isfinite(transformed_points).all(axis=1)
        & np.isfinite(target_normals).all(axis=1)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if np.count_nonzero(valid) < 100:
        return {
            'rank': 0,
            'condition_number': float('inf'),
            'normal_diversity': 0.0,
        }
    points = np.asarray(transformed_points[valid], dtype=float)
    normals = np.asarray(target_normals[valid], dtype=float)
    selected_weights = np.asarray(weights[valid], dtype=float)
    selected_weights /= float(np.sum(selected_weights))
    center = np.sum(points * selected_weights[:, None], axis=0)
    centered = points - center
    radius = float(np.sqrt(np.sum(selected_weights * np.sum(np.square(centered), axis=1))))
    radius = max(radius, 1e-9)
    jacobian = np.column_stack((np.cross(centered / radius, normals), normals))
    information = (jacobian * selected_weights[:, None]).T @ jacobian
    eigenvalues = np.maximum(0.0, np.linalg.eigvalsh(information))
    largest = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    threshold = max(1e-12, largest * 1e-6)
    rank = int(np.count_nonzero(eigenvalues > threshold))
    smallest = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    condition = largest / smallest if smallest > 1e-15 else float('inf')
    normal_information = (normals * selected_weights[:, None]).T @ normals
    normal_eigenvalues = np.maximum(0.0, np.linalg.eigvalsh(normal_information))
    normal_largest = float(normal_eigenvalues[-1]) if len(normal_eigenvalues) else 0.0
    normal_diversity = (
        float(normal_eigenvalues[0]) / normal_largest
        if normal_largest > 1e-15
        else 0.0
    )
    return {
        'rank': rank,
        'condition_number': float(condition),
        'normal_diversity': float(normal_diversity),
    }


def _box_points_in_target_frame(
    mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(mesh.get_axis_aligned_bounding_box().get_box_points(), dtype=float)
    if transform is not None:
        points = points @ transform[:3, :3].T + transform[:3, 3]
    return points


def _relative_displacement(points: np.ndarray, relative_transform: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    moved = points @ relative_transform[:3, :3].T + relative_transform[:3, 3]
    return float(np.max(np.linalg.norm(moved - points, axis=1)))


def _evaluate_high_precision_candidate(
    target_mesh: o3d.geometry.TriangleMesh,
    source_mesh: o3d.geometry.TriangleMesh,
    before_transform: np.ndarray,
    after_transform: np.ndarray,
    candidate_rmse_mm: float,
    config: AlignmentConfig,
) -> dict[str, object]:
    stable_distance = max(1e-6, float(config.high_precision_gate_stable_distance_mm))
    roi_distance = max(stable_distance, float(config.high_precision_gate_roi_distance_mm))
    normal_threshold = math.cos(math.radians(float(config.high_precision_normal_angle_degrees)))
    sample_limit = max(1_000, int(config.high_precision_max_vertices))
    source_points, source_normals, source_weights = _high_precision_source(
        source_mesh, sample_limit, config.random_seed + 3101
    )
    target_points, target_normals, target_weights = _high_precision_source(
        target_mesh, sample_limit, config.random_seed + 3102
    )
    target_scene = _distance_scene(target_mesh)
    source_scene = _distance_scene(source_mesh)

    direct_before = _mesh_correspondences(
        source_points, source_normals, before_transform, target_scene
    )
    direct_after = _mesh_correspondences(
        source_points, source_normals, after_transform, target_scene
    )
    inverse_before = np.linalg.inv(before_transform)
    inverse_after = np.linalg.inv(after_transform)
    reverse_before = _mesh_correspondences(
        target_points, target_normals, inverse_before, source_scene
    )
    reverse_after = _mesh_correspondences(
        target_points, target_normals, inverse_after, source_scene
    )

    direct_before_distances = direct_before[3]
    direct_after_distances = direct_after[3]
    reverse_before_distances = reverse_before[3]
    reverse_after_distances = reverse_after[3]
    direct_stable = (
        np.isfinite(direct_before_distances)
        & np.isfinite(direct_after_distances)
        & (direct_before_distances <= stable_distance)
        & (direct_after_distances <= stable_distance)
        & (direct_before[4][:, 1] >= normal_threshold)
        & (direct_after[4][:, 1] >= normal_threshold)
    )
    reverse_stable = (
        np.isfinite(reverse_before_distances)
        & np.isfinite(reverse_after_distances)
        & (reverse_before_distances <= stable_distance)
        & (reverse_after_distances <= stable_distance)
        & (reverse_before[4][:, 1] >= normal_threshold)
        & (reverse_after[4][:, 1] >= normal_threshold)
    )
    direct_roi = (
        np.isfinite(direct_before_distances)
        & (direct_before_distances <= roi_distance)
        & (direct_before[4][:, 1] >= normal_threshold)
    )
    reverse_roi = (
        np.isfinite(reverse_before_distances)
        & (reverse_before_distances <= roi_distance)
        & (reverse_before[4][:, 1] >= normal_threshold)
    )

    source_coverage = _weighted_ratio(direct_stable, source_weights)
    target_coverage = _weighted_ratio(reverse_stable, target_weights)
    source_area = _surface_area(source_mesh)
    target_area = _surface_area(target_mesh)
    area_ratio = max(source_area, target_area) / max(1e-12, min(source_area, target_area))
    local_mode = bool(
        config.partial_registration_enabled
        and area_ratio >= float(config.partial_area_ratio_threshold)
    )
    if local_mode and source_area <= target_area:
        mode = 'local_to_whole'
        local_coverage = source_coverage
        roi_coverage = _weighted_ratio(reverse_stable, target_weights, reverse_roi)
        local_spatial = _spatial_support_ratio(
            source_points, source_weights, direct_stable
        )
        roi_spatial = _spatial_support_ratio(
            target_points, target_weights, reverse_stable, reverse_roi
        )
        observability = _observability_metrics(
            direct_after[0], direct_after[2], source_weights, direct_stable
        )
        local_box = _box_points_in_target_frame(source_mesh, before_transform)
        whole_box = _box_points_in_target_frame(target_mesh)
    elif local_mode:
        mode = 'local_to_whole'
        local_coverage = target_coverage
        roi_coverage = _weighted_ratio(direct_stable, source_weights, direct_roi)
        local_spatial = _spatial_support_ratio(
            target_points, target_weights, reverse_stable
        )
        roi_spatial = _spatial_support_ratio(
            source_points, source_weights, direct_stable, direct_roi
        )
        observability = _observability_metrics(
            reverse_after[0], reverse_after[2], target_weights, reverse_stable
        )
        local_box = _box_points_in_target_frame(target_mesh)
        whole_box = _box_points_in_target_frame(source_mesh, before_transform)
    else:
        mode = 'whole_to_whole'
        local_coverage = min(source_coverage, target_coverage)
        roi_coverage = local_coverage
        local_spatial = _spatial_support_ratio(
            source_points, source_weights, direct_stable
        )
        roi_spatial = _spatial_support_ratio(
            target_points, target_weights, reverse_stable
        )
        observability = _observability_metrics(
            direct_after[0], direct_after[2], source_weights, direct_stable
        )
        local_box = _box_points_in_target_frame(source_mesh, before_transform)
        whole_box = np.vstack((local_box, _box_points_in_target_frame(target_mesh)))
    spatial_coverage = min(local_spatial, roi_spatial)

    before_metrics = _bidirectional_stable_metrics(
        direct_before_distances,
        reverse_before_distances,
        source_weights,
        target_weights,
        direct_stable,
        reverse_stable,
        stable_distance * 0.5,
    )
    after_metrics = _bidirectional_stable_metrics(
        direct_after_distances,
        reverse_after_distances,
        source_weights,
        target_weights,
        direct_stable,
        reverse_stable,
        stable_distance * 0.5,
    )

    relative = after_transform @ np.linalg.inv(before_transform)
    delta_rotation = _rotation_angle_degrees(relative)
    local_center = np.mean(local_box, axis=0)
    moved_center = local_center @ relative[:3, :3].T + relative[:3, 3]
    center_displacement = float(np.linalg.norm(moved_center - local_center))
    max_local_displacement = _relative_displacement(local_box, relative)
    max_extrapolated_displacement = _relative_displacement(whole_box, relative)

    reasons: list[str] = []
    if local_mode:
        if local_coverage < float(config.high_precision_gate_local_coverage_ratio):
            reasons.append('局部模型稳定覆盖率不足。')
        if roi_coverage < float(config.high_precision_gate_roi_coverage_ratio):
            reasons.append('整体模型对应 ROI 的稳定覆盖率不足。')
    else:
        threshold = float(config.high_precision_gate_whole_coverage_ratio)
        if source_coverage < threshold or target_coverage < threshold:
            reasons.append('双向全表面稳定覆盖率不足。')
    if spatial_coverage < float(config.high_precision_gate_spatial_coverage_ratio):
        reasons.append('稳定对应点的空间分布不足。')

    relative_improvement = max(0.0, float(config.high_precision_gate_min_relative_improvement))
    median_improved = after_metrics['median_mm'] <= before_metrics['median_mm'] * (1.0 - relative_improvement)
    robust_rms_improved = after_metrics['robust_rms_mm'] <= before_metrics['robust_rms_mm'] * (1.0 - relative_improvement)
    p90_preserved = after_metrics['p90_mm'] <= (
        before_metrics['p90_mm'] + float(config.high_precision_gate_p90_tolerance_mm)
    )
    if not median_improved:
        reasons.append('稳定区域中位误差没有改善。')
    if not robust_rms_improved:
        reasons.append('稳定区域稳健 RMS 没有改善。')
    if not p90_preserved:
        reasons.append('稳定区域 P90 出现恶化。')
    if max_local_displacement > float(config.high_precision_gate_max_local_displacement_mm):
        reasons.append('末级 ICP 在局部区域产生的位移过大。')
    if max_extrapolated_displacement > float(config.high_precision_gate_max_extrapolated_displacement_mm):
        reasons.append('末级 ICP 的远距离外推位移过大。')
    if (
        int(observability['rank']) < 6
        or float(observability['condition_number']) > float(config.high_precision_gate_max_condition_number)
        or float(observability['normal_diversity']) < float(config.high_precision_gate_min_normal_diversity)
    ):
        reasons.append('局部表面对旋转或滑动方向的约束不足。')

    accepted = bool(config.high_precision_gate_enabled and not reasons)
    if not config.high_precision_gate_enabled:
        accepted = True
        reasons = []
    selected_metrics = after_metrics if accepted else before_metrics
    return {
        'enabled': bool(config.high_precision_gate_enabled),
        'accepted': accepted,
        'selected_stage': 'high_precision_icp' if accepted else 'multiscale_icp',
        'mode': mode,
        'reasons': reasons,
        'stable_distance_mm': stable_distance,
        'roi_distance_mm': roi_distance,
        'source_stable_coverage_ratio': source_coverage,
        'target_stable_coverage_ratio': target_coverage,
        'local_stable_coverage_ratio': local_coverage,
        'target_roi_stable_coverage_ratio': roi_coverage,
        'spatial_coverage_ratio': spatial_coverage,
        'before_metrics': before_metrics,
        'candidate_metrics': after_metrics,
        'selected_metrics': selected_metrics,
        'candidate_point_to_surface_rmse_mm': float(candidate_rmse_mm),
        'delta_rotation_degrees': delta_rotation,
        'delta_center_displacement_mm': center_displacement,
        'delta_max_local_displacement_mm': max_local_displacement,
        'delta_max_extrapolated_displacement_mm': max_extrapolated_displacement,
        'observability': observability,
    }

def _point_to_mesh_refinement(
    target_mesh: o3d.geometry.TriangleMesh,
    source_mesh: o3d.geometry.TriangleMesh,
    initial_transform: np.ndarray,
    config: AlignmentConfig,
    progress: ProgressCallback | None = None,
) -> HighPrecisionRefinement:
    """Refine a good rigid pose against target triangles without voxelization.

    The objective is an area-weighted, robust point-to-plane residual where
    correspondences are closest points on target triangles rather than nearest
    target vertices.  When overlap is partial, the smaller surface drives the
    optimization so missing geometry does not dominate the result.
    """
    source_area = _surface_area(source_mesh)
    target_area = _surface_area(target_mesh)
    reverse = bool(config.partial_registration_enabled and source_area > target_area)
    if reverse:
        moving_mesh = target_mesh
        fixed_mesh = source_mesh
        transform = np.linalg.inv(np.asarray(initial_transform, dtype=float))
    else:
        moving_mesh = source_mesh
        fixed_mesh = target_mesh
        transform = np.asarray(initial_transform, dtype=float).copy()

    points, normals, area_weights = _high_precision_source(
        moving_mesh,
        config.high_precision_max_vertices,
        config.random_seed + 1701,
    )
    positive = np.isfinite(area_weights) & (area_weights > 0.0)
    if len(points) < 100 or np.count_nonzero(positive) < 100:
        return HighPrecisionRefinement(
            transformation=np.asarray(initial_transform, dtype=float),
            point_to_surface_rmse_mm=float('inf'),
            correspondence_count=0,
            iterations=0,
        )

    scene = _distance_scene(fixed_mesh)
    normal_threshold = math.cos(
        math.radians(float(config.high_precision_normal_angle_degrees))
    )
    total_iterations = 0
    final_rmse = float('inf')
    final_count = 0
    stages = tuple(
        distance for distance in config.high_precision_distance_stages_mm
        if np.isfinite(distance) and distance > 0.0
    )
    for stage_index, max_distance in enumerate(stages):
        stage_best = transform.copy()
        stage_best_rmse = float('inf')
        stage_best_count = 0
        previous_plane_rmse = float('inf')
        for _ in range(max(1, int(config.high_precision_iterations))):
            transformed, _, target_normals, distances, residual_data = _mesh_correspondences(
                points, normals, transform, scene
            )
            residuals = residual_data[:, 0]
            normal_agreement = residual_data[:, 1]
            valid = (
                positive
                & np.isfinite(distances)
                & np.isfinite(residuals)
                & np.isfinite(normal_agreement)
                & (distances <= max_distance)
                & (normal_agreement >= normal_threshold)
            )
            if np.count_nonzero(valid) < 100:
                break
            valid_distances = distances[valid]
            valid_residuals = residuals[valid]
            valid_weights = area_weights[valid]
            surface_rmse = float(
                np.sqrt(np.average(np.square(valid_distances), weights=valid_weights))
            )
            if surface_rmse < stage_best_rmse:
                stage_best = transform.copy()
                stage_best_rmse = surface_rmse
                stage_best_count = int(np.count_nonzero(valid))

            median = float(np.median(valid_residuals))
            mad = float(np.median(np.abs(valid_residuals - median)))
            robust_scale = 1.4826 * mad
            kernel = min(
                max_distance,
                max(
                    float(config.high_precision_robust_floor_mm),
                    4.0 * robust_scale,
                ),
            )
            normalized = np.abs(valid_residuals) / max(kernel, 1e-12)
            robust_weights = np.square(
                np.maximum(0.0, 1.0 - np.square(normalized))
            )
            combined_weights = valid_weights * robust_weights
            usable = combined_weights > 1e-12
            if np.count_nonzero(usable) < 100:
                combined_weights = valid_weights
                usable = combined_weights > 0.0

            p = transformed[valid][usable]
            n = target_normals[valid][usable]
            r = valid_residuals[usable]
            sqrt_weights = np.sqrt(combined_weights[usable])
            jacobian = np.column_stack((np.cross(p, n), n))
            weighted_jacobian = jacobian * sqrt_weights[:, None]
            weighted_rhs = -r * sqrt_weights
            step, _, rank, _ = np.linalg.lstsq(
                weighted_jacobian, weighted_rhs, rcond=None
            )
            if rank < 6 or not np.isfinite(step).all():
                break

            rotation_step = np.asarray(step[:3], dtype=float)
            translation_step = np.asarray(step[3:], dtype=float)
            rotation_limit = math.radians(1.0)
            rotation_norm = float(np.linalg.norm(rotation_step))
            if rotation_norm > rotation_limit:
                rotation_step *= rotation_limit / rotation_norm
            translation_norm = float(np.linalg.norm(translation_step))
            if translation_norm > max_distance:
                translation_step *= max_distance / translation_norm

            increment = np.eye(4, dtype=float)
            increment[:3, :3] = _rotation_from_vector(rotation_step)
            increment[:3, 3] = translation_step
            transform = increment @ transform
            total_iterations += 1
            plane_rmse = float(
                np.sqrt(np.average(np.square(r), weights=combined_weights[usable]))
            )
            if (
                abs(previous_plane_rmse - plane_rmse) <= 1e-11
                and float(np.linalg.norm(rotation_step)) <= 1e-10
                and float(np.linalg.norm(translation_step)) <= 1e-9
            ):
                break
            previous_plane_rmse = plane_rmse

        transformed, _, _, distances, residual_data = _mesh_correspondences(
            points, normals, transform, scene
        )
        valid = (
            positive
            & np.isfinite(distances)
            & np.isfinite(residual_data[:, 0])
            & np.isfinite(residual_data[:, 1])
            & (distances <= max_distance)
            & (residual_data[:, 1] >= normal_threshold)
        )
        if np.count_nonzero(valid) >= 100:
            candidate_rmse = float(
                np.sqrt(
                    np.average(
                        np.square(distances[valid]), weights=area_weights[valid]
                    )
                )
            )
            if candidate_rmse < stage_best_rmse:
                stage_best = transform.copy()
                stage_best_rmse = candidate_rmse
                stage_best_count = int(np.count_nonzero(valid))
        transform = stage_best
        final_rmse = stage_best_rmse
        final_count = stage_best_count
        _notify(
            progress,
            0.86 + 0.035 * (stage_index + 1) / max(1, len(stages)),
            f'全分辨率曲面精配准 {stage_index + 1}/{len(stages)}…',
        )

    if reverse:
        transform = np.linalg.inv(transform)
    return HighPrecisionRefinement(
        transformation=transform,
        point_to_surface_rmse_mm=final_rmse,
        correspondence_count=final_count,
        iterations=total_iterations,
    )

def _rotation_angle_degrees(transform: np.ndarray) -> float:
    rotation = transform[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))

def _aabb_overlap_ratio(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.ndarray) -> float:
    transformed = o3d.geometry.PointCloud(source)
    transformed.transform(transform)
    a = transformed.get_axis_aligned_bounding_box()
    b = target.get_axis_aligned_bounding_box()
    minimum = np.maximum(a.min_bound, b.min_bound)
    maximum = np.minimum(a.max_bound, b.max_bound)
    intersection_extent = np.maximum(0.0, maximum - minimum)
    intersection = float(np.prod(intersection_extent))
    volume_a = float(np.prod(np.maximum(0.0, a.get_extent())))
    volume_b = float(np.prod(np.maximum(0.0, b.get_extent())))
    denominator = min(volume_a, volume_b)
    return intersection / denominator if denominator > 0 else 0.0

def refine_registration(target_mesh: o3d.geometry.TriangleMesh, source_mesh: o3d.geometry.TriangleMesh, config: AlignmentConfig, progress: ProgressCallback | None=None) -> RegistrationRefinement:
    """Refine an already aligned mesh pair with robust multiscale ICP.

    The returned transform maps ``source_mesh`` into ``target_mesh``.  This is
    intentionally a local refinement: it starts at identity and is used after
    global alignment has made it possible to remove target-coordinate ROIs
    from both meshes.
    """
    started = time.perf_counter()
    if len(target_mesh.triangles) == 0 or len(source_mesh.triangles) == 0:
        raise ValueError('稳定表面为空，无法执行排除选区后的精配准。')
    target_diagonal = float(np.linalg.norm(target_mesh.get_axis_aligned_bounding_box().get_extent()))
    source_diagonal = float(np.linalg.norm(source_mesh.get_axis_aligned_bounding_box().get_extent()))
    voxel = config.effective_voxel(min(target_diagonal, source_diagonal))
    target_count, source_count, _ = _registration_sample_counts(target_mesh, source_mesh, config)
    target_raw = sample_registration_cloud(target_mesh, target_count, seed=config.random_seed + 101)
    source_raw = sample_registration_cloud(source_mesh, source_count, seed=config.random_seed + 102)
    reverse_registration = config.partial_registration_enabled and _surface_area(source_mesh) > _surface_area(target_mesh)
    if reverse_registration:
        registration_source_raw = target_raw
        registration_target_raw = source_raw
    else:
        registration_source_raw = source_raw
        registration_target_raw = target_raw
    estimator = _robust_estimator(config)
    transform = np.eye(4, dtype=float)
    final_source: o3d.geometry.PointCloud | None = None
    final_target: o3d.geometry.PointCloud | None = None
    final_distance = voxel
    for index, fraction in enumerate(config.voxel_fractions):
        level_voxel = voxel * fraction
        final_source = prepare_cloud(registration_source_raw, level_voxel, config.normal_radius_multiplier)
        final_target = prepare_cloud(registration_target_raw, level_voxel, config.normal_radius_multiplier)
        final_distance = level_voxel * config.correspondence_multipliers[index]
        result = o3d.pipelines.registration.registration_icp(final_source, final_target, final_distance, transform, estimator, o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-07, relative_rmse=1e-07, max_iteration=config.icp_iterations[index]))
        transform = np.asarray(result.transformation, dtype=float)
        _notify(progress, 0.9 + 0.035 * (index + 1) / len(config.voxel_fractions), '正在排除 DPlan 规划选区后精配准…')
    if final_source is None or final_target is None:
        raise ValueError('未配置精配准尺度。')
    evaluation = o3d.pipelines.registration.evaluate_registration(final_source, final_target, final_distance, transform)
    if reverse_registration:
        transform = np.linalg.inv(transform)
    return RegistrationRefinement(transformation=transform, fitness=float(evaluation.fitness), inlier_rmse_mm=float(evaluation.inlier_rmse), correspondence_count=len(evaluation.correspondence_set), elapsed_seconds=time.perf_counter() - started)

def _register_meshes_once(
    target_mesh: o3d.geometry.TriangleMesh,
    source_mesh: o3d.geometry.TriangleMesh,
    target_facts: MeshFacts,
    source_facts: MeshFacts,
    config: AlignmentConfig,
    progress: ProgressCallback | None = None,
    *,
    target_priority_faces: np.ndarray | None = None,
    source_priority_faces: np.ndarray | None = None,
) -> RegistrationResult:
    started = time.perf_counter()
    diagonal = min(target_facts.diagonal_mm, source_facts.diagonal_mm)
    voxel = config.effective_voxel(diagonal)
    _notify(progress, 0.05, '正在均匀采样 STL 表面…')
    target_sample_count, source_sample_count, area_mismatch = _registration_sample_counts(target_mesh, source_mesh, config)
    partial_mode = bool(config.partial_registration_enabled)
    target_raw = sample_registration_cloud(
        target_mesh,
        target_sample_count,
        seed=config.random_seed,
        priority_faces=target_priority_faces,
        priority_fraction=config.selection_priority_fraction,
    )
    source_raw = sample_registration_cloud(
        source_mesh,
        source_sample_count,
        seed=config.random_seed + 1,
        priority_faces=source_priority_faces,
        priority_fraction=config.selection_priority_fraction,
    )
    reverse_registration = partial_mode and _surface_area(source_mesh) > _surface_area(target_mesh)
    if reverse_registration:
        registration_source_raw = target_raw
        registration_target_raw = source_raw
        registration_source_mesh = target_mesh
        registration_target_mesh = source_mesh
    else:
        registration_source_raw = source_raw
        registration_target_raw = target_raw
        registration_source_mesh = source_mesh
        registration_target_mesh = target_mesh
    target_coarse = prepare_cloud(registration_target_raw, voxel, config.normal_radius_multiplier)
    source_coarse = prepare_cloud(registration_source_raw, voxel, config.normal_radius_multiplier)
    _notify(progress, 0.2, '正在提取 FPFH 几何特征…')
    target_feature = _features(target_coarse, voxel, config)
    source_feature = _features(source_coarse, voxel, config)
    _notify(progress, 0.35, '正在生成多个全局配准候选…')
    fgr = _fast_global_registration(source_coarse, target_coarse, source_feature, target_feature, voxel)
    coarse_candidates = [('fgr', np.asarray(fgr.transformation, dtype=float))]
    restart_count = max(1, int(config.global_registration_restarts))
    for restart in range(restart_count):
        o3d.utility.random.seed(config.random_seed + 7919 * (restart + 1))
        mutual_filter = restart % 3 == 0
        distance_multiplier = max(1.8 + 0.2 * (restart % 3), config.ransac_distance_multiplier) if partial_mode else config.ransac_distance_multiplier
        coarse = _global_registration(source_coarse, target_coarse, source_feature, target_feature, voxel, config, mutual_filter=mutual_filter, distance_multiplier=distance_multiplier)
        coarse_candidates.append((f"ransac_{restart + 1}_{('mutual' if mutual_filter else 'partial')}", np.asarray(coarse.transformation, dtype=float)))
    initial_candidates, initial_diagnostics = _select_initial_transforms(
        source_coarse,
        target_coarse,
        registration_source_mesh,
        registration_target_mesh,
        coarse_candidates,
        voxel,
        config,
        include_pca=not area_mismatch,
    )
    _notify(
        progress,
        0.55,
        (
            '彻底朝向检查完成，正在对优选姿态执行多尺度鲁棒 ICP…'
            if config.exhaustive_orientation_search
            else '正在对候选执行多尺度鲁棒 ICP…'
        ),
    )
    estimator = _robust_estimator(config)
    levels: list[tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, float, int]] = []
    for index, fraction in enumerate(config.voxel_fractions):
        level_voxel = voxel * fraction
        levels.append((prepare_cloud(registration_source_raw, level_voxel, config.normal_radius_multiplier), prepare_cloud(registration_target_raw, level_voxel, config.normal_radius_multiplier), level_voxel * config.correspondence_multipliers[index], config.icp_iterations[index]))
    target_scene = _distance_scene(registration_target_mesh)
    source_scene = (
        _distance_scene(registration_source_mesh)
        if config.exhaustive_orientation_search
        else None
    )
    fine_candidates: list[tuple[tuple[float, float, float, float, float, float], str, np.ndarray, object]] = []
    for candidate_index, (candidate_name, candidate_transform) in enumerate(initial_candidates):
        transform = np.asarray(candidate_transform, dtype=float)
        fine_result = fgr
        for source_level, target_level, max_distance, max_iterations in levels:
            fine_result = o3d.pipelines.registration.registration_icp(source_level, target_level, max_distance, transform, estimator, o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-07, relative_rmse=1e-07, max_iteration=max_iterations))
            transform = np.asarray(fine_result.transformation, dtype=float)
        support_values = _surface_support(
            registration_source_raw,
            target_scene,
            transform,
            config.coverage_distance_mm,
        )
        if source_scene is not None:
            support_values = _symmetric_support(
                support_values,
                _surface_support(
                    registration_target_raw,
                    source_scene,
                    np.linalg.inv(transform),
                    config.coverage_distance_mm,
                ),
            )
        tight, medium, support, clipped_mean, p90, p95 = support_values
        fine_candidates.append(((tight, medium, support, -clipped_mean, -p95, float(fine_result.fitness) - p90 * 0.001), candidate_name, transform, fine_result))
        _notify(progress, 0.58 + 0.28 * (candidate_index + 1) / max(1, len(initial_candidates)), f'候选精配准 {candidate_index + 1}/{len(initial_candidates)}…')
    fine_candidates.sort(key=lambda item: item[0], reverse=True)
    _, _, transform, fine_result = fine_candidates[0]
    final_source, final_target, final_distance, _ = levels[-1]
    candidate_diagnostics = tuple(initial_diagnostics) + tuple((CandidateDiagnostic(name=f'{name}_final', transformation=np.asarray(candidate_transform, dtype=float), fitness=float(candidate_result.fitness), inlier_rmse_mm=float(candidate_result.inlier_rmse)) for _, name, candidate_transform, candidate_result in fine_candidates))
    if reverse_registration:
        transform = np.linalg.inv(transform)
        candidate_diagnostics = tuple((CandidateDiagnostic(name=diagnostic.name, transformation=np.linalg.inv(diagnostic.transformation), fitness=diagnostic.fitness, inlier_rmse_mm=diagnostic.inlier_rmse_mm) for diagnostic in candidate_diagnostics))
    high_precision_decision: dict[str, object] | None = None
    if config.high_precision_refinement_enabled:
        before_high_precision = np.asarray(transform, dtype=float).copy()
        high_precision = _point_to_mesh_refinement(
            target_mesh,
            source_mesh,
            transform,
            config,
            progress,
        )
        if np.isfinite(high_precision.point_to_surface_rmse_mm):
            candidate_transform = np.asarray(high_precision.transformation, dtype=float)
            high_precision_decision = _evaluate_high_precision_candidate(
                target_mesh,
                source_mesh,
                before_high_precision,
                candidate_transform,
                high_precision.point_to_surface_rmse_mm,
                config,
            )
            high_precision_decision['candidate_transformation'] = candidate_transform
            high_precision_decision['selected_transformation'] = (
                candidate_transform
                if bool(high_precision_decision['accepted'])
                else before_high_precision
            )
            if bool(high_precision_decision['accepted']):
                transform = candidate_transform
                candidate_diagnostics = candidate_diagnostics + (
                    CandidateDiagnostic(
                        name='point_to_mesh_accepted',
                        transformation=transform,
                        fitness=1.0,
                        inlier_rmse_mm=high_precision.point_to_surface_rmse_mm,
                    ),
                )
            else:
                transform = before_high_precision
            _notify(
                progress,
                0.90,
                '末级高精度 ICP 已接受。'
                if bool(high_precision_decision['accepted'])
                else '末级高精度 ICP 未通过门控，已自动回退。',
            )
    evaluation_transform = np.linalg.inv(transform) if reverse_registration else transform
    evaluation = o3d.pipelines.registration.evaluate_registration(final_source, final_target, final_distance, evaluation_transform)
    correspondence_count = len(evaluation.correspondence_set)
    rotation = _rotation_angle_degrees(transform)
    translation = float(np.linalg.norm(transform[:3, 3]))
    overlap = _aabb_overlap_ratio(source_raw, target_raw, transform)
    metrics = RegistrationMetrics(fitness=float(evaluation.fitness), inlier_rmse_mm=float(evaluation.inlier_rmse), correspondence_count=correspondence_count, overlap_ratio=overlap, rotation_degrees=rotation, translation_mm=translation, candidate_diagnostics=candidate_diagnostics, high_precision_decision=high_precision_decision)
    aligned_for_quality = clone_mesh(source_mesh)
    aligned_for_quality.transform(transform)
    aligned_for_quality.compute_vertex_normals()
    quality = assess_registration_quality(target_mesh, aligned_for_quality, transform, overlap, candidate_diagnostics, coverage_distance_mm=config.coverage_distance_mm, min_directed_overlap=config.partial_overlap_threshold)
    confidence_names = {PositionConfidence.HIGH: '高', PositionConfidence.MEDIUM: '中', PositionConfidence.LOW: '低', PositionConfidence.FAILED: '失败'}
    warnings = list(target_facts.warnings + source_facts.warnings)
    if high_precision_decision is not None and not bool(high_precision_decision['accepted']):
        decision_reasons = '；'.join(str(value) for value in high_precision_decision.get('reasons', ()))
        warnings.append(
            '末级高精度 ICP 已自动回退。'
            + (f'原因：{decision_reasons}' if decision_reasons else '')
        )
    directed_partial_accepted = partial_mode and quality.directed_overlap_ratio >= config.partial_overlap_threshold
    if quality.position_confidence is PositionConfidence.FAILED:
        status = 'failed'
        confidence = '失败'
        warnings.extend(quality.reasons or ('共同表面过少或自动配准未找到可信解。',))
    else:
        confidence = confidence_names[quality.position_confidence]
        status = 'success' if quality.position_confidence is PositionConfidence.HIGH else 'warning'
        warnings.extend(quality.reasons)
        if (correspondence_count < 100 or metrics.fitness < config.min_fitness) and (not directed_partial_accepted):
            status = 'failed'
            confidence = '失败'
            warnings.append('共同表面过少或自动配准未找到可信解。')
        elif metrics.inlier_rmse_mm > config.max_inlier_rmse_mm:
            status = 'warning'
            confidence = '低'
            warnings.append('稳定对应点的配准误差偏高。')
    max_translation = diagonal * config.max_translation_diagonal_ratio
    if translation > max_translation:
        status = 'failed'
        confidence = '失败'
        warnings.append('求得的平移超出合理范围。')
    _notify(progress, 0.9, '配准完成，正在检查质量…')
    return RegistrationResult(transformation=transform, status=status, confidence=confidence, metrics=metrics, warnings=tuple(dict.fromkeys(warnings)), elapsed_seconds=time.perf_counter() - started, quality=quality)


def _priority_mask(
    value: np.ndarray | None,
    triangle_count: int,
) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool).reshape(-1)
    if len(mask) != triangle_count:
        raise ValueError("选区三角面掩码与模型不匹配。")
    return mask if np.any(mask) else None


def _face_subset_mesh(
    mesh: o3d.geometry.TriangleMesh,
    mask: np.ndarray | None,
) -> o3d.geometry.TriangleMesh:
    if mask is None:
        return clone_mesh(mesh)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    subset = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float).copy()),
        o3d.utility.Vector3iVector(triangles[selected].copy()),
    )
    subset.remove_unreferenced_vertices()
    subset.compute_triangle_normals()
    subset.compute_vertex_normals()
    return subset


def _selection_direction_metrics(
    query_mesh: o3d.geometry.TriangleMesh,
    query_mask: np.ndarray,
    reference_mesh: o3d.geometry.TriangleMesh,
    reference_mask: np.ndarray | None,
    transformation: np.ndarray,
    config: AlignmentConfig,
    seed: int,
) -> dict[str, float | int]:
    query = _face_subset_mesh(query_mesh, query_mask)
    reference = _face_subset_mesh(reference_mesh, reference_mask)
    count = max(1_000, int(config.selection_metric_sample_points))
    sampled = sample_registration_cloud(query, count, seed=seed)
    points = np.asarray(sampled.points, dtype=float)
    normals = np.asarray(sampled.normals, dtype=float)
    transform = np.asarray(transformation, dtype=float)
    transformed_points = points @ transform[:3, :3].T + transform[:3, 3]
    transformed_normals = normals @ transform[:3, :3].T
    scene = _distance_scene(reference)
    closest = scene.compute_closest_points(
        o3d.core.Tensor(transformed_points.astype(np.float32))
    )
    matches = closest["points"].numpy().astype(float)
    match_normals = closest["primitive_normals"].numpy().astype(float)
    distances = np.linalg.norm(transformed_points - matches, axis=1)
    normal_agreement = np.abs(
        np.einsum("ij,ij->i", transformed_normals, match_normals)
    )
    finite = np.isfinite(distances) & np.isfinite(normal_agreement)
    distances = distances[finite]
    normal_agreement = normal_agreement[finite]
    threshold = max(1e-6, float(config.coverage_distance_mm))
    similar = (distances <= threshold) & (
        normal_agreement >= math.cos(math.radians(45.0))
    )
    covariance = transformed_normals.T @ transformed_normals / max(len(transformed_normals), 1)
    eigenvalues = np.linalg.eigvalsh(covariance) if np.isfinite(covariance).all() else np.zeros(3)
    return {
        "sample_count": int(len(distances)),
        "coverage_ratio": float(np.mean(similar)) if len(similar) else 0.0,
        "median_mm": float(np.median(distances)) if len(distances) else float("inf"),
        "p90_mm": float(np.quantile(distances, 0.90)) if len(distances) else float("inf"),
        "rms_mm": (
            float(np.sqrt(np.mean(np.square(distances))))
            if len(distances)
            else float("inf")
        ),
        "normal_diversity": float(np.min(eigenvalues)) if len(eigenvalues) else 0.0,
    }


def _selection_candidate_metrics(
    target_mesh: o3d.geometry.TriangleMesh,
    source_mesh: o3d.geometry.TriangleMesh,
    target_mask: np.ndarray | None,
    source_mask: np.ndarray | None,
    transformation: np.ndarray,
    config: AlignmentConfig,
) -> dict[str, object]:
    directions: dict[str, dict[str, float | int]] = {}
    if source_mask is not None:
        directions["source_selection_to_target"] = _selection_direction_metrics(
            source_mesh,
            source_mask,
            target_mesh,
            target_mask,
            transformation,
            config,
            config.random_seed + 501,
        )
    if target_mask is not None:
        inverse = np.linalg.inv(np.asarray(transformation, dtype=float))
        directions["target_selection_to_source"] = _selection_direction_metrics(
            target_mesh,
            target_mask,
            source_mesh,
            source_mask,
            inverse,
            config,
            config.random_seed + 502,
        )
    values = list(directions.values())
    return {
        "directions": directions,
        "coverage_ratio": min(float(value["coverage_ratio"]) for value in values),
        "median_mm": max(float(value["median_mm"]) for value in values),
        "p90_mm": max(float(value["p90_mm"]) for value in values),
        "rms_mm": max(float(value["rms_mm"]) for value in values),
        "normal_diversity": min(float(value["normal_diversity"]) for value in values),
    }


def register_meshes(
    target_mesh: o3d.geometry.TriangleMesh,
    source_mesh: o3d.geometry.TriangleMesh,
    target_facts: MeshFacts,
    source_facts: MeshFacts,
    config: AlignmentConfig,
    progress: ProgressCallback | None = None,
    *,
    target_priority_faces: np.ndarray | None = None,
    source_priority_faces: np.ndarray | None = None,
) -> RegistrationResult:
    """Run the unchanged full-surface lane plus a soft selection-priority lane."""
    target_mask = _priority_mask(target_priority_faces, len(target_mesh.triangles))
    source_mask = _priority_mask(source_priority_faces, len(source_mesh.triangles))
    if target_mask is None and source_mask is None:
        return _register_meshes_once(
            target_mesh,
            source_mesh,
            target_facts,
            source_facts,
            config,
            progress,
        )

    def scaled_progress(offset: float, scale: float, lane: str):
        if progress is None:
            return None

        def callback(fraction: float, message: str) -> None:
            progress(
                offset + scale * max(0.0, min(1.0, float(fraction))),
                f"{lane}：{message}",
            )

        return callback

    baseline = _register_meshes_once(
        target_mesh,
        source_mesh,
        target_facts,
        source_facts,
        config,
        scaled_progress(0.0, 0.46, "全模型安全基线"),
    )
    priority = _register_meshes_once(
        target_mesh,
        source_mesh,
        target_facts,
        source_facts,
        config,
        scaled_progress(0.46, 0.50, "选区优先候选"),
        target_priority_faces=target_mask,
        source_priority_faces=source_mask,
    )
    baseline_roi = _selection_candidate_metrics(
        target_mesh,
        source_mesh,
        target_mask,
        source_mask,
        baseline.transformation,
        config,
    )
    priority_roi = _selection_candidate_metrics(
        target_mesh,
        source_mesh,
        target_mask,
        source_mask,
        priority.transformation,
        config,
    )

    selected_face_counts = {
        "target": int(np.count_nonzero(target_mask)) if target_mask is not None else 0,
        "source": int(np.count_nonzero(source_mask)) if source_mask is not None else 0,
    }
    counts_valid = all(
        count == 0 or count >= int(config.selection_min_faces)
        for count in selected_face_counts.values()
    )
    region_gate = (
        counts_valid
        and float(priority_roi["coverage_ratio"])
        >= float(config.selection_min_coverage_ratio)
        and float(priority_roi["normal_diversity"])
        >= float(config.selection_min_normal_diversity)
    )
    minimum_whole_overlap = max(
        0.02,
        float(baseline.metrics.overlap_ratio)
        * float(config.selection_whole_overlap_guard_ratio),
    )
    whole_guard = (
        np.isfinite(priority.transformation).all()
        and priority.metrics.translation_mm
        <= min(target_facts.diagonal_mm, source_facts.diagonal_mm)
        * config.max_translation_diagonal_ratio
        and priority.metrics.overlap_ratio >= minimum_whole_overlap
    )
    tolerance = float(config.selection_error_tolerance_ratio)
    not_worse = (
        float(priority_roi["coverage_ratio"])
        >= float(baseline_roi["coverage_ratio"]) - 0.02
        and float(priority_roi["median_mm"])
        <= float(baseline_roi["median_mm"]) * (1.0 + tolerance) + 1e-6
        and float(priority_roi["p90_mm"])
        <= float(baseline_roi["p90_mm"]) * (1.0 + tolerance) + 1e-6
    )
    use_priority = bool(region_gate and whole_guard and not_worse)
    reasons: list[str] = []
    if not counts_valid:
        reasons.append("选区面片数过少，已回退全模型基线。")
    if counts_valid and not region_gate:
        reasons.append("选区覆盖率或几何可观测性不足，已回退全模型基线。")
    if region_gate and not whole_guard:
        reasons.append("选区候选未通过全模型灾难性错位检查，已回退。")
    if region_gate and whole_guard and not not_worse:
        reasons.append("选区候选未优于全模型基线，已保留基线。")
    if use_priority:
        reasons.append("选区优先候选通过选区门控和全模型安全检查。")

    decision: dict[str, object] = {
        "enabled": True,
        "priority_fraction": float(config.selection_priority_fraction),
        "selected_face_counts": selected_face_counts,
        "baseline_metrics": baseline_roi,
        "priority_metrics": priority_roi,
        "region_gate_passed": bool(region_gate),
        "whole_model_guard_passed": bool(whole_guard),
        "priority_not_worse": bool(not_worse),
        "selected_lane": "selection_priority" if use_priority else "full_surface_baseline",
        "reasons": reasons,
    }
    chosen = priority if use_priority else baseline
    metrics = replace(chosen.metrics, selection_decision=decision)
    warnings = list(chosen.warnings)
    warnings.extend(reasons)
    status = chosen.status
    confidence = chosen.confidence
    if use_priority and chosen.status == "failed":
        # The full-surface quality report can reject legitimate partial overlap.
        # Passing the explicit ROI gate permits use only as a warning result.
        status = "warning"
        confidence = "中"
        warnings.append(
            "完整表面门控未通过，但操作者选区已通过独立门控；结果标记为警告。"
        )
    return replace(
        chosen,
        status=status,
        confidence=confidence,
        metrics=metrics,
        warnings=tuple(dict.fromkeys(warnings)),
        elapsed_seconds=baseline.elapsed_seconds + priority.elapsed_seconds,
    )
