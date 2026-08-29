from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import open3d as o3d

from ..mesh_io import sample_registration_cloud


class PositionConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"


@dataclass(frozen=True)
class CandidateDiagnostic:
    name: str
    transformation: np.ndarray
    fitness: float
    inlier_rmse_mm: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "transformation": self.transformation.tolist(),
            "fitness": self.fitness,
            "inlier_rmse_mm": self.inlier_rmse_mm,
        }


@dataclass(frozen=True)
class RegistrationQualityReport:
    position_confidence: PositionConfidence
    measurement_uncertainty_mm: float | None
    residual_median_mm: float | None
    residual_mad_mm: float | None
    residual_p90_mm: float | None
    residual_p95_mm: float | None
    low_error_ratio_015: float
    low_error_ratio_030: float
    source_coverage_ratio: float
    target_coverage_ratio: float
    spatial_coverage_ratio: float
    correspondence_concentration: float
    aabb_overlap_ratio: float
    normal_consistency_ratio: float
    candidate_consistency_ratio: float
    rotation_determinant: float
    rotation_orthogonality_error: float
    rotation_singular_values: tuple[float, float, float]
    reasons: tuple[str, ...]

    @property
    def directed_overlap_ratio(self) -> float:
        """Coverage of the better-supported surface direction."""
        return max(self.source_coverage_ratio, self.target_coverage_ratio)

    @property
    def overlap_direction(self) -> str:
        return (
            "source_to_target"
            if self.source_coverage_ratio >= self.target_coverage_ratio
            else "target_to_source"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "position_confidence": self.position_confidence.value,
            "measurement_uncertainty_mm": self.measurement_uncertainty_mm,
            "residual_median_mm": self.residual_median_mm,
            "residual_mad_mm": self.residual_mad_mm,
            "residual_p90_mm": self.residual_p90_mm,
            "residual_p95_mm": self.residual_p95_mm,
            "low_error_ratio_015": self.low_error_ratio_015,
            "low_error_ratio_030": self.low_error_ratio_030,
            "source_coverage_ratio": self.source_coverage_ratio,
            "target_coverage_ratio": self.target_coverage_ratio,
            "directed_overlap_ratio": self.directed_overlap_ratio,
            "overlap_direction": self.overlap_direction,
            "spatial_coverage_ratio": self.spatial_coverage_ratio,
            "correspondence_concentration": self.correspondence_concentration,
            "aabb_overlap_ratio": self.aabb_overlap_ratio,
            "normal_consistency_ratio": self.normal_consistency_ratio,
            "candidate_consistency_ratio": self.candidate_consistency_ratio,
            "rotation_determinant": self.rotation_determinant,
            "rotation_orthogonality_error": self.rotation_orthogonality_error,
            "rotation_singular_values": self.rotation_singular_values,
            "reasons": list(self.reasons),
            "interpretation": (
                "Geometric registration confidence and estimated local measurement uncertainty; "
                "not a deterministic clinical diagnosis or accuracy guarantee."
            ),
        }


def _tensor_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _sample(
    mesh: o3d.geometry.TriangleMesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sampled = sample_registration_cloud(mesh, count, seed=seed)
    points = np.asarray(sampled.points, dtype=np.float32)
    normals = np.asarray(sampled.normals, dtype=float)
    return points, normals


def _query(points: np.ndarray, normals: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closest = _tensor_scene(mesh).compute_closest_points(o3d.core.Tensor(points))
    target_points = closest["points"].numpy().astype(float)
    target_normals = closest["primitive_normals"].numpy().astype(float)
    distances = np.linalg.norm(points.astype(float) - target_points, axis=1)
    normal_agreement = np.abs(np.einsum("ij,ij->i", normals, target_normals))
    return distances, target_points, normal_agreement


def _spatial_metrics(points: np.ndarray, matched: np.ndarray, distances: np.ndarray, threshold: float) -> tuple[float, float]:
    inliers = distances <= threshold
    if np.sum(inliers) < 3:
        return 0.0, 1.0
    bounds_min = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - bounds_min, 1e-9)
    cells = np.floor((points[inliers] - bounds_min) / extent * 3.0).astype(int)
    cells = np.clip(cells, 0, 2)
    occupied = len(np.unique(cells, axis=0))
    coverage = occupied / 27.0
    center = points.mean(axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    spread = np.percentile(radius, 90) if len(radius) else 0.0
    inlier_spread = np.percentile(np.linalg.norm(matched[inliers] - center, axis=1), 90)
    concentration = 1.0 - min(1.0, inlier_spread / max(spread, 1e-9))
    return coverage, concentration


def _candidate_consistency(candidates: tuple[CandidateDiagnostic, ...], diagonal_mm: float) -> float:
    usable = [
        candidate
        for candidate in candidates
        if candidate.fitness >= 0.15
        and np.isfinite(candidate.inlier_rmse_mm)
    ]
    if not usable:
        return 0.0
    best = max(usable, key=lambda value: (value.fitness, -value.inlier_rmse_mm))
    # A clearly inferior fallback is not evidence of positional ambiguity.
    usable = [
        candidate
        for candidate in usable
        if candidate.fitness >= max(0.15, best.fitness - 0.15)
    ]
    if len(usable) == 1:
        return 1.0
    consistent = 0
    for candidate in usable:
        relative = candidate.transformation @ np.linalg.inv(best.transformation)
        rotation = relative[:3, :3]
        cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.rad2deg(np.arccos(cosine))
        translation = np.linalg.norm(relative[:3, 3])
        if angle <= 5.0 and translation <= max(0.5, diagonal_mm * 0.02):
            consistent += 1
    return consistent / len(usable)


def assess_registration_quality(
    target_mesh: o3d.geometry.TriangleMesh,
    aligned_source_mesh: o3d.geometry.TriangleMesh,
    transformation: np.ndarray,
    aabb_overlap_ratio: float,
    candidates: tuple[CandidateDiagnostic, ...] = (),
    sample_count: int = 8_000,
    coverage_distance_mm: float = 0.60,
    min_directed_overlap: float = 0.20,
) -> RegistrationQualityReport:
    transform = np.asarray(transformation, dtype=float)
    reasons: list[str] = []
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation)) if transform.shape == (4, 4) else float("nan")
    orthogonality = (
        float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
        if transform.shape == (4, 4)
        else float("inf")
    )
    singular_values = (
        tuple(float(value) for value in np.linalg.svd(rotation, compute_uv=False))
        if transform.shape == (4, 4) and np.isfinite(rotation).all()
        else (float("nan"),) * 3
    )
    rigid_valid = (
        transform.shape == (4, 4)
        and np.isfinite(transform).all()
        and determinant > 0
        and abs(determinant - 1.0) <= 0.02
        and orthogonality <= 0.03
        and max(abs(value - 1.0) for value in singular_values) <= 0.02
    )
    if not rigid_valid:
        reasons.append("变换包含镜像、缩放、非正交旋转或非有限数值。")

    count = max(500, sample_count)
    source_points, source_normals = _sample(aligned_source_mesh, count, 20260819)
    target_points, target_normals = _sample(target_mesh, count, 20260820)
    source_distances, source_matches, source_normal_agreement = _query(
        source_points, source_normals, target_mesh
    )
    target_distances, target_matches, target_normal_agreement = _query(
        target_points, target_normals, aligned_source_mesh
    )
    threshold = max(1e-6, float(coverage_distance_mm))
    normal_threshold = np.cos(np.deg2rad(45.0))
    source_similar = (
        (source_distances <= threshold)
        & (source_normal_agreement >= normal_threshold)
    )
    target_similar = (
        (target_distances <= threshold)
        & (target_normal_agreement >= normal_threshold)
    )
    source_coverage = float(np.mean(source_similar))
    target_coverage = float(np.mean(target_similar))
    if source_coverage >= target_coverage:
        dominant_distances = source_distances
        dominant_similar = source_similar
        dominant_normal_agreement = source_normal_agreement
    else:
        dominant_distances = target_distances
        dominant_similar = target_similar
        dominant_normal_agreement = target_normal_agreement
    finite_mask = np.isfinite(dominant_distances)
    finite = dominant_distances[finite_mask]
    finite_normal_agreement = dominant_normal_agreement[finite_mask]
    matched = dominant_distances[finite_mask & dominant_similar]
    if len(matched):
        median = float(np.median(matched))
        mad = float(np.median(np.abs(matched - median)))
        p90 = float(np.quantile(matched, 0.90))
        p95 = float(np.quantile(matched, 0.95))
        low015 = float(
            np.mean(
                (finite <= 0.15)
                & (finite_normal_agreement >= normal_threshold)
            )
        )
        low030 = float(
            np.mean(
                (finite <= 0.30)
                & (finite_normal_agreement >= normal_threshold)
            )
        )
        robust_scale = 1.4826 * mad
        stable_residuals = matched[
            matched <= max(threshold, median + 4.0 * robust_scale)
        ]
        stable_p90 = float(np.quantile(stable_residuals, 0.90)) if len(stable_residuals) else p90
        uncertainty = float(
            max(
                median + robust_scale,
                min(stable_p90, 1.75 * max(median, robust_scale)),
            )
        )
    else:
        median = mad = p90 = p95 = None
        low015 = low030 = 0.0
        uncertainty = None

    source_spatial, source_concentration = _spatial_metrics(
        source_points, source_matches, source_distances, threshold
    )
    target_spatial, target_concentration = _spatial_metrics(
        target_points, target_matches, target_distances, threshold
    )
    spatial_coverage = min(1.0, max(source_spatial, target_spatial) * 1.8)
    concentration = max(source_concentration, target_concentration)
    normal_values = np.concatenate(
        (
            source_normal_agreement[source_distances <= threshold],
            target_normal_agreement[target_distances <= threshold],
        )
    )
    normal_consistency = float(np.mean(normal_values >= np.cos(np.deg2rad(45)))) if len(normal_values) else 0.0
    diagonal = min(
        np.linalg.norm(target_mesh.get_axis_aligned_bounding_box().get_extent()),
        np.linalg.norm(aligned_source_mesh.get_axis_aligned_bounding_box().get_extent()),
    )
    candidate_consistency = _candidate_consistency(candidates, float(diagonal))

    hard_failure = (
        not rigid_valid
        or len(matched) < 100
        or max(source_coverage, target_coverage) < min_directed_overlap
    )
    if hard_failure:
        confidence = PositionConfidence.FAILED
        reasons.append("有效共同表面或刚性变换完整性不足。")
    else:
        score = 0
        score += 2 if median is not None and median <= 0.15 else 1 if median is not None and median <= 0.30 else 0
        score += 2 if low030 >= 0.50 else 1 if low030 >= 0.25 else 0
        score += 2 if max(source_coverage, target_coverage) >= max(0.50, min_directed_overlap) else 1 if max(source_coverage, target_coverage) >= min_directed_overlap else 0
        score += 1 if aabb_overlap_ratio >= 0.50 else 0
        score += 1 if normal_consistency >= 0.70 else 0
        score += 1 if spatial_coverage >= 0.35 and concentration <= 0.60 else 0
        score += 1 if candidate_consistency >= 0.50 else 0
        if score >= 8:
            confidence = PositionConfidence.HIGH
        elif score >= 5:
            confidence = PositionConfidence.MEDIUM
        else:
            confidence = PositionConfidence.LOW
        if concentration > 0.75:
            confidence = PositionConfidence.LOW
            reasons.append("低误差对应集中在较小区域。")
        if candidate_consistency <= 0.50 and len(candidates) >= 2:
            confidence = PositionConfidence.LOW
            reasons.append("多个初始化候选未收敛到近似变换。")
        if low030 < 0.25:
            reasons.append("低误差表面比例较少。")
        if min(source_coverage, target_coverage) < 0.10:
            reasons.append("双向表面覆盖不对称，局部测量需谨慎。")

    return RegistrationQualityReport(
        position_confidence=confidence,
        measurement_uncertainty_mm=uncertainty,
        residual_median_mm=median,
        residual_mad_mm=mad,
        residual_p90_mm=p90,
        residual_p95_mm=p95,
        low_error_ratio_015=low015,
        low_error_ratio_030=low030,
        source_coverage_ratio=source_coverage,
        target_coverage_ratio=target_coverage,
        spatial_coverage_ratio=spatial_coverage,
        correspondence_concentration=concentration,
        aabb_overlap_ratio=float(aabb_overlap_ratio),
        normal_consistency_ratio=normal_consistency,
        candidate_consistency_ratio=candidate_consistency,
        rotation_determinant=determinant,
        rotation_orthogonality_error=orthogonality,
        rotation_singular_values=singular_values,
        reasons=tuple(dict.fromkeys(reasons)),
    )
