from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlignmentConfig:
    """Algorithm parameters expressed in millimetres."""

    global_sample_points: int = 30_000
    metric_sample_points: int = 30_000
    base_voxel_mm: float = 0.45
    voxel_fractions: tuple[float, ...] = (1.0, 0.55, 0.28)
    correspondence_multipliers: tuple[float, ...] = (2.5, 1.8, 1.5)
    icp_iterations: tuple[int, ...] = (60, 45, 30)
    high_precision_refinement_enabled: bool = True
    high_precision_distance_stages_mm: tuple[float, ...] = (0.10, 0.05, 0.02)
    high_precision_iterations: int = 30
    high_precision_max_vertices: int = 400_000
    high_precision_normal_angle_degrees: float = 60.0
    high_precision_robust_floor_mm: float = 0.005
    high_precision_gate_enabled: bool = True
    high_precision_gate_stable_distance_mm: float = 0.05
    high_precision_gate_roi_distance_mm: float = 0.60
    high_precision_gate_whole_coverage_ratio: float = 0.70
    high_precision_gate_local_coverage_ratio: float = 0.75
    high_precision_gate_roi_coverage_ratio: float = 0.70
    high_precision_gate_spatial_coverage_ratio: float = 0.60
    high_precision_gate_min_relative_improvement: float = 0.01
    high_precision_gate_p90_tolerance_mm: float = 0.0005
    high_precision_gate_max_local_displacement_mm: float = 0.10
    high_precision_gate_max_extrapolated_displacement_mm: float = 0.10
    high_precision_gate_max_condition_number: float = 1_000_000.0
    high_precision_gate_min_normal_diversity: float = 0.005
    normal_radius_multiplier: float = 2.5
    feature_radius_multiplier: float = 5.0
    ransac_distance_multiplier: float = 1.6
    ransac_max_iterations: int = 80_000
    ransac_confidence: float = 0.999
    global_registration_restarts: int = 8
    final_candidate_count: int = 5
    exhaustive_orientation_search: bool = False
    exhaustive_orientation_angle_step_degrees: float = 30.0
    exhaustive_orientation_eigen_tolerance_ratio: float = 0.08
    exhaustive_orientation_max_candidates: int = 96
    exhaustive_orientation_finalist_count: int = 12
    robust_kernel_scale_mm: float = 0.35
    partial_registration_enabled: bool = True
    partial_area_ratio_threshold: float = 2.0
    partial_overlap_threshold: float = 0.20
    max_global_sample_multiplier: float = 6.0
    coverage_distance_mm: float = 0.60
    selection_priority_fraction: float = 0.70
    selection_metric_sample_points: int = 8_000
    selection_min_faces: int = 20
    selection_min_coverage_ratio: float = 0.60
    selection_min_normal_diversity: float = 0.001
    selection_whole_overlap_guard_ratio: float = 0.25
    selection_error_tolerance_ratio: float = 0.03
    min_fitness: float = 0.18
    max_inlier_rmse_mm: float = 0.80
    max_translation_diagonal_ratio: float = 3.0
    default_color_max_mm: float = 1.0
    random_seed: int = 20260807
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def effective_voxel(self, diagonal_mm: float) -> float:
        """Keep the default dental scale while tolerating small local scans."""
        if diagonal_mm <= 0:
            return self.base_voxel_mm
        return min(self.base_voxel_mm, max(0.20, diagonal_mm / 120.0))
