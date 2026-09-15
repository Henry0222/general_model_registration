# Integration API v1: generated signatures and fields

Generated from application 3.0.0 source. GUI imports do not initialize an application.

## auto_alignment.integration.core

### AlignmentConfig

```python
AlignmentConfig(algorithm_version: 'str' = '2.0', candidate_evaluation_points: 'int' = 8000, candidate_pool_finalists: 'int' = 6, candidate_feature_scales: 'tuple[float, ...]' = (1.0, 3.0), candidate_common_fractions: 'tuple[float, ...]' = (0.35, 0.65), candidate_common_iterations: 'int' = 16, candidate_primitive_enabled: 'bool' = True, candidate_gicp_enabled: 'bool' = True, global_sample_points: 'int' = 30000, metric_sample_points: 'int' = 30000, base_voxel_mm: 'float' = 0.45, voxel_fractions: 'tuple[float, ...]' = (1.0, 0.55, 0.28), correspondence_multipliers: 'tuple[float, ...]' = (2.5, 1.8, 1.5), icp_iterations: 'tuple[int, ...]' = (60, 45, 30), high_precision_refinement_enabled: 'bool' = True, high_precision_distance_stages_mm: 'tuple[float, ...]' = (0.1, 0.05, 0.02), high_precision_iterations: 'int' = 30, high_precision_max_vertices: 'int' = 400000, high_precision_normal_angle_degrees: 'float' = 60.0, high_precision_robust_floor_mm: 'float' = 0.005, high_precision_gate_enabled: 'bool' = True, high_precision_gate_stable_distance_mm: 'float' = 0.05, high_precision_gate_roi_distance_mm: 'float' = 0.6, high_precision_gate_whole_coverage_ratio: 'float' = 0.7, high_precision_gate_local_coverage_ratio: 'float' = 0.75, high_precision_gate_roi_coverage_ratio: 'float' = 0.7, high_precision_gate_spatial_coverage_ratio: 'float' = 0.6, high_precision_gate_min_relative_improvement: 'float' = 0.01, high_precision_gate_p90_tolerance_mm: 'float' = 0.0005, high_precision_gate_max_local_displacement_mm: 'float' = 0.1, high_precision_gate_max_extrapolated_displacement_mm: 'float' = 0.1, high_precision_gate_max_condition_number: 'float' = 1000000.0, high_precision_gate_min_normal_diversity: 'float' = 0.005, stable_region_enabled: 'bool' = True, stable_region_seed_distance_mm: 'float' = 0.6, stable_region_sigma_floor_mm: 'float' = 0.004, stable_region_k_sigma: 'float' = 3.0, stable_region_smoothing_rounds: 'int' = 3, stable_region_min_component_area_fraction: 'float' = 0.005, stable_region_min_area_fraction: 'float' = 0.15, stable_region_holdout_tolerance_sigma: 'float' = 0.25, stable_region_max_displacement_sigma: 'float' = 4.0, stable_region_min_consensus_retention: 'float' = 0.85, stable_region_bias_rounds: 'int' = 2, stable_region_bias_k_sigma: 'float' = 4.0, stable_region_passes: 'int' = 2, normal_radius_multiplier: 'float' = 2.5, feature_radius_multiplier: 'float' = 5.0, ransac_distance_multiplier: 'float' = 1.6, ransac_max_iterations: 'int' = 80000, ransac_confidence: 'float' = 0.999, global_registration_restarts: 'int' = 8, final_candidate_count: 'int' = 5, exhaustive_orientation_search: 'bool' = False, exhaustive_orientation_angle_step_degrees: 'float' = 30.0, exhaustive_orientation_eigen_tolerance_ratio: 'float' = 0.08, exhaustive_orientation_max_candidates: 'int' = 96, exhaustive_orientation_finalist_count: 'int' = 12, robust_kernel_scale_mm: 'float' = 0.35, partial_registration_enabled: 'bool' = True, partial_area_ratio_threshold: 'float' = 2.0, partial_overlap_threshold: 'float' = 0.2, max_global_sample_multiplier: 'float' = 6.0, coverage_distance_mm: 'float' = 0.6, selection_priority_fraction: 'float' = 0.7, selection_metric_sample_points: 'int' = 8000, selection_min_faces: 'int' = 20, selection_min_coverage_ratio: 'float' = 0.6, selection_min_normal_diversity: 'float' = 0.001, selection_whole_overlap_guard_ratio: 'float' = 0.25, selection_coverage_tolerance_ratio: 'float' = 0.02, selection_error_tolerance_ratio: 'float' = 0.03, min_fitness: 'float' = 0.18, max_inlier_rmse_mm: 'float' = 0.8, max_translation_diagonal_ratio: 'float' = 3.0, default_color_max_mm: 'float' = 1.0, random_seed: 'int' = 20260807, warnings: 'tuple[str, ...]' = <factory>, refinement_mode: 'str' = 'baseline') -> None
```

| Field | Type | Default |
|---|---|---|
| algorithm_version | str | '2.0' |
| candidate_evaluation_points | int | 8000 |
| candidate_pool_finalists | int | 6 |
| candidate_feature_scales | tuple[float, ...] | (1.0, 3.0) |
| candidate_common_fractions | tuple[float, ...] | (0.35, 0.65) |
| candidate_common_iterations | int | 16 |
| candidate_primitive_enabled | bool | True |
| candidate_gicp_enabled | bool | True |
| global_sample_points | int | 30000 |
| metric_sample_points | int | 30000 |
| base_voxel_mm | float | 0.45 |
| voxel_fractions | tuple[float, ...] | (1.0, 0.55, 0.28) |
| correspondence_multipliers | tuple[float, ...] | (2.5, 1.8, 1.5) |
| icp_iterations | tuple[int, ...] | (60, 45, 30) |
| high_precision_refinement_enabled | bool | True |
| high_precision_distance_stages_mm | tuple[float, ...] | (0.1, 0.05, 0.02) |
| high_precision_iterations | int | 30 |
| high_precision_max_vertices | int | 400000 |
| high_precision_normal_angle_degrees | float | 60.0 |
| high_precision_robust_floor_mm | float | 0.005 |
| high_precision_gate_enabled | bool | True |
| high_precision_gate_stable_distance_mm | float | 0.05 |
| high_precision_gate_roi_distance_mm | float | 0.6 |
| high_precision_gate_whole_coverage_ratio | float | 0.7 |
| high_precision_gate_local_coverage_ratio | float | 0.75 |
| high_precision_gate_roi_coverage_ratio | float | 0.7 |
| high_precision_gate_spatial_coverage_ratio | float | 0.6 |
| high_precision_gate_min_relative_improvement | float | 0.01 |
| high_precision_gate_p90_tolerance_mm | float | 0.0005 |
| high_precision_gate_max_local_displacement_mm | float | 0.1 |
| high_precision_gate_max_extrapolated_displacement_mm | float | 0.1 |
| high_precision_gate_max_condition_number | float | 1000000.0 |
| high_precision_gate_min_normal_diversity | float | 0.005 |
| stable_region_enabled | bool | True |
| stable_region_seed_distance_mm | float | 0.6 |
| stable_region_sigma_floor_mm | float | 0.004 |
| stable_region_k_sigma | float | 3.0 |
| stable_region_smoothing_rounds | int | 3 |
| stable_region_min_component_area_fraction | float | 0.005 |
| stable_region_min_area_fraction | float | 0.15 |
| stable_region_holdout_tolerance_sigma | float | 0.25 |
| stable_region_max_displacement_sigma | float | 4.0 |
| stable_region_min_consensus_retention | float | 0.85 |
| stable_region_bias_rounds | int | 2 |
| stable_region_bias_k_sigma | float | 4.0 |
| stable_region_passes | int | 2 |
| normal_radius_multiplier | float | 2.5 |
| feature_radius_multiplier | float | 5.0 |
| ransac_distance_multiplier | float | 1.6 |
| ransac_max_iterations | int | 80000 |
| ransac_confidence | float | 0.999 |
| global_registration_restarts | int | 8 |
| final_candidate_count | int | 5 |
| exhaustive_orientation_search | bool | False |
| exhaustive_orientation_angle_step_degrees | float | 30.0 |
| exhaustive_orientation_eigen_tolerance_ratio | float | 0.08 |
| exhaustive_orientation_max_candidates | int | 96 |
| exhaustive_orientation_finalist_count | int | 12 |
| robust_kernel_scale_mm | float | 0.35 |
| partial_registration_enabled | bool | True |
| partial_area_ratio_threshold | float | 2.0 |
| partial_overlap_threshold | float | 0.2 |
| max_global_sample_multiplier | float | 6.0 |
| coverage_distance_mm | float | 0.6 |
| selection_priority_fraction | float | 0.7 |
| selection_metric_sample_points | int | 8000 |
| selection_min_faces | int | 20 |
| selection_min_coverage_ratio | float | 0.6 |
| selection_min_normal_diversity | float | 0.001 |
| selection_whole_overlap_guard_ratio | float | 0.25 |
| selection_coverage_tolerance_ratio | float | 0.02 |
| selection_error_tolerance_ratio | float | 0.03 |
| min_fitness | float | 0.18 |
| max_inlier_rmse_mm | float | 0.8 |
| max_translation_diagonal_ratio | float | 3.0 |
| default_color_max_mm | float | 1.0 |
| random_seed | int | 20260807 |
| warnings | tuple[str, ...] | factory: <class 'tuple'> |
| refinement_mode | str | 'baseline' |

`effective_voxel(self, diagonal_mm: 'float') -> 'float'`

### MeshValidationError

### MeshFacts

```python
MeshFacts(path: 'str', vertices: 'int', triangles: 'int', diagonal_mm: 'float', bounds_min: 'tuple[float, float, float]', bounds_max: 'tuple[float, float, float]', warnings: 'tuple[str, ...]', normals_flipped: 'bool' = False) -> None
```

| Field | Type | Default |
|---|---|---|
| path | str | required |
| vertices | int | required |
| triangles | int | required |
| diagonal_mm | float | required |
| bounds_min | tuple[float, float, float] | required |
| bounds_max | tuple[float, float, float] | required |
| warnings | tuple[str, ...] | required |
| normals_flipped | bool | False |

`as_dict(self) -> 'dict[str, object]'`

### load_mesh

```python
load_mesh(path: 'str | Path', *, flip_normals: 'bool' = False) -> 'tuple[o3d.geometry.TriangleMesh, MeshFacts]'
```

### RegistrationMetrics

```python
RegistrationMetrics(fitness: 'float', inlier_rmse_mm: 'float', correspondence_count: 'int', overlap_ratio: 'float', rotation_degrees: 'float', translation_mm: 'float', candidate_diagnostics: 'tuple[CandidateDiagnostic, ...]' = (), high_precision_decision: 'dict[str, object] | None' = None, selection_decision: 'dict[str, object] | None' = None, candidate_selection: 'dict[str, object] | None' = None, refinement: 'dict[str, object] | None' = None) -> None
```

| Field | Type | Default |
|---|---|---|
| fitness | float | required |
| inlier_rmse_mm | float | required |
| correspondence_count | int | required |
| overlap_ratio | float | required |
| rotation_degrees | float | required |
| translation_mm | float | required |
| candidate_diagnostics | tuple[CandidateDiagnostic, ...] | () |
| high_precision_decision | dict[str, object] | None | None |
| selection_decision | dict[str, object] | None | None |
| candidate_selection | dict[str, object] | None | None |
| refinement | dict[str, object] | None | None |

`as_dict(self) -> 'dict[str, object]'`

### RegistrationResult

```python
RegistrationResult(transformation: 'np.ndarray', status: 'str', confidence: 'str', metrics: 'RegistrationMetrics', warnings: 'tuple[str, ...]', elapsed_seconds: 'float', quality: 'RegistrationQualityReport | None' = None, alternatives: 'tuple[tuple[str, RegistrationResult], ...]' = ()) -> None
```

| Field | Type | Default |
|---|---|---|
| transformation | np.ndarray | required |
| status | str | required |
| confidence | str | required |
| metrics | RegistrationMetrics | required |
| warnings | tuple[str, ...] | required |
| elapsed_seconds | float | required |
| quality | RegistrationQualityReport | None | None |
| alternatives | tuple[tuple[str, RegistrationResult], ...] | () |

### RegistrationQualityReport

```python
RegistrationQualityReport(position_confidence: 'PositionConfidence', measurement_uncertainty_mm: 'float | None', residual_median_mm: 'float | None', residual_mad_mm: 'float | None', residual_p90_mm: 'float | None', residual_p95_mm: 'float | None', low_error_ratio_015: 'float', low_error_ratio_030: 'float', source_coverage_ratio: 'float', target_coverage_ratio: 'float', spatial_coverage_ratio: 'float', correspondence_concentration: 'float', aabb_overlap_ratio: 'float', normal_consistency_ratio: 'float', candidate_consistency_ratio: 'float', rotation_determinant: 'float', rotation_orthogonality_error: 'float', rotation_singular_values: 'tuple[float, float, float]', reasons: 'tuple[str, ...]') -> None
```

| Field | Type | Default |
|---|---|---|
| position_confidence | PositionConfidence | required |
| measurement_uncertainty_mm | float | None | required |
| residual_median_mm | float | None | required |
| residual_mad_mm | float | None | required |
| residual_p90_mm | float | None | required |
| residual_p95_mm | float | None | required |
| low_error_ratio_015 | float | required |
| low_error_ratio_030 | float | required |
| source_coverage_ratio | float | required |
| target_coverage_ratio | float | required |
| spatial_coverage_ratio | float | required |
| correspondence_concentration | float | required |
| aabb_overlap_ratio | float | required |
| normal_consistency_ratio | float | required |
| candidate_consistency_ratio | float | required |
| rotation_determinant | float | required |
| rotation_orthogonality_error | float | required |
| rotation_singular_values | tuple[float, float, float] | required |
| reasons | tuple[str, ...] | required |

`as_dict(self) -> 'dict[str, object]'`

### PositionConfidence

```python
PositionConfidence(*values)
```

### RegistrationCancelled

### register_meshes

```python
register_meshes(target_mesh, source_mesh, target_facts: 'MeshFacts', source_facts: 'MeshFacts', config: 'AlignmentConfig', progress: 'Callable[[float, str], None] | None' = None, *, target_priority_faces: 'np.ndarray | None' = None, source_priority_faces: 'np.ndarray | None' = None, cancel: 'Callable[[], bool] | None' = None) -> 'RegistrationResult'
```

## auto_alignment.integration.review

### GeneralResultViewer

```python
GeneralResultViewer(data: 'ViewerData') -> 'None'
```

### ViewerData

```python
ViewerData(results_path: 'Path', target: 'o3d.geometry.TriangleMesh', aligned: 'o3d.geometry.TriangleMesh', signed_distances_mm: 'np.ndarray', deviation_scale: 'DeviationScale', direction_reversed: 'bool', registration_status: 'str | None' = None, registration_confidence: 'str | None' = None, registration_warnings: 'tuple[str, ...]' = (), review_only: 'bool' = False, target_path: 'Path | None' = None, aligned_path: 'Path | None' = None, annotation_file: 'Path | None' = None) -> None
```

| Field | Type | Default |
|---|---|---|
| results_path | Path | required |
| target | o3d.geometry.TriangleMesh | required |
| aligned | o3d.geometry.TriangleMesh | required |
| signed_distances_mm | np.ndarray | required |
| deviation_scale | DeviationScale | required |
| direction_reversed | bool | required |
| registration_status | str | None | None |
| registration_confidence | str | None | None |
| registration_warnings | tuple[str, ...] | () |
| review_only | bool | False |
| target_path | Path | None | None |
| aligned_path | Path | None | None |
| annotation_file | Path | None | None |

### configure_open3d_font

```python
configure_open3d_font(app: 'gui.Application', extra_text: 'str' = '') -> 'None'
```

### load_pair_viewer_data

```python
load_pair_viewer_data(target_path: 'str | Path', aligned_path: 'str | Path', *, minimum_nominal_mm: 'float' = -0.05, maximum_nominal_mm: 'float' = 0.05) -> 'ViewerData'
```

### load_viewer_data

```python
load_viewer_data(results_path: 'str | Path', *, target_override: 'str | Path | None' = None, aligned_override: 'str | Path | None' = None) -> 'ViewerData'
```

### run_general_result_viewer

```python
run_general_result_viewer(results_path: 'str | Path', *, target_override: 'str | Path | None' = None, aligned_override: 'str | Path | None' = None) -> 'None'
```

### run_pair_result_viewer

```python
run_pair_result_viewer(target_path: 'str | Path', aligned_path: 'str | Path') -> 'None'
```

### VIEWER_MANIFEST_SCHEMA_VERSION

```python
VIEWER_MANIFEST_SCHEMA_VERSION = 1
```

### RegistrationReviewSpec

```python
RegistrationReviewSpec(target_path: 'str | Path', aligned_path: 'str | Path', status: 'str' = 'unknown', position_confidence: 'str | None' = None, confidence_display: 'str | None' = None, warnings: 'tuple[str, ...]' = (), review_only: 'bool' = False, direction_reversed: 'bool' = False, minimum_nominal_mm: 'float' = -0.05, maximum_nominal_mm: 'float' = 0.05, target_sha256: 'str | None' = None, aligned_sha256: 'str | None' = None, annotations_path: 'str | Path | None' = None, manifest_path: 'str | Path | None' = None) -> None
```

| Field | Type | Default |
|---|---|---|
| target_path | str | Path | required |
| aligned_path | str | Path | required |
| status | str | 'unknown' |
| position_confidence | str | None | None |
| confidence_display | str | None | None |
| warnings | tuple[str, ...] | () |
| review_only | bool | False |
| direction_reversed | bool | False |
| minimum_nominal_mm | float | -0.05 |
| maximum_nominal_mm | float | 0.05 |
| target_sha256 | str | None | None |
| aligned_sha256 | str | None | None |
| annotations_path | str | Path | None | None |
| manifest_path | str | Path | None | None |

### build_review_manifest

```python
build_review_manifest(spec: 'RegistrationReviewSpec') -> 'dict'
```

### validate_review_manifest

```python
validate_review_manifest(payload: 'dict') -> 'None'
```

### write_review_manifest

```python
write_review_manifest(path: 'str | Path', spec: 'RegistrationReviewSpec') -> 'Path'
```

### load_review_manifest

```python
load_review_manifest(path: 'str | Path', *, target_override=None, aligned_override=None) -> 'RegistrationReviewSpec'
```

### run_registration_review

```python
run_registration_review(spec_or_path: 'RegistrationReviewSpec | str | Path') -> 'None'
```

## auto_alignment.integration.selection

### SelectionRegionSpec

```python
SelectionRegionSpec(key: 'str', label: 'str', color: 'tuple[float, float, float]') -> None
```

| Field | Type | Default |
|---|---|---|
| key | str | required |
| label | str | required |
| color | tuple[float, float, float] | required |

### RegionSelectionSnapshot

```python
RegionSelectionSnapshot(mesh_path: 'str', mesh_sha256: 'str', triangle_count: 'int', active_region: 'str', masks: 'dict[str, np.ndarray]') -> None
```

| Field | Type | Default |
|---|---|---|
| mesh_path | str | required |
| mesh_sha256 | str | required |
| triangle_count | int | required |
| active_region | str | required |
| masks | dict[str, np.ndarray] | required |

### RegionSelectionSession

```python
RegionSelectionSession(mesh_path: 'str | Path', triangle_count: 'int', regions: 'Sequence[SelectionRegionSpec]', *, initial_masks: 'Mapping[str, np.ndarray] | None' = None, active_region: 'str | None' = None, allow_overlap: 'bool' = False, on_change: 'Callable[[RegionSelectionSnapshot], None] | None' = None, on_save: 'Callable[[RegionSelectionSnapshot], None] | None' = None, mesh_sha256: 'str | None' = None)
```

`load(self, path: 'str | Path') -> 'None'`

`redo(self) -> 'None'`

`save(self, path: 'str | Path') -> 'Path'`

`select_region(self, region: 'str') -> 'None'`

`set_mask(self, region: 'str', mask: 'np.ndarray') -> 'None'`

`snapshot(self) -> 'RegionSelectionSnapshot'`

`undo(self) -> 'None'`

### MultiRegionSelectionViewer

```python
MultiRegionSelectionViewer(mesh_path, state_path, regions, *, initial_masks=None, active_region=None, allow_overlap=False, on_change=None, on_save=None, summary_provider=None, preloaded_mesh=None, preloaded_mesh_sha256=None)
```

`close(self)`

`get_snapshot(self)`

`redo(self)`

`reset_view(self)`

`save(self)`

`set_active_region(self, region)`

`set_region_mask(self, region, mask)`

`undo(self)`

### run_multi_region_selection_viewer

```python
run_multi_region_selection_viewer(mesh_path, state_path, regions, *, initial_masks=None, active_region=None, allow_overlap=False, on_change=None, on_save=None, summary_provider=None, preloaded_mesh=None, preloaded_mesh_sha256=None)
```

### ModelEditState

```python
ModelEditState(mesh_path: 'str', mesh_sha256: 'str', triangle_count: 'int', selected: 'np.ndarray', deleted: 'np.ndarray', updated_at: 'str') -> None
```

| Field | Type | Default |
|---|---|---|
| mesh_path | str | required |
| mesh_sha256 | str | required |
| triangle_count | int | required |
| selected | np.ndarray | required |
| deleted | np.ndarray | required |
| updated_at | str | required |

`as_dict(self) -> 'dict[str, object]'`

`normalized(self) -> "'ModelEditState'"`

### clone_state_with_masks

```python
clone_state_with_masks(state: 'ModelEditState', selected: 'np.ndarray', deleted: 'np.ndarray') -> 'ModelEditState'
```

### encode_face_ranges

```python
encode_face_ranges(indices) -> 'list[list[int]]'
```

### decode_face_ranges

```python
decode_face_ranges(ranges, triangle_count: 'int') -> 'np.ndarray'
```

### bounded_component_faces

```python
bounded_component_faces(mesh: 'o3d.geometry.TriangleMesh', selected: 'np.ndarray', deleted: 'np.ndarray') -> 'np.ndarray'
```

### REGION_SELECTION_SCHEMA_VERSION

```python
REGION_SELECTION_SCHEMA_VERSION = 1
```
