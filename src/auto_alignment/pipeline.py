from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from auto_alignment.comparison import ComparisonResult, compare_meshes
from auto_alignment.config import AlignmentConfig
from auto_alignment.exporters import export_results
from auto_alignment.mesh_io import MeshFacts, load_mesh
from auto_alignment.mesh_selection import apply_edit_state, load_edit_state, updated_mesh_facts
from auto_alignment.registration import RegistrationResult, register_meshes
from auto_alignment.refinement_modes import check_cancelled
ProgressCallback = Callable[[float, str], None]

@dataclass(frozen=True)
class AnalysisOutcome:
    registration: RegistrationResult
    comparison: ComparisonResult
    target_facts: MeshFacts
    source_facts: MeshFacts
    output_files: dict[str, Path]
    total_elapsed_seconds: float

def run_analysis_with_target(
    target_mesh,
    target_facts: MeshFacts,
    current_path: str | Path,
    output_dir: str | Path,
    color_max_mm: float = 1.0,
    config: AlignmentConfig | None = None,
    progress: ProgressCallback | None = None,
    green_tolerance_mm: float = 0.05,
    *,
    current_flip_normals: bool = False,
    minimum_nominal_mm: float | None = None,
    maximum_nominal_mm: float | None = None,
    target_archived_path: str | Path | None = None,
    target_edit_archived_path: str | Path | None = None,
    target_selected_faces: np.ndarray | None = None,
    current_edit_state_path: str | Path | None = None,
    cancel: Callable[[], bool] | None = None,
) -> AnalysisOutcome:
    config = config or AlignmentConfig()
    started = time.perf_counter()
    check_cancelled(cancel)
    if progress:
        progress(0.01, '正在读取和检查 STL…')
    current_mesh, source_facts = load_mesh(
        current_path,
        flip_normals=current_flip_normals,
    )
    source_selected_faces: np.ndarray | None = None
    source_edit_summary: dict[str, int] | None = None
    source_edit_payload: dict[str, object] | None = None
    if current_edit_state_path is not None:
        edit_state = load_edit_state(
            current_edit_state_path,
            current_path,
            len(current_mesh.triangles),
        )
        if np.any(edit_state.selected) or np.any(edit_state.deleted):
            applied = apply_edit_state(current_mesh, edit_state)
            current_mesh = applied.mesh
            source_selected_faces = applied.selected_faces
            source_edit_summary = {
                "selected_faces": applied.selected_count,
                "deleted_faces": applied.deleted_count,
            }
            source_edit_payload = edit_state.as_dict()
            source_facts = updated_mesh_facts(
                source_facts,
                current_mesh,
                edit_note=(
                    f"浮动模型工作副本：选区 {applied.selected_count} 面，"
                    f"删除 {applied.deleted_count} 面。"
                ),
            )
    registration = register_meshes(
        target_mesh,
        current_mesh,
        target_facts,
        source_facts,
        config,
        progress,
        target_priority_faces=target_selected_faces,
        source_priority_faces=source_selected_faces,
        **({"cancel": cancel} if cancel is not None else {}),
    )
    transform = np.asarray(registration.transformation, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError('自动配准未生成可供检查的有限 4×4 变换。')
    if progress:
        progress(
            0.92,
            (
                '质量门控未通过，正在生成最佳候选预览…'
                if not registration.succeeded
                else '正在计算有符号表面距离和方向颜色…'
            ),
        )
    comparison = compare_meshes(
        target_mesh,
        current_mesh,
        registration.transformation,
        config.metric_sample_points,
        color_max_mm,
        green_tolerance_mm,
        False,
        minimum_nominal_mm=minimum_nominal_mm,
        maximum_nominal_mm=maximum_nominal_mm,
    )
    check_cancelled(cancel)
    candidate_paths = {}
    candidate_files = {}
    def candidate_archive_path(value):
        if value is None or Path(value).is_absolute():
            return value
        return Path("../..") / value
    for name, candidate in registration.alternatives:
        check_cancelled(cancel)
        if progress:
            progress(0.94, f'正在保存 {name} 对比候选…')
        candidate_comparison = compare_meshes(
            target_mesh, current_mesh, candidate.transformation,
            config.metric_sample_points, color_max_mm, green_tolerance_mm, False,
            minimum_nominal_mm=minimum_nominal_mm, maximum_nominal_mm=maximum_nominal_mm,
        )
        check_cancelled(cancel)
        candidate_output = export_results(
            Path(output_dir) / "candidates" / name, target_facts, source_facts,
            candidate, candidate_comparison, time.perf_counter() - started,
            target_archived_path=candidate_archive_path(target_archived_path),
            target_edit_archived_path=candidate_archive_path(target_edit_archived_path),
            source_edit_state=source_edit_payload,
        )
        candidate_paths[name] = Path("candidates", name, "results.json").as_posix()
        candidate_files[f"candidate_{name}_results_json"] = candidate_output["results_json"]
    if progress:
        progress(0.98, '正在保存结果…')
    total_elapsed = time.perf_counter() - started
    files = export_results(
        output_dir,
        target_facts,
        source_facts,
        registration,
        comparison,
        total_elapsed,
        target_archived_path=target_archived_path,
        target_edit_archived_path=target_edit_archived_path,
        source_edit_state=source_edit_payload,
        candidate_results=candidate_paths,
        selection_info={
            "target_selected_faces": int(
                np.count_nonzero(target_selected_faces)
                if target_selected_faces is not None
                else 0
            ),
            "source_selected_faces": int(
                np.count_nonzero(source_selected_faces)
                if source_selected_faces is not None
                else 0
            ),
            **(source_edit_summary or {"selected_faces": 0, "deleted_faces": 0}),
        },
    )
    files.update(candidate_files)
    check_cancelled(cancel)
    if progress:
        progress(1.0, '完成')
    return AnalysisOutcome(registration=registration, comparison=comparison, target_facts=target_facts, source_facts=source_facts, output_files=files, total_elapsed_seconds=total_elapsed)


def run_analysis(target_path: str | Path, current_path: str | Path, output_dir: str | Path, color_max_mm: float=1.0, config: AlignmentConfig | None=None, progress: ProgressCallback | None=None, green_tolerance_mm: float=0.05, reverse_direction: bool=False, *, target_flip_normals: bool=False, current_flip_normals: bool=False, minimum_nominal_mm: float | None=None, maximum_nominal_mm: float | None=None, target_archived_path: str | Path | None=None, cancel: Callable[[], bool] | None=None) -> AnalysisOutcome:
    # reverse_direction is intentionally retained for API compatibility with
    # legacy callers. New 1.4 registrations always use the fixed mesh normals.
    del reverse_direction
    target_mesh, target_facts = load_mesh(
        target_path,
        flip_normals=target_flip_normals,
    )
    return run_analysis_with_target(
        target_mesh,
        target_facts,
        current_path,
        output_dir,
        color_max_mm,
        config,
        progress,
        green_tolerance_mm,
        current_flip_normals=current_flip_normals,
        minimum_nominal_mm=minimum_nominal_mm,
        maximum_nominal_mm=maximum_nominal_mm,
        target_archived_path=target_archived_path,
        **({"cancel": cancel} if cancel is not None else {}),
    )
