from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from auto_alignment.comparison import ComparisonResult, compare_meshes
from auto_alignment.config import AlignmentConfig
from auto_alignment.exporters import export_results
from auto_alignment.mesh_io import MeshFacts, load_mesh
from auto_alignment.registration import RegistrationResult, register_meshes
ProgressCallback = Callable[[float, str], None]

@dataclass(frozen=True)
class AnalysisOutcome:
    registration: RegistrationResult
    comparison: ComparisonResult
    target_facts: MeshFacts
    source_facts: MeshFacts
    output_files: dict[str, Path]
    total_elapsed_seconds: float

def run_analysis(target_path: str | Path, current_path: str | Path, output_dir: str | Path, color_max_mm: float=1.0, config: AlignmentConfig | None=None, progress: ProgressCallback | None=None, green_tolerance_mm: float=0.05, reverse_direction: bool=False) -> AnalysisOutcome:
    config = config or AlignmentConfig()
    started = time.perf_counter()
    if progress:
        progress(0.01, '正在读取和检查 STL…')
    target_mesh, target_facts = load_mesh(target_path)
    current_mesh, source_facts = load_mesh(current_path)
    registration = register_meshes(target_mesh, current_mesh, target_facts, source_facts, config, progress)
    if not registration.succeeded:
        details = '；'.join(registration.warnings) or '未知原因'
        raise RuntimeError(f'自动配准失败：{details}')
    if progress:
        progress(0.92, '正在计算有符号表面距离和方向颜色…')
    comparison = compare_meshes(target_mesh, current_mesh, registration.transformation, config.metric_sample_points, color_max_mm, green_tolerance_mm, reverse_direction)
    if progress:
        progress(0.98, '正在保存结果…')
    total_elapsed = time.perf_counter() - started
    files = export_results(output_dir, target_facts, source_facts, registration, comparison, total_elapsed)
    if progress:
        progress(1.0, '完成')
    return AnalysisOutcome(registration=registration, comparison=comparison, target_facts=target_facts, source_facts=source_facts, output_files=files, total_elapsed_seconds=total_elapsed)
