from __future__ import annotations
from .refinement_modes import RegistrationCancelled

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
from typing import Callable, Iterable

import numpy as np

from .batch_models import BatchItemResult, BatchOutcome, RegistrationJob
from .config import AlignmentConfig
from .exporters import _write_triangle_mesh, atomic_write_text, write_json
from .mesh_io import MeshFacts, load_mesh
from .mesh_selection import (
    apply_edit_state,
    load_edit_state,
    updated_mesh_facts,
)
from .pipeline import AnalysisOutcome, run_analysis_with_target
from .version import __version__


BatchProgressCallback = Callable[[int, float, str], None]
StopRequested = Callable[[], bool]


def create_batch_directory(
    output_parent: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now().astimezone()).strftime("%m%d%H%M")
    base = parent / f"align_{timestamp}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def _safe_stem(path: Path) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", path.stem).strip(" ._")
    return (cleaned or "model")[:80]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _mesh_lines(label: str, facts: MeshFacts, digest: str) -> list[str]:
    return [
        f"{label}路径：{facts.path}",
        f"{label}SHA-256：{digest}",
        f"{label}顶点/三角面：{facts.vertices} / {facts.triangles}",
        f"{label}包围盒对角线：{facts.diagonal_mm:.6f} mm",
        f"{label}翻转面朝向/法线：{'是' if facts.normals_flipped else '否'}",
    ]


def _log_metric(value: object) -> str:
    try:
        return f"{float(value):.9f} mm"
    except (TypeError, ValueError):
        return "未知"


def _success_log(
    outcome: AnalysisOutcome,
    *,
    target_digest: str,
    source_digest: str,
    started_at: str,
    finished_at: str,
    config: AlignmentConfig,
) -> str:
    registration = outcome.registration
    metrics = registration.metrics
    comparison = outcome.comparison.statistics
    decision = metrics.high_precision_decision or {}
    selection = metrics.selection_decision or {}
    selected = decision.get("selected_metrics") or {}
    bank = metrics.candidate_selection or {}
    refinement = metrics.refinement or {}
    review_only = not registration.succeeded
    lines = [
        (
            "通用模型自动配准质量失败预览日志"
            if review_only
            else "通用模型自动配准日志"
        ),
        f"软件版本：{__version__}",
        f"开始时间：{started_at}",
        f"完成时间：{finished_at}",
        *(
            (
                "",
                "重要：本次配准未通过质量门控。",
                "已保存算法找到的最佳候选位姿，仅供检查，不得作为正式配准结果。",
            )
            if review_only
            else ()
        ),
        "",
        *_mesh_lines("固定模型", outcome.target_facts, target_digest),
        *_mesh_lines("浮动模型", outcome.source_facts, source_digest),
        "",
        "配准参数：",
        f"  精修模式：{config.refinement_mode}",
        f"  表面采样点：{config.global_sample_points}",
        f"  RANSAC 最大迭代：{config.ransac_max_iterations}",
        (
            "  彻底检查可能朝向：启用"
            if config.exhaustive_orientation_search
            else "  彻底检查可能朝向：关闭"
        ),
        f"  轴对称角度步长：{config.exhaustive_orientation_angle_step_degrees:.3f}°",
        f"  覆盖距离：{config.coverage_distance_mm:.6f} mm",
        f"  最小有效覆盖率：{config.partial_overlap_threshold:.6f}",
        "",
        "配准结果：",
        f"  3.0 采用结果：{refinement.get('selected', 'initial')}",
        f"  精修状态/原因：{refinement.get('state', 'disabled')} / {refinement.get('reason', '')}",
        f"  精修计算耗时：{float(refinement.get('elapsed_seconds', 0)):.3f} 秒",
        f"  精修错误：{refinement.get('error') or refinement.get('assessment_failures') or '无'}",
        f"  状态：{registration.status}",
        f"  可信度：{registration.confidence}",
        f"  最终阶段：{('v3_' + str(refinement['selected'])) if refinement.get('selected', 'initial') != 'initial' else (selection.get('selected_stage') or decision.get('selected_stage', 'multiscale_icp'))}",
        f"  旋转角度：{metrics.rotation_degrees:.9f}°",
        f"  平移量：{metrics.translation_mm:.9f} mm",
        f"  ICP fitness：{metrics.fitness:.9f}",
        f"  ICP RMSE：{metrics.inlier_rmse_mm:.9f} mm",
        f"  包围盒重叠率：{metrics.overlap_ratio:.9f}",
        *((f"  稳定共同表面中位误差：{_log_metric(selected.get('median_mm'))}",
           f"  稳定共同表面 P90：{_log_metric(selected.get('p90_mm'))}") if not bank else ()),
        f"  模型全表面对称 RMS：{comparison.symmetric_rms_mm:.9f} mm",
        f"  模型全表面中位误差：{comparison.median_mm:.9f} mm",
        f"  模型全表面 HD95：{comparison.hd95_mm:.9f} mm",
        f"  模型全表面最大距离：{comparison.maximum_mm:.9f} mm",
        f"  总耗时：{outcome.total_elapsed_seconds:.3f} 秒",
        "",
        "变换矩阵（浮动模型 → 固定模型）：",
        np.array2string(registration.transformation, precision=12, suppress_small=False),
    ]
    reasons = [str(value) for value in decision.get("reasons", ())]
    if bank:
        chosen = next((item for item in bank.get("hypotheses", ()) if item.get("name") == bank.get("selected")), {})
        evaluation = chosen.get("evaluation") or {}
        mode = "共同表面支持" if bank.get("selected_mode") == "common_surface" else "整体形状拟合（共同表面证据不足）"
        lines.extend((
            "", "2.0 全阶段候选选择：",
            f"  采用候选：{bank.get('selected')}",
            f"  拟合依据：{mode}",
            f"  候选总数 / 未采用的精化更新：{bank.get('candidate_count')} / {bank.get('rejected_update_count')}",
            f"  固定评价点数（浮动 / 固定）：{bank.get('evaluation_points')}",
            f"  存在位置歧义：{'是' if bank.get('ambiguous') else '否'}",
            "  评价采用固定采样点和固定分母；属于内部几何检查，不代表已知真实位姿误差。",
        ))
        for label, direction in zip(("浮动→固定", "固定→浮动"), evaluation.get("directions", ())):
            lines.append(f"  {label}全采样中位距离 / P90：{_log_metric(direction.get('median_mm'))} / {_log_metric(direction.get('p90_mm'))}")
            lines.append(f"  {label}共同支持率 / 可观测秩：{float(direction.get('common_coverage', 0)):.6f} / {direction.get('observability_rank')}")
    if decision.get("mode") == "stable_region":
        initial_region = decision.get("initial_region") or {}
        refined_region = decision.get("refined_region") or initial_region
        holdout = (decision.get("candidate_metrics") or {}).get("source_holdout") or {}
        holdout_before = (decision.get("before_metrics") or {}).get("source_holdout") or {}
        lines.extend(
            (
                "",
                "自适应稳定区末级精配准：",
                f"  状态：{'已接受' if decision.get('accepted') else '未通过验证，已回退'}",
                f"  估计噪声 σ：{_log_metric(decision.get('sigma_mm'))}",
                f"  稳定判定阈值（k·σ）：{_log_metric(decision.get('stable_distance_mm'))}",
                f"  初始稳定区面积占比（浮动/固定）：{float(initial_region.get('source_stable_area_fraction', 0.0)):.3f} / {float(initial_region.get('target_stable_area_fraction', 0.0)):.3f}",
                f"  精配准后稳定区面积占比（浮动/固定）：{float(refined_region.get('source_stable_area_fraction', 0.0)):.3f} / {float(refined_region.get('target_stable_area_fraction', 0.0)):.3f}",
                f"  留出区中位误差（精配准前 → 后）：{_log_metric(holdout_before.get('median_mm'))} → {_log_metric(holdout.get('median_mm'))}",
                f"  留出区 P90（精配准前 → 后）：{_log_metric(holdout_before.get('p90_mm'))} → {_log_metric(holdout.get('p90_mm'))}",
                f"  稳定点最大位移：{_log_metric(decision.get('delta_max_local_displacement_mm'))}（允许 {_log_metric(decision.get('displacement_limit_mm'))}）",
            )
        )
    if reasons:
        lines.extend(("", "末级门控说明：", *(f"  - {reason}" for reason in reasons)))
    if selection.get("enabled"):
        counts = selection.get("selected_face_counts") or {}
        baseline_metrics = selection.get("baseline_metrics") or {}
        priority_metrics = selection.get("priority_metrics") or {}
        strict_metrics = selection.get("strict_metrics") or {}
        selected_metrics = selection.get("selected_metrics") or {}
        selected_lane = str(selection.get("selected_lane", "unknown"))
        lane_labels = {
            "full_surface_baseline": "全模型粗定位（选区指标最优）",
            "selection_weighted_global": "选区加权全局候选",
            "selection_strict_global": "严格选区全局候选",
            "selection_refined_baseline": "全模型粗定位 + 严格选区精配准",
            "selection_refined_weighted": "选区加权全局候选 + 严格选区精配准",
            "selection_refined_strict": "严格选区全局候选 + 严格选区精配准",
        }
        lines.extend(
            (
                "",
                "操作者选区主导配准：",
                f"  固定/浮动选区面片：{int(counts.get('target', 0))} / {int(counts.get('source', 0))}",
                "  决策策略：选区覆盖率与选区误差优先；完整模型仅作灾难性错位检查",
                f"  采用通道：{lane_labels.get(selected_lane, selected_lane)}",
                f"  全模型基线选区覆盖率/中位误差：{float(baseline_metrics.get('coverage_ratio', 0.0)):.6f} / {_log_metric(baseline_metrics.get('median_mm'))}",
                f"  选区加权全局候选覆盖率/中位误差：{float(priority_metrics.get('coverage_ratio', 0.0)):.6f} / {_log_metric(priority_metrics.get('median_mm'))}",
                f"  严格选区全局候选覆盖率/中位误差：{float(strict_metrics.get('coverage_ratio', 0.0)):.6f} / {_log_metric(strict_metrics.get('median_mm'))}",
                f"  最终选区覆盖率/中位误差/P90：{float(selected_metrics.get('coverage_ratio', 0.0)):.6f} / {_log_metric(selected_metrics.get('median_mm'))} / {_log_metric(selected_metrics.get('p90_mm'))}",
                f"  选区几何约束：{'有效' if selection.get('selection_geometry_valid') else '不足'}",
                f"  选区覆盖可信度：{'达到阈值' if selection.get('coverage_confident') else '低于阈值（仅降级警告，不自动回退）'}",
                f"  全模型灾难性错位检查：{'通过' if selection.get('whole_model_guard_passed') else '未通过'}",
            )
        )
        selection_reasons = [str(value) for value in selection.get("reasons", ())]
        if selection_reasons:
            lines.extend(("  决策说明：", *(f"    - {reason}" for reason in selection_reasons)))
    if registration.warnings:
        lines.extend(("", "警告：", *(f"  - {warning}" for warning in registration.warnings)))
    lines.append("")
    return "\n".join(lines)


def _failure_log(
    *,
    job: RegistrationJob,
    target_facts: MeshFacts,
    target_digest: str,
    started_at: str,
    finished_at: str,
    error: BaseException,
    details: str,
    stage: str,
) -> str:
    return "\n".join(
        [
            "通用模型自动配准失败日志",
            f"软件版本：{__version__}",
            f"开始时间：{started_at}",
            f"失败时间：{finished_at}",
            "",
            *_mesh_lines("固定模型", target_facts, target_digest),
            f"浮动模型路径：{job.source_path}",
            f"浮动模型翻转面朝向/法线：{'是' if job.flip_normals else '否'}",
            "",
            f"失败阶段：{stage}",
            f"错误类型：{type(error).__name__}",
            f"错误原因：{error}",
            "",
            "技术堆栈：",
            details.rstrip(),
            "",
        ]
    )


def _manifest_payload(
    *,
    batch_directory: Path,
    started_at: str,
    finished_at: str | None,
    target_facts: MeshFacts | None,
    target_digest: str | None,
    target_archive: Path | None,
    target_edit_archive: Path | None,
    config: AlignmentConfig,
    minimum_nominal_mm: float,
    maximum_nominal_mm: float,
    items: Iterable[BatchItemResult],
    stopped: bool,
) -> dict[str, object]:
    fixed: dict[str, object] | None = None
    if target_facts is not None:
        fixed = target_facts.as_dict()
        fixed.update(
            {
                "sha256": target_digest,
                "archived_path": (
                    target_archive.relative_to(batch_directory).as_posix()
                    if target_archive is not None
                    else None
                ),
                "edit_state_path": (
                    target_edit_archive.relative_to(batch_directory).as_posix()
                    if target_edit_archive is not None
                    else None
                ),
            }
        )
    return {
        "version": __version__,
        "schema_version": "1.4.2",
        "batch_name": batch_directory.name,
        "started_at": started_at,
        "finished_at": finished_at,
        "stopped": stopped,
        "fixed_model": fixed,
        "parameters": {
            "minimum_nominal_mm": minimum_nominal_mm,
            "maximum_nominal_mm": maximum_nominal_mm,
            "surface_sample_points": config.global_sample_points,
            "refinement_mode": config.refinement_mode,
            "ransac_max_iterations": config.ransac_max_iterations,
            "exhaustive_orientation_search": config.exhaustive_orientation_search,
            "exhaustive_orientation_angle_step_degrees": (
                config.exhaustive_orientation_angle_step_degrees
            ),
            "coverage_distance_mm": config.coverage_distance_mm,
            "minimum_overlap_ratio": config.partial_overlap_threshold,
        },
        "items": [item.as_dict() for item in items],
    }


def run_batch_analysis(
    target_path: str | Path,
    target_flip_normals: bool,
    jobs: Iterable[RegistrationJob],
    output_parent: str | Path,
    *,
    target_edit_state_path: str | Path | None = None,
    config: AlignmentConfig | None = None,
    minimum_nominal_mm: float = -0.05,
    maximum_nominal_mm: float = 0.05,
    color_max_mm: float = 1.0,
    progress: BatchProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
    item_finished: Callable[[BatchItemResult], None] | None = None,
) -> BatchOutcome:
    job_list = tuple(jobs)
    if not job_list:
        raise ValueError("至少需要一个浮动 STL。")
    if minimum_nominal_mm >= maximum_nominal_mm:
        raise ValueError("最小名义偏差必须小于最大名义偏差。")
    config = config or AlignmentConfig()
    started_perf = time.perf_counter()
    started_at = _iso_now()
    batch_directory = create_batch_directory(output_parent)
    manifest_path = batch_directory / "batch_results.json"
    batch_log_path = batch_directory / "batch.log"
    batch_lines = [
        "通用模型自动配准批次日志",
        f"软件版本：{__version__}",
        f"批次目录：{batch_directory}",
        f"开始时间：{started_at}",
        "",
    ]
    atomic_write_text(batch_log_path, "\n".join(batch_lines))

    target_facts: MeshFacts | None = None
    target_selected_faces: np.ndarray | None = None
    target_digest: str | None = None
    target_archive: Path | None = None
    target_edit_archive: Path | None = None
    results: list[BatchItemResult] = []
    stopped = False
    try:
        if progress:
            progress(0, 0.0, "正在读取固定 STL…")
        target_mesh, target_facts = load_mesh(
            target_path,
            flip_normals=target_flip_normals,
        )
        if target_edit_state_path is not None:
            edit_state = load_edit_state(
                target_edit_state_path,
                target_path,
                len(target_mesh.triangles),
            )
            if np.any(edit_state.selected) or np.any(edit_state.deleted):
                applied = apply_edit_state(target_mesh, edit_state)
                target_mesh = applied.mesh
                target_selected_faces = applied.selected_faces
                target_edit_archive = batch_directory / "fixed_target_edit_state.json"
                write_json(target_edit_archive, edit_state.as_dict())
                target_facts = updated_mesh_facts(
                    target_facts,
                    target_mesh,
                    edit_note=(
                        f"固定模型工作副本：选区 {applied.selected_count} 面，"
                        f"删除 {applied.deleted_count} 面。"
                    ),
                )
        target_digest = _sha256(Path(target_facts.path))
        target_archive = batch_directory / "fixed_target_used.stl"
        if not _write_triangle_mesh(target_archive, target_mesh):
            raise RuntimeError("无法归档实际参与配准的固定 STL。")
    except Exception as error:
        details = traceback.format_exc()
        batch_lines.extend(("固定模型读取失败。", f"{type(error).__name__}: {error}", details))
        atomic_write_text(batch_log_path, "\n".join(batch_lines))
        write_json(
            manifest_path,
            _manifest_payload(
                batch_directory=batch_directory,
                started_at=started_at,
                finished_at=_iso_now(),
                target_facts=target_facts,
                target_digest=target_digest,
                target_archive=target_archive,
                target_edit_archive=target_edit_archive,
                config=config,
                minimum_nominal_mm=minimum_nominal_mm,
                maximum_nominal_mm=maximum_nominal_mm,
                items=results,
                stopped=False,
            ),
        )
        raise

    assert target_facts is not None
    assert target_digest is not None
    assert target_archive is not None
    total_jobs = len(job_list)
    for position, job in enumerate(job_list):
        if stopped or (stop_requested is not None and stop_requested()):
            stopped = True
            for skipped in job_list[position:]:
                skipped_result = BatchItemResult(
                    index=skipped.index,
                    source_path=str(skipped.source_path),
                    source_name=skipped.source_path.name,
                    flip_normals=skipped.flip_normals,
                    status="skipped",
                    confidence="未执行",
                    elapsed_seconds=0.0,
                    output_directory=None,
                    results_json=None,
                    log_file=None,
                    error="操作者已停止批次。",
                )
                results.append(skipped_result)
                if item_finished is not None:
                    item_finished(skipped_result)
            batch_lines.append("操作者请求停止，未开始的模型已跳过。")
            break

        item_started_perf = time.perf_counter()
        item_started_at = _iso_now()
        item_directory = batch_directory / f"{job.index:02d}_{_safe_stem(job.source_path)}"
        item_directory.mkdir(parents=False, exist_ok=False)
        item_log = item_directory / "registration.log"
        atomic_write_text(
            item_log,
            "\n".join(
                (
                    "通用模型自动配准日志",
                    f"软件版本：{__version__}",
                    f"开始时间：{item_started_at}",
                    "状态：正在处理",
                    "",
                )
            ),
        )
        last_stage = "正在读取和检查浮动 STL"

        def item_progress(fraction: float, message: str) -> None:
            nonlocal last_stage
            last_stage = str(message)
            overall = (position + max(0.0, min(1.0, float(fraction)))) / total_jobs
            if progress:
                progress(job.index, overall, f"[{position + 1}/{total_jobs}] {message}")

        try:
            outcome = run_analysis_with_target(
                target_mesh,
                target_facts,
                job.source_path,
                item_directory,
                color_max_mm,
                config,
                item_progress,
                max(abs(minimum_nominal_mm), abs(maximum_nominal_mm)),
                current_flip_normals=job.flip_normals,
                minimum_nominal_mm=minimum_nominal_mm,
                maximum_nominal_mm=maximum_nominal_mm,
                target_archived_path=Path("..") / target_archive.name,
                target_edit_archived_path=(
                    Path("..") / target_edit_archive.name
                    if target_edit_archive is not None
                    else None
                ),
                target_selected_faces=target_selected_faces,
                current_edit_state_path=job.edit_state_path,
                **({"cancel": stop_requested} if stop_requested is not None else {}),
            )
            source_digest = _sha256(Path(outcome.source_facts.path))
            finished_at = _iso_now()
            atomic_write_text(
                item_log,
                _success_log(
                    outcome,
                    target_digest=target_digest,
                    source_digest=source_digest,
                    started_at=item_started_at,
                    finished_at=finished_at,
                    config=config,
                ),
            )
            stats = outcome.comparison.statistics
            decision = outcome.registration.metrics.high_precision_decision or {}
            selection = outcome.registration.metrics.selection_decision or {}
            refinement = outcome.registration.metrics.refinement or {}
            p90 = (decision.get("selected_metrics") or {}).get("p90_mm")
            result = BatchItemResult(
                index=job.index,
                source_path=str(job.source_path),
                source_name=job.source_path.name,
                flip_normals=job.flip_normals,
                status=outcome.registration.status,
                confidence=outcome.registration.confidence,
                elapsed_seconds=outcome.total_elapsed_seconds,
                output_directory=item_directory.relative_to(batch_directory).as_posix(),
                results_json=outcome.output_files["results_json"].relative_to(batch_directory).as_posix(),
                log_file=item_log.relative_to(batch_directory).as_posix(),
                error=(
                    "未通过质量门控：" + "；".join(outcome.registration.warnings)
                    if not outcome.registration.succeeded
                    else None
                ),
                review_only=not outcome.registration.succeeded,
                selection_enabled=bool(selection.get("enabled")),
                selection_lane=(
                    str(selection.get("selected_lane"))
                    if selection.get("selected_lane") is not None
                    else None
                ),
                symmetric_rms_mm=stats.symmetric_rms_mm,
                p90_mm=float(p90) if p90 is not None else None,
                hd95_mm=stats.hd95_mm,
                refinement_mode=config.refinement_mode,
                refinement_selected=str(refinement.get("selected", "initial")),
            )
            batch_lines.append(
                f"[{job.index:02d}] "
                + (
                    f"失败（已保存最佳候选预览）：{job.source_path.name}｜"
                    if result.review_only
                    else f"完成：{job.source_path.name}｜"
                )
                + f"状态={result.status}｜可信度={result.confidence}｜"
                + f"RMS={stats.symmetric_rms_mm:.6f} mm"
            )
        except RegistrationCancelled as error:
            stopped = True
            finished_at = _iso_now()
            atomic_write_text(item_log, f"配准已取消\n阶段：{last_stage}\n完成时间：{finished_at}\n")
            write_json(item_directory / "cancelled.json", {
                "version": __version__, "status": "cancelled", "stage": last_stage,
                "source_path": str(job.source_path), "started_at": item_started_at,
                "finished_at": finished_at, "error": str(error),
            })
            result = BatchItemResult(
                index=job.index, source_path=str(job.source_path), source_name=job.source_path.name,
                flip_normals=job.flip_normals, status="cancelled", confidence="—",
                elapsed_seconds=time.perf_counter() - item_started_perf,
                output_directory=item_directory.relative_to(batch_directory).as_posix(),
                results_json=None, log_file=item_log.relative_to(batch_directory).as_posix(),
                error=str(error), refinement_mode=config.refinement_mode,
            )
            batch_lines.append(f"[{job.index:02d}] 已取消：{job.source_path.name}")
        except Exception as error:
            details = traceback.format_exc()
            finished_at = _iso_now()
            atomic_write_text(
                item_log,
                _failure_log(
                    job=job,
                    target_facts=target_facts,
                    target_digest=target_digest,
                    started_at=item_started_at,
                    finished_at=finished_at,
                    error=error,
                    details=details,
                    stage=last_stage,
                ),
            )
            failure_path = item_directory / "failure.json"
            write_json(
                failure_path,
                {
                    "version": __version__,
                    "schema_version": "1.4.2",
                    "status": "failed",
                    "source_path": str(job.source_path),
                    "source_name": job.source_path.name,
                    "flip_normals": job.flip_normals,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_at": item_started_at,
                    "failed_at": finished_at,
                    "log_file": item_log.name,
                },
            )
            result = BatchItemResult(
                index=job.index,
                source_path=str(job.source_path),
                source_name=job.source_path.name,
                flip_normals=job.flip_normals,
                status="failed",
                confidence="失败",
                elapsed_seconds=time.perf_counter() - item_started_perf,
                output_directory=item_directory.relative_to(batch_directory).as_posix(),
                results_json=None,
                log_file=item_log.relative_to(batch_directory).as_posix(),
                error=f"{type(error).__name__}: {error}",
            )
            batch_lines.append(f"[{job.index:02d}] 失败：{job.source_path.name}｜{result.error}")
        results.append(result)
        if item_finished is not None:
            item_finished(result)
        atomic_write_text(batch_log_path, "\n".join((*batch_lines, "")))
        write_json(
            manifest_path,
            _manifest_payload(
                batch_directory=batch_directory,
                started_at=started_at,
                finished_at=None,
                target_facts=target_facts,
                target_digest=target_digest,
                target_archive=target_archive,
                target_edit_archive=target_edit_archive,
                config=config,
                minimum_nominal_mm=minimum_nominal_mm,
                maximum_nominal_mm=maximum_nominal_mm,
                items=results,
                stopped=stopped,
            ),
        )

    finished_at = _iso_now()
    elapsed = time.perf_counter() - started_perf
    batch_lines.extend(("", f"完成时间：{finished_at}", f"批次总耗时：{elapsed:.3f} 秒"))
    atomic_write_text(batch_log_path, "\n".join((*batch_lines, "")))
    write_json(
        manifest_path,
        _manifest_payload(
            batch_directory=batch_directory,
            started_at=started_at,
            finished_at=finished_at,
            target_facts=target_facts,
            target_digest=target_digest,
            target_archive=target_archive,
            target_edit_archive=target_edit_archive,
            config=config,
            minimum_nominal_mm=minimum_nominal_mm,
            maximum_nominal_mm=maximum_nominal_mm,
            items=results,
            stopped=stopped,
        ),
    )
    if progress:
        progress(0, 1.0, "批次处理完成。")
    return BatchOutcome(
        batch_directory=batch_directory,
        manifest_path=manifest_path,
        batch_log_path=batch_log_path,
        items=tuple(results),
        stopped=stopped,
        total_elapsed_seconds=elapsed,
    )
