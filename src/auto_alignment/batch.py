from __future__ import annotations

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

from .config import AlignmentConfig
from .exporters import _write_triangle_mesh, atomic_write_text, write_json
from .mesh_io import MeshFacts, load_mesh
from .pipeline import AnalysisOutcome, run_analysis_with_target
from .version import __version__


BatchProgressCallback = Callable[[int, float, str], None]
StopRequested = Callable[[], bool]


@dataclass(frozen=True)
class RegistrationJob:
    index: int
    source_path: Path
    flip_normals: bool = False


@dataclass(frozen=True)
class BatchItemResult:
    index: int
    source_path: str
    source_name: str
    flip_normals: bool
    status: str
    confidence: str
    elapsed_seconds: float
    output_directory: str | None
    results_json: str | None
    log_file: str | None
    error: str | None = None
    symmetric_rms_mm: float | None = None
    p90_mm: float | None = None
    hd95_mm: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "flip_normals": self.flip_normals,
            "status": self.status,
            "confidence": self.confidence,
            "elapsed_seconds": self.elapsed_seconds,
            "output_directory": self.output_directory,
            "results_json": self.results_json,
            "log_file": self.log_file,
            "error": self.error,
            "symmetric_rms_mm": self.symmetric_rms_mm,
            "p90_mm": self.p90_mm,
            "hd95_mm": self.hd95_mm,
        }


@dataclass(frozen=True)
class BatchOutcome:
    batch_directory: Path
    manifest_path: Path
    batch_log_path: Path
    items: tuple[BatchItemResult, ...]
    stopped: bool
    total_elapsed_seconds: float


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
    selected = decision.get("selected_metrics") or {}
    lines = [
        "通用模型自动配准日志",
        f"软件版本：{__version__}",
        f"开始时间：{started_at}",
        f"完成时间：{finished_at}",
        "",
        *_mesh_lines("固定模型", outcome.target_facts, target_digest),
        *_mesh_lines("浮动模型", outcome.source_facts, source_digest),
        "",
        "配准参数：",
        f"  表面采样点：{config.global_sample_points}",
        f"  RANSAC 最大迭代：{config.ransac_max_iterations}",
        f"  覆盖距离：{config.coverage_distance_mm:.6f} mm",
        f"  最小有效覆盖率：{config.partial_overlap_threshold:.6f}",
        "",
        "配准结果：",
        f"  状态：{registration.status}",
        f"  可信度：{registration.confidence}",
        f"  最终阶段：{decision.get('selected_stage', 'multiscale_icp')}",
        f"  旋转角度：{metrics.rotation_degrees:.9f}°",
        f"  平移量：{metrics.translation_mm:.9f} mm",
        f"  ICP fitness：{metrics.fitness:.9f}",
        f"  ICP RMSE：{metrics.inlier_rmse_mm:.9f} mm",
        f"  包围盒重叠率：{metrics.overlap_ratio:.9f}",
        f"  稳定共同表面中位误差：{_log_metric(selected.get('median_mm'))}",
        f"  稳定共同表面 P90：{_log_metric(selected.get('p90_mm'))}",
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
    if reasons:
        lines.extend(("", "末级门控说明：", *(f"  - {reason}" for reason in reasons)))
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
            }
        )
    return {
        "version": __version__,
        "schema_version": "1.4",
        "batch_name": batch_directory.name,
        "started_at": started_at,
        "finished_at": finished_at,
        "stopped": stopped,
        "fixed_model": fixed,
        "parameters": {
            "minimum_nominal_mm": minimum_nominal_mm,
            "maximum_nominal_mm": maximum_nominal_mm,
            "surface_sample_points": config.global_sample_points,
            "ransac_max_iterations": config.ransac_max_iterations,
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
    target_digest: str | None = None
    target_archive: Path | None = None
    results: list[BatchItemResult] = []
    stopped = False
    try:
        if progress:
            progress(0, 0.0, "正在读取固定 STL…")
        target_mesh, target_facts = load_mesh(
            target_path,
            flip_normals=target_flip_normals,
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
        if stop_requested is not None and stop_requested():
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
                symmetric_rms_mm=stats.symmetric_rms_mm,
                p90_mm=float(p90) if p90 is not None else None,
                hd95_mm=stats.hd95_mm,
            )
            batch_lines.append(
                f"[{job.index:02d}] 成功：{job.source_path.name}｜"
                f"状态={result.status}｜可信度={result.confidence}｜"
                f"RMS={stats.symmetric_rms_mm:.6f} mm"
            )
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
                    "schema_version": "1.4",
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
                config=config,
                minimum_nominal_mm=minimum_nominal_mm,
                maximum_nominal_mm=maximum_nominal_mm,
                items=results,
                stopped=False,
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
