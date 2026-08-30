from __future__ import annotations

import json
import os
from datetime import datetime
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .comparison import ComparisonResult
from .mesh_io import MeshFacts
from .registration import RegistrationResult
from .version import __version__


class ExportError(RuntimeError):
    pass


def _write_triangle_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> bool:
    """Atomically write a mesh, with an ASCII fallback for Open3D on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.stem}.{uuid.uuid4().hex}{path.suffix}"
    try:
        written = bool(
            o3d.io.write_triangle_mesh(str(temporary_path), mesh, write_ascii=False)
        )
    except (UnicodeError, RuntimeError):
        with tempfile.TemporaryDirectory(prefix="dental_stl_") as temporary:
            safe_path = Path(temporary) / f"output{path.suffix.lower()}"
            if not o3d.io.write_triangle_mesh(str(safe_path), mesh, write_ascii=False):
                return False
            shutil.copyfile(safe_path, temporary_path)
            written = True
    if not written:
        temporary_path.unlink(missing_ok=True)
        return False
    os.replace(temporary_path, path)
    return True


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法 JSON 序列化：{type(value).__name__}")


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, destination)
    return destination


def write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
    )


def export_results(
    output_dir: str | Path,
    target_facts: MeshFacts,
    source_facts: MeshFacts,
    registration: RegistrationResult,
    comparison: ComparisonResult,
    total_elapsed_seconds: float,
    *,
    target_archived_path: str | Path | None = None,
    target_edit_archived_path: str | Path | None = None,
    source_edit_state: dict[str, object] | None = None,
    selection_info: dict[str, object] | None = None,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    review_only = not registration.succeeded
    aligned_path = directory / (
        "best_candidate_FAILED_PREVIEW_ONLY.stl"
        if review_only
        else "aligned_current.stl"
    )
    colored_path = directory / (
        "best_candidate_colormap_FAILED_PREVIEW_ONLY.ply"
        if review_only
        else "comparison_colormap.ply"
    )
    transform_path = directory / (
        "best_candidate_transform_FAILED_PREVIEW_ONLY.json"
        if review_only
        else "transform.json"
    )
    results_path = directory / "results.json"
    source_edit_path: Path | None = None

    if not _write_triangle_mesh(aligned_path, comparison.aligned_source):
        raise ExportError(f"无法写入：{aligned_path}")
    if not _write_triangle_mesh(colored_path, comparison.colored_source):
        raise ExportError(f"无法写入：{colored_path}")
    if source_edit_state is not None:
        source_edit_path = directory / "moving_model_edit_state.json"
        write_json(source_edit_path, source_edit_state)

    transform_payload = {
        "version": __version__,
        "moving_model": source_facts.path,
        "fixed_model": target_facts.path,
        "units": "millimetres",
        "status": registration.status,
        "review_only": review_only,
        "notice": (
            "This transform failed quality gates and is for visual review only."
            if review_only
            else "Accepted registration transform."
        ),
        "transformation_current_to_target": registration.transformation,
    }
    write_json(transform_path, transform_payload)

    target_payload = target_facts.as_dict()
    if target_archived_path is not None:
        target_payload["archived_path"] = Path(target_archived_path).as_posix()
    if target_edit_archived_path is not None:
        target_payload["edit_state_path"] = Path(target_edit_archived_path).as_posix()
    scale_payload = (
        comparison.deviation_scale.as_dict()
        if comparison.deviation_scale is not None
        else {}
    )

    payload = {
        "version": __version__,
        "schema_version": "1.4.1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "review_only": review_only,
        "result_notice": (
            "配准未通过质量门控；当前文件是算法找到的最佳候选位姿，仅供检查，不得作为正式配准结果。"
            if review_only
            else "配准结果已通过质量判定。"
        ),
        "interpretation": "Signed target-normal distance: green is within tolerance, red is under-preparation, blue is over-preparation.",
        "color_mapping": {
            "green_rgb": [64, 255, 64],
            "green_range_mm": [
                -comparison.statistics.green_tolerance_mm,
                comparison.statistics.green_tolerance_mm,
            ],
            "positive_rgb": [255, 64, 64],
            "positive_meaning": "under-preparation",
            "negative_rgb": [64, 64, 255],
            "negative_meaning": "over-preparation",
            "saturation": 0.75,
            "color_limit_mm": comparison.statistics.color_max_mm,
            "direction_reversed": comparison.statistics.direction_reversed,
            "direction_basis": "closest target triangle normal",
            **scale_payload,
        },
        "target_mesh": target_payload,
        "current_mesh": source_facts.as_dict(),
        "model_editing": selection_info or {},
        "registration": {
            "status": registration.status,
            "confidence": registration.confidence,
            "metrics": registration.metrics.as_dict(),
            "quality": registration.quality.as_dict() if registration.quality else None,
            "elapsed_seconds": registration.elapsed_seconds,
            "warnings": list(registration.warnings),
        },
        "distance_statistics": comparison.statistics.as_dict(),
        "total_elapsed_seconds": total_elapsed_seconds,
        "outputs": {
            "aligned_current_stl": aligned_path.name,
            "comparison_colormap_ply": colored_path.name,
            "transform_json": transform_path.name,
            **(
                {"moving_model_edit_state_json": source_edit_path.name}
                if source_edit_path is not None
                else {}
            ),
        },
    }
    write_json(results_path, payload)
    return {
        "aligned_stl": aligned_path,
        "colored_ply": colored_path,
        "transform_json": transform_path,
        "results_json": results_path,
        **({"source_edit_state_json": source_edit_path} if source_edit_path else {}),
    }
