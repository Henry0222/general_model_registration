from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .comparison import ComparisonResult
from .mesh_io import MeshFacts
from .registration import RegistrationResult


class ExportError(RuntimeError):
    pass


def _write_triangle_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> bool:
    """Write through an ASCII temporary path for reliable Windows I/O."""
    try:
        return bool(o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False))
    except UnicodeError:
        with tempfile.TemporaryDirectory(prefix="dental_stl_") as temporary:
            safe_path = Path(temporary) / f"output{path.suffix.lower()}"
            if not o3d.io.write_triangle_mesh(str(safe_path), mesh, write_ascii=False):
                return False
            shutil.copyfile(safe_path, path)
            return True


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法 JSON 序列化：{type(value).__name__}")


def export_results(
    output_dir: str | Path,
    target_facts: MeshFacts,
    source_facts: MeshFacts,
    registration: RegistrationResult,
    comparison: ComparisonResult,
    total_elapsed_seconds: float,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    aligned_path = directory / "aligned_current.stl"
    colored_path = directory / "comparison_colormap.ply"
    transform_path = directory / "transform.json"
    results_path = directory / "results.json"

    if not _write_triangle_mesh(aligned_path, comparison.aligned_source):
        raise ExportError(f"无法写入：{aligned_path}")
    if not _write_triangle_mesh(colored_path, comparison.colored_source):
        raise ExportError(f"无法写入：{colored_path}")

    transform_payload = {
        "moving_model": "current_scan",
        "fixed_model": "target",
        "units": "millimetres",
        "transformation_current_to_target": registration.transformation,
    }
    transform_path.write_text(
        json.dumps(transform_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    payload = {
        "version": "1.3.0",
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
        },
        "target_mesh": target_facts.as_dict(),
        "current_mesh": source_facts.as_dict(),
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
        },
    }
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return {
        "aligned_stl": aligned_path,
        "colored_ply": colored_path,
        "transform_json": transform_path,
        "results_json": results_path,
    }
