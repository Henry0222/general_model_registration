from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass(frozen=True)
class HistoryRecord:
    timestamp: str
    source_name: str
    status: str
    confidence: str
    symmetric_rms_mm: float | None
    p90_mm: float | None
    hd95_mm: float | None
    flip_normals: bool
    directory: Path
    results_path: Path | None
    log_path: Path | None
    error: str | None = None


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def records_from_manifest(path: str | Path) -> list[HistoryRecord]:
    manifest_path = Path(path).resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    timestamp = str(payload.get("started_at") or base.name)
    records: list[HistoryRecord] = []
    for item in payload.get("items", []):
        directory_value = item.get("output_directory")
        directory = (
            (base / str(directory_value)).resolve()
            if directory_value
            else base
        )
        results_value = item.get("results_json")
        log_value = item.get("log_file")
        records.append(
            HistoryRecord(
                timestamp=timestamp,
                source_name=str(item.get("source_name") or item.get("source_path") or "未知模型"),
                status=str(item.get("status") or "unknown"),
                confidence=str(item.get("confidence") or "未知"),
                symmetric_rms_mm=_optional_float(item.get("symmetric_rms_mm")),
                p90_mm=_optional_float(item.get("p90_mm")),
                hd95_mm=_optional_float(item.get("hd95_mm")),
                flip_normals=bool(item.get("flip_normals", False)),
                directory=directory,
                results_path=(base / str(results_value)).resolve() if results_value else None,
                log_path=(base / str(log_value)).resolve() if log_value else None,
                error=str(item.get("error")) if item.get("error") else None,
            )
        )
    return records


def record_from_results(path: str | Path) -> HistoryRecord:
    results_path = Path(path).resolve(strict=True)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    registration = payload.get("registration") or {}
    metrics = registration.get("metrics") or {}
    decision = metrics.get("high_precision_decision") or {}
    selected = decision.get("selected_metrics") or {}
    statistics = payload.get("distance_statistics") or {}
    source = payload.get("current_mesh") or {}
    log_path = results_path.parent / "registration.log"
    return HistoryRecord(
        timestamp=str(
            payload.get("created_at")
            or datetime.fromtimestamp(results_path.stat().st_mtime)
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        source_name=Path(str(source.get("path") or results_path.parent.name)).name,
        status=str(registration.get("status") or "success"),
        confidence=str(registration.get("confidence") or "未知"),
        symmetric_rms_mm=_optional_float(statistics.get("symmetric_rms_mm")),
        p90_mm=_optional_float(selected.get("p90_mm")),
        hd95_mm=_optional_float(statistics.get("hd95_mm")),
        flip_normals=bool(source.get("normals_flipped", False)),
        directory=results_path.parent,
        results_path=results_path,
        log_path=log_path if log_path.is_file() else None,
    )


def scan_history(root: str | Path) -> list[HistoryRecord]:
    directory = Path(root)
    if not directory.is_dir():
        return []
    records: list[HistoryRecord] = []
    seen_results: set[Path] = set()
    for manifest in sorted(directory.glob("align_*/batch_results.json"), reverse=True):
        try:
            manifest_records = records_from_manifest(manifest)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        records.extend(manifest_records)
        seen_results.update(
            record.results_path for record in manifest_records if record.results_path
        )
    for results_path in sorted(directory.glob("**/results.json"), reverse=True):
        resolved = results_path.resolve()
        if resolved in seen_results:
            continue
        try:
            records.append(record_from_results(resolved))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return records
