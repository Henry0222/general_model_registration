"""Public review manifests. Parsing and writing never initialize an Open3D GUI."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import threading

from ..file_io import atomic_write_text

VIEWER_MANIFEST_SCHEMA_VERSION = 1
_LEGACY_RESULT_SCHEMAS = {"1.4.0", "1.4.1", "1.4.2"}
_VIEWER_EXPORTS = {"ViewerData", "GeneralResultViewer", "load_viewer_data", "load_pair_viewer_data",
                   "run_general_result_viewer", "run_pair_result_viewer", "configure_open3d_font"}
__all__ = sorted(_VIEWER_EXPORTS) + ["VIEWER_MANIFEST_SCHEMA_VERSION", "RegistrationReviewSpec",
          "build_review_manifest", "validate_review_manifest", "write_review_manifest",
          "load_review_manifest", "run_registration_review"]


def __getattr__(name):
    if name in _VIEWER_EXPORTS:
        return getattr(importlib.import_module("..result_viewer", __package__), name)
    raise AttributeError(name)


@dataclass(frozen=True)
class RegistrationReviewSpec:
    target_path: str | Path
    aligned_path: str | Path
    status: str = "unknown"
    position_confidence: str | None = None
    confidence_display: str | None = None
    warnings: tuple[str, ...] = ()
    review_only: bool = False
    direction_reversed: bool = False
    minimum_nominal_mm: float = -0.05
    maximum_nominal_mm: float = 0.05
    target_sha256: str | None = None
    aligned_sha256: str | None = None
    annotations_path: str | Path | None = None
    manifest_path: str | Path | None = None


def _confidence(value):
    return getattr(value, "value", value)


def build_review_manifest(spec: RegistrationReviewSpec) -> dict:
    payload = dict(schema_version=VIEWER_MANIFEST_SCHEMA_VERSION,
                   target_mesh={"path": str(spec.target_path)},
                   outputs={"aligned_current_stl": str(spec.aligned_path)},
                   registration={"status": spec.status, "confidence": spec.confidence_display,
                                 "warnings": list(spec.warnings), "position_confidence": _confidence(spec.position_confidence)},
                   review_only=spec.review_only,
                   distance_statistics={"direction_reversed": spec.direction_reversed},
                   color_mapping={"configured_minimum_nominal_mm": spec.minimum_nominal_mm,
                                  "configured_maximum_nominal_mm": spec.maximum_nominal_mm})
    if spec.target_sha256 is not None: payload["target_mesh"]["sha256"] = spec.target_sha256
    if spec.aligned_sha256 is not None: payload["aligned_sha256"] = spec.aligned_sha256
    if spec.annotations_path is not None: payload["viewer_annotations_path"] = str(spec.annotations_path)
    validate_review_manifest(payload)
    return payload


def validate_review_manifest(payload: dict) -> None:
    if not isinstance(payload, dict): raise ValueError("Review manifest must be an object")
    version = payload.get("schema_version", 1)  # Pre-schema viewer sidecars are legacy v1.
    if not ((type(version) is int and version == VIEWER_MANIFEST_SCHEMA_VERSION)
            or (isinstance(version, str) and version in _LEGACY_RESULT_SCHEMAS)):
        raise ValueError(f"Unsupported viewer manifest schema: {version!r}")
    try:
        paths = [payload["target_mesh"]["path"], payload["outputs"]["aligned_current_stl"]]
        if any(not isinstance(p, str) or not p.strip() for p in paths): raise ValueError("Empty/non-string mesh path")
        registration = payload.get("registration") or {}
        if registration.get("status", "unknown") not in {"success", "warning", "failed", "unknown", None}:
            raise ValueError("Unsupported registration status")
        confidence = registration.get("position_confidence")
        if confidence not in {None, "high", "medium", "low", "failed"}:
            raise ValueError("Unsupported position_confidence")
        warnings = registration.get("warnings", [])
        if not isinstance(warnings, (list, tuple)) or any(not isinstance(w, str) for w in warnings):
            raise ValueError("warnings must be a sequence of strings")
        for value in (payload["target_mesh"].get("sha256"), payload.get("aligned_sha256")):
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)):
                raise ValueError("Invalid SHA-256")
        mapping = payload.get("color_mapping") or {}
        low = float(mapping.get("configured_minimum_nominal_mm", -.05))
        high = float(mapping.get("configured_maximum_nominal_mm", .05))
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("Nominal bounds must be finite and increasing")
        direction = (payload.get("distance_statistics") or {}).get("direction_reversed", False)
        if type(direction) is not bool or type(payload.get("review_only", False)) is not bool:
            raise ValueError("Review flags must be boolean")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed review manifest: {exc}") from exc


def _path(base, value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def _verify(path, expected):
    if not path.is_file(): raise FileNotFoundError(f"找不到查看模型：{path}")
    if expected:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
        if digest.hexdigest().lower() != expected.lower():
            raise ValueError(f"模型 SHA-256 不匹配：{path}")


def _resolve(payload, path, target_override=None, aligned_override=None):
    validate_review_manifest(payload)
    base = path.parent
    target_record = payload["target_mesh"]
    target = _path(base, target_record["path"])
    archived = target_record.get("archived_path")
    if archived and _path(base, archived).is_file(): target = _path(base, archived)
    if target_override is not None: target = Path(target_override).expanduser().resolve()
    aligned = (_path(base, payload["outputs"]["aligned_current_stl"]) if aligned_override is None
               else Path(aligned_override).expanduser().resolve())
    _verify(target, target_record.get("sha256"))
    _verify(aligned, payload.get("aligned_sha256"))
    registration = payload.get("registration") or {}
    mapping = payload.get("color_mapping") or {}
    status = registration.get("status") or "unknown"
    annotations = _path(base, payload.get("viewer_annotations_path", "viewer_annotations.json"))
    if annotations in {target, aligned, path}:
        raise ValueError("Annotation path must not overwrite an input or manifest")
    return RegistrationReviewSpec(target, aligned, status, registration.get("position_confidence"),
        registration.get("confidence"), tuple(registration.get("warnings", [])),
        payload.get("review_only", status == "failed"),
        (payload.get("distance_statistics") or {}).get("direction_reversed", False),
        float(mapping.get("configured_minimum_nominal_mm", -.05)),
        float(mapping.get("configured_maximum_nominal_mm", .05)),
        target_record.get("sha256"), payload.get("aligned_sha256"), annotations, path)


def write_review_manifest(path: str | Path, spec: RegistrationReviewSpec) -> Path:
    path = Path(path).expanduser().resolve()
    payload = build_review_manifest(spec)
    resolved = _resolve(payload, path)
    if path in {resolved.target_path, resolved.aligned_path}:
        raise ValueError("Manifest must not overwrite an input mesh")
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def load_review_manifest(path: str | Path, *, target_override=None, aligned_override=None) -> RegistrationReviewSpec:
    path = Path(path).expanduser().resolve(strict=True)
    return _resolve(json.loads(path.read_text(encoding="utf-8")), path, target_override, aligned_override)


def run_registration_review(spec_or_path: RegistrationReviewSpec | str | Path) -> None:
    """Start a standalone viewer on the main thread; a spec needs manifest_path."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Open3D viewers must start on the main thread")
    if isinstance(spec_or_path, RegistrationReviewSpec):
        if spec_or_path.manifest_path is None:
            raise ValueError("An in-memory review spec requires manifest_path for persistent annotations")
        path = write_review_manifest(spec_or_path.manifest_path, spec_or_path)
    else:
        path = spec_or_path
    from ..result_viewer import run_general_result_viewer
    run_general_result_viewer(path)
