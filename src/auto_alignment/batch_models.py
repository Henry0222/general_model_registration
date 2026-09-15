"""Lightweight batch messages shared by the GUI and numerical worker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegistrationJob:
    index: int
    source_path: Path
    flip_normals: bool = False
    edit_state_path: Path | None = None


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
    review_only: bool = False
    selection_enabled: bool = False
    selection_lane: str | None = None
    symmetric_rms_mm: float | None = None
    p90_mm: float | None = None
    hd95_mm: float | None = None
    refinement_mode: str = "baseline"
    refinement_selected: str = "initial"

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
            "review_only": self.review_only,
            "selection_enabled": self.selection_enabled,
            "selection_lane": self.selection_lane,
            "symmetric_rms_mm": self.symmetric_rms_mm,
            "p90_mm": self.p90_mm,
            "hd95_mm": self.hd95_mm,
            "refinement_mode": self.refinement_mode,
            "refinement_selected": self.refinement_selected,
        }


@dataclass(frozen=True)
class BatchOutcome:
    batch_directory: Path
    manifest_path: Path
    batch_log_path: Path
    items: tuple[BatchItemResult, ...]
    stopped: bool
    total_elapsed_seconds: float
