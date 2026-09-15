"""Headless multi-region state and a lazily imported Open3D selection widget."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence
import numpy as np

from ..file_io import atomic_write_text
from ..mesh_selection import (ModelEditState, clone_state_with_masks, bounded_component_faces,
                              mesh_file_sha256, _encode_ranges, _decode_ranges)

REGION_SELECTION_SCHEMA_VERSION = 1
__all__ = ["SelectionRegionSpec", "RegionSelectionSnapshot", "RegionSelectionSession",
           "MultiRegionSelectionViewer", "run_multi_region_selection_viewer", "ModelEditState",
           "clone_state_with_masks", "encode_face_ranges", "decode_face_ranges", "bounded_component_faces",
           "REGION_SELECTION_SCHEMA_VERSION"]


def encode_face_ranges(indices) -> list[list[int]]:
    """Encode face INDICES, not a boolean mask; delegates to the legacy codec."""
    values = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices)
    if values.ndim != 1 or (values.size and (values.dtype.kind not in "iu" or np.any(values < 0))):
        raise ValueError("Face indices must be a 1D sequence of nonnegative integers")
    return _encode_ranges(values)


def decode_face_ranges(ranges, triangle_count: int) -> np.ndarray:
    """Decode inclusive ranges to a fresh bool mask; malformed ranges fail."""
    if type(triangle_count) is not int or triangle_count < 0 or not isinstance(ranges, list):
        raise ValueError("Invalid face ranges or triangle_count")
    for item in ranges:
        if (not isinstance(item, list) or len(item) != 2 or any(type(i) is not int for i in item)
                or not 0 <= item[0] <= item[1] < triangle_count):
            raise ValueError("Face range outside the baseline mesh")
    return _decode_ranges(ranges, triangle_count)


@dataclass(frozen=True)
class SelectionRegionSpec:
    key: str
    label: str
    color: tuple[float, float, float]


@dataclass(frozen=True)
class RegionSelectionSnapshot:
    mesh_path: str
    mesh_sha256: str
    triangle_count: int
    active_region: str
    masks: dict[str, np.ndarray]


class RegionSelectionSession:
    """Every edit/switch/undo/redo commits atomically, then emits on_change.

    Saves write atomically, then emit on_save. Callback errors propagate; the
    committed in-memory state remains available and dirty. Callbacks cannot
    reenter mutation methods. Every outward snapshot owns copies of all masks.
    """
    def __init__(self, mesh_path: str | Path, triangle_count: int,
                 regions: Sequence[SelectionRegionSpec], *, initial_masks: Mapping[str, np.ndarray] | None = None,
                 active_region: str | None = None, allow_overlap: bool = False,
                 on_change: Callable[[RegionSelectionSnapshot], None] | None = None,
                 on_save: Callable[[RegionSelectionSnapshot], None] | None = None,
                 mesh_sha256: str | None = None):
        self.mesh_path = str(Path(mesh_path).expanduser().resolve(strict=True))
        digest = mesh_file_sha256(self.mesh_path) if mesh_sha256 is None else str(mesh_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError("mesh_sha256 must be a 64-character hexadecimal digest")
        self.mesh_sha256 = digest.lower()
        if type(triangle_count) is not int or triangle_count <= 0:
            raise ValueError("triangle_count must be positive")
        self.triangle_count = triangle_count
        self.regions = tuple(regions)
        keys = [r.key for r in self.regions]
        if not keys or len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys):
            raise ValueError("At least one uniquely named region is required")
        for region in self.regions:
            color = np.asarray(region.color, dtype=float)
            if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
                raise ValueError("Region colors must contain three values in [0, 1]")
        self.allow_overlap = bool(allow_overlap)
        self._active = active_region or keys[0]
        if self._active not in keys: raise ValueError("Unknown active region")
        initial_masks = {} if initial_masks is None else initial_masks
        if set(initial_masks) - set(keys): raise ValueError("Unknown initial region")
        self._masks = {k: self._mask(initial_masks.get(k, np.zeros(triangle_count, bool))) for k in keys}
        self._validate_overlap(self._masks)
        self._undo, self._redo = [], []
        self._on_change, self._on_save = on_change, on_save
        self._notifying = False
        self.dirty = True

    def _guard(self):
        if self._notifying: raise RuntimeError("Selection callbacks cannot reenter a state mutation")

    def _mask(self, mask):
        value = np.asarray(mask)
        if value.dtype != np.bool_ or value.shape != (self.triangle_count,):
            raise ValueError(f"Region mask must be a 1D bool array of length {self.triangle_count}")
        return value.copy()

    def _validate_overlap(self, masks):
        if not self.allow_overlap and np.any(np.sum(np.stack(list(masks.values())), axis=0) > 1):
            raise ValueError("选区与其他区域重叠：整个操作未应用")

    def snapshot(self) -> RegionSelectionSnapshot:
        return RegionSelectionSnapshot(self.mesh_path, self.mesh_sha256, self.triangle_count,
                                       self._active, {k: v.copy() for k, v in self._masks.items()})

    @property
    def can_undo(self): return bool(self._undo)

    @property
    def can_redo(self): return bool(self._redo)

    def _emit(self, callback):
        if callback is None: return
        self._notifying = True
        try:
            callback(self.snapshot())
        except Exception:
            self.dirty = True
            raise
        finally:
            self._notifying = False

    def _commit(self, masks, active):
        self._guard()
        self._validate_overlap(masks)
        if active == self._active and all(np.array_equal(masks[k], self._masks[k]) for k in masks):
            return
        self._undo.append(self.snapshot())
        self._undo = self._undo[-100:]
        self._redo.clear()
        self._masks, self._active = masks, active
        self.dirty = True
        self._emit(self._on_change)

    def set_mask(self, region: str, mask: np.ndarray) -> None:
        if region not in self._masks: raise ValueError("Unknown region")
        masks = {k: v.copy() for k, v in self._masks.items()}
        masks[region] = self._mask(mask)
        self._commit(masks, self._active)

    def select_region(self, region: str) -> None:
        if region not in self._masks: raise ValueError("Unknown region")
        self._commit({k: v.copy() for k, v in self._masks.items()}, region)

    def _restore(self, snap):
        self._active = snap.active_region
        self._masks = {k: v.copy() for k, v in snap.masks.items()}
        self.dirty = True
        self._emit(self._on_change)

    def undo(self) -> None:
        self._guard()
        if self._undo:
            self._redo.append(self.snapshot())
            self._restore(self._undo.pop())

    def redo(self) -> None:
        self._guard()
        if self._redo:
            self._undo.append(self.snapshot())
            self._restore(self._redo.pop())

    def save(self, path: str | Path) -> Path:
        self._guard()
        path = Path(path).expanduser().resolve()
        if path == Path(self.mesh_path): raise ValueError("Selection state must not overwrite the source mesh")
        if mesh_file_sha256(self.mesh_path) != self.mesh_sha256:
            raise ValueError("Source mesh changed; selection state retained in memory")
        payload = dict(schema_version=REGION_SELECTION_SCHEMA_VERSION, mesh_path=self.mesh_path,
                       mesh_sha256=self.mesh_sha256, triangle_count=self.triangle_count,
                       indexing="load_mesh baseline triangles before edit deletions",
                       active_region=self._active, allow_overlap=self.allow_overlap,
                       masks={k: encode_face_ranges(np.flatnonzero(v)) for k, v in self._masks.items()})
        try:
            atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
            self._emit(self._on_save)
        except Exception:
            self.dirty = True
            raise
        self.dirty = False
        return path

    def load(self, path: str | Path) -> None:
        """Restore matching persisted state; reject stale data without changes."""
        self._guard()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != REGION_SELECTION_SCHEMA_VERSION:
            raise ValueError("Unsupported region selection schema")
        if payload.get("mesh_sha256") != self.mesh_sha256 or payload.get("triangle_count") != self.triangle_count:
            raise ValueError("Selection belongs to a different mesh")
        if set(payload["masks"]) != set(self._masks) or payload["active_region"] not in self._masks:
            raise ValueError("Saved region definitions do not match")
        masks = {k: decode_face_ranges(v, self.triangle_count) for k, v in payload["masks"].items()}
        self._validate_overlap(masks)
        self._masks, self._active = masks, payload["active_region"]
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        self._emit(self._on_change)


def __getattr__(name):
    if name in {"MultiRegionSelectionViewer", "run_multi_region_selection_viewer"}:
        from . import _selection_viewer
        return getattr(_selection_viewer, name)
    raise AttributeError(name)
