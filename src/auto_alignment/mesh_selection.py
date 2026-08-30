"""Persistent, non-destructive triangle selection and mesh-edit helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

from .mesh_io import MeshFacts, clone_mesh


EDIT_STATE_SCHEMA = "1.4.1"


def mesh_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_edit_state_path(mesh_path: str | Path) -> Path:
    """Return a stable per-path cache file without writing beside the input STL."""
    local = Path(os.environ.get("LOCALAPPDATA", "")).expanduser()
    if not str(local) or not local.is_absolute():
        local = Path.home() / "AppData" / "Local"
    normalized = os.path.normcase(str(Path(mesh_path).expanduser().resolve()))
    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return local / "GeneralModelRegistration" / "model_edits" / f"{key}.json"


def _encode_ranges(indices: Iterable[int]) -> list[list[int]]:
    values = np.unique(np.fromiter((int(value) for value in indices), dtype=np.int64))
    if not len(values):
        return []
    starts = np.r_[True, np.diff(values) != 1]
    ends = np.r_[np.diff(values) != 1, True]
    return [
        [int(first), int(last)]
        for first, last in zip(values[starts], values[ends])
    ]


def _decode_ranges(value: object, triangle_count: int) -> np.ndarray:
    mask = np.zeros(max(0, int(triangle_count)), dtype=bool)
    if not isinstance(value, list):
        return mask
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            first, last = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        first = max(0, first)
        last = min(len(mask) - 1, last)
        if first <= last:
            mask[first : last + 1] = True
    return mask


@dataclass(frozen=True)
class ModelEditState:
    mesh_path: str
    mesh_sha256: str
    triangle_count: int
    selected: np.ndarray
    deleted: np.ndarray
    updated_at: str

    @classmethod
    def empty(cls, mesh_path: str | Path, triangle_count: int) -> "ModelEditState":
        resolved = Path(mesh_path).expanduser().resolve(strict=True)
        count = int(triangle_count)
        return cls(
            str(resolved),
            mesh_file_sha256(resolved),
            count,
            np.zeros(count, dtype=bool),
            np.zeros(count, dtype=bool),
            datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    def normalized(self) -> "ModelEditState":
        count = max(0, int(self.triangle_count))
        selected = np.asarray(self.selected, dtype=bool).reshape(-1)
        deleted = np.asarray(self.deleted, dtype=bool).reshape(-1)
        if len(selected) != count or len(deleted) != count:
            raise ValueError("模型编辑状态与三角面数量不匹配。")
        selected = selected.copy()
        deleted = deleted.copy()
        selected[deleted] = False
        return replace(self, selected=selected, deleted=deleted)

    def as_dict(self) -> dict[str, object]:
        state = self.normalized()
        return {
            "schema_version": EDIT_STATE_SCHEMA,
            "mesh_path": state.mesh_path,
            "mesh_sha256": state.mesh_sha256,
            "triangle_count": state.triangle_count,
            "selected_ranges": _encode_ranges(np.flatnonzero(state.selected)),
            "deleted_ranges": _encode_ranges(np.flatnonzero(state.deleted)),
            "selected_count": int(np.count_nonzero(state.selected)),
            "deleted_count": int(np.count_nonzero(state.deleted)),
            "updated_at": state.updated_at,
        }


def load_edit_state(
    state_path: str | Path,
    mesh_path: str | Path,
    triangle_count: int,
) -> ModelEditState:
    """Load a compatible state; return a clean state if the STL changed."""
    empty = ModelEditState.empty(mesh_path, triangle_count)
    path = Path(state_path)
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("schema_version")) != EDIT_STATE_SCHEMA:
            return empty
        if str(payload.get("mesh_sha256")) != empty.mesh_sha256:
            return empty
        if int(payload.get("triangle_count", -1)) != empty.triangle_count:
            return empty
        return ModelEditState(
            empty.mesh_path,
            empty.mesh_sha256,
            empty.triangle_count,
            _decode_ranges(payload.get("selected_ranges"), empty.triangle_count),
            _decode_ranges(payload.get("deleted_ranges"), empty.triangle_count),
            str(payload.get("updated_at") or empty.updated_at),
        ).normalized()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return empty


def save_edit_state(path: str | Path, state: ModelEditState) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = replace(
        state.normalized(),
        updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    temporary = destination.parent / f".{destination.name}.tmp"
    temporary.write_text(
        json.dumps(normalized.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


@dataclass(frozen=True)
class AppliedModelEdits:
    mesh: o3d.geometry.TriangleMesh
    selected_faces: np.ndarray
    original_face_indices: np.ndarray
    selected_count: int
    deleted_count: int


def apply_edit_state(
    mesh: o3d.geometry.TriangleMesh,
    state: ModelEditState,
) -> AppliedModelEdits:
    state = state.normalized()
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) != state.triangle_count:
        raise ValueError("模型编辑状态与当前 STL 不匹配。")
    kept = np.flatnonzero(~state.deleted)
    if len(kept) < 4:
        raise ValueError("删除后有效三角面过少，无法用于配准。")
    edited = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float).copy()),
        o3d.utility.Vector3iVector(triangles[kept].copy()),
    )
    edited.remove_unreferenced_vertices()
    edited.compute_triangle_normals()
    edited.compute_vertex_normals()
    selected = np.asarray(state.selected[kept], dtype=bool)
    return AppliedModelEdits(
        edited,
        selected,
        kept,
        int(np.count_nonzero(selected)),
        int(np.count_nonzero(state.deleted)),
    )


def updated_mesh_facts(
    facts: MeshFacts,
    mesh: o3d.geometry.TriangleMesh,
    *,
    edit_note: str | None = None,
) -> MeshFacts:
    bounds = mesh.get_axis_aligned_bounding_box()
    diagonal = float(np.linalg.norm(np.asarray(bounds.get_extent(), dtype=float)))
    warnings = list(facts.warnings)
    if edit_note:
        warnings.append(edit_note)
    return replace(
        facts,
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        diagonal_mm=diagonal,
        bounds_min=tuple(float(value) for value in bounds.min_bound),
        bounds_max=tuple(float(value) for value in bounds.max_bound),
        warnings=tuple(warnings),
    )


def _component_labels(
    mesh: o3d.geometry.TriangleMesh,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Connected faces across ordinary two-face edges; stop at mesh boundaries."""
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    available = np.asarray(available, dtype=bool).reshape(-1)
    if len(available) != len(triangles):
        raise ValueError("三角面可用掩码长度不正确。")
    active = np.flatnonzero(available)
    labels = np.full(len(triangles), -1, dtype=np.int64)
    if not len(active):
        return labels, np.empty(0, dtype=float)

    active_triangles = triangles[active]
    edge_corners = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    edges = np.sort(active_triangles[:, edge_corners], axis=2).reshape((-1, 2))
    owners = np.repeat(np.arange(len(active), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    parent = np.arange(len(active), dtype=np.int64)

    def find(value: int) -> int:
        root = int(value)
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    starts = np.r_[0, 1 + np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1))]
    ends = np.r_[starts[1:], len(edges)]
    for first, last in zip(starts, ends):
        # Exactly two incident faces is an ordinary traversable mesh edge.
        if last - first == 2:
            union(int(owners[first]), int(owners[first + 1]))

    roots = np.fromiter((find(index) for index in range(len(active))), dtype=np.int64)
    _, local_labels = np.unique(roots, return_inverse=True)
    labels[active] = local_labels

    vertices = np.asarray(mesh.vertices, dtype=float)
    corners = vertices[active_triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    component_areas = np.bincount(
        local_labels,
        weights=np.where(np.isfinite(areas), areas, 0.0),
    )
    return labels, component_areas


def bounded_component_faces(
    mesh: o3d.geometry.TriangleMesh,
    selected: np.ndarray,
    deleted: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    deleted = np.asarray(deleted, dtype=bool).reshape(-1)
    available = ~deleted
    labels, _ = _component_labels(mesh, available)
    seed_labels = np.unique(labels[selected & available])
    seed_labels = seed_labels[seed_labels >= 0]
    return np.isin(labels, seed_labels) & available


def floating_component_faces(
    mesh: o3d.geometry.TriangleMesh,
    deleted: np.ndarray,
) -> np.ndarray:
    """Return every disconnected component except the largest by surface area."""
    deleted = np.asarray(deleted, dtype=bool).reshape(-1)
    available = ~deleted
    labels, component_areas = _component_labels(mesh, available)
    remove = np.zeros(len(deleted), dtype=bool)
    if len(component_areas) <= 1:
        return remove
    main = int(np.argmax(component_areas))
    remove = available & (labels >= 0) & (labels != main)
    return remove


def clone_state_with_masks(
    state: ModelEditState,
    selected: np.ndarray,
    deleted: np.ndarray,
) -> ModelEditState:
    return replace(
        state,
        selected=np.asarray(selected, dtype=bool).copy(),
        deleted=np.asarray(deleted, dtype=bool).copy(),
    ).normalized()
