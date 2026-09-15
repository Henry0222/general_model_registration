"""Rigid registration in mm: target first, source second, target = T @ source."""
from __future__ import annotations
from typing import Callable
import numpy as np

from ..config import AlignmentConfig
from ..mesh_io import MeshValidationError, MeshFacts, load_mesh
from ..quality import RegistrationQualityReport, PositionConfidence
from ..registration import RegistrationMetrics, RegistrationResult
from ..registration import register_meshes as _register_meshes
from ..refinement_modes import RegistrationCancelled

__all__ = ["AlignmentConfig", "MeshValidationError", "MeshFacts", "load_mesh",
           "RegistrationMetrics", "RegistrationResult", "RegistrationQualityReport",
           "PositionConfidence", "RegistrationCancelled", "register_meshes"]


def _mask(value, count, name):
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.ndim != 1 or len(array) != count:
        raise ValueError(f"{name} must be a 1D bool mask of length {count}")
    return array.copy()


def register_meshes(target_mesh, source_mesh, target_facts: MeshFacts,
                    source_facts: MeshFacts, config: AlignmentConfig,
                    progress: Callable[[float, str], None] | None = None, *,
                    target_priority_faces: np.ndarray | None = None,
                    source_priority_faces: np.ndarray | None = None,
                    cancel: Callable[[], bool] | None = None) -> RegistrationResult:
    """Delegate without changing solver/config; validate the public boundary.

    Masks index the meshes returned by load_mesh, before any face deletion.
    Transformation acts on homogeneous COLUMN vectors. Rotation is in degrees.
    success/warning count as succeeded; confidence is display text, quality may
    be None, and quality.position_confidence is the machine-readable enum.
    """
    target_mask = _mask(target_priority_faces, len(target_mesh.triangles), "target_priority_faces")
    source_mask = _mask(source_priority_faces, len(source_mesh.triangles), "source_priority_faces")
    def notify(fraction, message):
        fraction = float(fraction)
        if not np.isfinite(fraction):
            raise ValueError("Nonfinite progress fraction")
        progress(max(0., min(1., fraction)), str(message))
    result = _register_meshes(target_mesh, source_mesh, target_facts, source_facts,
                              config, notify if progress is not None else None,
                              target_priority_faces=target_mask, source_priority_faces=source_mask,
                              **({"cancel": cancel} if cancel is not None else {}))
    if result.status not in {"success", "warning", "failed"}:
        raise ValueError(f"Unsupported registration status: {result.status!r}")
    matrix = np.asarray(result.transformation)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-5)):
        raise ValueError("Solver returned a non-rigid homogeneous transformation")
    return result
