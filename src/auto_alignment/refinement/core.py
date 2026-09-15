"""Geometry-only entry point: no paths, labels, or experiment metadata."""
from __future__ import annotations

import time
import numpy as np

from ..candidate_selection import rigid_valid
from . import _router


def refine_meshes(source_mesh, target_mesh, initial, mode, checkpoint=None):
    if mode not in {"A", "B", "compare", "auto"}:
        raise ValueError(f"Unknown refinement mode: {mode!r}")
    if not rigid_valid(initial):
        raise ValueError("Refinement requires a finite rigid initial transform")
    started = time.perf_counter()
    if checkpoint:
        checkpoint(0., "正在准备几何精修…")
    if mode in {"compare", "auto"}:
        result, _ = _router.run_meshes(
            source_mesh, target_mesh, initial, full=mode == "compare", checkpoint=checkpoint,
        )
        selected = "initial" if mode == "compare" else result["route"]
    else:
        ctx = _router.Context(source_mesh, target_mesh, initial)
        result = {"matrices": {"initial": np.asarray(initial)}, "timings": {}}
        if mode == "A":
            _, gate = ctx.witness(ctx.T0)
            transform, meta, _ = ctx.a(gate, checkpoint=checkpoint)
            result.update(A=meta, gates={"initial": gate})
        else:
            transform, meta, _ = ctx.b(checkpoint=checkpoint)
            result["B"] = meta
        result["matrices"][mode] = ctx.Ci @ transform @ ctx.C
        selected = mode
        result["reason"] = "explicit_mode"
    matrices = {name: np.asarray(matrix).copy() for name, matrix in result["matrices"].items()
                if name in {"initial", "A", "B"}}
    if not all(rigid_valid(matrix) for matrix in matrices.values()):
        raise ValueError("Refinement produced a non-rigid or nonfinite transform")
    result = {key: value for key, value in result.items() if key != "matrices"}
    result.update(mode=mode, selected=selected, elapsed_seconds=time.perf_counter() - started,
                  experimental=mode == "auto", rigid_output=True)
    if checkpoint:
        checkpoint(1., "几何精修完成，正在复核候选…")
    return matrices, result
