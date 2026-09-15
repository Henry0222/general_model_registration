"""Synthetic registration benchmark with known ground truth.

Takes one real scan (any STL), applies a known rigid transform, and then
corrupts the moved copy with the three kinds of change that break rigid
registration in dental work:

* ``local``   - one large-amplitude local change (a region pushed outward).
* ``gingiva`` - broad, low-amplitude normal offset over a wide band, like
                soft-tissue swelling between two scans.
* ``scale``   - uniform scaling, like resin shrinkage on a print.

Noise is added to both surfaces.  Because the transform is known, the pose
error of any registration can be measured directly and compared across
algorithm variants.  Run with ``--help`` for options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_alignment.config import AlignmentConfig  # noqa: E402
from auto_alignment.mesh_io import MeshFacts, load_mesh  # noqa: E402
from auto_alignment.registration import register_meshes  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    local_mm: float
    gingiva_mm: float
    gingiva_fraction: float
    scale: float
    noise_mm: float


CASES = (
    Case("clean", 0.0, 0.0, 0.0, 1.0, 0.02),
    Case("local_large", 1.5, 0.0, 0.0, 1.0, 0.02),
    Case("gingiva_broad_small", 0.0, 0.15, 0.35, 1.0, 0.02),
    Case("scale_shrink", 0.0, 0.0, 0.0, 0.999, 0.02),
    Case("mixed", 1.0, 0.12, 0.30, 0.9995, 0.03),
)


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    k = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _facts(mesh: o3d.geometry.TriangleMesh, name: str) -> MeshFacts:
    bounds = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent(), dtype=float)
    return MeshFacts(
        path=name,
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        diagonal_mm=float(np.linalg.norm(extent)),
        bounds_min=tuple(float(v) for v in bounds.min_bound),
        bounds_max=tuple(float(v) for v in bounds.max_bound),
        warnings=(),
    )


def corrupt(mesh: o3d.geometry.TriangleMesh, case: Case, rng: np.random.Generator) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Return a corrupted copy and a per-vertex mask of truly changed vertices."""
    out = o3d.geometry.TriangleMesh(mesh)
    out.compute_vertex_normals()
    vertices = np.asarray(out.vertices, dtype=float).copy()
    normals = np.asarray(out.vertex_normals, dtype=float)
    changed = np.zeros(len(vertices), dtype=bool)
    centre = vertices.mean(axis=0)
    extent = vertices.max(axis=0) - vertices.min(axis=0)

    if case.local_mm > 0:
        # A blob of radius ~12% of the model extent, pushed out along normals
        # with a smooth falloff: a crown-sized change.
        seed = vertices[rng.integers(len(vertices))]
        radius = 0.12 * float(np.max(extent))
        distance = np.linalg.norm(vertices - seed, axis=1)
        weight = np.clip(1.0 - distance / radius, 0.0, 1.0) ** 2
        vertices += normals * (case.local_mm * weight)[:, None]
        changed |= weight > 0.05

    if case.gingiva_mm > 0:
        # A broad band over the lowest fraction of the model (by the axis
        # with the smallest extent, approximating "towards the gingiva").
        axis = int(np.argmin(extent))
        threshold = np.quantile(vertices[:, axis], case.gingiva_fraction)
        band = vertices[:, axis] <= threshold
        # Smooth ramp so the change has no sharp edge.
        depth = np.clip((threshold - vertices[:, axis]) / (0.25 * extent[axis] + 1e-9), 0.0, 1.0)
        vertices += normals * (case.gingiva_mm * depth * band)[:, None]
        changed |= band & (depth > 0.2)

    if case.scale != 1.0:
        vertices = centre + (vertices - centre) * case.scale
        # Scaling changes everything slightly; mark nothing as "changed"
        # because this is exactly the case where nothing is locally wrong.

    if case.noise_mm > 0:
        vertices += rng.normal(0.0, case.noise_mm, size=vertices.shape)

    out.vertices = o3d.utility.Vector3dVector(vertices)
    out.compute_vertex_normals()
    return out, changed


def pose_error(estimate: np.ndarray, truth: np.ndarray, points: np.ndarray) -> dict[str, float]:
    # points are in the target frame; compare both poses in that frame.
    relative = estimate @ np.linalg.inv(truth)
    cosine = float(np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    moved = points @ relative[:3, :3].T + relative[:3, 3]
    displacement = np.linalg.norm(moved - points, axis=1)
    return {
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
        "translation_mm": float(np.linalg.norm(relative[:3, 3])),
        "surface_rms_mm": float(np.sqrt(np.mean(np.square(displacement)))),
        "surface_max_mm": float(np.max(displacement)),
    }


def run(mesh_path: Path, cases: tuple[Case, ...], config: AlignmentConfig, seed: int, repeats: int) -> list[dict[str, object]]:
    base, _ = load_mesh(mesh_path)
    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []
    for case in cases:
        for repeat in range(repeats):
            target = o3d.geometry.TriangleMesh(base)
            tv = np.asarray(target.vertices, dtype=float).copy()
            tv += rng.normal(0.0, case.noise_mm, size=tv.shape)
            target.vertices = o3d.utility.Vector3dVector(tv)
            target.compute_vertex_normals()

            moving, changed = corrupt(base, case, rng)
            truth = np.eye(4)
            truth[:3, :3] = _rotation(rng.normal(size=3), float(rng.uniform(8.0, 25.0)))
            truth[:3, 3] = rng.uniform(-6.0, 6.0, size=3)
            # truth maps moving -> target frame; so place moving at inv(truth)
            moving.transform(np.linalg.inv(truth))

            started = time.perf_counter()
            result = register_meshes(target, moving, _facts(target, "target"), _facts(moving, "moving"), config)
            elapsed = time.perf_counter() - started
            stable_points = np.asarray(base.vertices, dtype=float)[~changed]
            error = pose_error(result.transformation, truth, stable_points)
            decision = result.metrics.high_precision_decision or {}
            results.append(
                {
                    "case": case.name,
                    "repeat": repeat,
                    "status": result.status,
                    "confidence": result.confidence,
                    "hp_accepted": decision.get("accepted"),
                    "hp_stage": decision.get("selected_stage"),
                    "elapsed_s": round(elapsed, 2),
                    **{k: round(v, 6) for k, v in error.items()},
                }
            )
            print(
                f"{case.name:22s} r{repeat} {result.status:8s} conf={result.confidence} "
                f"hp={decision.get('accepted')} rot={error['rotation_deg']:.4f}deg "
                f"trans={error['translation_mm']:.4f}mm surfRMS={error['surface_rms_mm']:.4f}mm "
                f"max={error['surface_max_mm']:.4f}mm {elapsed:.1f}s",
                flush=True,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--restarts", type=int, default=None)
    args = parser.parse_args()
    cases = tuple(c for c in CASES if args.cases is None or c.name in args.cases)
    config = AlignmentConfig() if args.restarts is None else AlignmentConfig(global_registration_restarts=args.restarts)
    results = run(args.mesh, cases, config, args.seed, args.repeats)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
