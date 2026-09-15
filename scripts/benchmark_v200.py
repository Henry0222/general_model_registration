"""Run unmarked registration; use reference matrices only AFTER inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import numpy as np
import open3d as o3d
from auto_alignment.config import AlignmentConfig
from auto_alignment.mesh_io import load_mesh, sample_registration_cloud
from auto_alignment.registration import register_meshes
try:
    from auto_alignment.pose_evaluation import matrix_pose_error, imposed_parameter_comparison
except ModuleNotFoundError as error:
    if error.name != "auto_alignment.pose_evaluation":
        raise
    # A frozen pre-2.0 source snapshot has no new offline evaluator. Load only
    # the reference-metric helper from this checkout, never its newer solver.
    import importlib.util
    metric_path = Path(__file__).resolve().parents[1]/"src/auto_alignment/pose_evaluation.py"
    metric_spec = importlib.util.spec_from_file_location("offline_pose_metrics", metric_path)
    metric_module = importlib.util.module_from_spec(metric_spec)
    metric_spec.loader.exec_module(metric_module)
    matrix_pose_error = metric_module.matrix_pose_error
    imposed_parameter_comparison = metric_module.imposed_parameter_comparison
from auto_alignment.version import __version__


def cases(root):
    result = []
    for folder, translation, angles in (
        ("0829TEST_SQUARE", (30, -40, 15), (8, 168, 285)),
        ("0903Test-square", (100, -60, 500), (58, 47, 268)),
    ):
        base = root/folder
        for source in sorted(base.glob("*.stl")):
            if source.name == "立方体原位.stl":
                continue
            result.append(dict(id=folder+"/"+source.stem, target=base/"立方体原位.stl",
                               source=source, gold=(translation, angles), reference_kind="known_rigid_motion"))
    base = root/"90°和 100μm层厚（无金标准）"
    for source in sorted(base.glob("*.stl")):
        if source.name != "牙冠CAD设计.stl":
            result.append(dict(id="crown_scans/"+source.stem, target=base/"牙冠CAD设计.stl",
                               source=source, gold=None, reference_kind="no_ground_truth"))
    base = root/"拆冠数据（无金标准）"
    result.append(dict(id="crown_removal", target=base/"理想位置.stl", source=base/"实际位置.stl",
                       gold=None, reference_kind="no_ground_truth"))
    base = root/"咬合调整数据"
    result.append(dict(id="bite_adjustment", target=base/"李主任16冠.stl", source=base/"李主任16冠调冠前.stl",
                       gold=base/"金标准/01_李主任16冠调冠前/transform.json", reference_kind="provided_registration_reference"))
    return result


def reference_matrix(gold):
    if gold is None:
        return None
    if isinstance(gold, Path):
        return np.asarray(json.loads(gold.read_text(encoding="utf-8-sig"))["transformation_current_to_target"])
    translation, angles = gold
    matrix = np.eye(4)
    matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(angles)).T
    matrix[:3, 3] = -np.asarray(translation)
    return matrix


def pose_error(transform, reference, points):
    delta = np.asarray(transform) @ np.linalg.inv(reference)
    distances = np.linalg.norm(points @ delta[:3, :3].T + delta[:3, 3] - points, axis=1)
    return dict(pose_rms_mm=float(np.sqrt(np.mean(distances**2))), pose_p95_mm=float(np.quantile(distances, .95)),
                rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--algorithm")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = AlignmentConfig(**({"algorithm_version": args.algorithm} if args.algorithm else {}))
    targets = {}
    rows = []
    for index, case in enumerate(cases(args.data.resolve())):
        if args.filter and args.filter not in case["id"]:
            continue
        directory = args.output / f"{index+1:02d}"
        directory.mkdir(exist_ok=True)
        output = directory/"result.json"
        if args.resume and output.exists():
            rows.append(json.loads(output.read_text(encoding="utf-8")))
            continue
        print("START", index+1, case["id"], flush=True)
        started = time.perf_counter()
        try:
            if case["target"] not in targets:
                targets[case["target"]] = load_mesh(case["target"])
            target, target_facts = targets[case["target"]]
            source, source_facts = load_mesh(case["source"])
            loaded = time.perf_counter()
            print("LOADED", index+1, f"{loaded-started:.2f}s", flush=True)
            last = [time.perf_counter()]
            def progress(fraction, message):
                if time.perf_counter()-last[0] > 35:
                    print("PROGRESS", index+1, round(fraction, 2), message, flush=True)
                    last[0] = time.perf_counter()
            # This call receives geometry and configuration only. No reference or annotations.
            registration = register_meshes(target, source, target_facts, source_facts, cfg, progress)
            row = dict(id=case["id"], version=__version__, source=str(case["source"]), target=str(case["target"]),
                       reference_kind=case["reference_kind"], status=registration.status, confidence=registration.confidence,
                       load_seconds=loaded-started, registration_seconds=registration.elapsed_seconds,
                       transformation=registration.transformation.tolist(), warnings=list(registration.warnings),
                       metrics=registration.metrics.as_dict(), quality=registration.quality.as_dict() if registration.quality else None)
            # Only now may reference data enter evaluation.
            reference = reference_matrix(case["gold"])
            if reference is not None:
                row["matrix_pose_error"] = matrix_pose_error(registration.transformation, reference)
                if isinstance(case["gold"], tuple):
                    row["imposed_parameter_comparison"] = imposed_parameter_comparison(registration.transformation, *case["gold"])
                points = np.asarray(sample_registration_cloud(target, 20000, seed=20260905).points)
                row["pose"] = pose_error(registration.transformation, reference, points)
                candidates = [{"name": c.name, **pose_error(c.transformation, reference, points)}
                              for c in registration.metrics.candidate_diagnostics]
                row["candidate_pose_errors"] = candidates
                row["oracle_candidate_for_diagnosis_only"] = min(candidates, key=lambda c: c["pose_p95_mm"]) if candidates else None
                row["reference_matrix"] = reference.tolist()
            aligned = o3d.geometry.TriangleMesh(source)
            aligned.transform(registration.transformation)
            aligned.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(directory/"aligned.stl"), aligned)
        except Exception:
            row = dict(id=case["id"], status="error", error=traceback.format_exc())
        row["total_seconds"] = time.perf_counter()-started
        output.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=lambda x: x.tolist()), encoding="utf-8")
        rows.append(row)
        print("DONE", json.dumps({k: row[k] for k in ("id", "status", "pose", "total_seconds") if k in row}, ensure_ascii=False), flush=True)
        (args.output/"summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2,
                                                           default=lambda x: x.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
