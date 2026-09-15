"""Convert completed benchmark poses to ordinary app review files, without refitting."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from auto_alignment.comparison import compare_meshes
from auto_alignment.exporters import export_results
from auto_alignment.file_io import write_json
from auto_alignment.history import scan_history
from auto_alignment.mesh_io import load_mesh
from auto_alignment.quality import CandidateDiagnostic, PositionConfidence, RegistrationQualityReport
from auto_alignment.registration import RegistrationMetrics, RegistrationResult


def restore_registration(row):
    metrics = dict(row["metrics"])
    metrics["candidate_diagnostics"] = tuple(
        CandidateDiagnostic(**{**item, "transformation": np.asarray(item["transformation"])})
        for item in metrics.get("candidate_diagnostics", ())
    )
    quality = row.get("quality")
    if quality:
        quality = {field.name: quality[field.name] for field in fields(RegistrationQualityReport)}
        quality["position_confidence"] = PositionConfidence(quality["position_confidence"])
        quality = RegistrationQualityReport(**quality)
    return RegistrationResult(np.asarray(row["transformation"]), row["status"], row["confidence"],
                              RegistrationMetrics(**metrics), tuple(row["warnings"]),
                              row["registration_seconds"], quality)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    targets, checks = {}, []
    for path in sorted(args.input.glob("[0-9][0-9]/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["status"] == "error":
            raise RuntimeError(f"Cannot export failed execution: {row['id']}")
        if row["target"] not in targets:
            targets[row["target"]] = load_mesh(row["target"])
        target, target_facts = targets[row["target"]]
        source, source_facts = load_mesh(row["source"])
        registration = restore_registration(row)
        comparison = compare_meshes(target, source, registration.transformation, 20000, 1.0, .05)
        directory = args.output/path.parent.name
        outputs = export_results(directory, target_facts, source_facts, registration, comparison,
                                 row["total_seconds"], target_archived_path=row["target"])
        write_json(directory/"offline_evaluation.json", {key: row[key] for key in (
            "id", "reference_kind", "pose", "reference_matrix", "oracle_candidate_for_diagnosis_only",
            "matrix_pose_error", "imposed_parameter_comparison"
        ) if key in row})
        checks.append(dict(id=row["id"], status=row["status"],
                           results_json=str(outputs["results_json"].resolve()),
                           distances=comparison.statistics.as_dict()))
        assert all(file.is_file() and file.stat().st_size for file in outputs.values())
        print("EXPORTED", path.parent.name, row["id"], flush=True)
    records = scan_history(args.output)
    assert len(records) == len(checks), (len(records), len(checks))
    write_json(args.output/"review_index.json", dict(records=checks, verified_history_count=len(records),
               note="Exported from completed benchmark matrices without re-registration. Fixed-model paths refer to original local input data."))
    print("VERIFIED", len(records), "history records", flush=True)


if __name__ == "__main__":
    main()
