from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import open3d as o3d

from auto_alignment.candidate_selection import CandidateBank, FixedEvaluator, MeshCache, observability
from auto_alignment.registration import _principal_axis_candidates
from auto_alignment.stable_region import estimate_noise_sigma


def shape():
    mesh = o3d.geometry.TriangleMesh.create_box(4, 6, 9)
    sphere = o3d.geometry.TriangleMesh.create_sphere(2, resolution=10)
    sphere.translate((5, 1, 3))
    mesh += sphere
    mesh.compute_vertex_normals()
    return mesh


def test_sampling_keeps_triangle_identity_and_barycentric_coordinates():
    mesh = shape()
    cache = MeshCache(mesh)
    samples = cache.sample(2500, 71)
    corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[samples.triangle_ids]]
    reconstructed = np.einsum("ij,ijk->ik", samples.barycentric, corners)
    assert np.allclose(reconstructed, samples.points)
    assert np.allclose(samples.barycentric.sum(axis=1), 1)
    assert cache.sample(2500, 71) is samples


def test_bad_refinement_remains_diagnostic_but_cannot_replace_parent():
    cache = MeshCache(shape())
    evaluator = FixedEvaluator(cache, cache, 1500, 17)
    bank = CandidateBank(evaluator)
    parent = bank.add("correct", "global", np.eye(4))
    bad = np.eye(4)
    bad[:3, 3] = (0.4, 0.2, -0.3)
    child = bank.add("drifted", "fine", bad, parent, guarded=True)
    assert not child.accepted
    assert bank.best() is parent
    assert len(bank.candidates) == 2
    assert child.parent == parent.name
    assert child.evaluation["fixed_denominators"] == parent.evaluation["fixed_denominators"]
    assert child.evaluation["directions"][0]["matched_count"] < parent.evaluation["directions"][0]["matched_count"]


def test_same_pose_with_different_origins_has_same_evaluation():
    mesh = shape()
    cache = MeshCache(mesh)
    transform = np.eye(4)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([.003, -.002, .004])
    transform[:3, 3] = [.02, -.01, .03]
    before = FixedEvaluator(cache, cache, 2200, 17).evaluate(transform)
    shift = np.array([10000., -50000., 8000.])
    shifted = o3d.geometry.TriangleMesh(mesh).translate(shift)
    moved_cache = MeshCache(shifted)
    conjugated = transform.copy()
    conjugated[:3, 3] += shift-transform[:3, :3]@shift
    after = FixedEvaluator(moved_cache, moved_cache, 2200, 17).evaluate(conjugated)
    assert abs(after["score"]-before["score"]) < 1e-5
    assert np.allclose(after["scales_mm"], before["scales_mm"])


def test_flat_surface_does_not_fully_constrain_rigid_pose():
    rng = np.random.default_rng(19)
    points = rng.normal(size=(300, 3))
    points[:, 2] = 0
    normals = np.tile([0., 0., 1.], (len(points), 1))
    rank, _ = observability(points, normals, np.ones(len(points)))
    assert rank == 3


def test_distributed_exact_component_is_explicitly_reported():
    cache = MeshCache(shape())
    result = FixedEvaluator(cache, cache, 1500, 29).evaluate(np.eye(4))
    assert all(d["precise_consensus"] for d in result["directions"])
    assert all(d["precision_rank"] == 6 for d in result["directions"])


def test_noise_estimator_never_reintroduces_excluded_points():
    rng = np.random.default_rng(21)
    valid = rng.normal(0, .02, 10000)
    expected = estimate_noise_sigma(valid, np.ones(len(valid), bool), floor_mm=.001)
    residual = np.r_[valid, np.zeros(100000)]
    result = estimate_noise_sigma(residual, np.r_[np.ones(len(valid), bool), np.zeros(100000, bool)], floor_mm=.001)
    assert result == expected


def test_pca_always_has_four_proper_rotations():
    rng = np.random.default_rng(42)
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.normal(size=(300, 3))*[1, 2, 4]))
    for _ in range(20):
        target = o3d.geometry.PointCloud(source)
        target.rotate(o3d.geometry.get_rotation_matrix_from_xyz(rng.uniform(-3, 3, 3)))
        candidates = _principal_axis_candidates(source, target)
        assert len(candidates) == 4
        assert all(abs(np.linalg.det(c[:3, :3])-1) < 1e-8 for c in candidates)


def test_main_window_import_does_not_load_numerical_stack():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"), QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c",
                             "import sys; import auto_alignment.gui; assert 'open3d' not in sys.modules; assert 'numpy' not in sys.modules"],
                            env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
