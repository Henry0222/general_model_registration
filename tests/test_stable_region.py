from __future__ import annotations

import numpy as np
import open3d as o3d

from auto_alignment.config import AlignmentConfig
from auto_alignment.quality.models import CandidateDiagnostic, _candidate_consistency
from auto_alignment.registration import _distance_scene, _stable_region_refinement
from auto_alignment.stable_region import (
    connected_labels,
    estimate_noise_sigma,
    estimate_stable_region,
    face_adjacency_pairs,
    holdout_split,
    smooth_face_mask,
)


def _bumpy_surface(size: float = 30.0, resolution: int = 90, seed: int = 3) -> o3d.geometry.TriangleMesh:
    """A gently undulating open sheet with enough curvature to fix six DOF."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(-size / 2, size / 2, resolution)
    x, y = np.meshgrid(axis, axis)
    z = (
        2.0 * np.sin(x / 5.0) * np.cos(y / 6.0)
        + 1.2 * np.cos(x / 3.0 + 1.0)
        + 0.6 * np.sin(y / 2.5)
    )
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    index = np.arange(resolution * resolution).reshape(resolution, resolution)
    a = index[:-1, :-1].ravel()
    b = index[:-1, 1:].ravel()
    c = index[1:, :-1].ravel()
    d = index[1:, 1:].ravel()
    triangles = np.vstack((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    del rng
    return mesh


def test_connected_labels_matches_breadth_first_search() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        count = int(rng.integers(1, 300))
        pairs = rng.integers(0, count, size=(int(rng.integers(0, 500)), 2))
        roots = connected_labels(count, pairs)
        adjacency: list[list[int]] = [[] for _ in range(count)]
        for first, second in pairs:
            adjacency[first].append(second)
            adjacency[second].append(first)
        component = -np.ones(count, dtype=int)
        label = 0
        for start in range(count):
            if component[start] >= 0:
                continue
            stack = [start]
            component[start] = label
            while stack:
                node = stack.pop()
                for neighbour in adjacency[node]:
                    if component[neighbour] < 0:
                        component[neighbour] = label
                        stack.append(neighbour)
            label += 1
        _, ours = np.unique(roots, return_inverse=True)
        _, theirs = np.unique(component, return_inverse=True)
        for value in np.unique(ours):
            assert len(set(theirs[ours == value])) == 1
        for value in np.unique(theirs):
            assert len(set(ours[theirs == value])) == 1


def test_noise_sigma_ignores_changed_surface() -> None:
    rng = np.random.default_rng(5)
    noise = rng.normal(0.0, 0.02, size=20_000)
    changed = rng.normal(0.3, 0.05, size=6_000)
    residuals = np.concatenate((noise, changed))
    seed = np.abs(residuals) < 0.6
    sigma = estimate_noise_sigma(residuals, seed, floor_mm=0.001)
    assert 0.016 < sigma < 0.026, sigma


def test_smooth_face_mask_removes_islands_and_fills_holes() -> None:
    mesh = _bumpy_surface(resolution=40)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    stable = vertices[:, 0] < 0.0
    rng = np.random.default_rng(2)
    noisy = stable.copy()
    flip = rng.random(len(noisy)) < 0.03
    noisy[flip] = ~noisy[flip]
    areas = np.full(len(triangles), 1.0)
    faces = smooth_face_mask(triangles, noisy, areas, rounds=3, min_component_area_fraction=0.01)
    truth = stable[triangles].mean(axis=1) >= 0.5
    agreement = np.mean(faces == truth)
    assert agreement > 0.97, agreement
    assert len(face_adjacency_pairs(triangles)) > 0


def test_holdout_split_is_spatially_interleaved() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-10, 10, size=(5000, 3))
    eligible = np.ones(len(points), dtype=bool)
    first, second = holdout_split(points, eligible, seed=0)
    assert not np.any(first & second)
    assert np.all(first | second)
    assert 0.35 < first.mean() < 0.65
    # both halves span the whole extent
    for half in (first, second):
        assert np.all(points[half].max(axis=0) > 8.0)
        assert np.all(points[half].min(axis=0) < -8.0)


def test_stable_region_excludes_broad_low_amplitude_change() -> None:
    target = _bumpy_surface()
    source = o3d.geometry.TriangleMesh(target)
    vertices = np.asarray(source.vertices).copy()
    normals = np.asarray(source.vertex_normals)
    band = vertices[:, 1] > 5.0
    vertices[band] += normals[band] * 0.12
    rng = np.random.default_rng(9)
    vertices += rng.normal(0.0, 0.01, size=vertices.shape)
    source.vertices = o3d.utility.Vector3dVector(vertices)
    source.compute_vertex_normals()
    estimate = estimate_stable_region(
        target,
        source,
        np.eye(4),
        target_scene=_distance_scene(target),
        source_scene=_distance_scene(source),
        initial_distance_mm=0.6,
        normal_angle_degrees=60.0,
        sigma_floor_mm=0.001,
        k_sigma=3.0,
        bias_rounds=2,
        bias_k_sigma=4.0,
    )
    assert 0.008 < estimate.sigma_mm < 0.03, estimate.sigma_mm
    triangles = np.asarray(source.triangles)
    face_band = band[triangles].any(axis=1)
    leak = np.count_nonzero(estimate.source_faces & face_band) / max(1, np.count_nonzero(estimate.source_faces))
    kept = np.count_nonzero(estimate.source_faces & ~face_band) / max(1, np.count_nonzero(~face_band))
    assert leak < 0.05, leak
    assert kept > 0.8, kept


def test_stable_region_refinement_recovers_pose_despite_swollen_band() -> None:
    target = _bumpy_surface()
    source = o3d.geometry.TriangleMesh(target)
    vertices = np.asarray(source.vertices).copy()
    normals = np.asarray(source.vertex_normals)
    band = vertices[:, 1] > 4.0
    vertices[band] += normals[band] * 0.10
    rng = np.random.default_rng(11)
    vertices += rng.normal(0.0, 0.01, size=vertices.shape)
    source.vertices = o3d.utility.Vector3dVector(vertices)
    source.compute_vertex_normals()
    # Start slightly off the true pose, as multiscale ICP would leave it.
    before = np.eye(4)
    before[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(np.deg2rad((0.04, -0.03, 0.02)))
    before[:3, 3] = (0.02, -0.015, 0.03)
    config = AlignmentConfig(high_precision_max_vertices=50_000, stable_region_bias_rounds=2)
    after, decision = _stable_region_refinement(target, source, before, config, None, incoming_resolution_mm=0.19)
    assert decision["accepted"], decision["reasons"]
    stable_points = np.asarray(target.vertices)[~band]
    moved = stable_points @ after[:3, :3].T + after[:3, 3]
    error = np.linalg.norm(moved - stable_points, axis=1)
    moved_before = stable_points @ before[:3, :3].T + before[:3, 3]
    error_before = np.linalg.norm(moved_before - stable_points, axis=1)
    assert np.sqrt(np.mean(error**2)) < 0.25 * np.sqrt(np.mean(error_before**2))
    assert np.sqrt(np.mean(error**2)) < 0.01


def test_candidate_consistency_ignores_derived_diagnostics() -> None:
    identity = np.eye(4)
    far = np.eye(4)
    far[:3, 3] = (30.0, 0.0, 0.0)
    candidates = (
        CandidateDiagnostic("fgr", identity, 0.78, 0.33),
        CandidateDiagnostic("ransac_1", far, 0.80, 0.40),
        CandidateDiagnostic("pca_1", far, 0.79, 0.40),
        CandidateDiagnostic("fgr_final", identity, 0.33, 0.13),
        CandidateDiagnostic("stable_region_accepted", identity, 1.0, 0.03),
    )
    assert _candidate_consistency(candidates, 90.0) < 0.75
