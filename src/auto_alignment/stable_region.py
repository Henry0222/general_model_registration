"""Data-driven stable-region estimation for rigid registration.

Given a current rigid pose, this module decides which parts of two surfaces
are the *same* surface and which have genuinely changed, without any fixed
millimetre threshold.  The noise level is estimated from the data, the
stable/changed decision is smoothed over face adjacency so it forms coherent
patches rather than salt-and-pepper labels, and tiny islands are dropped.
The resulting face masks drive the final high-precision refinement and its
gate, so a broad low-amplitude change (soft-tissue swelling, print
shrinkage) can no longer bias the pose the way a single distance cut-off
lets it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class StableRegionEstimate:
    sigma_mm: float
    threshold_mm: float
    source_faces: np.ndarray
    target_faces: np.ndarray
    source_area_fraction: float
    target_area_fraction: float
    source_raw_fraction: float
    target_raw_fraction: float

    def summary(self) -> dict[str, float]:
        return {
            "sigma_mm": float(self.sigma_mm),
            "threshold_mm": float(self.threshold_mm),
            "source_stable_area_fraction": float(self.source_area_fraction),
            "target_stable_area_fraction": float(self.target_area_fraction),
            "source_raw_inlier_fraction": float(self.source_raw_fraction),
            "target_raw_inlier_fraction": float(self.target_raw_fraction),
            "source_stable_faces": int(np.count_nonzero(self.source_faces)),
            "target_stable_faces": int(np.count_nonzero(self.target_faces)),
        }


def face_adjacency_pairs(triangles: np.ndarray) -> np.ndarray:
    """Return (M, 2) face index pairs sharing an ordinary two-face edge."""
    triangles = np.asarray(triangles, dtype=np.int64)
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int64)
    edge_corners = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    edges = np.sort(triangles[:, edge_corners], axis=2).reshape((-1, 2))
    owners = np.repeat(np.arange(len(triangles), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(edges)])
    shared = starts[counts == 2]
    return np.column_stack((owners[shared], owners[shared + 1]))


def connected_labels(count: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(count, dtype=np.int64)
    if not len(pairs):
        return parent
    first = pairs[:, 0]
    second = pairs[:, 1]
    while True:
        root_first = parent[first]
        root_second = parent[second]
        low = np.minimum(root_first, root_second)
        high = np.maximum(root_first, root_second)
        changed = low != high
        if not np.any(changed):
            break
        np.minimum.at(parent, high[changed], low[changed])
        while True:
            grand = parent[parent]
            if np.array_equal(grand, parent):
                break
            parent = grand
    return parent


def estimate_noise_sigma(
    signed_residuals: np.ndarray,
    initial_inliers: np.ndarray,
    *,
    floor_mm: float,
    rounds: int = 3,
    k: float = 3.0,
) -> float:
    """Robust noise scale of point-to-plane residuals on the consensus set.

    Starts from a generous inlier set and tightens to ``k`` sigma a few
    times.  Distances of genuinely changed surface are far outside the
    noise band, so they fall out of the consensus set instead of inflating
    the estimate.
    """
    residuals = np.asarray(signed_residuals, dtype=float)
    eligible = np.asarray(initial_inliers, dtype=bool) & np.isfinite(residuals)
    inliers = eligible.copy()
    sigma = float(floor_mm)
    for _ in range(max(1, int(rounds))):
        if np.count_nonzero(inliers) < 100:
            break
        selected = residuals[inliers]
        centre = float(np.median(selected))
        mad = float(np.median(np.abs(selected - centre)))
        sigma = max(float(floor_mm), 1.4826 * mad)
        inliers = eligible & (np.abs(residuals - centre) <= k * sigma)
    return sigma


def face_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    corners = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    return np.where(np.isfinite(areas), areas, 0.0)


def smooth_face_mask(
    triangles: np.ndarray,
    vertex_stable: np.ndarray,
    areas: np.ndarray,
    *,
    rounds: int = 3,
    min_component_area_fraction: float = 0.005,
    adjacency: MeshAdjacency | None = None,
) -> np.ndarray:
    """Turn per-vertex stability into coherent stable face patches.

    A face starts with the fraction of its stable corners, then repeatedly
    blends with the mean of its edge-neighbours.  Isolated stable faces in
    a changed region fade out; isolated holes in a stable region fill in.
    Finally components smaller than ``min_component_area_fraction`` of the
    total stable area are dropped so a few coincidental matches inside a
    changed region cannot anchor the pose.
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    vertex_stable = np.asarray(vertex_stable, dtype=bool)
    score = vertex_stable[triangles].mean(axis=1)
    if adjacency is None:
        pairs = face_adjacency_pairs(triangles)
        degree = np.bincount(pairs.ravel(), minlength=len(triangles)).astype(float)
    else:
        pairs = adjacency.face_pairs
        degree = adjacency.face_degree
    if len(pairs):
        for _ in range(max(0, int(rounds))):
            neighbour_sum = np.bincount(pairs[:, 0], weights=score[pairs[:, 1]], minlength=len(triangles))
            neighbour_sum += np.bincount(pairs[:, 1], weights=score[pairs[:, 0]], minlength=len(triangles))
            neighbour_mean = np.where(degree > 0, neighbour_sum / np.maximum(degree, 1.0), score)
            score = 0.5 * score + 0.5 * neighbour_mean
    stable = score >= 0.5
    if not np.any(stable) or not len(pairs):
        return stable
    stable_ids = np.flatnonzero(stable)
    keep_pair = stable[pairs[:, 0]] & stable[pairs[:, 1]]
    local_index = np.full(len(triangles), -1, dtype=np.int64)
    local_index[stable_ids] = np.arange(len(stable_ids))
    local_pairs = local_index[pairs[keep_pair]]
    roots = connected_labels(len(stable_ids), local_pairs)
    _, labels = np.unique(roots, return_inverse=True)
    component_area = np.bincount(labels, weights=areas[stable_ids])
    total = float(np.sum(component_area))
    if total <= 0.0:
        return stable
    small = component_area < float(min_component_area_fraction) * total
    stable[stable_ids[small[labels]]] = False
    return stable


def _closest_surface(
    points: np.ndarray,
    normals: np.ndarray,
    transform: np.ndarray,
    scene: o3d.t.geometry.RaycastingScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moved = points @ transform[:3, :3].T + transform[:3, 3]
    moved_normals = normals @ transform[:3, :3].T
    closest = scene.compute_closest_points(o3d.core.Tensor(moved.astype(np.float32)))
    hit_points = closest["points"].numpy().astype(float)
    hit_normals = closest["primitive_normals"].numpy().astype(float)
    delta = moved - hit_points
    distance = np.linalg.norm(delta, axis=1)
    residual = np.einsum("ij,ij->i", delta, hit_normals)
    agreement = np.abs(np.einsum("ij,ij->i", moved_normals, hit_normals))
    return distance, residual, agreement


def vertex_adjacency_pairs(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    if not len(triangles):
        return np.empty((0, 2), dtype=np.int64)
    edges = np.sort(triangles[:, ((0, 1), (1, 2), (2, 0))], axis=2).reshape((-1, 2))
    # Pack both ends into one integer so a single 1-D unique suffices; the
    # row-wise unique on an (N, 2) array is an order of magnitude slower.
    span = int(triangles.max()) + 1
    packed = np.unique(edges[:, 0] * span + edges[:, 1])
    return np.column_stack((packed // span, packed % span))


@dataclass
class MeshAdjacency:
    """Per-mesh connectivity that is expensive to build and never changes."""

    triangles: np.ndarray
    face_pairs: np.ndarray
    vertex_pairs: np.ndarray
    face_degree: np.ndarray

    @classmethod
    def build(cls, triangles: np.ndarray) -> "MeshAdjacency":
        triangles = np.asarray(triangles, dtype=np.int64)
        face_pairs = face_adjacency_pairs(triangles)
        degree = np.bincount(face_pairs.ravel(), minlength=len(triangles)).astype(float)
        return cls(triangles, face_pairs, vertex_adjacency_pairs(triangles), degree)


def local_mean_residual(
    triangles: np.ndarray,
    residuals: np.ndarray,
    valid: np.ndarray,
    *,
    rounds: int,
    vertex_pairs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbourhood-averaged signed residual and the count it averaged over.

    A broad low-amplitude change (soft-tissue swelling, print shrinkage) is
    invisible per vertex because each residual sits inside the noise band,
    yet its sign is consistent over the whole patch.  Averaging over a few
    rings of neighbours shrinks the noise by roughly the square root of the
    neighbourhood size while leaving a systematic offset untouched, which
    is exactly what separates the two.
    """
    residuals = np.asarray(residuals, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    pairs = vertex_adjacency_pairs(triangles) if vertex_pairs is None else vertex_pairs
    count = len(residuals)
    total = np.where(valid, residuals, 0.0)
    weight = valid.astype(float)
    if not len(pairs) or rounds <= 0:
        return np.where(weight > 0, total / np.maximum(weight, 1e-12), np.nan), weight
    a, b = pairs[:, 0], pairs[:, 1]
    for _ in range(int(rounds)):
        neighbour_total = np.bincount(a, weights=total[b], minlength=count) + np.bincount(b, weights=total[a], minlength=count)
        neighbour_weight = np.bincount(a, weights=weight[b], minlength=count) + np.bincount(b, weights=weight[a], minlength=count)
        total = total + neighbour_total
        weight = weight + neighbour_weight
    return np.where(weight > 0, total / np.maximum(weight, 1e-12), np.nan), weight


def _stable_vertices(
    distance: np.ndarray,
    residual: np.ndarray,
    agreement: np.ndarray,
    threshold_mm: float,
    normal_threshold: float,
) -> np.ndarray:
    return (
        np.isfinite(distance)
        & np.isfinite(residual)
        & (distance <= threshold_mm)
        & (agreement >= normal_threshold)
    )


def _bias_free_vertices(
    triangles: np.ndarray,
    residual: np.ndarray,
    point_stable: np.ndarray,
    sigma: float,
    *,
    rounds: int,
    bias_k_sigma: float,
    vertex_pairs: np.ndarray | None = None,
) -> np.ndarray:
    if rounds <= 0 or bias_k_sigma <= 0.0:
        return point_stable
    mean, weight = local_mean_residual(
        triangles, residual, point_stable, rounds=rounds, vertex_pairs=vertex_pairs
    )
    # The mean of n noisy samples has spread sigma/sqrt(n); flag a patch
    # whose mean exceeds bias_k_sigma of that spread.
    spread = float(sigma) / np.sqrt(np.maximum(weight, 1.0))
    biased = np.isfinite(mean) & (np.abs(mean) > float(bias_k_sigma) * spread)
    return point_stable & ~biased


def estimate_stable_region(
    target_mesh: o3d.geometry.TriangleMesh,
    source_mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
    *,
    target_scene: o3d.t.geometry.RaycastingScene,
    source_scene: o3d.t.geometry.RaycastingScene,
    initial_distance_mm: float,
    normal_angle_degrees: float,
    sigma_floor_mm: float,
    k_sigma: float = 3.0,
    smoothing_rounds: int = 3,
    min_component_area_fraction: float = 0.005,
    fixed_sigma_mm: float | None = None,
    bias_rounds: int = 4,
    bias_k_sigma: float = 4.0,
    source_adjacency: MeshAdjacency | None = None,
    target_adjacency: MeshAdjacency | None = None,
) -> StableRegionEstimate:
    """Estimate noise and coherent stable patches on both surfaces.

    ``transform`` maps ``source_mesh`` into the target frame.  The moving
    surface is compared against the fixed one and vice versa so that a
    region present on only one side is unstable on both.  ``fixed_sigma_mm``
    skips the noise estimate so two poses can be compared at one threshold.
    """
    matrix = np.asarray(transform, dtype=float)
    inverse = np.linalg.inv(matrix)
    normal_threshold = float(np.cos(np.deg2rad(normal_angle_degrees)))

    source_vertices = np.asarray(source_mesh.vertices, dtype=float)
    source_normals = np.asarray(source_mesh.vertex_normals, dtype=float)
    target_vertices = np.asarray(target_mesh.vertices, dtype=float)
    target_normals = np.asarray(target_mesh.vertex_normals, dtype=float)
    source_triangles = np.asarray(source_mesh.triangles, dtype=np.int64)
    target_triangles = np.asarray(target_mesh.triangles, dtype=np.int64)

    s_distance, s_residual, s_agreement = _closest_surface(
        source_vertices, source_normals, matrix, target_scene
    )
    t_distance, t_residual, t_agreement = _closest_surface(
        target_vertices, target_normals, inverse, source_scene
    )
    initial = float(initial_distance_mm)
    seed = np.concatenate(
        (
            (s_distance <= initial) & (s_agreement >= normal_threshold),
            (t_distance <= initial) & (t_agreement >= normal_threshold),
        )
    )
    sigma = (
        float(fixed_sigma_mm)
        if fixed_sigma_mm is not None
        else estimate_noise_sigma(
            np.concatenate((s_residual, t_residual)),
            seed,
            floor_mm=sigma_floor_mm,
            k=k_sigma,
        )
    )
    threshold = float(k_sigma) * sigma
    source_vertex_stable = _stable_vertices(s_distance, s_residual, s_agreement, threshold, normal_threshold)
    target_vertex_stable = _stable_vertices(t_distance, t_residual, t_agreement, threshold, normal_threshold)
    source_vertex_stable = _bias_free_vertices(
        source_triangles, s_residual, source_vertex_stable, sigma, rounds=bias_rounds, bias_k_sigma=bias_k_sigma,
        vertex_pairs=None if source_adjacency is None else source_adjacency.vertex_pairs,
    )
    target_vertex_stable = _bias_free_vertices(
        target_triangles, t_residual, target_vertex_stable, sigma, rounds=bias_rounds, bias_k_sigma=bias_k_sigma,
        vertex_pairs=None if target_adjacency is None else target_adjacency.vertex_pairs,
    )

    source_areas = face_areas(source_vertices, source_triangles)
    target_areas = face_areas(target_vertices, target_triangles)
    source_faces = smooth_face_mask(
        source_triangles,
        source_vertex_stable,
        source_areas,
        rounds=smoothing_rounds,
        min_component_area_fraction=min_component_area_fraction,
        adjacency=source_adjacency,
    )
    target_faces = smooth_face_mask(
        target_triangles,
        target_vertex_stable,
        target_areas,
        rounds=smoothing_rounds,
        min_component_area_fraction=min_component_area_fraction,
        adjacency=target_adjacency,
    )

    def fraction(areas: np.ndarray, mask: np.ndarray) -> float:
        total = float(np.sum(areas))
        return float(np.sum(areas[mask]) / total) if total > 0 else 0.0

    return StableRegionEstimate(
        sigma_mm=sigma,
        threshold_mm=threshold,
        source_faces=source_faces,
        target_faces=target_faces,
        source_area_fraction=fraction(source_areas, source_faces),
        target_area_fraction=fraction(target_areas, target_faces),
        source_raw_fraction=fraction(source_areas, source_vertex_stable[source_triangles].all(axis=1)),
        target_raw_fraction=fraction(target_areas, target_vertex_stable[target_triangles].all(axis=1)),
    )


def vertex_weights_from_faces(
    triangles: np.ndarray,
    face_mask: np.ndarray,
    vertex_count: int,
) -> np.ndarray:
    """Vertex multiplier: 1 when any incident face is stable, else 0."""
    triangles = np.asarray(triangles, dtype=np.int64)
    stable = np.zeros(int(vertex_count), dtype=bool)
    stable[np.unique(triangles[np.asarray(face_mask, dtype=bool)])] = True
    return stable.astype(float)


def holdout_split(
    points: np.ndarray,
    eligible: np.ndarray,
    seed: int,
    cells: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Split eligible points into two spatially interleaved halves.

    A random per-point split would let the fit half "see" the geometry of
    every holdout point through its immediate neighbours.  Splitting by a
    coarse checkerboard of spatial cells makes the two halves genuinely
    different pieces of surface while still covering the same overall
    extent, so a good fit on one half must generalise to the other.
    """
    eligible = np.asarray(eligible, dtype=bool)
    points = np.asarray(points, dtype=float)
    if np.count_nonzero(eligible) == 0:
        return eligible.copy(), eligible.copy()
    selected = points[eligible]
    minimum = selected.min(axis=0)
    extent = np.maximum(selected.max(axis=0) - minimum, 1e-9)
    cell = np.floor((selected - minimum) / extent * cells).astype(np.int64)
    cell = np.clip(cell, 0, cells - 1)
    parity = (cell.sum(axis=1) + int(seed)) % 2 == 0
    first = np.zeros(len(points), dtype=bool)
    second = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(eligible)
    first[ids[parity]] = True
    second[ids[~parity]] = True
    return first, second
