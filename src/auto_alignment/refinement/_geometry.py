"""Audited geometric calculations, packaged without experiment dependencies."""
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ..candidate_selection import MeshCache

POINTS = 4096
REGIONS = 192
REGION_POINTS = 16
REGION_STEPS = 10
FINAL_CANDIDATES = 9
REFINE_STEPS = 8
SAMPLE_SEED = 2026091215

def closest(points, triangles):
    """One corresponding point/triangle per row; plane plus three closed edges."""
    p = np.asarray(points, np.float64)
    t = np.asarray(triangles, np.float64)
    a, bb, c = (t[:, 0], t[:, 1], t[:, 2])
    ab = bb - a
    ac = c - a
    n = np.cross(ab, ac)
    length = np.linalg.norm(n, axis=1)
    unit = np.divide(n, length[:, None], out=np.zeros_like(n), where=length[:, None] > 0)
    plane = p - np.einsum('ij,ij->i', p - a, unit)[:, None] * unit
    inside = length > 0
    for u, v in [(a, bb), (bb, c), (c, a)]:
        inside &= np.einsum('ij,ij->i', np.cross(v - u, plane - u), unit) >= 0
    result = plane.copy()
    distance = np.sum((p - plane) ** 2, axis=1)
    distance[~inside] = np.inf
    for u, v in [(a, bb), (bb, c), (c, a)]:
        edge = v - u
        den = np.sum(edge * edge, axis=1)
        s = np.divide(np.einsum('ij,ij->i', p - u, edge), den, out=np.zeros(len(p)), where=den > 0)
        q = u + np.clip(s, 0, 1)[:, None] * edge
        d = np.sum((p - q) ** 2, axis=1)
        better = d < distance
        result[better] = q[better]
        distance[better] = d[better]
    return (result, distance)

def target_frame(target_mesh):
    """Target-only principal axes, signed by the farthest projected vertex.

    Reject a numerically ambiguous axis/sign instead of silently choosing a world
    axis. This fixes joint-coordinate dependence for non-degenerate input shapes;
    it is not a claim of unique orientation for a perfectly symmetric object.
    """
    v = np.asarray(target_mesh.vertices)
    f = np.asarray(target_mesh.triangles)
    p = v[f]
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) * 0.5
    w = np.zeros(len(v))
    np.add.at(w, f.ravel(), np.repeat(area / 3, 3))
    w /= w.sum()
    c = w @ v
    z = v - c
    cov = z.T @ (w[:, None] * z)
    values, axes = np.linalg.eigh(cov)
    values = values[::-1]
    axes = axes[:, ::-1]
    gaps = (values[:-1] - values[1:]) / np.maximum(values[:-1], 1e-30)
    if np.min(gaps) < 1e-06:
        raise ValueError('ambiguous target principal axes')
    signs = []
    for k in range(2):
        projection = z @ axes[:, k]
        i = int(np.argmax(np.abs(projection)))
        positive = float(projection.max())
        negative = float(-projection.min())
        margin = abs(positive - negative) / max(positive, negative, 1e-30)
        if margin < 1e-08:
            raise ValueError('ambiguous target axis sign')
        if projection[i] < 0:
            axes[:, k] *= -1
        signs.append(dict(vertex=i, relative_extreme_margin=margin))
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    axes[:, 2] /= np.linalg.norm(axes[:, 2])
    C = np.eye(4)
    C[:3, :3] = axes.T
    C[:3, 3] = -axes.T @ c
    return (C, dict(eigenvalues=values, relative_eigenvalue_gaps=gaps, signs=signs, center_mm=c))

def mesh(v, f):
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))

def moved_mesh(m, H):
    return mesh(apply(np.asarray(m.vertices), H), np.asarray(m.triangles))

def apply(p, T):
    return p @ T[:3, :3].T + T[:3, 3]

def fps(p, k):
    ids = []
    center = p.mean(0)
    i = int(np.argmax(np.sum((p - center) ** 2, 1)))
    distance = np.full(len(p), np.inf)
    for _ in range(k):
        ids.append(i)
        distance = np.minimum(distance, np.sum((p - p[i]) ** 2, 1))
        i = int(np.argmax(distance))
    return np.asarray(ids)

def query(cache, q):
    _, _, ix = cache.query(q)
    if np.any(ix < 0) or np.any(ix >= len(cache.triangles)):
        raise ValueError("最近曲面查询未返回有效面片。")
    tri = cache.vertices[cache.triangles[ix]]
    y, _ = closest(q, tri)
    return (y, cache.normals[ix], ix)

def update(x, y, n, T, origin, weights=None):
    q = apply(x, T)
    w = np.ones(len(x)) if weights is None else weights
    A = np.column_stack((np.cross(q - origin, n), n))
    rhs = np.sum((y - q) * n, 1)
    length = np.sqrt(np.sum(A * A * w[:, None], 0)).clip(1e-12)
    B = A / length
    sv = np.linalg.svd(B * np.sqrt(w[:, None]), compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-07)) if sv[0] > 0 else 0
    if rank < 6 or (np.count_nonzero(w) > 0 and w.sum() ** 2 / max(w @ w, 1e-30) < 12):
        return (T.copy(), dict(rank=rank, held=True, delta=np.zeros(6)))
    delta = np.linalg.solve(B.T @ (B * w[:, None]) + 1e-09 * np.eye(6), B.T @ (w * rhs)) / length
    delta[:3] *= min(1, np.deg2rad(2) / max(np.linalg.norm(delta[:3]), 1e-30))
    delta[3:] *= min(1, 1 / max(np.linalg.norm(delta[3:]), 1e-30))
    R = Rotation.from_rotvec(delta[:3]).as_matrix()
    U = T.copy()
    U[:3, :3] = R @ T[:3, :3]
    U[:3, 3] = R @ (T[:3, 3] - origin) + origin + delta[3:]
    return (U, dict(rank=rank, held=False, delta=delta))

def support_features(x, T, target, origin, radius):
    q = apply(x, T)
    y, n, ids = query(target, q)
    d = np.linalg.norm(q - y, axis=1)
    res = q - y
    unit = max(radius, 1e-08)
    quant = np.quantile(d, [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 1])
    thresholds = [1e-06, 1e-05, 0.0001, 0.001, 0.003, 0.01, 0.03]
    counts = np.asarray([np.mean(d < v) for v in thresholds])
    spread = []
    for v in [1e-05, 0.001, 0.01]:
        mask = d < v
        spread.append(float(np.sqrt(np.mean(np.sum((q[mask] - q[mask].mean(0)) ** 2, 1))) / unit) if mask.sum() >= 3 else 0.0)
    signed = np.sum(res * n, 1)
    features = np.r_[np.log1p(quant / 0.01), counts, spread, np.mean(res, 0), np.std(res, 0), np.quantile(signed, [0.01, 0.1, 0.5, 0.9, 0.99]), np.mean(n, 0), np.std(n, 0)]
    score = float(np.round(counts[1] * min(spread[0], 1.5), 12))
    return (features, score, dict(q=q, y=y, n=n, triangle_ids=ids, distance=d))

def candidates(source, target, initial, checkpoint=None):
    """Inputs are MeshCache objects and one local rigid matrix, exclusively."""
    begin = time.perf_counter()
    sample = source.sample(POINTS, SAMPLE_SEED)
    x = sample.points
    origin = target.origin
    radius = float(np.sqrt(np.mean(np.sum((target.vertices - origin) ** 2, 1))))
    selected = fps(x, REGIONS)
    _, neighbors = cKDTree(x).query(x[selected], k=REGION_POINTS, workers=1)
    Ts = np.repeat(np.asarray(initial)[None], REGIONS, axis=0)
    traces = []
    for step in range(REGION_STEPS):
        if checkpoint:
            checkpoint(0.05 + 0.3 * step / REGION_STEPS, "A：搜索局部共同表面…")
        qs = np.stack([apply(x[ix], T) for ix, T in zip(neighbors, Ts)])
        ys, ns, ids = query(target, qs.reshape(-1, 3))
        ys = ys.reshape(REGIONS, REGION_POINTS, 3)
        ns = ns.reshape(REGIONS, REGION_POINTS, 3)
        ids = ids.reshape(REGIONS, REGION_POINTS)
        new = []
        deltas = []
        ranks = []
        for i, ix in enumerate(neighbors):
            U, info = update(x[ix], ys[i], ns[i], Ts[i], qs[i].mean(0))
            rel = U @ np.linalg.inv(initial)
            if np.linalg.norm(rel[:3, :3] @ origin + rel[:3, 3] - origin) > 3 or np.linalg.norm(Rotation.from_matrix(rel[:3, :3]).as_rotvec()) > np.deg2rad(5):
                U = Ts[i].copy()
                info['held'] = True
            new.append(U)
            deltas.append(info['delta'])
            ranks.append(info['rank'])
        traces.append(dict(T=Ts.copy(), q=qs, y=ys, n=ns, ids=ids, delta=np.asarray(deltas), rank=np.asarray(ranks)))
        Ts = np.asarray(new)
    coarse = []
    coarse_support = []
    coarse_features = []
    xs = x[::4]
    for i, T in enumerate(Ts):
        if checkpoint and i % 12 == 0:
            checkpoint(0.35 + 0.25 * i / REGIONS, "A：比较局部候选…")
        f, s, d = support_features(xs, T, target, origin, radius)
        coarse.append((s, i))
        coarse_support.append(d)
        coarse_features.append(f)
    winners = [i for _, i in sorted(coarse, key=lambda z: (-z[0], z[1]))[:FINAL_CANDIDATES - 1]]
    pool = [np.asarray(initial).copy()] + [Ts[i].copy() for i in winners]
    finaltrace = []
    for j, T in enumerate(pool):
        if checkpoint:
            checkpoint(0.65 + 0.3 * j / len(pool), "A：精修共同表面候选…")
        if j == 0:
            continue
        for step in range(REFINE_STEPS):
            q = apply(x, T)
            y, n, ids = query(target, q)
            dist = np.linalg.norm(q - y, axis=1)
            cut = [0.01, 0.003, 0.001, 0.0003, 0.0001, 3e-05, 1e-05, 1e-05][step]
            w = (dist < cut).astype(float)
            U, info = update(x, y, n, T, origin, w)
            finaltrace.append(dict(candidate=j, step=step, T=T.copy(), y=y, n=n, ids=ids, w=w, delta=info['delta'], rank=info['rank']))
            T = U
        pool[j] = T
    features = []
    scores = []
    support = []
    twists = []
    for T in pool:
        f, s, d = support_features(x, T, target, origin, radius)
        rel = T @ np.linalg.inv(initial)
        omega = Rotation.from_matrix(rel[:3, :3]).as_rotvec()
        shift = rel[:3, :3] @ origin + rel[:3, 3] - origin
        motion = np.r_[omega, shift]
        features.append(np.r_[f, omega, shift, np.linalg.norm(omega), np.linalg.norm(shift)])
        scores.append(s)
        support.append(d)
        twists.append(motion)
    return (dict(x=x, source_triangle_ids=sample.triangle_ids, source_barycentric=sample.barycentric, region_indices=neighbors, initial=np.asarray(initial), origin=origin, radius=radius, CANDIDATES=np.asarray(pool), X=np.asarray(features), scores=np.asarray(scores), twists=np.asarray(twists), chosen_geometry=int(np.argmax(scores)), coarse_transforms=Ts, coarse_scores=np.asarray(coarse), coarse_features=np.asarray(coarse_features), coarse_selected=np.asarray(winners), seconds=time.perf_counter() - begin), dict(region=traces, refine=finaltrace, support=support, coarse_support=coarse_support))
