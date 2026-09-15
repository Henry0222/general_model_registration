"""Fixed 40-step scalar normal field; output transformation is rigid."""
import numpy as np
from . import _geometry as geo

def vfeatures(v, f):
    cc = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = np.linalg.norm(cc, axis=1) / 2
    mass = np.bincount(f.ravel(), weights=np.repeat(area / 3, 3), minlength=len(v))
    mass /= mass.sum()
    nn = np.zeros_like(v)
    for j in range(3):
        np.add.at(nn, f[:, j], cc)
    nn /= np.maximum(np.linalg.norm(nn, axis=1)[:, None], 1e-30)
    return (mass, nn)

def basis(v, mass):
    center = mass @ v
    nearest = np.sum((v - center) ** 2, axis=1)
    ids = [int(np.argmax(nearest))]
    nearest = np.sum((v - v[ids[0]]) ** 2, axis=1)
    for k in range(1, 16):
        ids.append(int(np.argmax(nearest)))
        nearest = np.minimum(nearest, np.sum((v - v[ids[-1]]) ** 2, axis=1))
    centers = v[ids]
    pair = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    width = float(np.median(pair[np.triu_indices(16, 1)]))
    raw = np.exp(-np.sum((v[:, None] - centers[None]) ** 2, axis=-1) / (2 * width ** 2))
    mean = mass @ raw
    scale = np.sqrt(mass @ (raw - mean) ** 2).clip(1e-12)
    phi = np.c_[np.ones(len(v)), (raw - mean) / scale]
    gram = phi.T @ (mass[:, None] * phi)
    penalty = gram.copy()
    penalty[0, :] = 0
    penalty[:, 0] = 0
    ev, U = np.linalg.eigh(penalty)
    Q = np.diag(np.sqrt(np.maximum(ev, 0))) @ U.T
    return (phi, gram, Q, dict(center_ids=ids, centers=centers, width=width, mean=mean, scale=scale))

def refine(x, n, phi, gram, Q, source, S0, origin, mode, checkpoint=None):
    k = {'rigid': 0, 'constant': 1, 'smooth17': 17}[mode]
    P = phi[:, :k]
    K = gram[:k, :k]
    reg = Q[:, :k] if k == 17 else np.zeros((k, k))
    theta = np.zeros(k)
    S = S0.copy()
    trace = []
    status = 'final_40'
    for step in range(40):
        if checkpoint:
            checkpoint(step / 40, "B：拟合连续表面偏差…")
        d = P @ theta
        z = x + d[:, None] * n
        q = geo.apply(z, S)
        y, ns, ids = geo.query(source, q)
        res = np.sum((q - y) * ns, axis=1)
        o = geo.apply(origin[None], S)[0]
        J = np.c_[np.cross(q - o, ns), ns]
        a = np.sum(n @ S[:3, :3].T * ns, axis=1)
        J = np.c_[J, a[:, None] * P]
        M = J / np.sqrt(len(x))
        rhs = -res / np.sqrt(len(x))
        regular = np.c_[np.zeros((len(reg), 6)), np.sqrt(0.01) * reg]
        A = np.vstack([M, regular])
        b = np.r_[rhs, -np.sqrt(0.01) * (reg @ theta)]
        scale = np.linalg.norm(A, axis=0).clip(1e-12)
        delta, _, rank, sv = np.linalg.lstsq(A / scale, b, rcond=1e-09)
        delta /= scale
        raw = delta.copy()
        data_sv = np.linalg.svd(M / np.linalg.norm(M, axis=0).clip(1e-12), compute_uv=False)
        if rank < 6 + k:
            delta *= 0
            status = 'rank_warning'
        delta[:3] *= min(1, np.deg2rad(2) / max(np.linalg.norm(delta[:3]), 1e-30))
        delta[3:6] *= min(1, 1 / max(np.linalg.norm(delta[3:6]), 1e-30))
        if k:
            delta[6:] *= min(1, 1 / max(float(np.sqrt(max(delta[6:] @ K @ delta[6:], 0))), 1e-30))
            nexttheta = theta + delta[6:]
            nexttheta *= min(1, 2 / max(float(np.sqrt(max(nexttheta @ K @ nexttheta, 0))), 1e-30))
            delta[6:] = nexttheta - theta
        trace.append(dict(S=S.copy(), theta=theta.copy(), q=q, y=y, n=ns, ids=ids, raw_delta=raw, delta=delta, rank=rank, singular=sv, data_singular=data_sv, objective=float(np.mean(res ** 2) + 0.01 * np.sum((reg @ theta) ** 2))))
        R = geo.Rotation.from_rotvec(delta[:3]).as_matrix()
        S[:3, :3] = R @ S[:3, :3]
        S[:3, 3] = R @ (S[:3, 3] - o) + o + delta[3:6]
        theta += delta[6:]
        if status == 'rank_warning':
            break
    cache = {key: np.stack([r[key] for r in trace]) for key in trace[0]}
    cache.update(final_S=S, final_theta=theta)
    return (np.linalg.inv(S), cache, dict(status=status, iterations=len(trace), field_rms=float(np.sqrt(max(theta @ K @ theta, 0))) if k else 0.0))
