"""Geometry-only router with common rigid holdout metrics for all candidates.

Surface quantiles are evidence, not proof of pose accuracy. Keep the original
pose unless the refinement has sufficient support; never score B's fitted field.
"""
import time
import numpy as np
from . import _geometry as geo
from . import _normal_field as nf
HOLDOUT_SEED = 2026091507
HOLDOUT_N = 1024
ROUTING_POLICY = 'rigid_bidirectional_quantiles_v1'
DISTANCE_TOL_MM = 1e-6


def distance_quantiles(q, y, normal_dot):
    distances = np.linalg.norm(q - y, axis=1)
    median, p90, p95 = np.quantile(distances, [.5, .9, .95])
    return dict(median_mm=float(median), p90_mm=float(p90), p95_mm=float(p95),
                median_normal_dot=float(np.median(normal_dot)))


def quantiles_improve(candidate, baseline):
    """Both directions must improve centrally without worsening either tail."""
    for direction in ('source_to_target', 'target_to_source'):
        c, b = candidate[direction], baseline[direction]
        values = [x[key] for x in (c, b) for key in ('median_mm', 'p90_mm', 'p95_mm')]
        if not np.isfinite(values).all() or min(values) < 0:
            return False
        if b['median_mm'] <= DISTANCE_TOL_MM or c['median_mm'] > .8 * b['median_mm']:
            return False
        if any(c[key] > b[key] + DISTANCE_TOL_MM for key in ('p90_mm', 'p95_mm')):
            return False
    return True

def support(q, y, n, origin, radius):
    hit = np.linalg.norm(q - y, axis=1) < 1e-06
    p = q[hit]
    nn = n[hit]
    J = np.c_[np.cross(p - origin, nn), nn]
    s = np.linalg.norm(J, axis=0).clip(1e-12)
    sv = np.linalg.svd(J / s, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-06)) if len(sv) and sv[0] > 0 else 0
    spread = float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1))) / radius) if len(p) else 0.0
    return dict(keep_initial=len(p) >= 64 and rank == 6 and (spread >= 0.25), count=len(p), rank=rank, spread=spread)

class Context:

    def __init__(self, source_mesh, target_mesh, initial):
        self.sm = source_mesh
        self.tm = target_mesh
        self.C, _ = geo.target_frame(target_mesh)
        self.Ci = np.linalg.inv(self.C)
        self.source = geo.MeshCache(geo.moved_mesh(source_mesh, self.C))
        self.target = geo.MeshCache(geo.moved_mesh(target_mesh, self.C))
        self.T0 = self.C @ np.asarray(initial) @ self.Ci
        self.initial = np.asarray(initial)
        self.origin = self.target.origin
        self.radius = float(np.sqrt(np.mean(np.sum((self.target.vertices - self.origin) ** 2, axis=1))))
        self.x = self.source.sample(4096, geo.SAMPLE_SEED).points
        self.validation = None

    def witness(self, T):
        q = geo.apply(self.x, T)
        y, n, ids = geo.query(self.target, q)
        return (dict(q=q, y=y, n=n), support(q, y, n, self.origin, self.radius))

    def a(self, g0, checkpoint=None):
        if g0['keep_initial']:
            return (self.T0.copy(), dict(skipped=True), {})
        d, tr = geo.candidates(self.source, self.target, self.T0, checkpoint=checkpoint)
        T = d['CANDIDATES'][d['chosen_geometry']]
        return (T, dict(skipped=False, chosen=int(d['chosen_geometry']), scores=d['scores']), dict(A_candidates=d['CANDIDATES'], A_scores=d['scores']))

    def b(self, checkpoint=None):
        mass, vn = nf.vfeatures(self.target.vertices, self.target.triangles)
        vp, K, Q, info = nf.basis(self.target.vertices, mass)
        sample = self.target.sample(4096, 2026091391)
        phi = np.einsum('nj,njk->nk', sample.barycentric, vp[self.target.triangles[sample.triangle_ids]])
        n = np.einsum('nj,njk->nk', sample.barycentric, vn[self.target.triangles[sample.triangle_ids]])
        n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-30)
        T, trace, meta = nf.refine(sample.points, n, phi, K, Q, self.source, np.linalg.inv(self.T0), self.origin, 'smooth17', checkpoint=checkpoint)
        self.validation = (vn, vp, K, trace['final_theta'])
        cache = {f'B_{k}': trace[k] for k in ['S', 'theta', 'raw_delta', 'delta', 'rank', 'singular', 'data_singular', 'objective', 'final_S', 'final_theta']}
        return (T, meta, cache)

    def validate(self, A, B):
        # These samples are independent of either solver's samples. In
        # particular, self.validation (B's fitted field) is never consulted.
        samples = {
            'target_to_source': self.target.sample(HOLDOUT_N, HOLDOUT_SEED),
            'source_to_target': self.source.sample(HOLDOUT_N, HOLDOUT_SEED + 1),
        }
        stats = dict(policy=ROUTING_POLICY, distance='euclidean_nearest_surface_mm',
                     field_compensation=False, samples_per_direction=HOLDOUT_N)
        cache = {}
        for name, T in [('initial', self.T0), ('A', A), ('B', B)]:
            stats[name] = {}
            for direction, sam in samples.items():
                forward = direction == 'source_to_target'
                mesh = self.source if forward else self.target
                destination = self.target if forward else self.source
                transform = T if forward else np.linalg.inv(T)
                n = mesh.normals[sam.triangle_ids]
                q = geo.apply(sam.points, transform)
                y, ns, ids = geo.query(destination, q)
                transformed_normal = n @ transform[:3, :3].T
                normal_dot = np.sum(transformed_normal * ns, axis=1)
                stats[name][direction] = distance_quantiles(q, y, normal_dot)
                for k, v in dict(q=q, y=y, n=ns, transformed_normal=transformed_normal).items():
                    cache[f'validation_{name}_{direction}_{k}'] = v
        rel = B @ np.linalg.inv(self.T0)
        omega = geo.Rotation.from_matrix(rel[:3, :3]).as_rotvec()
        shift = rel[:3, :3] @ self.origin + rel[:3, 3] - self.origin
        stats['correction_mm'] = float(np.linalg.norm(shift))
        stats['correction_deg'] = float(np.rad2deg(np.linalg.norm(omega)))
        return (stats, cache)

def decide(g0, ga, bmeta, validation):
    if g0['keep_initial']:
        return ('initial', 'initial', 'initial_geometry_hold')
    if ga['keep_initial']:
        return ('A', 'A', 'A_geometry_support')
    accept = (
        bmeta['status'] == 'final_40' and bmeta['iterations'] == 40
        and 0 <= validation['correction_mm'] <= 3
        and 0 <= validation['correction_deg'] <= 5
        and all(.8 <= validation['B'][direction]['median_normal_dot'] <= 1 + 1e-12
                for direction in ('source_to_target', 'target_to_source'))
        and quantiles_improve(validation['B'], validation['initial'])
        and quantiles_improve(validation['B'], validation['A'])
    )
    return ('B' if accept else 'initial', 'B',
            'B_rigid_quantiles_accept' if accept else 'B_rigid_quantiles_reject')

def run_meshes(sm, tm, initial, full=True, checkpoint=None):
    start = time.perf_counter()
    times = {}
    mark = time.perf_counter()
    ctx = Context(sm, tm, initial)
    times['context'] = time.perf_counter() - mark
    mark = time.perf_counter()
    w0, g0 = ctx.witness(ctx.T0)
    times['gate_initial'] = time.perf_counter() - mark
    mark = time.perf_counter()
    A, ameta, cache = ctx.a(g0, checkpoint=(lambda f, m: checkpoint(0.1 + 0.45 * f, m)) if checkpoint else None)
    wa, ga = ctx.witness(A)
    times['A_and_gate'] = time.perf_counter() - mark
    need_b = not g0['keep_initial'] and (not ga['keep_initial'])
    if full or need_b:
        mark = time.perf_counter()
        B, bmeta, bc = ctx.b(checkpoint=(lambda f, m: checkpoint(0.55 + 0.4 * f, m)) if checkpoint else None)
        times['B'] = time.perf_counter() - mark
        cache.update(bc)
        mark = time.perf_counter()
        validation, vc = ctx.validate(A, B)
        times['validation'] = time.perf_counter() - mark
        cache.update(vc)
        auto, force, reason = decide(g0, ga, bmeta, validation)
    else:
        B = None
        bmeta = None
        validation = None
        times.update(B=0.0, validation=0.0)
        auto, force, reason = decide(g0, ga, {}, {})
    for prefix, w in [('initial', w0), ('A', wa)]:
        for k, v in w.items():
            cache[prefix + '_' + k] = v
    cache.update(C=ctx.C, Cinv=ctx.Ci, origin=ctx.origin, radius=np.asarray(ctx.radius), initial_local=ctx.T0, A_local=A)
    if B is not None:
        cache['B_local'] = B
    matrices = {'initial': np.asarray(initial), 'A': ctx.Ci @ A @ ctx.C}
    if B is not None:
        matrices['B'] = ctx.Ci @ B @ ctx.C
    matrices.update(auto=matrices[auto].copy(), auto_force_b=matrices[force].copy())
    times['all_branches'] = time.perf_counter() - start
    times['auto_compute_estimate'] = times['context'] + times['gate_initial'] + (0 if g0['keep_initial'] else times['A_and_gate']) + (times['B'] + times['validation'] if need_b else 0)
    result = dict(matrices=matrices, route=auto, force_route=force, reason=reason, gates=dict(initial=g0, A=ga), A=ameta, B=bmeta, validation=validation, timings=times, full_comparison=full)
    return (result, cache)
