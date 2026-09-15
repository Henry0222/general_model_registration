"""Fixed geometry evaluation and a non-destructive rigid-pose candidate bank.

No experiment labels or reference transforms are accepted by this module.
Evaluation samples are fixed per pair. They are an internal model check, not
independent measurement ground truth. Per-hypothesis unmatched points remain
in the denominator, including at surface boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class SurfaceSamples:
    points: np.ndarray
    normals: np.ndarray
    triangle_ids: np.ndarray
    barycentric: np.ndarray
    blocks: np.ndarray

    def cloud(self):
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(self.points)
        result.normals = o3d.utility.Vector3dVector(self.normals)
        return result


class MeshCache:
    """Run-local cache. Input geometry must not be mutated during its lifetime."""

    def __init__(self, mesh):
        started = time.perf_counter()
        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        corners = self.vertices[self.triangles]
        cross = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        self.areas = lengths*.5
        self.area = float(self.areas.sum())
        if not np.isfinite(self.area) or self.area <= 0:
            raise ValueError("模型没有有效曲面面积。")
        self.normals = cross/np.maximum(lengths[:, None], 1e-30)
        self.origin = np.sum(corners.mean(axis=1)*self.areas[:, None], axis=0)/self.area
        local = o3d.geometry.TriangleMesh(mesh)
        local.translate(-self.origin)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(local))
        self.minimum = self.vertices.min(axis=0)
        self.extent = np.maximum(np.ptp(self.vertices, axis=0), 1e-9)
        self.diagonal = float(np.linalg.norm(self.extent))
        self._samples = {}
        self._clouds = {}
        self.build_seconds = time.perf_counter()-started
        self.queries = 0

    def sample(self, count, seed, priority=None):
        key = (int(count), int(seed), None if priority is None else np.packbits(priority).tobytes())
        if key in self._samples:
            return self._samples[key]
        rng = np.random.default_rng(seed)
        weights = self.areas.copy()
        if priority is not None:
            priority = np.asarray(priority, dtype=bool)
            if priority.shape != weights.shape:
                raise ValueError("选区与曲面面片数不匹配。")
            if np.any(priority) and np.any(~priority):
                weights[priority] *= .7/max(weights[priority].sum(), 1e-30)
                weights[~priority] *= .3/max(weights[~priority].sum(), 1e-30)
        weights /= weights.sum()
        ids = rng.choice(len(weights), max(200, int(count)), p=weights)
        root = np.sqrt(rng.random(len(ids)))
        second = rng.random(len(ids))
        barycentric = np.column_stack((1-root, root*(1-second), root*second))
        points = np.einsum("ij,ijk->ik", barycentric, self.vertices[self.triangles[ids]])
        cells = np.clip(((points-self.minimum)/self.extent*4).astype(int), 0, 3)
        blocks = cells[:, 0]+4*cells[:, 1]+16*cells[:, 2]
        value = SurfaceSamples(points, self.normals[ids], ids, barycentric, blocks)
        self._samples[key] = value
        return value

    def query(self, points):
        self.queries += 1
        result = self.scene.compute_closest_points(o3d.core.Tensor((points-self.origin).astype(np.float32)))
        ids = result["primitive_ids"].numpy().astype(np.int64)
        valid = ids < len(self.triangles)
        nearest = result["points"].numpy().astype(float)+self.origin
        normals = np.zeros_like(nearest)
        normals[valid] = self.normals[ids[valid]]
        return nearest, normals, ids


def rigid_valid(matrix):
    matrix = np.asarray(matrix)
    return bool(matrix.shape == (4, 4) and np.isfinite(matrix).all()
                and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
                and np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-5)
                and abs(np.linalg.det(matrix[:3, :3])-1) < 1e-5)


def observability(points, normals, weights):
    valid = np.isfinite(weights) & (weights > 1e-12)
    if valid.sum() < 12:
        return 0, float("inf")
    points, normals, weights = points[valid], normals[valid], weights[valid]
    weights = weights/weights.sum()
    centered = points-np.sum(points*weights[:, None], axis=0)
    radius = max(float(np.sqrt(np.sum(weights*np.sum(centered**2, axis=1)))), 1e-9)
    jacobian = np.column_stack((np.cross(centered, normals)/radius, normals))
    eigenvalues = np.linalg.eigvalsh((jacobian*weights[:, None]).T@jacobian)
    cutoff = max(float(eigenvalues[-1])*1e-6, 1e-12)
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    condition = float(eigenvalues[-1]/max(eigenvalues[0], 1e-30))
    return rank, condition


def block_means(values, blocks):
    counts = np.bincount(blocks, minlength=64)
    total = np.bincount(blocks, weights=values, minlength=64)
    return total[counts > 0]/counts[counts > 0]


class FixedEvaluator:
    def __init__(self, source, target, count, seed, partial=True):
        self.source, self.target = source, target
        self.samples = (source.sample(count, seed+701), target.sample(count, seed+702))
        diagonal = min(source.diagonal, target.diagonal)
        self.scales = np.maximum(diagonal*np.array([.0003, .001, .003, .01, .03]), 1e-5)
        # A distributed, fully constraining near-exact component is stronger
        # evidence than a tiny improvement in the broad deformation fit.
        # This geometric precision band is fixed for the whole candidate bank.
        self.precision_scale = max(diagonal*1e-6, 1e-6)
        self.scale_weights = np.array([4., 3., 2., 1., 1.])/11
        ratio = max(source.area, target.area)/min(source.area, target.area)
        if partial and ratio > 1.8:
            self.direction_weights = np.array([.8, .2] if source.area < target.area else [.2, .8])
        else:
            self.direction_weights = np.array([.5, .5])

    def evaluate(self, transform):
        if not rigid_valid(transform):
            raise ValueError("候选不是合法的刚体矩阵。")
        directions = []
        block_scores = []
        for direction, (samples, fixed, matrix) in enumerate((
            (self.samples[0], self.target, transform),
            (self.samples[1], self.source, np.linalg.inv(transform)),
        )):
            points = samples.points@matrix[:3, :3].T+matrix[:3, 3]
            nearest, normals, _ = fixed.query(points)
            distance = np.linalg.norm(points-nearest, axis=1)
            distance = np.nan_to_num(distance, nan=np.inf, posinf=np.inf)
            agreement = np.abs(np.einsum("ij,ij->i", samples.normals@matrix[:3, :3].T, normals))
            angular = np.clip((agreement-.5)/.5, 0, 1)
            response = np.exp(-.5*np.minimum(distance[:, None]/self.scales, 40)**2)*angular[:, None]
            per_point = response@self.scale_weights
            precision_response = np.exp(-.5*np.minimum(distance/self.precision_scale, 40)**2)*angular
            precision_mask = (distance <= self.precision_scale*3) & (agreement >= .5)
            precision_rank, _ = observability(points, normals, precision_mask.astype(float))
            precision_blocks = block_means(precision_mask.astype(float), samples.blocks)
            precise_consensus = bool(precision_rank == 6 and precision_mask.mean() >= .15
                                     and np.count_nonzero(precision_blocks >= .2) >= 4)
            precision_bonus = .25*precision_response if precise_consensus else np.zeros(len(distance))
            macro_loss = np.minimum(distance/self.scales[-1], 3)/3
            local_score = per_point+precision_bonus-.08*macro_loss
            blocks = block_means(local_score, samples.blocks)
            score = .5*float(local_score.mean())+.5*float(blocks.mean())
            tight = (distance <= self.scales[1]) & (agreement >= .5)
            rank, condition = observability(points, normals, tight.astype(float))
            directions.append(dict(
                score=score, count=len(distance), matched_count=int(tight.sum()),
                precision_coverage=float(precision_mask.mean()), precision_rank=precision_rank,
                precise_consensus=precise_consensus, precision_scale_mm=self.precision_scale,
                common_coverage=float(tight.mean()), support_by_scale=response.mean(axis=0).tolist(),
                median_mm=float(np.median(distance)), p90_mm=float(np.quantile(distance, .9)),
                rms_mm=float(np.sqrt(np.mean(distance**2))),
                observability_rank=rank, condition_number=condition if np.isfinite(condition) else None,
                supported_blocks=int(np.count_nonzero(block_means(tight.astype(float), samples.blocks) >= .2)),
                occupied_blocks=len(blocks),
            ))
            block_scores.append(blocks*self.direction_weights[direction]*2)
        scores = np.concatenate(block_scores)
        score = float(sum(w*d["score"] for w, d in zip(self.direction_weights, directions)))
        reliable = any(d["observability_rank"] == 6 and d["common_coverage"] >= .15
                       and d["supported_blocks"] >= 4 for d in directions)
        return dict(score=score, mode="common_surface" if reliable else "shape_fallback",
                    directions=directions, scales_mm=self.scales.tolist(),
                    fixed_denominators=[d["count"] for d in directions],
                    _block_scores=scores)

    def displacement(self, first, second):
        points = self.samples[0].points
        difference = points@first[:3, :3].T+first[:3, 3]-(points@second[:3, :3].T+second[:3, 3])
        return float(np.quantile(np.linalg.norm(difference, axis=1), .95))


@dataclass
class PoseCandidate:
    name: str
    stage: str
    parent: str | None
    transformation: np.ndarray
    evaluation: dict
    accepted: bool
    reason: str

    def as_dict(self):
        return dict(name=self.name, stage=self.stage, parent=self.parent,
                    transformation=self.transformation.tolist(), accepted=self.accepted, reason=self.reason,
                    evaluation={k: v for k, v in self.evaluation.items() if not k.startswith("_")})


class CandidateBank:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.candidates = []

    def add(self, name, stage, transform, parent=None, guarded=False):
        transform = np.asarray(transform, dtype=float).copy()
        evaluation = self.evaluator.evaluate(transform)
        accepted, reason = True, "independent hypothesis retained"
        if guarded and parent is not None:
            difference = evaluation["_block_scores"]-parent.evaluation["_block_scores"]
            uncertainty = float(np.std(difference)/np.sqrt(max(len(difference), 1)))
            tolerance = max(1e-7, .10*uncertainty)
            gain = evaluation["score"]-parent.evaluation["score"]
            accepted = gain > tolerance
            reason = "fixed evaluation improved" if accepted else "parent retained: improvement not resolved"
            evaluation["gain_over_parent"] = gain
            evaluation["comparison_tolerance"] = tolerance
        item = PoseCandidate(name, stage, parent.name if parent else None, transform, evaluation, accepted, reason)
        self.candidates.append(item)
        return item

    def finalists(self, count, minimum_displacement=None):
        separation = self.evaluator.scales[1] if minimum_displacement is None else minimum_displacement
        selected = []
        for item in sorted((c for c in self.candidates if c.accepted), key=lambda c: c.evaluation["score"], reverse=True):
            if not any(self.evaluator.displacement(item.transformation, other.transformation) < separation for other in selected):
                selected.append(item)
            if len(selected) >= count:
                break
        return selected

    def best(self):
        return max((c for c in self.candidates if c.accepted), key=lambda c: c.evaluation["score"])
