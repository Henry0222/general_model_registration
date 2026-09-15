from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from auto_alignment.config import AlignmentConfig
from auto_alignment.feature_matching import FeatureMatchCache
from auto_alignment.registration import _global_registration


def feature(values):
    result = o3d.pipelines.registration.Feature()
    result.data = np.asarray(values, dtype=float)
    return result


@pytest.mark.parametrize("kind", ["random", "ties", "fallback", "small"])
def test_cached_matches_equal_native_open3d_including_order_and_fallback(kind):
    rng = np.random.default_rng(47)
    source = rng.normal(size=(33, 100))
    target = rng.normal(size=(33, 71))
    if kind == "ties":
        source[:, 1:20] = source[:, :1]
        target[:, :20] = source[:, :1]
    elif kind == "fallback":
        source[:] = 0
        target[:] = 0
    elif kind == "small":
        source, target = source[:, :5], target[:, :3]
    sf, tf = feature(source), feature(target)
    cache = FeatureMatchCache(sf, tf)
    for mutual in (False, True, True, False, True):
        expected = o3d.pipelines.registration.correspondences_from_features(
            sf, tf, mutual_filter=mutual)
        actual = cache.get(mutual)
        assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_eight_restarts_search_each_feature_direction_only_once(monkeypatch):
    rng = np.random.default_rng(71)
    sf, tf = feature(rng.normal(size=(33, 91))), feature(rng.normal(size=(33, 60)))
    native = o3d.pipelines.registration.correspondences_from_features
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return native(*args, **kwargs)

    monkeypatch.setattr(o3d.pipelines.registration, "correspondences_from_features", counted)
    cache = FeatureMatchCache(sf, tf)
    for restart in range(8):
        cache.get(restart % 3 != 0)
    assert len(calls) == 2
    assert cache.searches == 2
    # A different registration run must not reuse a previous model's indices.
    FeatureMatchCache(tf, sf).get(False)
    assert len(calls) == 3


def test_cached_ransac_retains_rigid_transform_recovery():
    rng = np.random.default_rng(93)
    points = rng.normal(size=(140, 3))
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    source.normals = o3d.utility.Vector3dVector(points / np.linalg.norm(points, axis=1)[:, None])
    expected = np.eye(4)
    expected[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([.3, -.5, .8])
    expected[:3, 3] = [4, -7, 3]
    target = o3d.geometry.PointCloud(source).transform(expected)
    sf = feature(rng.normal(size=(33, len(points))))
    matches = FeatureMatchCache(sf, sf).get(True)
    config = AlignmentConfig(ransac_max_iterations=500)
    for cached in (None, matches):
        o3d.utility.random.seed(29)
        result = _global_registration(source, target, sf, sf, .2, config,
                                      correspondences=cached)
        assert result.fitness == 1.0
        assert np.allclose(result.transformation, expected, atol=1e-10)
