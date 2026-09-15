"""Run-local feature matches shared by RANSAC restarts.

Matches retain Open3D 0.19's ordering, mutual filtering and small-set fallback.
Only nearest-neighbour work is cached; every RANSAC restart still runs.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


class FeatureMatchCache:
    """Own one immutable pair of feature descriptors for a registration run."""

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self._forward = None
        self._mutual = None
        self.searches = 0

    def get(self, mutual_filter):
        if self._forward is None:
            self._forward = o3d.pipelines.registration.correspondences_from_features(
                self.source, self.target, mutual_filter=False)
            self.searches += 1
        if not mutual_filter:
            return self._forward
        if self._mutual is None:
            reverse = o3d.pipelines.registration.correspondences_from_features(
                self.target, self.source, mutual_filter=False)
            self.searches += 1
            forward_pairs = np.asarray(self._forward)
            reverse_pairs = np.asarray(reverse)
            mutual = forward_pairs[
                reverse_pairs[forward_pairs[:, 1], 1] == forward_pairs[:, 0]]
            # Open3D uses int(float(0.1) * num_source_points), including the
            # float32 product and integer truncation at the fallback boundary.
            minimum = int(np.float32(.1) * np.float32(self.source.num()))
            self._mutual = (o3d.utility.Vector2iVector(mutual)
                            if len(mutual) >= minimum else self._forward)
        return self._mutual
