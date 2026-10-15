#!/usr/bin/env python3

import unittest

import numpy as np
from scipy.spatial import cKDTree

from build_tsdf_mesh import adaptive_frame_ids, near_weak_support


class AdaptiveSupportTest(unittest.TestCase):
    def test_reserves_holdout_without_dropping_baseline(self) -> None:
        fusion, holdout = adaptive_frame_ids([0, 3, 6, 9], list(range(10)), 2)
        self.assertTrue({0, 3, 6, 9}.issubset(fusion))
        self.assertEqual(len(holdout), 2)
        self.assertFalse(set(fusion) & set(holdout))
        self.assertEqual(set(fusion) | set(holdout), set(range(10)))

    def test_selects_only_points_near_weak_observations(self) -> None:
        tree = cKDTree(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        selected = near_weak_support(
            np.array([[0.004, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 1.0, 1.006]]),
            tree, 0.0075,
        )
        self.assertEqual(selected.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
