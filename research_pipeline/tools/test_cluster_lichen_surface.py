#!/usr/bin/env python3

import unittest

import numpy as np

from cluster_lichen_surface import local_surface_mean, robust_standardize, smooth_triangle_labels


class SurfaceClusterTest(unittest.TestCase):
    def test_local_mean_only_uses_connected_faces(self):
        triangles = np.asarray([[0, 1, 2], [1, 2, 3], [4, 5, 6]])
        values = np.asarray([[1.0], [3.0], [9.0]])
        result = local_surface_mean(values, triangles)[:, 0]
        self.assertAlmostEqual(result[2], 9.0)
        self.assertGreater(result[0], 1.0)
        self.assertLess(result[1], 3.0)

    def test_standardization_handles_constant_feature(self):
        result, center, scale = robust_standardize(np.asarray([[1.0, 2.0], [3.0, 2.0]]))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(scale[1], 1.0)
        self.assertEqual(center[1], 2.0)

    def test_smoothing_does_not_cross_disconnected_faces(self):
        triangles = np.asarray([[0, 1, 2], [1, 2, 3], [4, 5, 6]])
        labels = np.asarray([0, 1, 1], dtype=np.int16)
        observed = np.ones(3, dtype=bool)
        result = smooth_triangle_labels(triangles, labels, observed, 2, 1)
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[2], 1)


if __name__ == "__main__":
    unittest.main()
