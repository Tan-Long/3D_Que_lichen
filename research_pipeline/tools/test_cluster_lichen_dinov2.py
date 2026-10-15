#!/usr/bin/env python3

import unittest

import numpy as np

from cluster_lichen_dinov2 import (
    MODEL_DIMENSIONS,
    MODEL_SPECS,
    fit_embedding_clusters,
    fit_projected_clusters,
    sample_patch_features,
    load_model,
)


class DinoSurfaceClusterTest(unittest.TestCase):
    def test_supported_model_dimensions(self):
        self.assertEqual(MODEL_DIMENSIONS["dinov2_vits14"], 384)
        self.assertEqual(MODEL_DIMENSIONS["dinov2_vitb14"], 768)
        self.assertEqual(MODEL_DIMENSIONS["dinov2_vitl14"], 1024)
        self.assertEqual(MODEL_DIMENSIONS["dinov3_vits16"], 384)
        self.assertEqual(MODEL_DIMENSIONS["dinov3_vitb16"], 768)
        self.assertEqual(MODEL_DIMENSIONS["dinov3_vitl16"], 1024)
        self.assertEqual(MODEL_SPECS["dinov3_vitl16"][1:4], (16, 512, 384))

    def test_dinov3_requires_approved_weights(self):
        with self.assertRaisesRegex(ValueError, "approved weights"):
            load_model("cpu", "dinov3_vits16")

    def test_reclusters_cached_projected_features(self):
        features = np.asarray([[0, 0], [0.1, 0], [9.9, 10], [10, 10]], dtype=np.float32)
        labels, centers = fit_projected_clusters(features, 2, 100, 7)
        self.assertEqual(centers.shape, (2, 2))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_patch_lookup_maps_full_image_corners(self):
        patches = np.arange(2 * 3 * 2).reshape(2, 3, 2)
        result = sample_patch_features(
            patches, np.asarray([0.0, 599.0]), np.asarray([0.0, 399.0]), 600, 400)
        np.testing.assert_array_equal(result[0], patches[0, 0])
        np.testing.assert_array_equal(result[1], patches[1, 2])

    def test_embedding_clusters_separate_two_groups(self):
        rng = np.random.default_rng(3)
        features = np.vstack((rng.normal(-2, 0.05, (20, 6)),
                              rng.normal(2, 0.05, (20, 6)))).astype(np.float32)
        labels, fit = fit_embedding_clusters(features, 2, 3, 40, 7)
        self.assertEqual(len(set(labels[:20])), 1)
        self.assertEqual(len(set(labels[20:])), 1)
        self.assertNotEqual(labels[0], labels[-1])
        self.assertEqual(fit["projected_features"].shape, (40, 3))


if __name__ == "__main__":
    unittest.main()
