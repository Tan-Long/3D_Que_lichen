#!/usr/bin/env python3

import unittest

import numpy as np

from train_material_probe import (
    SOFT_TARGETS,
    fit_probe,
    probability_colors,
    reconstruct_features,
)


class MaterialProbeTest(unittest.TestCase):
    def test_mixed_labels_are_ordered_soft_targets(self) -> None:
        self.assertEqual(
            [SOFT_TARGETS[name] for name in
             ("bark", "mixed_bark", "mixed", "mixed_lichen", "lichen")],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
        self.assertNotIn("vegetation", SOFT_TARGETS)

    def test_inverse_pca_reconstruction_is_normalized(self) -> None:
        projected = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        features = reconstruct_features(projected, np.zeros((1, 2), dtype=np.float32),
                                        np.eye(2, dtype=np.float32))
        np.testing.assert_allclose(features, np.eye(2), atol=1e-6)

    def test_probe_orders_bark_and_lichen_features(self) -> None:
        features = np.asarray([[-1.0], [-0.8], [0.8], [1.0]], dtype=np.float32)
        targets = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        coefficients = fit_probe(features, targets, np.ones(4, dtype=np.float32), 0.01)
        predictions = np.column_stack((features, np.ones(4))) @ coefficients
        self.assertLess(predictions[:2].mean(), predictions[2:].mean())

    def test_probability_colors_reserve_green_for_vegetation(self) -> None:
        colors = probability_colors(np.asarray([0.0, 0.5, 1.0, np.nan]),
                                    np.asarray([False, False, False, True]))
        np.testing.assert_allclose(colors[3], [0.20, 0.55, 0.27])
        self.assertFalse(np.array_equal(colors[0], colors[2]))


if __name__ == "__main__":
    unittest.main()
