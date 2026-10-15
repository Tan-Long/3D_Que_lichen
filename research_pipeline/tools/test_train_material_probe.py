#!/usr/bin/env python3

import unittest

import numpy as np

from train_material_probe import (
    PURE_TARGETS,
    fit_probe,
    output_complete,
    probability_colors,
    production_uids,
    reconstruct_features,
    source_dir_for,
)


class MaterialProbeTest(unittest.TestCase):
    def test_only_pure_labels_are_training_targets(self) -> None:
        self.assertEqual(PURE_TARGETS, {"bark": 0.0, "lichen": 1.0})

    def test_finds_direct_then_cached_k3_artifact(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "tree" / "dinov3_vitl16_k3_cached"
            cached.mkdir(parents=True)
            (cached / "surface_cluster_triangle_data.npz").touch()
            self.assertEqual(source_dir_for("tree", root), cached)
            direct = root / "tree" / "dinov3_vitl16_k3"
            direct.mkdir()
            (direct / "surface_cluster_triangle_data.npz").touch()
            self.assertEqual(source_dir_for("tree", root), direct)

    def test_reads_production_uids_and_requires_all_resume_outputs(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "production.csv"
            manifest.write_text("scan_uid,scan\nfirst,Tree 1\nsecond,Tree 2\n", encoding="utf-8")
            self.assertEqual(production_uids(manifest), ["first", "second"])
            output = root / "first" / "material_probe_pure_v2"
            output.mkdir(parents=True)
            for name in ("material_probability_mesh.ply", "material_probabilities.npz"):
                (output / name).touch()
            self.assertFalse(output_complete("first", root))
            (output / "material_probe_metrics.json").touch()
            self.assertTrue(output_complete("first", root))

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
