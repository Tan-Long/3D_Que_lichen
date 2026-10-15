#!/usr/bin/env python3
"""Regression checks for centreline fitting on cluttered trunk slices."""

from __future__ import annotations

import unittest

import numpy as np

from measure_mesh_surface import estimate_centerline
from build_tsdf_mesh import clean_centerline


class CenterlineRegressionTest(unittest.TestCase):
    def test_keeps_consistent_circle_when_attached_clutter_lowers_inlier_fraction(self) -> None:
        rng = np.random.default_rng(2205)
        vertices = []
        for height in np.arange(0.2, 1.3, 0.1):
            angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
            trunk = np.column_stack((
                0.06 * np.cos(angles),
                np.full_like(angles, height),
                0.06 * np.sin(angles),
            ))
            # Points attached to the same component (bark flares/branch fragments)
            # make the true circle less than half of the slice population.
            clutter_angles = rng.uniform(0, 2 * np.pi, 170)
            clutter_radii = rng.uniform(0.10, 0.22, 170)
            clutter = np.column_stack((
                clutter_radii * np.cos(clutter_angles),
                rng.uniform(height - 0.045, height + 0.045, 170),
                clutter_radii * np.sin(clutter_angles),
            ))
            vertices.extend((trunk, clutter))
        cloud = np.vstack(vertices)
        result = clean_centerline(
            estimate_centerline(cloud, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        )
        self.assertGreaterEqual(len(result["slices"]), 8)
        self.assertLess(abs(np.median([s["radius_m"] for s in result["slices"]]) - 0.06), 0.01)


if __name__ == "__main__":
    unittest.main()
