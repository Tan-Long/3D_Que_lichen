#!/usr/bin/env python3

import unittest

import numpy as np

from build_observed_cylinder_pilot import cylindrical_candidates, voxel_reduce


class ObservedCylinderPilotTest(unittest.TestCase):
    def test_voxel_support_counts_distinct_frames(self) -> None:
        points = np.array([[.001, 0, 0], [.002, 0, 0], [.003, 0, 0], [.011, 0, 0]])
        reduced, support = voxel_reduce(points, np.array([1, 1, 2, 1]), .01)
        self.assertEqual(len(reduced), 2)
        self.assertEqual(support.tolist(), [2, 1])

    def test_triangulation_does_not_bridge_unobserved_wedge(self) -> None:
        angles = np.deg2rad(np.r_[np.arange(-150, -20, 5), np.arange(20, 151, 5)])
        heights = np.arange(.2, .61, .01)
        theta, height = np.meshgrid(angles, heights)
        points = np.column_stack((.08 * np.cos(theta.ravel()), height.ravel(),
                                  .08 * np.sin(theta.ravel())))
        slices = [{"height_m": .2, "x": 0.0, "y": 0.0},
                  {"height_m": .4, "x": 0.0, "y": 0.0},
                  {"height_m": .6, "x": 0.0, "y": 0.0}]
        triangles = cylindrical_candidates(
            points, np.zeros(3), np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), slices,
            max_angle_deg=8.0, max_height_edge=.015, max_edge=.02)
        triangle_angles = np.rad2deg(np.arctan2(points[triangles, 2], points[triangles, 0]))
        self.assertFalse(np.any(np.ptp(triangle_angles, axis=1) > 20))


if __name__ == "__main__":
    unittest.main()
