import unittest

import numpy as np

from build_tsdf_mesh import select_consistent_frames


class FrameConsistencyTest(unittest.TestCase):
    def test_drops_minority_radial_shell(self) -> None:
        shape = (2, 2, 20)
        good = np.array([0, 20, 40, 60]) + 10
        bad = np.array([0, 20, 40, 60]) + 15
        observations = {frame: good for frame in range(8)}
        observations.update({8: bad, 9: bad})

        selected, report = select_consistent_frames(observations, shape, 0.55, 2, 0.6)

        self.assertEqual(selected, list(range(8)))
        self.assertTrue(report["applied"])
        self.assertEqual(report["dropped_frame_ids"], [8, 9])

    def test_aborts_when_filter_would_remove_too_many_frames(self) -> None:
        shape = (1, 1, 20)
        observations = {0: np.array([1]), 1: np.array([1]), 2: np.array([10])}

        selected, report = select_consistent_frames(observations, shape, 0.55, 1, 0.8)

        self.assertEqual(selected, [0, 1, 2])
        self.assertFalse(report["applied"])


if __name__ == "__main__":
    unittest.main()
