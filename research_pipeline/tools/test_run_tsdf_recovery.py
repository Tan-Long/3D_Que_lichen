#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from run_tsdf_recovery import select_deferred_and_low_support


class RunTsdfRecoveryTest(unittest.TestCase):
    def test_selects_union_of_deferred_and_low_support(self) -> None:
        rows = [
            {"scan_uid": "deferred", "usable_geometry": "True", "duplicate_canonical": "True"},
            {"scan_uid": "low", "usable_geometry": "True", "duplicate_canonical": "True"},
            {"scan_uid": "good", "usable_geometry": "True", "duplicate_canonical": "True"},
            {"scan_uid": "duplicate", "usable_geometry": "True", "duplicate_canonical": "False"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for uid, supported in (("low", 4), ("good", 8), ("duplicate", 1)):
                output = root / uid / "v5mm_step12_pixel2_conf1_r22"
                output.mkdir(parents=True)
                (output / "report.json").write_text(json.dumps({
                    "observed_voxels": 10,
                    "observed_voxels_multi_frame": supported,
                }))
            selected, stats = select_deferred_and_low_support(rows, {"deferred"}, root, 0.5)
        self.assertEqual({row["scan_uid"] for row in selected}, {"deferred", "low"})
        self.assertEqual(stats["low_support"], 1)
        self.assertEqual(stats["missing_support_reports"], 1)


if __name__ == "__main__":
    unittest.main()
