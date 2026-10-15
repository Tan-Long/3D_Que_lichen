#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

from run_dinov2_expansion import load_all_production_rows, load_selected_rows


class DinoExpansionTest(unittest.TestCase):
    def test_all_production_rows_get_unknown_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "production.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scan_uid", "scan"])
                writer.writeheader()
                writer.writerow({"scan_uid": "a", "scan": "A"})
            rows = load_all_production_rows(path)
            self.assertEqual(rows, [{"scan_uid": "a", "scan": "A", "label": "unknown"}])

    def test_selection_preserves_order_and_adds_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scan_uid", "label"])
                writer.writeheader()
                writer.writerows([{"scan_uid": "b", "label": "unknown"},
                                  {"scan_uid": "a", "label": "lichen_present"}])
            with (root / "production.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scan_uid", "scan"])
                writer.writeheader()
                writer.writerows([{"scan_uid": "a", "scan": "A"},
                                  {"scan_uid": "b", "scan": "B"}])
            rows = load_selected_rows(root / "selection.csv", root / "production.csv")
            self.assertEqual([row["scan_uid"] for row in rows], ["b", "a"])
            self.assertEqual(rows[0]["label"], "unknown")


if __name__ == "__main__":
    unittest.main()
