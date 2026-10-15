import json
import tempfile
import unittest
from pathlib import Path

from serve_deferred_gallery import (
    CLUSTER_SCANS,
    MATERIAL_REVIEW_PATH,
    REVIEW_PATHS,
    ROOT,
    cluster_review_path,
    dinov3_k3_catalog,
    merged_reviews,
    record_cluster_review,
    record_material_review,
    record_review,
    record_tree_coverage_review,
    review_group,
)


class ReviewStoreTest(unittest.TestCase):
    def test_model_comparison_viewer_lists_only_dinov3_backbones(self) -> None:
        html = (ROOT / "surface_cluster_gallery.html").read_text(encoding="utf-8")
        self.assertIn("dinov3compare", html)
        self.assertIn("dinov3label", html)
        self.assertIn("dinov3ksweep", html)
        self.assertIn("variantParam==='material'", html)
        self.assertIn("material_probe_pure_v2", html)
        self.assertIn('id="sample-search"', html)
        self.assertIn('id="sample-filter"', html)
        self.assertIn('class="inspector"', html)
        self.assertIn('id="review-progress"', html)
        self.assertIn("function visibleIndices()", html)
        self.assertIn("key==='q'", html)
        self.assertIn("<kbd>Q</kbd> đổi lớp", html)
        self.assertIn("material_probability_mesh.ply", html)
        self.assertIn('id="material-review"', html)
        self.assertIn("/api/material-reviews", html)
        self.assertIn('data-material-band=', html)
        self.assertIn("['review_band','Cần xem'", html)
        self.assertIn("variantParam==='dinov3full'", html)
        self.assertIn("/api/dinov3-k3-catalog", html)
        self.assertIn("Bao phủ địa y toàn thân", html)
        self.assertIn("isLabel?'dinov3_vitl16_k3_cached'", html)
        self.assertNotIn('data-model="dinov2_', html)
        self.assertIn("Cây bụi/nền", html)
        self.assertIn("['mixed_lichen','Lẫn · thiên địa y']", html)
        self.assertIn("['mixed_bark','Lẫn · thiên vỏ']", html)
        self.assertIn("['mixed','Lẫn cân bằng']", html)
        for variant in ("dinov3_vits16_k5", "dinov3_vitb16_k5", "dinov3_vitl16_k5"):
            self.assertIn(f'data-model="{variant}"', html)
        for variant in ("dinov3_vitl16_k2_cached", "dinov3_vitl16_k3_cached",
                        "dinov3_vitl16_k4_cached", "dinov3_vitl16_k5"):
            self.assertIn(f'data-kmodel="{variant}"', html)
        self.assertNotIn('data-kmodel="dinov3_vitl16_k8_cached"', html)

    def test_surface_cluster_pilot_contains_expansion_set(self) -> None:
        self.assertGreaterEqual(len(CLUSTER_SCANS), 137)

    def test_full_k3_catalog_contains_every_approved_tree(self) -> None:
        catalog = dinov3_k3_catalog()
        self.assertEqual(catalog["ready"], 137)
        self.assertEqual(catalog["total"], 137)
        self.assertTrue(all(row["variant"].startswith("dinov3_vitl16_k3")
                            for row in catalog["samples"]))

    def test_persists_and_replaces_a_valid_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            allowed = {"scan-1": "Tree 1"}
            record_review(path, allowed, "scan-1", "approved", "first")
            record_review(path, allowed, "scan-1", "rejected", "second")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["reviews"]["scan-1"]["decision"], "rejected")
            self.assertEqual(saved["reviews"]["scan-1"]["updated_at"], "second")

    def test_rejects_unknown_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                record_review(Path(directory) / "reviews.json", {}, "unknown", "approved")

    def test_accepts_needs_review_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            review = record_review(path, {"scan-1": "Tree 1"}, "scan-1", "needs_review")
            self.assertEqual(review["decision"], "needs_review")

    def test_material_review_store_is_separate(self) -> None:
        self.assertEqual(MATERIAL_REVIEW_PATH.name, "material_reviews.json")
        self.assertNotIn(MATERIAL_REVIEW_PATH, REVIEW_PATHS.values())

    def test_persists_material_decision_by_band(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "material.json"
            scan_uid = next(iter(CLUSTER_SCANS))
            record_material_review(path, scan_uid, "review_band", "mixed_bark", "now")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                saved["reviews"][scan_uid]["review_band"]["decision"], "mixed_bark")

    def test_rejects_unknown_material_band(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "band must be"):
                record_material_review(
                    Path(directory) / "material.json", next(iter(CLUSTER_SCANS)),
                    "whole_tree", "bark")

    def test_selects_review_group_from_query(self) -> None:
        self.assertEqual(review_group("/api/reviews"), "deferred")
        self.assertEqual(review_group("/api/reviews?group=pass"), "pass")
        self.assertEqual(review_group("/api/reviews?group=all"), "all")
        with self.assertRaises(ValueError):
            review_group("/api/reviews?group=unknown")

    def test_merges_existing_review_groups(self) -> None:
        merged = merged_reviews(
            {"reviews": {"pass-1": {"decision": "approved"}}},
            {"reviews": {"deferred-1": {"decision": "needs_review"}}},
        )
        self.assertEqual(set(merged["reviews"]), {"pass-1", "deferred-1"})

    def test_merge_keeps_latest_duplicate_decision(self) -> None:
        merged = merged_reviews(
            {"reviews": {"scan-1": {"decision": "needs_review", "updated_at": "second"}}},
            {"reviews": {"scan-1": {"decision": "rejected", "updated_at": "first"}}},
        )
        self.assertEqual(merged["reviews"]["scan-1"]["decision"], "needs_review")

    def test_persists_cluster_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster_reviews.json"
            scan_uid = next(iter(CLUSTER_SCANS))
            record_cluster_review(path, scan_uid, 3, "lichen", "now")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["reviews"][scan_uid]["3"]["decision"], "lichen")

    def test_accepts_vegetation_cluster_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan_uid = next(iter(CLUSTER_SCANS))
            review = record_cluster_review(
                Path(directory) / "reviews.json", scan_uid, 0, "vegetation", "now")
            self.assertEqual(review["decision"], "vegetation")

    def test_accepts_mixed_cluster_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan_uid = next(iter(CLUSTER_SCANS))
            review = record_cluster_review(
                Path(directory) / "reviews.json", scan_uid, 0, "mixed", "now",
                cluster_count=3)
            self.assertEqual(review["decision"], "mixed")

    def test_accepts_dominant_mixed_cluster_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            scan_uid = next(iter(CLUSTER_SCANS))
            for cluster, decision in enumerate(("mixed_lichen", "mixed_bark")):
                review = record_cluster_review(
                    path, scan_uid, cluster, decision, "now", cluster_count=3)
                self.assertEqual(review["decision"], decision)

    def test_persists_tree_coverage_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan_uid = next(iter(CLUSTER_SCANS))
            review = record_tree_coverage_review(
                Path(directory) / "coverage.json", scan_uid, "full", "now")
            self.assertEqual(review["decision"], "full")

    def test_k3_review_rejects_fourth_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan_uid = next(iter(CLUSTER_SCANS))
            with self.assertRaisesRegex(ValueError, "0 to 2"):
                record_cluster_review(
                    Path(directory) / "reviews.json", scan_uid, 3, "lichen",
                    cluster_count=3)

    def test_separates_classical_and_dinov3_cluster_reviews(self) -> None:
        classical = cluster_review_path("/api/cluster-reviews")
        dinov3 = cluster_review_path("/api/dinov3-cluster-reviews")
        dinov3_k3 = cluster_review_path("/api/dinov3-k3-cluster-reviews")
        self.assertIsNotNone(classical)
        self.assertIsNotNone(dinov3)
        self.assertIsNotNone(dinov3_k3)
        self.assertNotEqual(classical, dinov3)
        self.assertNotEqual(dinov3, dinov3_k3)

    def test_rejects_invalid_cluster_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scan_uid = next(iter(CLUSTER_SCANS))
            with self.assertRaises(ValueError):
                record_cluster_review(Path(directory) / "reviews.json", scan_uid, 5, "lichen")
            with self.assertRaises(ValueError):
                record_cluster_review(Path(directory) / "reviews.json", scan_uid, 0, "moss")


if __name__ == "__main__":
    unittest.main()
