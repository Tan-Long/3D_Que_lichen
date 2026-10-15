import json
import tempfile
import unittest
from pathlib import Path

from finalize_baseline_manifest import RUN_NAME, build_manifest


class FinalizeBaselineManifestTest(unittest.TestCase):
    def test_uses_latest_review_and_exact_baseline_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            manifests.mkdir()
            row = {"scan_uid": "scan-1", "scan": "Tree 1", "source": "new",
                   "raw_path": "/raw/tree-1", "qa_status": "pass", "qa_reasons": ""}
            (manifests / "geometry_qa_split_all.json").write_text(
                json.dumps({"pass": [row], "deferred": []}), encoding="utf-8")
            (manifests / "deferred_manual_review.json").write_text(json.dumps({"reviews": {
                "scan-1": {"decision": "approved", "updated_at": "second"}}}), encoding="utf-8")
            (manifests / "pass_manual_review.json").write_text(json.dumps({"reviews": {
                "scan-1": {"decision": "rejected", "updated_at": "first"}}}), encoding="utf-8")
            output = root / "artifacts/meshes_all/scan-1" / RUN_NAME
            output.mkdir(parents=True)
            for name in ("trunk_mesh_tsdf.ply", "metrics.json", "preview.png"):
                (output / name).write_text("x", encoding="utf-8")

            manifest = build_manifest(root)

            self.assertEqual(manifest["counts"]["downstream_geometry"], 1)
            self.assertEqual(manifest["rows"][0]["manual_decision"], "approved")
            self.assertEqual(Path(manifest["rows"][0]["mesh_path"]).parent.name, RUN_NAME)


if __name__ == "__main__":
    unittest.main()
