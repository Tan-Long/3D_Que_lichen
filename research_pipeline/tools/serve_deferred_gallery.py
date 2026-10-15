#!/usr/bin/env python3
"""Serve the deferred mesh reviewer and persist local review decisions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "manifests/geometry_qa_split_all.json"
PRODUCTION_PATH = ROOT / "manifests/production_baseline_v1_approved.csv"
REVIEW_PATHS = {
    "deferred": ROOT / "manifests/deferred_manual_review.json",
    "pass": ROOT / "manifests/pass_manual_review.json",
}
DECISIONS = {"approved", "needs_review", "rejected"}
CLUSTER_REVIEW_PATHS = {
    "/api/cluster-reviews": ROOT / "artifacts/surface_cluster_pilot/cluster_reviews_k5.json",
    "/api/dinov3-cluster-reviews": ROOT / "artifacts/surface_cluster_pilot/cluster_reviews_dinov3_k5.json",
    "/api/dinov3-k3-cluster-reviews": ROOT / "artifacts/surface_cluster_pilot/cluster_reviews_dinov3_k3.json",
}
CLUSTER_DECISIONS = {
    "lichen", "bark", "vegetation", "mixed_lichen", "mixed_bark", "mixed", "uncertain"
}
TREE_COVERAGE_PATH = ROOT / "artifacts/surface_cluster_pilot/tree_coverage_reviews.json"
TREE_COVERAGE_DECISIONS = {"none", "sparse", "patchy", "high", "full", "uncertain"}
MATERIAL_REVIEW_PATH = ROOT / "artifacts/surface_cluster_pilot/material_reviews.json"
CLUSTER_SCANS = {
    "new-1-tra-van-tra-my-video-20260630-113436-6541ffc0": "1-tra-van-tra-my__video_20260630_113436",
    "new-cay-0002-1905-xh-2-981697e8": "cay_0002_1905_XH-2",
    "new-cay-0003-1905-xh-3-997db922": "cay_0003_1905_XH-3",
    "new-cay-0005-2105-m-1-1-strayscanner-export-20260522-231359-25566b4c": "cay_0005_2105_M-1.1",
    "new-cay-0006-2105-cl-6-strayscanner-export-20260522-231359-17346938": "cay_0006_2105_CL-6",
    "new-2-tra-van-tra-my-video-20260630-113709-ab0797e9": "2-tra-van-tra-my_video_20260630_113709",
    "new-cay-0002-2005-nh-3-a0d9b571": "cay_0002_2005_NH-3",
    "new-cay-0003-2005-nh-2-a4b75b5f": "cay_0003_2005_NH-2",
    "new-cay-0004-2005-nh-4-fc65eb5c": "cay_0004_2005_NH-4",
    "new-cay-0004-1805-b79e331a": "cay_0004_1805",
    "new-cay-0006-2005-cc-1-892f15bf": "cay_0006_2005_CC-1",
    "new-cay-0007-1805-9d18cdb1": "cay_0007_1805",
    "new-cay-0009-1805-95ebe75a": "cay_0009_1805",
    "new-cay-0012-1805-0f950e4a": "cay_0012_1805",
    "new-cay-0013-1805-52942c11": "cay_0013_1805",
}


def production_scans(path: Path = PRODUCTION_PATH) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["scan_uid"]: row["scan"] for row in csv.DictReader(handle)}


CLUSTER_SCANS.update(production_scans())


def read_reviews(path: Path) -> dict:
    if not path.exists():
        return {"reviews": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data.get("reviews"), dict) else {"reviews": {}}


def merged_reviews(*values: dict) -> dict:
    reviews = {}
    for value in values:
        for scan_uid, review in value.get("reviews", {}).items():
            current = reviews.get(scan_uid)
            if current is None or review.get("updated_at", "") > current.get("updated_at", ""):
                reviews[scan_uid] = review
    return {"reviews": reviews}


def reviews_for_group(group: str) -> dict:
    if group == "all":
        return merged_reviews(*(read_reviews(path) for path in REVIEW_PATHS.values()))
    return read_reviews(REVIEW_PATHS[group])


def record_review(path: Path, allowed: dict[str, str], scan_uid: str,
                  decision: str, updated_at: str | None = None) -> dict:
    if scan_uid not in allowed:
        raise ValueError("scan_uid is not in the selected QA group")
    if decision not in DECISIONS:
        raise ValueError("decision must be approved, needs_review, or rejected")
    timestamp = updated_at or datetime.now(timezone.utc).isoformat()
    data = read_reviews(path)
    review = {"scan_uid": scan_uid, "scan": allowed[scan_uid],
              "decision": decision, "updated_at": timestamp}
    data["reviews"][scan_uid] = review
    data["updated_at"] = timestamp
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return review


def record_cluster_review(path: Path, scan_uid: str, cluster: int,
                          decision: str, updated_at: str | None = None,
                          cluster_count: int = 5) -> dict:
    if scan_uid not in CLUSTER_SCANS:
        raise ValueError("scan_uid is not in the surface-cluster pilot")
    if not isinstance(cluster, int) or not 0 <= cluster < cluster_count:
        raise ValueError(f"cluster must be an integer from 0 to {cluster_count - 1}")
    if decision not in CLUSTER_DECISIONS:
        raise ValueError(
            "decision must be lichen, bark, vegetation, mixed_lichen, mixed_bark, mixed, or uncertain"
        )
    timestamp = updated_at or datetime.now(timezone.utc).isoformat()
    data = read_reviews(path)
    scan_reviews = data["reviews"].setdefault(scan_uid, {})
    review = {"scan_uid": scan_uid, "scan": CLUSTER_SCANS[scan_uid],
              "cluster": cluster, "decision": decision, "updated_at": timestamp}
    scan_reviews[str(cluster)] = review
    data["updated_at"] = timestamp
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return review


def record_tree_coverage_review(path: Path, scan_uid: str, decision: str,
                                updated_at: str | None = None) -> dict:
    if scan_uid not in CLUSTER_SCANS:
        raise ValueError("scan_uid is not in the approved production set")
    if decision not in TREE_COVERAGE_DECISIONS:
        raise ValueError("decision must be none, sparse, patchy, high, full, or uncertain")
    timestamp = updated_at or datetime.now(timezone.utc).isoformat()
    data = read_reviews(path)
    review = {"scan_uid": scan_uid, "scan": CLUSTER_SCANS[scan_uid],
              "decision": decision, "updated_at": timestamp}
    data["reviews"][scan_uid] = review
    data["updated_at"] = timestamp
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return review


def dinov3_k3_catalog(path: Path = PRODUCTION_PATH) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples = []
    for row in rows:
        uid = row["scan_uid"]
        variants = ("dinov3_vitl16_k3", "dinov3_vitl16_k3_cached")
        variant = next((name for name in variants if (
            ROOT / "artifacts/surface_cluster_pilot" / uid / name /
            "surface_cluster_metrics.json").is_file()), None)
        if variant is not None:
            samples.append({"uid": uid, "name": row["scan"], "prior": "baseline approved",
                            "variant": variant})
    return {"samples": samples, "ready": len(samples), "total": len(rows)}


def review_group(request_path: str) -> str:
    group = parse_qs(urlsplit(request_path).query).get("group", ["deferred"])[0]
    if group not in (*REVIEW_PATHS, "all"):
        raise ValueError("group must be deferred, pass, or all")
    return group


def cluster_review_path(request_path: str) -> Path | None:
    return CLUSTER_REVIEW_PATHS.get(request_path.split("?", 1)[0])


def group_scans(group: str) -> dict[str, str]:
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    if group == "all":
        return {row["scan_uid"]: row["scan"] for key in REVIEW_PATHS for row in data[key]}
    return {row["scan_uid"]: row["scan"] for row in data[group]}


def review_path(group: str, scan_uid: str) -> Path:
    if group != "all":
        return REVIEW_PATHS[group]
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    for key, path in REVIEW_PATHS.items():
        if any(row["scan_uid"] == scan_uid for row in data[key]):
            return path
    raise ValueError("scan_uid is not in the selected QA group")


class GalleryHandler(SimpleHTTPRequestHandler):
    def send_json(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        cluster_path = cluster_review_path(self.path)
        if cluster_path is not None:
            self.send_json(200, read_reviews(cluster_path))
            return
        request_path = self.path.split("?", 1)[0]
        if request_path == "/api/tree-coverage-reviews":
            self.send_json(200, read_reviews(TREE_COVERAGE_PATH))
            return
        if request_path == "/api/material-reviews":
            self.send_json(200, read_reviews(MATERIAL_REVIEW_PATH))
            return
        if request_path == "/api/dinov3-k3-catalog":
            self.send_json(200, dinov3_k3_catalog())
            return
        if request_path == "/api/reviews":
            try:
                group = review_group(self.path)
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, reviews_for_group(group))
            return
        super().do_GET()

    def do_POST(self) -> None:
        cluster_path = cluster_review_path(self.path)
        if cluster_path is not None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4096:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                review = record_cluster_review(
                    cluster_path, payload.get("scan_uid"),
                    payload.get("cluster"), payload.get("decision"),
                    cluster_count=3 if "dinov3-k3" in self.path else 5)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, {"review": review})
            return
        if self.path.split("?", 1)[0] == "/api/tree-coverage-reviews":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4096:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                review = record_tree_coverage_review(
                    TREE_COVERAGE_PATH, payload.get("scan_uid"), payload.get("decision"))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, {"review": review})
            return
        if self.path.split("?", 1)[0] == "/api/material-reviews":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4096:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                review = record_review(
                    MATERIAL_REVIEW_PATH, CLUSTER_SCANS,
                    payload.get("scan_uid"), payload.get("decision"))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, {"review": review})
            return
        if self.path.split("?", 1)[0] != "/api/reviews":
            self.send_json(404, {"error": "not found"})
            return
        try:
            group = review_group(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            review = record_review(review_path(group, payload.get("scan_uid")), group_scans(group),
                                   payload.get("scan_uid"), payload.get("decision"))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        self.send_json(200, {"review": review})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = partial(GalleryHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"QA reviewer: http://127.0.0.1:{args.port}/deferred_gallery.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
