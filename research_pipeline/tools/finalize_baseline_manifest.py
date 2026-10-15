#!/usr/bin/env python3
"""Freeze the reviewed baseline TSDF outputs for downstream work."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from serve_deferred_gallery import merged_reviews, read_reviews


RUN_NAME = "tensor_v5mm_step3_conf1_adaptive_r35"
VERSION = "baseline-v1-20261012"


def build_manifest(root: Path) -> dict:
    qa = json.loads((root / "manifests/geometry_qa_split_all.json").read_text(encoding="utf-8"))
    qa_rows = qa["pass"] + qa["deferred"]
    reviews = merged_reviews(
        read_reviews(root / "manifests/deferred_manual_review.json"),
        read_reviews(root / "manifests/pass_manual_review.json"),
    )["reviews"]
    qa_uids = {row["scan_uid"] for row in qa_rows}
    if set(reviews) != qa_uids:
        raise RuntimeError(
            f"review coverage mismatch: missing={len(qa_uids - set(reviews))}, "
            f"unexpected={len(set(reviews) - qa_uids)}"
        )

    rows = []
    for qa_row in sorted(qa_rows, key=lambda row: row["scan_uid"]):
        uid = qa_row["scan_uid"]
        output = root / "artifacts/meshes_all" / uid / RUN_NAME
        paths = {
            "mesh_path": output / "trunk_mesh_tsdf.ply",
            "metrics_path": output / "metrics.json",
            "preview_path": output / "preview.png",
            "surface_area_path": output / "surface_area_by_side.json",
        }
        decision = reviews[uid]["decision"]
        geometry_ready = paths["mesh_path"].is_file() and paths["metrics_path"].is_file()
        if decision == "approved" and not geometry_ready:
            raise RuntimeError(f"approved scan is missing baseline geometry: {uid}")
        rows.append({
            "scan_uid": uid,
            "scan": qa_row["scan"],
            "source": qa_row["source"],
            "raw_path": qa_row["raw_path"],
            "manual_decision": decision,
            "manual_updated_at": reviews[uid].get("updated_at", ""),
            "automatic_qa": qa_row["qa_status"],
            "automatic_qa_reasons": qa_row["qa_reasons"],
            "production_variant": "baseline",
            "production_run": RUN_NAME,
            "downstream_geometry": decision == "approved" and geometry_ready,
            "downstream_measurement": (decision == "approved" and geometry_ready
                                       and paths["surface_area_path"].is_file()),
            "measurement_blocker": ("terrain direction undefined; upslope/downslope not invented"
                                    if decision == "approved" and geometry_ready
                                    and not paths["surface_area_path"].is_file() else ""),
            **{key: str(path) for key, path in paths.items()},
        })

    decisions = Counter(row["manual_decision"] for row in rows)
    return {
        "version": VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "production_variant": "baseline only",
            "production_run": RUN_NAME,
            "downstream_geometry_rule": "manual_decision == approved and baseline mesh+metrics exist",
            "experimental_outputs": "excluded",
        },
        "counts": {
            "total": len(rows),
            "approved": decisions["approved"],
            "needs_review": decisions["needs_review"],
            "rejected": decisions["rejected"],
            "downstream_geometry": sum(row["downstream_geometry"] for row in rows),
            "downstream_measurement": sum(row["downstream_measurement"] for row in rows),
        },
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = build_manifest(root)
    destination = root / "manifests/production_baseline_v1.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    csv_path = root / "manifests/production_baseline_v1.csv"
    approved_path = root / "manifests/production_baseline_v1_approved.csv"
    write_csv(csv_path, manifest["rows"])
    write_csv(
        approved_path,
        [row for row in manifest["rows"] if row["downstream_geometry"]],
    )
    lock = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (destination, csv_path, approved_path)}
    (root / "manifests/production_baseline_v1.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
