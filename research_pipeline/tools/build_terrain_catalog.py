#!/usr/bin/env python3
"""Build trunk/terrain metadata for every canonical geometry-usable capture.

New captures use sparse ARKit scene points and the validated terrain fitter.
Legacy captures lack point_cloud_raw.ply, so their sparse scene cloud is
reconstructed directly from raw depth, confidence, calibration and odometry.
No historical mesh or result from the old pipeline is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "prototypes/upslope_downslope"
sys.path.insert(0, str(PROTO))
from generate_prototype import analyze_points, analyze_scan, read_xyz_csv  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from build_tsdf_mesh import common_frame_ids, load_frames  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def raw_rgbd_points(scan: Path, frame_step: int = 24, pixel_step: int = 4) -> np.ndarray:
    """Reconstruct a sparse scene point cloud directly from immutable raw RGB-D."""
    frames = load_frames(scan / "odometry.csv")
    ids = common_frame_ids(scan, frames, frame_step)
    fallback = np.loadtxt(scan / "camera_matrix.csv", delimiter=",")
    first = np.asarray(Image.open(scan / "depth" / f"{ids[0]:06d}.png"))
    depth_h, depth_w = first.shape
    # Stray intrinsics are stored for RGB. Legacy exports use 1920x1440.
    rgb_w, rgb_h = 1920, 1440
    yy, xx = np.meshgrid(np.arange(0, depth_h, pixel_step),
                         np.arange(0, depth_w, pixel_step), indexing="ij")
    points = []
    for frame_id in ids:
        depth = np.asarray(Image.open(scan / "depth" / f"{frame_id:06d}.png"), dtype=np.float32) / 1000
        confidence = np.asarray(Image.open(scan / "confidence" / f"{frame_id:06d}.png"))
        z = depth[yy, xx]
        conf = confidence[yy, xx]
        k = (frames[frame_id]["intrinsic_rgb"]
             if frames[frame_id]["intrinsic_rgb"] is not None else fallback).copy()
        k[0, :] *= depth_w / rgb_w
        k[1, :] *= depth_h / rgb_h
        valid = (z > 0.15) & (z <= 2.5) & (conf >= 1)
        camera = np.column_stack(((xx[valid] - k[0, 2]) * z[valid] / k[0, 0],
                                  (yy[valid] - k[1, 2]) * z[valid] / k[1, 1], z[valid]))
        transform = frames[frame_id]["transform"]
        points.append(camera @ transform[:3, :3].T + transform[:3, 3])
    cloud = np.concatenate(points) if points else np.empty((0, 3))
    if len(cloud) > 600_000:
        rng = np.random.default_rng(7719)
        cloud = cloud[rng.choice(len(cloud), 600_000, replace=False)]
    return cloud


def legacy_raw_record(scan: Path, seed: int) -> dict:
    points = raw_rgbd_points(scan)
    cameras = read_xyz_csv(scan / "odometry.csv")
    # ARKit world coordinates are gravity aligned; legacy exports have no location.csv.
    gravity_down = np.array([0.0, -1.0, 0.0])
    record = analyze_points(scan.name, points, cameras, gravity_down, seed)
    record["source"] = "legacy_raw_rgbd"
    record["gravity_source"] = "ARKit gravity-aligned world Y (location.csv unavailable)"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/scans.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/terrain/terrain_all.json")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["usable_geometry"] == "True" and row["duplicate_canonical"] == "True"]
    existing = {}
    if args.resume and args.output.exists():
        existing = {item["name"]: item for item in json.loads(args.output.read_text(encoding="utf-8"))}
    output = []
    failures = []
    for index, row in enumerate(rows, 1):
        scan = Path(row["raw_path"])
        if scan.name in existing:
            record = existing[scan.name]
            print(f"[{index}/{len(rows)}] {scan.name} / cached", flush=True)
        else:
            print(f"[{index}/{len(rows)}] {scan.name} / {row['source']}", flush=True)
            try:
                record = (analyze_scan(scan, 4100 + index * 17) if row["source"] == "new"
                          else legacy_raw_record(scan, 4100 + index * 17))
                record["source"] = row["source"] if row["source"] == "new" else record["source"]
            except Exception as exc:
                record = {"name": scan.name, "source": row["source"], "error": str(exc),
                          "variants": {"adaptive": {"valid": False, "reason": str(exc)}}}
                failures.append({"scan": scan.name, "error": str(exc)})
        output.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    failure_path = args.output.with_name(args.output.stem + "_failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = sum(bool(item.get("variants", {}).get("adaptive", {}).get("valid")) for item in output)
    print(f"terrain_catalog total={len(output)} valid={valid} failed={len(output)-valid}")


if __name__ == "__main__":
    main()
