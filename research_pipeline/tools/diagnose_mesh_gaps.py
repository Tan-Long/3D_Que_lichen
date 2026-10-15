#!/usr/bin/env python3
"""Fast differential diagnostic for a holey preview and an incomplete TSDF mesh.

The command intentionally exits 1 when it reproduces either reported symptom.
It does not rebuild a TSDF volume; it compares mesh topology/projection and
counts the raw LiDAR samples that survive each ROI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tsdf_mesh import common_frame_ids, load_frames, load_terrain, unit  # noqa: E402


def mesh_projection(mesh_path: Path, metrics: dict, output: Path) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    base = np.asarray(metrics["terrain"]["base"], dtype=float)
    up = unit(np.asarray(metrics["terrain"]["up"], dtype=float))
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.85:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = unit(seed - (seed @ up) * up)
    e2 = unit(np.cross(up, e1))
    xy = np.column_stack(((vertices - base) @ e1, (vertices - base) @ e2))
    lo, hi = np.quantile(xy, [0.005, 0.995], axis=0)
    span = np.maximum(hi - lo, 1e-5)
    scale = min(560.0 / span[0], 560.0 / span[1])
    pix = np.rint((xy - (lo + hi) / 2) * scale + 320).astype(np.int32)

    filled = np.zeros((640, 640), np.uint8)
    points = np.zeros_like(filled)
    for tri in triangles:
        cv2.fillConvexPoly(filled, pix[tri], 255)
    for p in pix:
        if 0 <= p[0] < 640 and 0 <= p[1] < 640:
            cv2.circle(points, tuple(p), 1, 255, -1, lineType=cv2.LINE_AA)
    inside = filled > 0
    point_coverage = float(np.count_nonzero((points > 0) & inside) / max(1, inside.sum()))
    canvas = np.hstack((points, filled))
    cv2.putText(canvas, "VERTEX PREVIEW", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, 180, 2)
    cv2.putText(canvas, "FILLED TRIANGLES", (655, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, 180, 2)
    cv2.imwrite(str(output), canvas)

    edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in triangles:
        edges[tuple(sorted((int(a), int(b))))] += 1
        edges[tuple(sorted((int(b), int(c))))] += 1
        edges[tuple(sorted((int(c), int(a))))] += 1
    boundary = np.asarray([edge for edge, count in edges.items() if count == 1], dtype=int)
    height = (vertices - base) @ up
    if len(boundary):
        middle_height = height[boundary].mean(axis=1)
        at_cut_end = ((middle_height <= height.min() + .02)
                      | (middle_height >= height.max() - .02))
        interior_boundary_fraction = float((~at_cut_end).mean())
    else:
        interior_boundary_fraction = 0.0
    return {
        "top_triangle_pixels": int(inside.sum()),
        "top_vertex_preview_coverage_of_triangle_projection": point_coverage,
        "preview_gap_fraction": 1.0 - point_coverage,
        "boundary_edges": int(len(boundary)),
        "interior_boundary_fraction": interior_boundary_fraction,
        "diagnostic_image": str(output),
    }


def roi_counts(scan: Path, terrain_path: Path, step: int = 12) -> dict:
    frames = load_frames(scan / "odometry.csv")
    ids = common_frame_ids(scan, frames, step)
    terrain = load_terrain(terrain_path, scan.name)
    fallback = np.loadtxt(scan / "camera_matrix.csv", delimiter=",")
    cap = cv2.VideoCapture(str(scan / "rgb.mp4"))
    rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    first = np.asarray(Image.open(scan / "depth" / f"{ids[0]:06d}.png"))
    depth_h, depth_w = first.shape
    px, py = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    scenarios = {
        "baseline_r22_d2_conf1": (.22, 2.0, 1),
        "wider_r35": (.35, 2.0, 1),
        "wider_r50": (.50, 2.0, 1),
        "farther_d25": (.22, 2.5, 1),
        "low_conf0": (.22, 2.0, 0),
        "all_depth_no_roi": (np.inf, 2.0, 1),
    }
    counts = {name: [] for name in scenarios}
    for frame_id in ids:
        depth = np.asarray(Image.open(scan / "depth" / f"{frame_id:06d}.png"), dtype=np.float32) / 1000
        confidence = np.asarray(Image.open(scan / "confidence" / f"{frame_id:06d}.png"))
        frame = frames[frame_id]
        k = (frame["intrinsic_rgb"] if frame["intrinsic_rgb"] is not None else fallback).copy()
        k[0, :] *= depth_w / rgb_w
        k[1, :] *= depth_h / rgb_h
        cam = np.stack(((px - k[0, 2]) * depth / k[0, 0],
                        (py - k[1, 2]) * depth / k[1, 1], depth), axis=-1).reshape(-1, 3)
        world = cam @ frame["transform"][:3, :3].T + frame["transform"][:3, 3]
        delta = world - terrain["center"]
        horizontal = delta - np.outer(delta @ terrain["up"], terrain["up"])
        radius = np.linalg.norm(horizontal, axis=1).reshape(depth_h, depth_w)
        height = ((world - terrain["base"]) @ terrain["up"]).reshape(depth_h, depth_w)
        clearance = ((world - terrain["base"]) @ terrain["ground_normal"]).reshape(depth_h, depth_w)
        vertical = (height >= .15) & (height <= 2.2) & (clearance >= .15)
        for name, (roi_radius, max_depth, min_conf) in scenarios.items():
            valid = ((depth > 0) & (depth <= max_depth) & (confidence >= min_conf))
            if np.isfinite(roi_radius):
                valid &= vertical & (radius <= roi_radius)
            counts[name].append(int(valid.sum()))
    return {
        name: {
            "median_valid_pixels": float(np.median(values)),
            "empty_frames": int(np.count_nonzero(np.asarray(values) == 0)),
            "total_valid_pixels": int(np.sum(values)),
        }
        for name, values in counts.items()
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good-metrics", type=Path, required=True)
    parser.add_argument("--bad-metrics", type=Path, required=True)
    parser.add_argument("--fixed-bad-metrics", type=Path,
                        help="Optional rebuilt result; makes this a post-fix regression check")
    parser.add_argument("--good-scan", type=Path, required=True)
    parser.add_argument("--bad-scan", type=Path, required=True)
    parser.add_argument("--terrain-results", type=Path,
                        default=root / "prototypes/upslope_downslope/upslope_downslope_prototype.json")
    parser.add_argument("--output", type=Path,
                        default=root / "artifacts/meshes/mesh_gap_diagnosis.json")
    args = parser.parse_args()
    good = json.loads(args.good_metrics.read_text(encoding="utf-8"))
    bad = json.loads(args.bad_metrics.read_text(encoding="utf-8"))
    image_path = args.output.with_suffix(".png")
    report = {
        "good_mesh_projection": mesh_projection(
            args.good_metrics.parent / "trunk_mesh_tsdf.ply", good, image_path),
        "raw_roi_counts": {
            "good": roi_counts(args.good_scan, args.terrain_results),
            "bad": roi_counts(args.bad_scan, args.terrain_results),
        },
        "existing_mesh_qa": {
            "good": {"mesh": good["mesh"], "heldout_depth": good["heldout_depth"]},
            "bad": {"mesh": bad["mesh"], "heldout_depth": bad["heldout_depth"]},
        },
    }
    projection = report["good_mesh_projection"]
    bad_qa = bad["mesh"]
    bad_depth = bad["heldout_depth"]
    preview_is_point_rendered = good.get("config", {}).get("preview_rendering") != "filled_triangles"
    symptoms = {
        # A 15% unpainted fraction is readily visible as pinholes at this scale,
        # but filled-triangle previews no longer expose those sampling gaps.
        "good_preview_looks_holey": (preview_is_point_rendered
                                     and projection["preview_gap_fraction"] > .15),
        "bad_mesh_is_incomplete": (bad_qa["azimuth_coverage_median"] < .50
                                   or bad_depth["mesh_hit_fraction"] < .80),
    }
    if args.fixed_bad_metrics:
        fixed = json.loads(args.fixed_bad_metrics.read_text(encoding="utf-8"))
        recovery = {
            "height_span_gain_m": fixed["mesh"]["height_span_m"] - bad_qa["height_span_m"],
            "azimuth_coverage_gain": (fixed["mesh"]["azimuth_coverage_median"]
                                      - bad_qa["azimuth_coverage_median"]),
            "single_component": fixed["mesh"]["components"] == 1,
            "adaptive_centerline": fixed.get("config", {}).get("roi_mode") == "adaptive_centerline",
        }
        report["fixed_bad_recovery"] = recovery
        symptoms["bad_mesh_is_incomplete"] = not (
            recovery["height_span_gain_m"] >= .15
            and fixed["mesh"]["azimuth_coverage_median"] >= .50
            and recovery["single_component"]
            and recovery["adaptive_centerline"]
        )
    report["symptoms_reproduced"] = symptoms
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"symptoms_reproduced": symptoms,
                      "good_projection": projection,
                      "raw_roi_counts": report["raw_roi_counts"]},
                     ensure_ascii=False, indent=2))
    return 1 if any(symptoms.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
