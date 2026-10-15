#!/usr/bin/env python3
"""Build a colour TSDF trunk mesh directly from a Stray Scanner capture.

This pilot tool streams rgb.mp4, masks RGB-D samples in a world-space cylinder
around the verified tree centre, fuses them with Open3D ScalableTSDFVolume, and
writes the mesh plus auditable QA metrics outside the immutable raw-data tree.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from measure_mesh_surface import (MIN_CENTERLINE_INLIER_FRACTION, centers_at,
                                  estimate_centerline)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def load_frames(path: Path) -> dict[int, dict]:
    frames = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            frame = int(row["frame"])
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat([
                float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
            ]).as_matrix()
            transform[:3, 3] = [float(row["x"]), float(row["y"]), float(row["z"])]
            intrinsic = None
            try:
                intrinsic = np.array([[float(row["fx"]), 0.0, float(row["cx"])],
                                      [0.0, float(row["fy"]), float(row["cy"])],
                                      [0.0, 0.0, 1.0]])
            except (KeyError, TypeError, ValueError):
                pass
            frames[frame] = {"transform": transform, "intrinsic_rgb": intrinsic}
    return frames


def load_terrain(path: Path, scan_name: str) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        if record["name"] == scan_name:
            fit = record["variants"]["adaptive"]
            if not fit.get("valid"):
                raise ValueError(f"terrain fit invalid for {scan_name}")
            return {
                "center": np.asarray(record["center"], dtype=float),
                "base": np.asarray(fit["base"], dtype=float),
                "up": -unit(np.asarray(record["gravity"], dtype=float)),
                "ground_normal": unit(np.asarray(fit["normal"], dtype=float)),
                "downslope": unit(np.asarray(fit["down"], dtype=float)),
                "terrain_quality_ok": bool(fit.get("quality_ok", False)),
                "terrain_source_variant": fit.get("source_variant"),
            }
    raise KeyError(f"scan not present in terrain results: {scan_name}")


def common_frame_ids(scan: Path, frames: dict[int, dict], step: int) -> list[int]:
    depth = {int(p.stem) for p in (scan / "depth").glob("*.png") if p.stem.isdigit()}
    confidence = {int(p.stem) for p in (scan / "confidence").glob("*.png") if p.stem.isdigit()}
    ids = sorted(set(frames) & depth & confidence)
    return ids[::step]


def adaptive_frame_ids(
    baseline_ids: list[int], all_ids: list[int], holdout_count: int
) -> tuple[list[int], list[int]]:
    """Add non-baseline frames while reserving independent frames for QA."""
    baseline = set(baseline_ids)
    candidates = [frame_id for frame_id in all_ids if frame_id not in baseline]
    if not candidates or holdout_count <= 0:
        return sorted(set(all_ids)), []
    picks = np.linspace(0, len(candidates) - 1, min(holdout_count, len(candidates))).astype(int)
    holdout = {candidates[index] for index in np.unique(picks)}
    return sorted(baseline | (set(candidates) - holdout)), sorted(holdout)


def weak_observed_tree(path: Path) -> tuple[cKDTree, int, int]:
    cloud = o3d.io.read_point_cloud(str(path))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    if not len(points) or len(colors) != len(points):
        raise RuntimeError(f"observed support cloud has no coloured points: {path}")
    # The diagnostic encodes support below two frames as half-green; >=2 is full-green.
    weak = colors[:, 1] < 0.75
    if not weak.any():
        raise RuntimeError(f"observed support cloud has no weak points: {path}")
    return cKDTree(points[weak]), int(weak.sum()), int(len(points))


def near_weak_support(points: np.ndarray, tree: cKDTree, radius: float) -> np.ndarray:
    distances, _ = tree.query(points, distance_upper_bound=radius, workers=-1)
    return np.isfinite(distances)


def select_consistent_frames(
    frame_voxels: dict[int, np.ndarray], shape: tuple[int, int, int],
    minimum_score: float, tolerance_bins: int, minimum_keep_fraction: float,
) -> tuple[list[int], dict]:
    """Keep frames agreeing with the dominant measured radius in each theta/height bin."""
    if not frame_voxels:
        return [], {"applied": False, "reason": "no frame observations"}
    counts = np.zeros(int(np.prod(shape)), dtype=np.uint16)
    for voxels in frame_voxels.values():
        counts += np.bincount(voxels, minlength=len(counts)).astype(np.uint16)
    counts = counts.reshape(shape)
    dominant_radius = counts.argmax(axis=2)
    peak_support = counts.max(axis=2)
    radius_bins = shape[2]
    scores = {}
    for frame_id, voxels in frame_voxels.items():
        radial = voxels % radius_bins
        angular_height = voxels // radius_bins
        theta = angular_height // shape[1]
        height = angular_height % shape[1]
        eligible = peak_support[theta, height] >= 2
        scores[frame_id] = (float((np.abs(radial[eligible] - dominant_radius[theta[eligible], height[eligible]])
                                  <= tolerance_bins).mean()) if eligible.any() else 0.0)
    selected = sorted(frame_id for frame_id, score in scores.items() if score >= minimum_score)
    required = math.ceil(len(frame_voxels) * minimum_keep_fraction)
    passing = len(selected)
    applied = passing >= required
    if not applied:
        selected = sorted(frame_voxels)
    values = np.asarray(list(scores.values()), dtype=float)
    return selected, {
        "applied": applied,
        "reason": None if applied else f"only {passing} frames passed; required {required}",
        "frames_scored": len(frame_voxels),
        "frames_selected": len(selected),
        "minimum_score": minimum_score,
        "score_median": float(np.median(values)),
        "score_p10": float(np.quantile(values, 0.1)),
        "score_min": float(values.min()),
        "dropped_frame_ids": sorted(set(frame_voxels) - set(selected)),
    }


def consistency_frame_ids(
    scan: Path, frames: dict[int, dict], frame_ids: list[int], terrain: dict,
    intrinsic_rgb: np.ndarray, rgb_size: tuple[int, int], depth_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[list[int], dict]:
    """Build a compact cylindrical observation histogram without creating surfaces."""
    depth_w, depth_h = depth_size
    rgb_w, rgb_h = rgb_size
    pixel_y, pixel_x = np.mgrid[0:depth_h:args.consistency_pixel_step,
                               0:depth_w:args.consistency_pixel_step]
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, terrain["up"])) > 0.85:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = unit(seed - np.dot(seed, terrain["up"]) * terrain["up"])
    e2 = unit(np.cross(terrain["up"], e1))
    theta_bins = 72
    height_bins = math.ceil((args.max_height - args.min_height) / 0.05)
    radial_step = 0.005
    radial_bins = math.ceil(args.roi_search_radius / radial_step) + 1
    shape = (theta_bins, height_bins, radial_bins)
    observations = {}
    for frame_id in frame_ids:
        depth = np.asarray(Image.open(scan / "depth" / f"{frame_id:06d}.png"), dtype=np.float32)[
            ::args.consistency_pixel_step, ::args.consistency_pixel_step] / 1000.0
        confidence = np.asarray(Image.open(scan / "confidence" / f"{frame_id:06d}.png"))[
            ::args.consistency_pixel_step, ::args.consistency_pixel_step]
        frame = frames[frame_id]
        K = (frame["intrinsic_rgb"] if frame["intrinsic_rgb"] is not None else intrinsic_rgb).copy()
        K[0, :] *= depth_w / rgb_w
        K[1, :] *= depth_h / rgb_h
        cam_x = (pixel_x - K[0, 2]) * depth / K[0, 0]
        cam_y = (pixel_y - K[1, 2]) * depth / K[1, 1]
        camera_points = np.stack((cam_x, cam_y, depth), axis=-1).reshape(-1, 3)
        transform = frame["transform"]
        world = camera_points @ transform[:3, :3].T + transform[:3, 3]
        delta = world - terrain["center"]
        height = (world - terrain["base"]) @ terrain["up"]
        clearance = (world - terrain["base"]) @ terrain["ground_normal"]
        x, y = delta @ e1, delta @ e2
        radius = np.hypot(x, y)
        valid = ((depth.ravel() > 0) & (depth.ravel() <= args.max_depth)
                 & (confidence.ravel() >= args.confidence)
                 & (radius <= args.roi_search_radius)
                 & (height >= args.min_height) & (height < args.max_height)
                 & (clearance >= args.min_height))
        if not valid.any():
            continue
        theta_index = np.floor(((np.arctan2(y[valid], x[valid]) + 2 * np.pi) % (2 * np.pi))
                               / (2 * np.pi) * theta_bins).astype(np.int32)
        height_index = np.floor((height[valid] - args.min_height) / 0.05).astype(np.int32)
        radius_index = np.minimum(np.floor(radius[valid] / radial_step).astype(np.int32), radial_bins - 1)
        observations[frame_id] = np.unique(
            (theta_index * height_bins + height_index) * radial_bins + radius_index)
    selected, report = select_consistent_frames(
        observations, shape, args.frame_consistency_min,
        max(1, round(args.frame_consistency_tolerance / radial_step)),
        args.frame_consistency_min_keep,
    )
    report.update({
        "enabled": True,
        "radial_bin_m": radial_step,
        "angular_bin_degrees": 360 / theta_bins,
        "height_bin_m": 0.05,
        "tolerance_m": args.frame_consistency_tolerance,
        "pixel_step": args.consistency_pixel_step,
    })
    return selected, report


def observed_support_qa(
    mesh: o3d.geometry.TriangleMesh, path: Path | None, radius: float
) -> dict:
    if path is None:
        return {"enabled": False}
    cloud = o3d.io.read_point_cloud(str(path))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    weak = colors[:, 1] < 0.75
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distance = scene.compute_distance(
        o3d.core.Tensor(points.astype(np.float32))).numpy()
    return {
        "enabled": True,
        "distance_threshold_m": radius,
        "all_points_near_mesh_fraction": float((distance <= radius).mean()),
        "weak_points_near_mesh_fraction": float((distance[weak] <= radius).mean()),
        "strong_points_near_mesh_fraction": float((distance[~weak] <= radius).mean()),
        "weak_point_distance_p90_m": float(np.quantile(distance[weak], 0.9)),
    }


def mesh_qa(
    mesh: o3d.geometry.TriangleMesh,
    terrain: dict | None = None,
    centerline: dict | None = None,
) -> dict:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles):
        return {"vertices": int(len(vertices)), "triangles": int(len(triangles))}
    edges = Counter()
    for a, b, c in triangles:
        edges[tuple(sorted((int(a), int(b))))] += 1
        edges[tuple(sorted((int(b), int(c))))] += 1
        edges[tuple(sorted((int(c), int(a))))] += 1
    boundary = sum(count == 1 for count in edges.values())
    nonmanifold = sum(count > 2 for count in edges.values())
    clusters, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    result = {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "surface_area_m2": float(mesh.get_surface_area()),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
        "boundary_edges": int(boundary),
        "nonmanifold_edges": int(nonmanifold),
        "components": int(len(counts)),
        "largest_component_fraction": float(counts.max() / len(triangles)) if len(counts) else 0.0,
        "is_edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()),
        "is_watertight": bool(mesh.is_watertight()),
    }
    if terrain is not None:
        up = terrain["up"]
        if centerline is None:
            seed = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(seed, up)) > 0.85:
                seed = np.array([0.0, 0.0, 1.0])
            e1 = unit(seed - np.dot(seed, up) * up)
            e2 = unit(np.cross(up, e1))
            delta = vertices - terrain["center"]
            radial_xy = np.column_stack((delta @ e1, delta @ e2))
        else:
            e1, e2 = centerline["e1"], centerline["e2"]
            relative = vertices - centerline["base"]
            xy = np.column_stack((relative @ e1, relative @ e2))
            radial_xy = xy - centers_at(centerline, vertices)
        height = (vertices - terrain["base"]) @ up
        angles = (np.arctan2(radial_xy[:, 1], radial_xy[:, 0]) + 2 * np.pi) % (2 * np.pi)
        coverages = []
        if np.ptp(height) > 0.1:
            bins = np.arange(height.min(), height.max() + 0.1001, 0.1)
            for low, high in zip(bins[:-1], bins[1:]):
                selected = angles[(height >= low) & (height < high)]
                if len(selected) >= 100:
                    sectors = np.unique(np.floor(selected / (2 * np.pi / 36)).astype(int))
                    coverages.append(len(sectors) / 36.0)
        result["height_span_m"] = float(np.ptp(height))
        result["azimuth_coverage_median"] = float(np.median(coverages)) if coverages else 0.0
        result["azimuth_coverage_p10"] = float(np.quantile(coverages, 0.1)) if coverages else 0.0
        result["evaluated_height_slices"] = len(coverages)
    return result


def heldout_depth_qa(
    mesh: o3d.geometry.TriangleMesh,
    scan: Path,
    frames: dict[int, dict],
    integrated_ids: list[int],
    terrain: dict,
    fallback_intrinsic: np.ndarray,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    args: argparse.Namespace,
    centerline: dict | None = None,
) -> dict:
    candidates = sorted(set(frames) - set(integrated_ids))
    if not candidates or not len(mesh.triangles):
        return {"frames": 0}
    pick = np.linspace(0, len(candidates) - 1, min(args.heldout_frames, len(candidates))).astype(int)
    selected_ids = [candidates[i] for i in np.unique(pick)]
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    rgb_w, rgb_h = rgb_size
    depth_w, depth_h = depth_size
    pixel_x, pixel_y = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    residuals = []
    total_valid = total_hit = within_10 = within_20 = 0
    used = 0
    for frame_id in selected_ids:
        depth_path = scan / "depth" / f"{frame_id:06d}.png"
        confidence_path = scan / "confidence" / f"{frame_id:06d}.png"
        if not depth_path.exists() or not confidence_path.exists():
            continue
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
        confidence = np.asarray(Image.open(confidence_path))
        frame = frames[frame_id]
        K_rgb = frame["intrinsic_rgb"] if frame["intrinsic_rgb"] is not None else fallback_intrinsic
        K = K_rgb.copy()
        K[0, :] *= depth_w / rgb_w
        K[1, :] *= depth_h / rgb_h
        z = depth
        cam_x = (pixel_x - K[0, 2]) * z / K[0, 0]
        cam_y = (pixel_y - K[1, 2]) * z / K[1, 1]
        cam_points = np.stack((cam_x, cam_y, z), axis=-1).reshape(-1, 3)
        transform = frame["transform"]
        world = cam_points @ transform[:3, :3].T + transform[:3, 3]
        radius = radial_distance(world, terrain, centerline)
        height = (world - terrain["base"]) @ terrain["up"]
        clearance = (world - terrain["base"]) @ terrain["ground_normal"]
        valid = ((depth > 0) & (depth <= args.max_depth) & (confidence >= args.eval_confidence)
                 & (radius.reshape(depth_h, depth_w) <= args.roi_radius)
                 & (height.reshape(depth_h, depth_w) >= args.min_height)
                 & (height.reshape(depth_h, depth_w) <= args.max_height)
                 & (clearance.reshape(depth_h, depth_w) >= args.min_height))
        if not valid.any():
            continue
        rays = scene.create_rays_pinhole(
            o3d.core.Tensor(K, dtype=o3d.core.float64),
            o3d.core.Tensor(np.linalg.inv(transform), dtype=o3d.core.float64),
            depth_w, depth_h,
        )
        predicted = scene.cast_rays(rays)["t_hit"].numpy()
        hit = valid & np.isfinite(predicted)
        difference = np.abs(predicted[hit] - depth[hit])
        total_valid += int(valid.sum())
        total_hit += int(hit.sum())
        within_10 += int((difference <= 0.010).sum())
        within_20 += int((difference <= 0.020).sum())
        residuals.append(difference)
        used += 1
    all_residuals = np.concatenate(residuals) if residuals else np.array([])
    return {
        "frames": used,
        "confidence_min": args.eval_confidence,
        "valid_depth_pixels": total_valid,
        "mesh_hit_fraction": float(total_hit / total_valid) if total_valid else 0.0,
        "within_10mm_fraction": float(within_10 / total_valid) if total_valid else 0.0,
        "within_20mm_fraction": float(within_20 / total_valid) if total_valid else 0.0,
        "absolute_residual_median_m": float(np.median(all_residuals)) if len(all_residuals) else None,
        "absolute_residual_p90_m": float(np.quantile(all_residuals, 0.9)) if len(all_residuals) else None,
    }


def remove_small_components(mesh: o3d.geometry.TriangleMesh, minimum: int) -> o3d.geometry.TriangleMesh:
    if not len(mesh.triangles):
        return mesh
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    result = copy.deepcopy(mesh)
    result.remove_triangles_by_mask(counts[clusters] < minimum)
    result.remove_unreferenced_vertices()
    return result


def radial_distance(points: np.ndarray, terrain: dict, centerline: dict | None = None) -> np.ndarray:
    if centerline is None:
        delta = points - terrain["center"]
        horizontal = delta - np.outer(delta @ terrain["up"], terrain["up"])
        return np.linalg.norm(horizontal, axis=1)
    relative = points - centerline["base"]
    xy = np.column_stack((relative @ centerline["e1"], relative @ centerline["e2"]))
    return np.linalg.norm(xy - centers_at(centerline, points), axis=1)


def select_trunk_component(mesh: o3d.geometry.TriangleMesh, terrain: dict) -> tuple[o3d.geometry.TriangleMesh, dict]:
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    candidates = []
    for label, count in enumerate(counts):
        indices = np.unique(triangles[labels == label])
        height = (vertices[indices] - terrain["base"]) @ terrain["up"]
        span = float(np.ptp(height))
        # A trunk is both large and vertically persistent; this rejects a nearby operator.
        score = float(count) * max(span, 0.05)
        candidates.append({"label": int(label), "triangles": int(count),
                           "height_span_m": span, "score": score})
    selected = max(candidates, key=lambda item: item["score"])
    result = copy.deepcopy(mesh)
    result.remove_triangles_by_mask(labels != selected["label"])
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    return result, {"selected": selected, "candidates": candidates}


def clean_centerline(centerline: dict) -> dict:
    """Reject branch-sized circle fits while retaining the leaning trunk path."""
    radii = np.asarray([item["radius_m"] for item in centerline["slices"]])
    median = float(np.median(radii))
    limit = max(0.08, median * 2.5)
    slices = [item for item in centerline["slices"]
              if item["radius_m"] <= limit
              and item["inlier_fraction"] >= MIN_CENTERLINE_INLIER_FRACTION]
    if len(slices) < 3:
        raise RuntimeError("fewer than three stable adaptive centreline slices")
    return {**centerline, "slices": slices}


def make_preview(mesh: o3d.geometry.TriangleMesh, base: np.ndarray, up: np.ndarray, path: Path) -> None:
    points = np.asarray(mesh.vertices) - base
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    if not len(points):
        return
    if len(colors) != len(points):
        colors = np.full((len(points), 3), 0.7)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, up)) > 0.85:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = unit(seed - np.dot(seed, up) * up)
    e2 = unit(np.cross(up, e1))
    xyz = np.column_stack((points @ e1, points @ e2, points @ up))
    panels = [(0, 2, "SIDE A"), (1, 2, "SIDE B"), (0, 1, "TOP")]
    canvas = np.full((720, 1440, 3), 18, np.uint8)
    for panel, (ax, ay, label) in enumerate(panels):
        x0, width = panel * 480, 480
        values = xyz[:, [ax, ay]]
        lo, hi = np.quantile(values, [0.005, 0.995], axis=0)
        span = np.maximum(hi - lo, 1e-4)
        scale = min(410 / span[0], 620 / span[1])
        px = ((values[:, 0] - (lo[0] + hi[0]) / 2) * scale + x0 + width / 2).astype(int)
        py = (680 - (values[:, 1] - lo[1]) * scale).astype(int)
        depth_axis = 1 if ax == 0 else 0
        order = np.argsort(xyz[triangles].mean(axis=1)[:, depth_axis])
        for triangle_index in order:
            triangle = triangles[triangle_index]
            polygon = np.column_stack((px[triangle], py[triangle])).astype(np.int32)
            if (polygon[:, 0].max() < x0 + 5 or polygon[:, 0].min() >= x0 + width - 5
                    or polygon[:, 1].max() < 30 or polygon[:, 1].min() >= 700):
                continue
            bgr = tuple(int(value) for value in np.clip(
                colors[triangle].mean(axis=0)[::-1] * 255, 0, 255))
            cv2.fillConvexPoly(canvas, polygon, bgr, lineType=cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 + 18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 240, 235), 2)
        if panel:
            cv2.line(canvas, (x0, 0), (x0, 720), (60, 75, 70), 1)
    cv2.imwrite(str(path), canvas)


def build(args: argparse.Namespace) -> dict:
    scan = args.scan_path.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    terrain = load_terrain(args.terrain_results, scan.name)
    frames = load_frames(scan / "odometry.csv")
    baseline_ids = common_frame_ids(scan, frames, args.frame_step)
    if not baseline_ids:
        raise RuntimeError("no common odometry/depth/confidence frames")

    weak_tree = None
    weak_points = observed_points = 0
    adaptive_holdout: list[int] = []
    fusion_ids = baseline_ids
    if args.observed_support is not None:
        weak_tree, weak_points, observed_points = weak_observed_tree(args.observed_support)
        fusion_ids, adaptive_holdout = adaptive_frame_ids(
            baseline_ids, common_frame_ids(scan, frames, 1), args.heldout_frames)

    intrinsic_rgb = np.loadtxt(scan / "camera_matrix.csv", delimiter=",")
    first_depth = np.asarray(Image.open(scan / "depth" / f"{baseline_ids[0]:06d}.png"))
    depth_h, depth_w = first_depth.shape
    cap = cv2.VideoCapture(str(scan / "rgb.mp4"))
    video_available = cap.isOpened()
    if not video_available and not args.geometry_only:
        raise RuntimeError("cannot open rgb.mp4")
    rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if video_available else args.rgb_width
    rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if video_available else args.rgb_height
    if args.evaluate_existing is not None:
        cap.release()
        mesh = o3d.io.read_triangle_mesh(str(args.evaluate_existing))
        if not len(mesh.triangles):
            raise RuntimeError(f"cannot read triangle mesh: {args.evaluate_existing}")
        metrics_path = out / "metrics.json"
        result = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {
            "scan": scan.name,
            "method": "existing mesh evaluation",
            "config": {},
        }
        result["mesh"] = mesh_qa(mesh, terrain)
        result["heldout_depth"] = heldout_depth_qa(
            mesh, scan, frames, baseline_ids, terrain, intrinsic_rgb,
            (rgb_w, rgb_h), (depth_w, depth_h), args
        )
        result["observed_support_qa"] = observed_support_qa(
            mesh, args.observed_support, args.support_radius)
        make_preview(mesh, terrain["base"], terrain["up"], out / "preview.png")
        result.setdefault("config", {})["preview_rendering"] = "filled_triangles"
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    fallback_K = intrinsic_rgb.copy()
    fallback_K[0, :] *= depth_w / rgb_w
    fallback_K[1, :] *= depth_h / rgb_h
    consistency_qa = {"enabled": bool(args.frame_consistency_filter), "applied": False}
    if args.frame_consistency_filter:
        filtered_ids, consistency_qa = consistency_frame_ids(
            scan, frames, fusion_ids, terrain, intrinsic_rgb,
            (rgb_w, rgb_h), (depth_w, depth_h), args)
        if consistency_qa["applied"]:
            fusion_ids = filtered_ids
    if args.backend == "legacy":
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.voxel,
            sdf_trunc=args.voxel * args.trunc_mult,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
    else:
        volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=args.voxel,
            block_resolution=args.block_resolution,
            block_count=args.block_count,
            device=o3d.core.Device("CPU:0"),
        )
    target = set(fusion_ids)
    baseline_set = set(baseline_ids)
    integrated = 0
    adaptive_integrated = 0
    adaptive_valid_pixels = 0
    skipped_empty = 0
    valid_pixels = []
    start = time.perf_counter()
    pixel_x, pixel_y = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    frame_index = -1
    while target:
        if video_available:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_index += 1
        else:
            frame_index = min(target)
            bgr = None
        if frame_index not in target:
            continue
        target.remove(frame_index)
        depth = np.asarray(Image.open(scan / "depth" / f"{frame_index:06d}.png"), dtype=np.float32) / 1000.0
        confidence = np.asarray(Image.open(scan / "confidence" / f"{frame_index:06d}.png"))
        valid = (depth > 0) & (depth <= args.max_depth) & (confidence >= args.confidence)

        frame = frames[frame_index]
        transform = frame["transform"]
        K_rgb = frame["intrinsic_rgb"] if frame["intrinsic_rgb"] is not None else intrinsic_rgb
        K = K_rgb.copy()
        K[0, :] *= depth_w / rgb_w
        K[1, :] *= depth_h / rgb_h
        z = depth
        cam_x = (pixel_x - K[0, 2]) * z / K[0, 0]
        cam_y = (pixel_y - K[1, 2]) * z / K[1, 1]
        cam_points = np.stack((cam_x, cam_y, z), axis=-1).reshape(-1, 3)
        world = cam_points @ transform[:3, :3].T + transform[:3, 3]
        delta = world - terrain["center"]
        height = (world - terrain["base"]) @ terrain["up"]
        ground_clearance = (world - terrain["base"]) @ terrain["ground_normal"]
        horizontal = delta - np.outer(delta @ terrain["up"], terrain["up"])
        radius = np.linalg.norm(horizontal, axis=1)
        fusion_radius = args.roi_search_radius if args.adaptive_roi else args.roi_radius
        roi = ((radius <= fusion_radius) & (ground_clearance >= args.min_height)
               & (height >= args.min_height)
               & (height <= args.max_height)).reshape(depth_h, depth_w)
        valid &= roi
        if weak_tree is not None and frame_index not in baseline_set:
            selected = np.flatnonzero(valid.ravel())
            keep = near_weak_support(world[selected], weak_tree, args.support_radius)
            adaptive_mask = np.zeros(valid.size, dtype=bool)
            adaptive_mask[selected[keep]] = True
            valid &= adaptive_mask.reshape(depth_h, depth_w)
            adaptive_valid_pixels += int(valid.sum())
        depth[~valid] = 0.0
        valid_pixels.append(int(valid.sum()))
        if not valid.any():
            skipped_empty += 1
            continue
        if bgr is None:
            rgb_small = np.zeros((depth_h, depth_w, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
        extrinsic = np.linalg.inv(transform)
        if args.backend == "legacy":
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_small.astype(np.uint8)),
                o3d.geometry.Image(depth.astype(np.float32)),
                depth_scale=1.0,
                depth_trunc=args.max_depth,
                convert_rgb_to_intensity=False,
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                depth_w, depth_h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            )
            volume.integrate(rgbd, intrinsic, extrinsic)
        else:
            depth_mm = np.rint(depth * 1000.0).astype(np.uint16)
            depth_t = o3d.t.geometry.Image(o3d.core.Tensor(depth_mm))
            color_t = o3d.t.geometry.Image(o3d.core.Tensor(rgb_small.astype(np.uint8)))
            intrinsic_t = o3d.core.Tensor(K, dtype=o3d.core.float64)
            extrinsic_t = o3d.core.Tensor(extrinsic, dtype=o3d.core.float64)
            try:
                blocks = volume.compute_unique_block_coordinates(
                    depth_t, intrinsic_t, extrinsic_t, 1000.0, args.max_depth, args.trunc_mult
                )
                volume.integrate(
                    blocks, depth_t, color_t, intrinsic_t, intrinsic_t, extrinsic_t,
                    1000.0, args.max_depth, args.trunc_mult
                )
            except RuntimeError as exc:
                if "No block is touched" in str(exc):
                    skipped_empty += 1
                    continue
                raise
        integrated += 1
        adaptive_integrated += int(frame_index not in baseline_set)
        if integrated == 1 or integrated % 100 == 0:
            print(f"integrated {integrated}/{len(fusion_ids)} frames", flush=True)
    cap.release()
    mesh = volume.extract_triangle_mesh()
    if args.backend == "tensor":
        mesh = mesh.to_legacy()
    mesh.compute_vertex_normals()
    if not len(mesh.vertices):
        raise RuntimeError("TSDF produced an empty mesh")

    vertices = np.asarray(mesh.vertices)
    delta = vertices - terrain["center"]
    horizontal = delta - np.outer(delta @ terrain["up"], terrain["up"])
    height = (vertices - terrain["base"]) @ terrain["up"]
    ground_clearance = (vertices - terrain["base"]) @ terrain["ground_normal"]
    fusion_radius = args.roi_search_radius if args.adaptive_roi else args.roi_radius
    remove = ((np.linalg.norm(horizontal, axis=1) > fusion_radius)
              | (ground_clearance < args.min_height) | (height < args.min_height)
              | (height > args.max_height))
    mesh.remove_vertices_by_mask(remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh = remove_small_components(mesh, args.min_component)
    mesh.compute_vertex_normals()

    centerline = None
    adaptive_qa = {"enabled": bool(args.adaptive_roi)}
    if args.adaptive_roi:
        mesh, component_qa = select_trunk_component(mesh, terrain)
        centerline = clean_centerline(
            estimate_centerline(np.asarray(mesh.vertices), terrain["base"], terrain["up"])
        )
        vertices = np.asarray(mesh.vertices)
        mesh.remove_vertices_by_mask(radial_distance(vertices, terrain, centerline) > args.roi_radius)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        mesh = remove_small_components(mesh, args.min_component)
        mesh.compute_vertex_normals()
        adaptive_qa.update({
            "search_radius_m": args.roi_search_radius,
            "final_radius_m": args.roi_radius,
            "component_selection": component_qa,
            "centerline_slices": centerline["slices"],
        })

    mesh_path = out / "trunk_mesh_tsdf.ply"
    preview_path = out / "preview.png"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_colors=True, write_vertex_normals=True)
    make_preview(mesh, terrain["base"], terrain["up"], preview_path)
    elapsed = time.perf_counter() - start
    result = {
        "scan": scan.name,
        "method": ("Open3D Tensor VoxelBlockGrid TSDF" if args.backend == "tensor"
                   else "Open3D legacy ScalableTSDFVolume"),
        "config": {
            "voxel_m": args.voxel,
            "sdf_trunc_m": args.voxel * args.trunc_mult,
            "frame_step": args.frame_step,
            "confidence_min": args.confidence,
            "heldout_confidence_min": args.eval_confidence,
            "max_depth_m": args.max_depth,
            "roi_radius_m": args.roi_radius,
            "roi_mode": "adaptive_centerline" if args.adaptive_roi else "fixed_cylinder",
            "roi_search_radius_m": args.roi_search_radius if args.adaptive_roi else None,
            "height_range_m": [args.min_height, args.max_height],
            "min_component_triangles": args.min_component,
            "intrinsics_mode": "per_frame_with_camera_matrix_fallback",
            "color_usage": ("unavailable; geometry-only blank colour"
                            if not video_available else
                            "preview_only; geometry/semantic measurements remain separate"),
            "preview_rendering": "filled_triangles",
            "observed_support": str(args.observed_support) if args.observed_support else None,
            "support_radius_m": args.support_radius if args.observed_support else None,
            "frame_consistency_filter": bool(args.frame_consistency_filter),
        },
        "terrain": {
            "quality_ok": terrain["terrain_quality_ok"],
            "source_variant": terrain["terrain_source_variant"],
            "center": terrain["center"].tolist(),
            "base": terrain["base"].tolist(),
            "up": terrain["up"].tolist(),
        },
        "frames_available": len(fusion_ids),
        "frames_integrated": integrated,
        "frames_skipped_empty_roi": skipped_empty,
        "adaptive_support": {
            "enabled": args.observed_support is not None,
            "observed_points": observed_points,
            "weak_points": weak_points,
            "baseline_frames": len(baseline_ids),
            "extra_frames_integrated": adaptive_integrated,
            "extra_valid_depth_pixels": adaptive_valid_pixels,
            "heldout_frames": adaptive_holdout,
        },
        "frame_consistency": consistency_qa,
        "observed_support_qa": observed_support_qa(mesh, args.observed_support, args.support_radius),
        "valid_pixels_median": float(np.median(valid_pixels)) if valid_pixels else 0,
        "elapsed_seconds": elapsed,
        "adaptive_roi": adaptive_qa,
        "mesh": mesh_qa(mesh, terrain, centerline),
        "heldout_depth": heldout_depth_qa(
            mesh, scan, frames, fusion_ids, terrain, intrinsic_rgb,
            (rgb_w, rgb_h), (depth_w, depth_h), args, centerline
        ),
        "outputs": {"mesh": str(mesh_path), "preview": str(preview_path)},
    }
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terrain-results", type=Path,
                        default=root / "prototypes/upslope_downslope/upslope_downslope_prototype.json")
    parser.add_argument("--voxel", type=float, default=0.005)
    parser.add_argument("--backend", choices=("tensor", "legacy"), default="tensor")
    parser.add_argument("--trunc-mult", type=float, default=4.0)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--eval-confidence", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--roi-radius", type=float, default=0.22)
    parser.add_argument("--adaptive-roi", action="store_true",
                        help="Fuse a wider search cylinder, isolate the trunk, then crop around a fitted centreline")
    parser.add_argument("--roi-search-radius", type=float, default=0.35)
    parser.add_argument("--min-height", type=float, default=0.15)
    parser.add_argument("--max-height", type=float, default=2.20)
    parser.add_argument("--min-component", type=int, default=200)
    parser.add_argument("--block-resolution", type=int, choices=(8, 16), default=8)
    parser.add_argument("--block-count", type=int, default=20000)
    parser.add_argument("--heldout-frames", type=int, default=12)
    parser.add_argument("--observed-support", type=Path,
                        help="Observed-point PLY; add unused frames only near points supported by <2 frames")
    parser.add_argument("--support-radius", type=float, default=0.0075,
                        help="World-space radius around weak observed points to receive extra depth frames")
    parser.add_argument("--frame-consistency-filter", action="store_true",
                        help="Experimental: drop frames disagreeing with the dominant measured cylindrical surface")
    parser.add_argument("--frame-consistency-min", type=float, default=0.55,
                        help="Minimum fraction of observations agreeing with dominant measured radii")
    parser.add_argument("--frame-consistency-tolerance", type=float, default=0.012,
                        help="Radial agreement tolerance in metres")
    parser.add_argument("--frame-consistency-min-keep", type=float, default=0.60,
                        help="Abort filtering if fewer than this fraction of frames remain")
    parser.add_argument("--consistency-pixel-step", type=int, default=2,
                        help="Depth-pixel stride used only by the frame consistency diagnostic")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Allow depth-only fusion when rgb.mp4 is unreadable")
    parser.add_argument("--rgb-width", type=int, default=1920,
                        help="Calibration image width used with --geometry-only")
    parser.add_argument("--rgb-height", type=int, default=1440,
                        help="Calibration image height used with --geometry-only")
    parser.add_argument("--evaluate-existing", type=Path,
                        help="Skip fusion; recompute preview and QA for an existing mesh")
    args = parser.parse_args()
    if (args.voxel <= 0 or args.frame_step < 1 or args.trunc_mult < 2 or args.support_radius <= 0
            or args.consistency_pixel_step < 1):
        parser.error("voxel > 0, frame-step >= 1, and trunc-mult >= 2 are required")
    if not 0 < args.frame_consistency_min <= 1 or not 0 < args.frame_consistency_min_keep <= 1:
        parser.error("frame consistency fractions must be in (0, 1]")
    if args.frame_consistency_tolerance <= 0:
        parser.error("frame-consistency-tolerance must be positive")
    if args.observed_support is not None and not args.observed_support.is_file():
        parser.error("observed-support must be an existing PLY file")
    if args.adaptive_roi and args.roi_search_radius < args.roi_radius:
        parser.error("roi-search-radius must be >= roi-radius in adaptive mode")
    return args


if __name__ == "__main__":
    build(parse_args())
