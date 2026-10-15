#!/usr/bin/env python3
"""Build an observed-point diagnostic and a conservatively supported cylinder mesh."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import Delaunay

from build_tsdf_mesh import common_frame_ids, load_frames, load_terrain, unit


def local_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.85:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = unit(seed - (seed @ up) * up)
    return e1, unit(np.cross(up, e1))


def center_at(heights: np.ndarray, slices: list[dict]) -> np.ndarray:
    source_h = np.asarray([item["height_m"] for item in slices])
    return np.column_stack((
        np.interp(heights, source_h, [item["x"] for item in slices]),
        np.interp(heights, source_h, [item["y"] for item in slices]),
    ))


def voxel_reduce(points: np.ndarray, frame_ids: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    cells = np.floor(points / voxel).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_cells), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse)
    reduced = sums / counts[:, None]
    pairs = np.unique(np.column_stack((inverse, frame_ids)), axis=0)
    support = np.bincount(pairs[:, 0], minlength=len(unique_cells))
    return reduced, support


def cylindrical_candidates(
    points: np.ndarray,
    base: np.ndarray,
    up: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    slices: list[dict],
    max_angle_deg: float,
    max_height_edge: float,
    max_edge: float,
) -> np.ndarray:
    relative = points - base
    height = relative @ up
    xy = np.column_stack((relative @ e1, relative @ e2))
    centered = xy - center_at(height, slices)
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    median_radius = float(np.median(np.linalg.norm(centered, axis=1)))
    angular_limit = np.deg2rad(max_angle_deg)
    seam = theta > np.pi - angular_limit
    seam_other = theta < -np.pi + angular_limit
    expanded_index = np.concatenate((np.arange(len(points)), np.flatnonzero(seam), np.flatnonzero(seam_other)))
    expanded_theta = np.concatenate((theta, theta[seam] - 2 * np.pi, theta[seam_other] + 2 * np.pi))
    uv = np.column_stack((expanded_theta * median_radius, height[expanded_index]))
    simplices = Delaunay(uv).simplices
    original = expanded_index[simplices]
    unique_vertices = np.all(np.diff(np.sort(original, axis=1), axis=1) != 0, axis=1)
    angular_span = np.ptp(expanded_theta[simplices], axis=1)
    height_span = np.ptp(height[original], axis=1)
    edges = np.stack((
        np.linalg.norm(points[original[:, 0]] - points[original[:, 1]], axis=1),
        np.linalg.norm(points[original[:, 1]] - points[original[:, 2]], axis=1),
        np.linalg.norm(points[original[:, 2]] - points[original[:, 0]], axis=1),
    ), axis=1)
    keep = (unique_vertices & (angular_span <= angular_limit)
            & (height_span <= max_height_edge) & (edges.max(axis=1) <= max_edge))
    triangles = np.sort(original[keep], axis=1)
    return np.unique(triangles, axis=0)


def observed_triangle_support(
    centroids: np.ndarray,
    scan: Path,
    frames: dict[int, dict],
    frame_ids: list[int],
    fallback_intrinsic: np.ndarray,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    confidence_min: int,
    tolerance: float,
) -> np.ndarray:
    support = np.zeros(len(centroids), dtype=np.uint16)
    depth_w, depth_h = depth_size
    rgb_w, rgb_h = rgb_size
    for frame_id in frame_ids:
        depth = np.asarray(Image.open(scan / "depth" / f"{frame_id:06d}.png"), dtype=np.float32) / 1000.0
        confidence = np.asarray(Image.open(scan / "confidence" / f"{frame_id:06d}.png"))
        intrinsic = (frames[frame_id]["intrinsic_rgb"]
                     if frames[frame_id]["intrinsic_rgb"] is not None else fallback_intrinsic).copy()
        intrinsic[0, :] *= depth_w / rgb_w
        intrinsic[1, :] *= depth_h / rgb_h
        world_to_camera = np.linalg.inv(frames[frame_id]["transform"])
        camera = centroids @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        positive = camera[:, 2] > 0
        px = np.rint(intrinsic[0, 0] * camera[:, 0] / np.maximum(camera[:, 2], 1e-6)
                      + intrinsic[0, 2]).astype(int)
        py = np.rint(intrinsic[1, 1] * camera[:, 1] / np.maximum(camera[:, 2], 1e-6)
                      + intrinsic[1, 2]).astype(int)
        inside = positive & (px >= 0) & (px < depth_w) & (py >= 0) & (py < depth_h)
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        measured = depth[py[indices], px[indices]]
        confident = confidence[py[indices], px[indices]] >= confidence_min
        agrees = confident & (measured > 0) & (np.abs(measured - camera[indices, 2]) <= tolerance)
        support[indices[agrees]] += 1
    return support


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-path", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terrain-results", type=Path,
                        default=root / "artifacts/terrain/terrain_all_runtime.json")
    parser.add_argument("--frame-step", type=int, default=12)
    parser.add_argument("--pixel-step", type=int, default=2)
    parser.add_argument("--voxel", type=float, default=0.005)
    parser.add_argument("--confidence", type=int, default=1)
    parser.add_argument("--min-point-support", type=int, default=2)
    parser.add_argument("--points-only", action="store_true",
                        help="Write the observed cloud and report without triangulating a mesh")
    parser.add_argument("--min-triangle-support", type=int, default=2)
    parser.add_argument("--support-tolerance", type=float, default=0.012)
    parser.add_argument("--roi-radius", type=float, default=0.22)
    parser.add_argument("--max-angle-deg", type=float, default=12.0)
    parser.add_argument("--max-height-edge", type=float, default=0.025)
    parser.add_argument("--max-edge", type=float, default=0.03)
    parser.add_argument("--rgb-width", type=int, default=1920)
    parser.add_argument("--rgb-height", type=int, default=1440)
    args = parser.parse_args()

    started = time.perf_counter()
    scan = args.scan_path.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    slices = baseline.get("adaptive_roi", {}).get("centerline_slices", [])
    if len(slices) < 3:
        raise RuntimeError("baseline has fewer than three adaptive centreline slices")
    terrain = load_terrain(args.terrain_results, scan.name)
    frames = load_frames(scan / "odometry.csv")
    frame_ids = common_frame_ids(scan, frames, args.frame_step)
    fallback = np.loadtxt(scan / "camera_matrix.csv", delimiter=",")
    first_depth = np.asarray(Image.open(scan / "depth" / f"{frame_ids[0]:06d}.png"))
    depth_h, depth_w = first_depth.shape
    cap = cv2.VideoCapture(str(scan / "rgb.mp4"))
    rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else args.rgb_width
    rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else args.rgb_height
    cap.release()
    base, up = terrain["base"], terrain["up"]
    e1, e2 = local_basis(up)
    all_points, all_frames = [], []

    for ordinal, frame_id in enumerate(frame_ids, 1):
        depth = np.asarray(Image.open(scan / "depth" / f"{frame_id:06d}.png"), dtype=np.float32)[::args.pixel_step, ::args.pixel_step] / 1000.0
        confidence = np.asarray(Image.open(scan / "confidence" / f"{frame_id:06d}.png"))[::args.pixel_step, ::args.pixel_step]
        intrinsic = (frames[frame_id]["intrinsic_rgb"]
                     if frames[frame_id]["intrinsic_rgb"] is not None else fallback).copy()
        intrinsic[0, :] *= depth_w / rgb_w
        intrinsic[1, :] *= depth_h / rgb_h
        yy, xx = np.mgrid[0:depth_h:args.pixel_step, 0:depth_w:args.pixel_step]
        camera = np.column_stack(((xx.ravel() - intrinsic[0, 2]) * depth.ravel() / intrinsic[0, 0],
                                  (yy.ravel() - intrinsic[1, 2]) * depth.ravel() / intrinsic[1, 1],
                                  depth.ravel()))
        world = camera @ frames[frame_id]["transform"][:3, :3].T + frames[frame_id]["transform"][:3, 3]
        height = (world - base) @ up
        relative = world - base
        xy = np.column_stack((relative @ e1, relative @ e2))
        radius = np.linalg.norm(xy - center_at(height, slices), axis=1)
        valid = ((depth.ravel() > 0) & (depth.ravel() <= 2.0)
                 & (confidence.ravel() >= args.confidence)
                 & (height >= 0.15) & (height <= 2.20) & (radius <= args.roi_radius))
        all_points.append(world[valid])
        all_frames.append(np.full(valid.sum(), frame_id, dtype=np.int32))
        if ordinal == 1 or ordinal % 50 == 0:
            print(f"observed {ordinal}/{len(frame_ids)} frames", flush=True)

    points, point_support = voxel_reduce(
        np.concatenate(all_points), np.concatenate(all_frames), args.voxel)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    scaled_support = np.clip(point_support / max(args.min_point_support, 1), 0, 1)
    cloud.colors = o3d.utility.Vector3dVector(
        np.column_stack((1 - scaled_support, scaled_support, np.full(len(points), 0.25))))
    o3d.io.write_point_cloud(str(output / "observed_points.ply"), cloud)

    report = {
        "scan": scan.name,
        "method": ("observed points diagnostic" if args.points_only else
                   "observed points + cylindrical Delaunay with depth support mask"),
        "config": vars(args) | {"scan_path": str(scan), "baseline_metrics": str(args.baseline_metrics),
                                "output_dir": str(output), "terrain_results": str(args.terrain_results)},
        "frames_checked": len(frame_ids),
        "observed_voxels": len(points),
        "observed_voxels_multi_frame": int(np.count_nonzero(point_support >= args.min_point_support)),
        "point_support_median_frames": float(np.median(point_support)),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {"observed_points": str(output / "observed_points.ply")},
    }
    if args.points_only:
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    supported_points = points[point_support >= args.min_point_support]
    candidates = cylindrical_candidates(
        supported_points, base, up, e1, e2, slices,
        args.max_angle_deg, args.max_height_edge, args.max_edge)
    centroids = supported_points[candidates].mean(axis=1)
    triangle_support = observed_triangle_support(
        centroids, scan, frames, frame_ids, fallback, (rgb_w, rgb_h), (depth_w, depth_h),
        args.confidence, args.support_tolerance)
    triangles = candidates[triangle_support >= args.min_triangle_support]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(supported_points), o3d.utility.Vector3iVector(triangles))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output / "cylindrical_supported_mesh.ply"), mesh)

    report.update({
        "candidate_triangles": len(candidates),
        "kept_triangles": len(triangles),
        "rejected_by_depth_support": int(len(candidates) - len(triangles)),
        "triangle_support_median_frames": float(np.median(triangle_support)) if len(triangle_support) else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {**report["outputs"], "mesh": str(output / "cylindrical_supported_mesh.ply")},
    })
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
