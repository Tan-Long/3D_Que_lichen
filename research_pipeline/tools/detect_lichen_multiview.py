#!/usr/bin/env python3
"""Fuse full-resolution multi-view RGB onto a QA-approved mesh and run HSV baseline.

This is an auditable semantic baseline, not validated ground truth. White/cream
lichen candidates are reported separately from green biological material that
can include moss, algae, green lichen, or vegetation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tsdf_mesh import load_frames, make_preview, unit  # noqa: E402
from measure_mesh_surface import centers_at, triangle_areas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

CLASS_NAMES = ["bark", "lichen_pale_candidate", "green_biota_candidate", "uncertain"]
CLASS_COLORS = np.asarray([
    [0.42, 0.25, 0.12], [1.00, 0.55, 0.00], [0.10, 0.85, 0.25], [0.65, 0.20, 0.85]
])


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    image = np.clip(np.rint(rgb[:, None, :] * 255), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, 0].astype(np.float32)
    hsv[:, 0] *= 2.0
    hsv[:, 1:] /= 255.0
    return hsv


def semantic_baseline(points: np.ndarray, rgb: np.ndarray, count: np.ndarray,
                      color_std: np.ndarray, min_views: int) -> tuple[np.ndarray, dict]:
    hsv = rgb_to_hsv(rgb)
    hue, saturation, value = hsv.T
    observed = count >= min_views
    usable_v = value[observed]
    bark_p20 = float(np.percentile(usable_v, 20)) if len(usable_v) else 0.0
    global_std = float(np.std(usable_v)) if len(usable_v) else 0.0

    k = min(31, len(points))
    tree = cKDTree(points)
    _, neighbors = tree.query(points, k=k, workers=-1)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    neighbors = neighbors[:, 1:] if neighbors.shape[1] > 1 else neighbors
    local_v = value[neighbors].mean(axis=1)
    local_s = saturation[neighbors].mean(axis=1)
    local_std = value[neighbors].std(axis=1)
    v_contrast = value - local_v
    s_drop = local_s - saturation

    if bark_p20 < 0.32:
        absolute_v, absolute_s = max(0.55, bark_p20 + 1.8 * global_std), 0.28
        sigma_k, sigma_base, required_s_drop = 0.8, 0.03, 0.03
    elif bark_p20 < 0.45:
        absolute_v, absolute_s = max(0.63, bark_p20 + 2.1 * global_std), 0.24
        sigma_k, sigma_base, required_s_drop = 1.0, 0.04, 0.08
    else:
        absolute_v, absolute_s = max(0.70, bark_p20 + 2.5 * global_std), 0.20
        sigma_k, sigma_base, required_s_drop = 1.2, 0.05, 0.08

    green_hue = (hue >= 60) & (hue <= 155)
    obvious_vegetation = green_hue & (saturation > 0.25)
    local_pale = ((v_contrast > np.maximum(sigma_base, sigma_k * local_std))
                  & (saturation < local_s - required_s_drop)
                  & (saturation < 0.42) & ~obvious_vegetation)
    absolute_pale = ((value >= absolute_v) & (saturation < absolute_s)
                     & ~obvious_vegetation)
    pale = observed & (local_pale | absolute_pale)
    green = observed & green_hue & (saturation >= 0.25) & (value >= 0.30) & (value <= 0.92) & ~pale

    unstable_color = color_std > 0.08
    labels = np.zeros(len(points), dtype=np.uint8)
    labels[pale] = 1
    labels[green] = 2
    labels[~observed | unstable_color] = 3
    return labels, {
        "bark_value_p20": bark_p20,
        "value_std": global_std,
        "absolute_pale_thresholds": {"value_min": absolute_v, "saturation_max": absolute_s},
        "min_views": min_views,
        "uncertainty_feature": "weighted RGB chromaticity standard deviation",
        "unstable_color_std_threshold": 0.08,
    }


def duplicate_triangle_mesh(vertices: np.ndarray, triangles: np.ndarray,
                            labels: np.ndarray) -> o3d.geometry.TriangleMesh:
    expanded = vertices[triangles].reshape(-1, 3)
    faces = np.arange(len(expanded), dtype=np.int32).reshape(-1, 3)
    colors = np.repeat(CLASS_COLORS[labels], 3, axis=0)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(expanded),
                                    o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


def load_side_model(side_path: Path, terrain_path: Path, scan_name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    side = json.loads(side_path.read_text(encoding="utf-8"))
    records = json.loads(terrain_path.read_text(encoding="utf-8"))
    record = next(item for item in records if item["name"] == scan_name)
    fit = record["variants"]["adaptive"]
    base = np.asarray(fit["base"], dtype=float)
    up = -unit(np.asarray(record["gravity"], dtype=float))
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.85:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = unit(seed - (seed @ up) * up)
    e2 = unit(np.cross(up, e1))
    centerline = {"base": base, "up": up, "e1": e1, "e2": e2,
                  "slices": side["centerline"]["slices"]}
    down = unit(np.asarray(fit["down"], dtype=float))
    down_xy = np.asarray([down @ e1, down @ e2])
    return centerline, down_xy, up


def aggregate_area(areas: np.ndarray, labels: np.ndarray, downslope: np.ndarray) -> dict:
    result = {}
    for side_name, side_mask in (("overall", np.ones(len(labels), dtype=bool)),
                                 ("upslope", ~downslope), ("downslope", downslope)):
        total = float(areas[side_mask].sum())
        class_area = {name: float(areas[side_mask & (labels == index)].sum())
                      for index, name in enumerate(CLASS_NAMES)}
        confident = total - class_area["uncertain"]
        lichen = class_area["lichen_pale_candidate"]
        result[side_name] = {
            "observed_surface_cm2": total * 10000,
            "confidently_classified_cm2": confident * 10000,
            "class_area_cm2": {key: value * 10000 for key, value in class_area.items()},
            "lichen_coverage_observed_pct": 100 * lichen / total if total else None,
            "lichen_coverage_classified_pct": 100 * lichen / confident if confident else None,
        }
    return result


def run(args: argparse.Namespace) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    centers = vertices[triangles].mean(axis=1)
    normals = np.asarray(mesh.triangle_normals)
    frames = load_frames(args.scan / "odometry.csv")
    fallback_k = np.loadtxt(args.scan / "camera_matrix.csv", delimiter=",")
    depth_ids = {int(path.stem) for path in (args.scan / "depth").glob("*.png") if path.stem.isdigit()}
    confidence_ids = {int(path.stem) for path in (args.scan / "confidence").glob("*.png") if path.stem.isdigit()}
    selected = set(sorted(set(frames) & depth_ids & confidence_ids)[::args.frame_step])

    weighted_rgb = np.zeros((len(triangles), 3), dtype=np.float64)
    weighted_rgb2 = np.zeros_like(weighted_rgb)
    weighted_chroma = np.zeros_like(weighted_rgb)
    weighted_chroma2 = np.zeros_like(weighted_rgb)
    weight_sum = np.zeros(len(triangles), dtype=np.float64)
    observations = np.zeros(len(triangles), dtype=np.uint16)
    cap = cv2.VideoCapture(str(args.scan / "rgb.mp4"))
    rgb_w, rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = -1
    used_frames = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id not in selected:
            continue
        depth = np.asarray(Image.open(args.scan / "depth" / f"{frame_id:06d}.png"), dtype=np.float32) / 1000
        confidence = np.asarray(Image.open(args.scan / "confidence" / f"{frame_id:06d}.png"))
        depth_h, depth_w = depth.shape
        frame = frames[frame_id]
        transform = frame["transform"]
        camera = (centers - transform[:3, 3]) @ transform[:3, :3]
        z = camera[:, 2]
        k = frame["intrinsic_rgb"] if frame["intrinsic_rgb"] is not None else fallback_k
        u = k[0, 0] * camera[:, 0] / np.maximum(z, 1e-8) + k[0, 2]
        v = k[1, 1] * camera[:, 1] / np.maximum(z, 1e-8) + k[1, 2]
        ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
        du = np.clip(np.rint(u * depth_w / rgb_w).astype(int), 0, depth_w - 1)
        dv = np.clip(np.rint(v * depth_h / rgb_h).astype(int), 0, depth_h - 1)
        reference_depth = depth[dv, du]
        conf = confidence[dv, du]
        direction = transform[:3, 3] - centers
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
        incidence = np.abs(np.sum(normals * direction, axis=1))
        valid = ((z > 0) & (ui >= 0) & (ui < rgb_w) & (vi >= 0) & (vi < rgb_h)
                 & (reference_depth > 0) & (conf >= args.confidence)
                 & (np.abs(reference_depth - z) <= args.depth_tolerance)
                 & (incidence >= args.min_incidence))
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[vi[indices], ui[indices]].astype(np.float64) / 255
        chroma = rgb / np.maximum(rgb.sum(axis=1, keepdims=True), 1e-8)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        sharpness_weight = np.clip(sharpness / 100.0, 0.15, 1.0)
        weights = incidence[indices] * (1.0 + 0.5 * (conf[indices] == 2)) * sharpness_weight
        weighted_rgb[indices] += rgb * weights[:, None]
        weighted_rgb2[indices] += rgb * rgb * weights[:, None]
        weighted_chroma[indices] += chroma * weights[:, None]
        weighted_chroma2[indices] += chroma * chroma * weights[:, None]
        weight_sum[indices] += weights
        observations[indices] += 1
        used_frames += 1
    cap.release()

    fused = np.divide(weighted_rgb, weight_sum[:, None], out=np.zeros_like(weighted_rgb),
                      where=weight_sum[:, None] > 0)
    mean_chroma = np.divide(weighted_chroma, weight_sum[:, None], out=np.zeros_like(weighted_chroma),
                            where=weight_sum[:, None] > 0)
    chroma_variance = np.divide(
        weighted_chroma2, weight_sum[:, None], out=np.zeros_like(weighted_chroma2),
        where=weight_sum[:, None] > 0) - mean_chroma * mean_chroma
    color_std = np.sqrt(np.maximum(chroma_variance, 0)).mean(axis=1)
    labels, thresholds = semantic_baseline(centers, fused, observations, color_std, args.min_views)

    centerline, down_xy, up = load_side_model(args.side_metrics, args.terrain_results, args.scan.name)
    relative = centers - centerline["base"]
    xy = np.column_stack((relative @ centerline["e1"], relative @ centerline["e2"]))
    radial = xy - centers_at(centerline, centers)
    downslope = radial @ down_xy >= 0
    areas = triangle_areas(vertices, triangles)
    area = aggregate_area(areas, labels, downslope)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_mesh = duplicate_triangle_mesh(vertices, triangles, labels)
    mesh_path = args.output_dir / "lichen_hsv_semantic_mesh.ply"
    preview_path = args.output_dir / "lichen_hsv_preview.png"
    o3d.io.write_triangle_mesh(str(mesh_path), semantic_mesh, write_vertex_colors=True)
    make_preview(semantic_mesh, centerline["base"], up, preview_path)
    np.savez_compressed(args.output_dir / "lichen_hsv_triangle_labels.npz",
                        labels=labels, observations=observations, fused_rgb=fused.astype(np.float32),
                        color_std=color_std.astype(np.float32), downslope=downslope, areas_m2=areas)
    result = {
        "scan": args.scan.name,
        "status": "baseline_only_not_ground_truth",
        "method": "full_resolution_multiview_RGB_to_mesh + legacy_HSV_local_contrast",
        "classes": CLASS_NAMES,
        "config": {"frame_step": args.frame_step, "confidence_min": args.confidence,
                   "depth_tolerance_m": args.depth_tolerance,
                   "min_incidence_cosine": args.min_incidence, "min_views": args.min_views},
        "frames_selected": len(selected), "frames_contributing": used_frames,
        "triangle_observation": {
            "total": int(len(triangles)),
            "observed_at_least_once_fraction": float((observations > 0).mean()),
            "observed_min_views_fraction": float((observations >= args.min_views).mean()),
            "median_views": float(np.median(observations)),
        },
        "adaptive_hsv": thresholds,
        "area": area,
        "outputs": {"semantic_mesh": str(mesh_path), "preview": str(preview_path),
                    "triangle_data": str(args.output_dir / "lichen_hsv_triangle_labels.npz")},
    }
    (args.output_dir / "lichen_hsv_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--side-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terrain-results", type=Path,
                        default=root / "prototypes/upslope_downslope/upslope_downslope_prototype.json")
    parser.add_argument("--frame-step", type=int, default=24)
    parser.add_argument("--confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--depth-tolerance", type=float, default=0.035)
    parser.add_argument("--min-incidence", type=float, default=0.20)
    parser.add_argument("--min-views", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
