#!/usr/bin/env python3
"""Fuse DINO patch features onto one production mesh and cluster its surface.

This is an unlabeled visual pilot, not a biological lichen classifier.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tsdf_mesh import load_frames  # noqa: E402
from cluster_lichen_surface import (  # noqa: E402
    PALETTE,
    colored_triangle_mesh,
    smooth_triangle_labels,
)
from measure_mesh_surface import triangle_areas  # noqa: E402


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_SPECS = {
    "dinov2_vits14": (384, 14, 518, 392, "DINOv2", "facebookresearch/dinov2"),
    "dinov2_vitb14": (768, 14, 518, 392, "DINOv2", "facebookresearch/dinov2"),
    "dinov2_vitl14": (1024, 14, 518, 392, "DINOv2", "facebookresearch/dinov2"),
    "dinov3_vits16": (384, 16, 512, 384, "DINOv3", "facebookresearch/dinov3"),
    "dinov3_vitb16": (768, 16, 512, 384, "DINOv3", "facebookresearch/dinov3"),
    "dinov3_vitl16": (1024, 16, 512, 384, "DINOv3", "facebookresearch/dinov3"),
}
MODEL_DIMENSIONS = {model: spec[0] for model, spec in MODEL_SPECS.items()}


def sample_patch_features(patches: np.ndarray, u: np.ndarray, v: np.ndarray,
                          image_width: int, image_height: int) -> np.ndarray:
    """Nearest-patch lookup for projected full-resolution image coordinates."""
    grid_h, grid_w = patches.shape[:2]
    px = np.clip((u * grid_w / image_width).astype(np.int32), 0, grid_w - 1)
    py = np.clip((v * grid_h / image_height).astype(np.int32), 0, grid_h - 1)
    return patches[py, px]


def fit_embedding_clusters(features: np.ndarray, clusters: int, dimensions: int,
                           sample_limit: int, seed: int) -> tuple[np.ndarray, dict]:
    if len(features) < clusters:
        raise ValueError("not enough observed triangles for clustering")
    rng = np.random.default_rng(seed)
    fit_indices = (rng.choice(len(features), sample_limit, replace=False)
                   if len(features) > sample_limit else np.arange(len(features)))
    fit = np.ascontiguousarray(features[fit_indices], dtype=np.float32)
    dimensions = min(dimensions, fit.shape[0] - 1, fit.shape[1])
    mean, eigenvectors = cv2.PCACompute(fit, mean=None, maxComponents=dimensions)
    projected = np.ascontiguousarray((features - mean) @ eigenvectors.T, dtype=np.float32)
    labels, kmeans_centers = fit_projected_clusters(projected, clusters, sample_limit, seed)
    return labels, {
        "pca_mean": mean.astype(np.float32),
        "pca_components": eigenvectors.astype(np.float32),
        "kmeans_centers": kmeans_centers,
        "projected_features": projected,
    }


def fit_projected_clusters(projected: np.ndarray, clusters: int,
                           sample_limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster cached PCA features without running DINO inference again."""
    if len(projected) < clusters:
        raise ValueError("not enough projected features for clustering")
    rng = np.random.default_rng(seed)
    fit_indices = (rng.choice(len(projected), sample_limit, replace=False)
                   if len(projected) > sample_limit else np.arange(len(projected)))
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, _, kmeans_centers = cv2.kmeans(
        projected[fit_indices], clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = np.empty(len(projected), dtype=np.int16)
    for start in range(0, len(projected), 8192):
        chunk = projected[start:start + 8192]
        distance = ((chunk[:, None, :] - kmeans_centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + len(chunk)] = distance.argmin(axis=1)
    return labels, kmeans_centers.astype(np.float32)


def load_model(device: str, model_name: str, model_repo: Path | None = None,
               model_weights: Path | None = None):
    if model_name.startswith("dinov3_"):
        if model_repo is None or model_weights is None:
            raise ValueError("DINOv3 requires the official local repository and approved weights")
        sys.path.insert(0, str(model_repo))
        backbones = importlib.import_module("dinov3.hub.backbones")
        model = getattr(backbones, model_name)(weights=str(model_weights))
        return model.eval().to(device)
    import torch

    if model_repo is None:
        model = torch.hub.load(
            "facebookresearch/dinov2", model_name, trust_repo=True, verbose=True)
    else:
        model = torch.hub.load(
            str(model_repo), model_name, source="local", verbose=True)
    return model.eval().to(device)


def extract_patches(model, rgb: np.ndarray, device: str,
                    width: int, height: int, patch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        tokens = model.forward_features(tensor)["x_norm_patchtokens"]
        tokens = functional.normalize(tokens, dim=-1)
    grid_h, grid_w = height // patch_size, width // patch_size
    return tokens[0].reshape(grid_h, grid_w, -1).float().cpu().numpy()


def run(args: argparse.Namespace) -> dict:
    dimension, patch_size, default_width, default_height, family, source = MODEL_SPECS[args.model]
    width, height = args.width or default_width, args.height or default_height
    if width % patch_size or height % patch_size:
        raise ValueError(f"{family} image width and height must be divisible by patch size {patch_size}")
    started = time.monotonic()
    import torch

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    centers = vertices[triangles].mean(axis=1)
    normals = np.asarray(mesh.triangle_normals)
    frames = load_frames(args.scan / "odometry.csv")
    fallback_k = np.loadtxt(args.scan / "camera_matrix.csv", delimiter=",")
    candidates = sorted(frames)[::args.frame_step]
    if args.max_frames:
        candidates = candidates[:args.max_frames]
    selected = [frame_id for frame_id in candidates
                if (args.scan / "depth" / f"{frame_id:06d}.png").is_file()
                and (args.scan / "confidence" / f"{frame_id:06d}.png").is_file()]
    selected_set = set(selected)

    model = load_model(device, args.model, args.model_repo, args.model_weights)
    feature_sum = np.zeros((len(triangles), dimension), dtype=np.float32)
    rgb_sum = np.zeros((len(triangles), 3), dtype=np.float32)
    weight_sum = np.zeros(len(triangles), dtype=np.float32)
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
        if frame_id not in selected_set:
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
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patches = extract_patches(model, rgb, device, width, height, patch_size)
        features = sample_patch_features(patches, u[indices], v[indices], rgb_w, rgb_h)
        pixel_rgb = rgb[vi[indices], ui[indices]].astype(np.float32) / 255
        weights = incidence[indices] * (1.0 + 0.5 * (conf[indices] == 2))
        feature_sum[indices] += features * weights[:, None]
        rgb_sum[indices] += pixel_rgb * weights[:, None]
        weight_sum[indices] += weights
        observations[indices] += 1
        used_frames += 1
        print(f"frame {used_frames}/{len(selected)} id={frame_id} visible={len(indices)}", flush=True)
    cap.release()

    observed = observations >= args.min_views
    fused = np.divide(feature_sum, weight_sum[:, None], out=np.zeros_like(feature_sum),
                      where=weight_sum[:, None] > 0)
    norm = np.linalg.norm(fused, axis=1, keepdims=True)
    fused = np.divide(fused, norm, out=np.zeros_like(fused), where=norm > 0)
    raw_labels, fit = fit_embedding_clusters(
        fused[observed], args.clusters, args.pca_dimensions, args.fit_sample_limit, args.seed)

    if args.rgb_triangle_data is not None:
        rgb_data = np.load(args.rgb_triangle_data)
        fused_rgb = np.asarray(rgb_data["fused_rgb"], dtype=np.float32)
        rgb_source = str(args.rgb_triangle_data)
    else:
        fused_rgb = np.divide(rgb_sum, weight_sum[:, None], out=np.zeros_like(rgb_sum),
                              where=weight_sum[:, None] > 0)
        rgb_source = f"same {family} projection frames"
    if len(fused_rgb) != len(triangles):
        raise ValueError("RGB triangle data does not match the production mesh")
    areas = triangle_areas(vertices, triangles)
    lightness = cv2.cvtColor(np.clip(fused_rgb[:, None, :], 0, 1), cv2.COLOR_RGB2LAB)[:, 0, 0]
    order = np.argsort([lightness[observed][raw_labels == cluster].mean()
                        for cluster in range(args.clusters)])
    remap = np.empty(args.clusters, dtype=np.int16)
    remap[order] = np.arange(args.clusters)
    labels = np.full(len(triangles), -1, dtype=np.int16)
    labels[observed] = remap[raw_labels]
    labels = smooth_triangle_labels(
        triangles, labels, observed, args.clusters, args.smooth_iterations)

    face_colors = np.full((len(triangles), 3), 0.16, dtype=np.float64)
    face_colors[observed] = PALETTE[labels[observed]]
    rgb_colors = np.clip(fused_rgb.astype(np.float64), 0, 1)
    rgb_colors[~observed] = 0.16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_mesh_path = args.output_dir / "surface_clusters_mesh.ply"
    rgb_mesh_path = args.output_dir / "surface_fused_rgb_mesh.ply"
    o3d.io.write_triangle_mesh(
        str(cluster_mesh_path), colored_triangle_mesh(vertices, triangles, face_colors),
        write_vertex_colors=True)
    o3d.io.write_triangle_mesh(
        str(rgb_mesh_path), colored_triangle_mesh(vertices, triangles, rgb_colors),
        write_vertex_colors=True)
    np.savez_compressed(
        args.output_dir / "surface_cluster_triangle_data.npz",
        labels=labels, observations=observations, areas_m2=areas,
        fused_rgb=fused_rgb,
        projected_features=fit.pop("projected_features"), **fit)

    observed_area = float(areas[observed].sum())
    rows = []
    for cluster in range(args.clusters):
        mask = observed & (labels == cluster)
        area = float(areas[mask].sum())
        rows.append({
            "cluster": cluster,
            "triangles": int(mask.sum()),
            "area_cm2": area * 10000,
            "observed_area_pct": 100 * area / observed_area if observed_area else None,
            "mean_rgb": fused_rgb[mask].mean(axis=0).tolist(),
            "display_color": PALETTE[cluster].tolist(),
        })
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = peak_rss / (1024 ** 2 if platform.system() == "Darwin" else 1024)
    result = {
        "status": "candidate_clusters_not_ground_truth",
        "method": "DINO_multiview_patch_features_to_production_mesh",
        "config": {
            "model": f"{source}:{args.model}",
            "device": device,
            "frame_step": args.frame_step,
            "image_size": [width, height],
            "patch_size": patch_size,
            "confidence_min": args.confidence,
            "depth_tolerance_m": args.depth_tolerance,
            "min_incidence_cosine": args.min_incidence,
            "min_views": args.min_views,
            "clusters": args.clusters,
            "pca_dimensions": args.pca_dimensions,
            "smooth_iterations": args.smooth_iterations,
            "seed": args.seed,
            "rgb_source": rgb_source,
        },
        "frames_selected": len(selected),
        "frames_contributing": used_frames,
        "triangles": int(len(triangles)),
        "observed_triangles": int(observed.sum()),
        "observed_fraction": float(observed.mean()),
        "observed_area_cm2": observed_area * 10000,
        "clusters": rows,
        "runtime": {"seconds": time.monotonic() - started, "peak_rss_mb": peak_rss_mb},
        "outputs": {
            "cluster_mesh": str(cluster_mesh_path),
            "fused_rgb_mesh": str(rgb_mesh_path),
            "triangle_data": str(args.output_dir / "surface_cluster_triangle_data.npz"),
        },
    }
    (args.output_dir / "surface_cluster_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--rgb-triangle-data", type=Path,
                        help="Optional pre-fused RGB; omitted to fuse RGB during DINO projection")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=48)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--model", choices=tuple(MODEL_DIMENSIONS), default="dinov2_vits14")
    parser.add_argument("--model-repo", type=Path,
                        help="Local official DINO repository")
    parser.add_argument("--model-weights", type=Path,
                        help="Approved official DINOv3 checkpoint")
    parser.add_argument("--confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--depth-tolerance", type=float, default=0.035)
    parser.add_argument("--min-incidence", type=float, default=0.20)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--pca-dimensions", type=int, default=16)
    parser.add_argument("--fit-sample-limit", type=int, default=20000)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
