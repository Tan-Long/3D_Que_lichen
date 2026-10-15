#!/usr/bin/env python3
"""Cluster multi-view surface appearance without semantic training labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


PALETTE = np.asarray([
    [0.89, 0.34, 0.31], [0.96, 0.68, 0.25], [0.47, 0.75, 0.34],
    [0.20, 0.70, 0.72], [0.28, 0.49, 0.84], [0.58, 0.40, 0.82],
    [0.88, 0.43, 0.72], [0.63, 0.67, 0.70], [0.70, 0.52, 0.32],
    [0.39, 0.77, 0.58], [0.48, 0.60, 0.90], [0.79, 0.55, 0.86],
], dtype=np.float64)


def local_surface_mean(values: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Average face values through shared vertices without crossing mesh gaps."""
    vertex_count = int(triangles.max()) + 1
    flat_vertices = triangles.reshape(-1)
    counts = np.bincount(flat_vertices, minlength=vertex_count)
    vertex_values = np.zeros((vertex_count, values.shape[1]), dtype=np.float64)
    repeated = np.repeat(values, 3, axis=0)
    for channel in range(values.shape[1]):
        vertex_values[:, channel] = np.bincount(
            flat_vertices, weights=repeated[:, channel], minlength=vertex_count)
    vertex_values /= np.maximum(counts[:, None], 1)
    return vertex_values[triangles].mean(axis=1)


def robust_standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(features, axis=0)
    scale = np.median(np.abs(features - center), axis=0) * 1.4826
    scale[scale < 1e-6] = 1.0
    return ((features - center) / scale).astype(np.float32), center, scale


def smooth_triangle_labels(triangles: np.ndarray, labels: np.ndarray,
                           observed: np.ndarray, clusters: int,
                           iterations: int) -> np.ndarray:
    """Use shared-vertex majority votes; never smooth across disconnected faces."""
    result = labels.copy()
    vertex_count = int(triangles.max()) + 1
    for _ in range(iterations):
        votes = np.zeros((vertex_count, clusters), dtype=np.uint16)
        for cluster in range(clusters):
            faces = observed & (result == cluster)
            votes[:, cluster] = np.bincount(
                triangles[faces].reshape(-1), minlength=vertex_count)
        vertex_has_votes = votes.sum(axis=1) > 0
        vertex_labels = votes.argmax(axis=1)
        candidate = vertex_labels[triangles]
        valid = vertex_has_votes[triangles]
        for cluster in range(clusters):
            majority = ((candidate == cluster) & valid).sum(axis=1) >= 2
            result[observed & majority] = cluster
    return result


def colored_triangle_mesh(vertices: np.ndarray, triangles: np.ndarray,
                          face_colors: np.ndarray) -> o3d.geometry.TriangleMesh:
    expanded = vertices[triangles].reshape(-1, 3)
    faces = np.arange(len(expanded), dtype=np.int32).reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(expanded), o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(face_colors, 3, axis=0))
    mesh.compute_vertex_normals()
    return mesh


def cluster_surface(mesh_path: Path, triangle_data_path: Path, output_dir: Path,
                    clusters: int = 8, min_views: int = 3,
                    smooth_iterations: int = 2, seed: int = 7) -> dict:
    if clusters < 2 or clusters > len(PALETTE):
        raise ValueError(f"clusters must be between 2 and {len(PALETTE)}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    data = np.load(triangle_data_path)
    fused_rgb = np.asarray(data["fused_rgb"], dtype=np.float32)
    observations = np.asarray(data["observations"])
    areas = np.asarray(data["areas_m2"], dtype=np.float64)
    if len(triangles) != len(fused_rgb):
        raise ValueError("triangle data does not match the production mesh")
    observed = observations >= min_views
    if int(observed.sum()) < clusters:
        raise ValueError("not enough observed triangles for clustering")

    rgb_image = np.clip(fused_rgb[:, None, :], 0, 1).astype(np.float32)
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)[:, 0]
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)[:, 0]
    local_lab = local_surface_mean(lab.astype(np.float64), triangles)
    local_delta = lab - local_lab
    local_distance = np.linalg.norm(local_delta, axis=1, keepdims=True)
    hue = np.deg2rad(hsv[:, 0:1])
    features = np.column_stack((
        lab, np.sin(hue[:, 0]), np.cos(hue[:, 0]), hsv[:, 1], hsv[:, 2],
        local_delta[:, 0], local_distance[:, 0],
    ))
    standardized, feature_center, feature_scale = robust_standardize(features[observed])
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, raw_labels, _ = cv2.kmeans(
        standardized, clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    raw_labels = raw_labels[:, 0]

    mean_lightness = np.asarray([
        float(lab[observed][raw_labels == cluster, 0].mean())
        for cluster in range(clusters)
    ])
    order = np.argsort(mean_lightness)
    remap = np.empty(clusters, dtype=np.int16)
    remap[order] = np.arange(clusters)
    labels = np.full(len(triangles), -1, dtype=np.int16)
    labels[observed] = remap[raw_labels]
    labels = smooth_triangle_labels(
        triangles, labels, observed, clusters, smooth_iterations)

    cluster_colors = PALETTE[:clusters]
    face_colors = np.full((len(triangles), 3), 0.16, dtype=np.float64)
    face_colors[observed] = cluster_colors[labels[observed]]
    fused_colors = np.clip(fused_rgb.astype(np.float64), 0, 1)
    fused_colors[~observed] = 0.16
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_mesh_path = output_dir / "surface_clusters_mesh.ply"
    fused_mesh_path = output_dir / "surface_fused_rgb_mesh.ply"
    o3d.io.write_triangle_mesh(
        str(cluster_mesh_path), colored_triangle_mesh(vertices, triangles, face_colors),
        write_vertex_colors=True)
    o3d.io.write_triangle_mesh(
        str(fused_mesh_path), colored_triangle_mesh(vertices, triangles, fused_colors),
        write_vertex_colors=True)
    np.savez_compressed(
        output_dir / "surface_cluster_triangle_data.npz", labels=labels,
        observations=observations, fused_rgb=fused_rgb, areas_m2=areas,
        feature_center=feature_center, feature_scale=feature_scale)

    observed_area = float(areas[observed].sum())
    rows = []
    for cluster in range(clusters):
        mask = observed & (labels == cluster)
        area = float(areas[mask].sum())
        rows.append({
            "cluster": cluster,
            "triangles": int(mask.sum()),
            "area_cm2": area * 10000,
            "observed_area_pct": 100 * area / observed_area if observed_area else None,
            "mean_rgb": fused_rgb[mask].mean(axis=0).tolist(),
            "mean_lab": lab[mask].mean(axis=0).tolist(),
            "mean_saturation": float(hsv[mask, 1].mean()),
            "display_color": cluster_colors[cluster].tolist(),
        })
    result = {
        "status": "candidate_clusters_not_ground_truth",
        "method": "multiview_RGB_to_mesh + Lab_HSV_local_contrast_kmeans + mesh_adjacency_smoothing",
        "config": {"clusters": clusters, "min_views": min_views,
                   "smooth_iterations": smooth_iterations, "seed": seed},
        "mesh": str(mesh_path), "triangle_data": str(triangle_data_path),
        "triangles": int(len(triangles)),
        "observed_triangles": int(observed.sum()),
        "observed_area_cm2": observed_area * 10000,
        "clusters": rows,
        "outputs": {"cluster_mesh": str(cluster_mesh_path),
                    "fused_rgb_mesh": str(fused_mesh_path),
                    "triangle_data": str(output_dir / "surface_cluster_triangle_data.npz")},
    }
    (output_dir / "surface_cluster_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--triangle-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(cluster_surface(
        args.mesh, args.triangle_data, args.output_dir, args.clusters,
        args.min_views, args.smooth_iterations, args.seed),
        ensure_ascii=False, indent=2))
