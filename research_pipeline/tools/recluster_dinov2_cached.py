#!/usr/bin/env python3
"""Recluster cached DINO triangle features without rerunning inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from cluster_lichen_dinov2 import fit_projected_clusters
from cluster_lichen_surface import PALETTE, colored_triangle_mesh, smooth_triangle_labels


def recluster(mesh_path: Path, source_dir: Path, output_dir: Path,
              clusters: int, smooth_iterations: int, seed: int) -> dict:
    if not 2 <= clusters <= len(PALETTE):
        raise ValueError(f"clusters must be between 2 and {len(PALETTE)}")
    started = time.monotonic()
    source_metrics = json.loads((source_dir / "surface_cluster_metrics.json").read_text())
    source_data = np.load(source_dir / "surface_cluster_triangle_data.npz")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    observations = np.asarray(source_data["observations"])
    fused_rgb = np.asarray(source_data["fused_rgb"], dtype=np.float32)
    areas = np.asarray(source_data["areas_m2"], dtype=np.float64)
    projected = np.asarray(source_data["projected_features"], dtype=np.float32)
    min_views = int(source_metrics["config"]["min_views"])
    observed = observations >= min_views
    if len(triangles) != len(observations) or int(observed.sum()) != len(projected):
        raise ValueError("cached features do not match the production mesh")

    raw_labels, centers = fit_projected_clusters(projected, clusters, 20000, seed)
    lightness = cv2.cvtColor(np.clip(fused_rgb[:, None, :], 0, 1), cv2.COLOR_RGB2LAB)[:, 0, 0]
    order = np.argsort([lightness[observed][raw_labels == cluster].mean()
                        for cluster in range(clusters)])
    remap = np.empty(clusters, dtype=np.int16)
    remap[order] = np.arange(clusters)
    labels = np.full(len(triangles), -1, dtype=np.int16)
    labels[observed] = remap[raw_labels]
    labels = smooth_triangle_labels(
        triangles, labels, observed, clusters, smooth_iterations)

    face_colors = np.full((len(triangles), 3), 0.16, dtype=np.float64)
    face_colors[observed] = PALETTE[labels[observed]]
    rgb_colors = np.clip(fused_rgb.astype(np.float64), 0, 1)
    rgb_colors[~observed] = 0.16
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_mesh_path = output_dir / "surface_clusters_mesh.ply"
    rgb_mesh_path = output_dir / "surface_fused_rgb_mesh.ply"
    o3d.io.write_triangle_mesh(
        str(cluster_mesh_path), colored_triangle_mesh(vertices, triangles, face_colors),
        write_vertex_colors=True)
    o3d.io.write_triangle_mesh(
        str(rgb_mesh_path), colored_triangle_mesh(vertices, triangles, rgb_colors),
        write_vertex_colors=True)
    np.savez_compressed(
        output_dir / "surface_cluster_triangle_data.npz", labels=labels,
        observations=observations, areas_m2=areas, fused_rgb=fused_rgb,
        projected_features=projected, pca_mean=source_data["pca_mean"],
        pca_components=source_data["pca_components"], kmeans_centers=centers)

    observed_area = float(areas[observed].sum())
    rows = []
    for cluster in range(clusters):
        mask = observed & (labels == cluster)
        area = float(areas[mask].sum())
        rows.append({
            "cluster": cluster,
            "triangles": int(mask.sum()),
            "area_cm2": area * 10000,
            "observed_area_pct": 100 * area / observed_area,
            "mean_rgb": fused_rgb[mask].mean(axis=0).tolist(),
            "display_color": PALETTE[cluster].tolist(),
        })
    result = {
        "status": "candidate_clusters_not_ground_truth",
        "method": "cached_DINO_projected_features_recluster",
        "source": str(source_dir),
        "config": {**source_metrics["config"], "clusters": clusters,
                   "smooth_iterations": smooth_iterations, "seed": seed},
        "triangles": int(len(triangles)),
        "observed_triangles": int(observed.sum()),
        "observed_fraction": float(observed.mean()),
        "observed_area_cm2": observed_area * 10000,
        "clusters": rows,
        "runtime": {"seconds": time.monotonic() - started},
        "outputs": {"cluster_mesh": str(cluster_mesh_path),
                    "fused_rgb_mesh": str(rgb_mesh_path),
                    "triangle_data": str(output_dir / "surface_cluster_triangle_data.npz")},
    }
    (output_dir / "surface_cluster_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(recluster(args.mesh, args.source_dir, args.output_dir,
                               args.clusters, args.smooth_iterations, args.seed),
                     ensure_ascii=False, indent=2))
