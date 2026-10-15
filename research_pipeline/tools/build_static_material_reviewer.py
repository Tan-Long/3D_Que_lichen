#!/usr/bin/env python3
"""Build a compact static material reviewer for Vercel Hobby deployments."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts/surface_cluster_pilot"
PRODUCTION_PATH = ROOT / "manifests/production_baseline_v1_approved.csv"


def write_packed(path: Path, vertices: np.ndarray, triangles: np.ndarray,
                 rgb_colors: np.ndarray, material_colors: np.ndarray) -> None:
    faces = len(triangles)
    if rgb_colors.shape != (faces, 3) or material_colors.shape != (faces, 3):
        raise ValueError("face colors must match triangle count")
    vertices = np.asarray(vertices, dtype=np.float32)
    minimum = vertices.min(axis=0)
    scale = vertices.max(axis=0) - minimum
    scale[scale == 0] = 1
    quantized = np.rint((vertices - minimum) / scale * 65535).astype("<u2")
    with gzip.open(path, "wb", compresslevel=6) as handle:
        handle.write(b"QLM2")
        handle.write(struct.pack("<II", len(vertices), faces))
        handle.write(np.asarray(minimum, dtype="<f4").tobytes())
        handle.write(np.asarray(scale, dtype="<f4").tobytes())
        handle.write(quantized.tobytes())
        handle.write(b"\0" * (-quantized.nbytes % 4))
        handle.write(np.asarray(triangles, dtype="<u4").tobytes())
        handle.write(np.asarray(rgb_colors, dtype=np.uint8).tobytes())
        handle.write(np.asarray(material_colors, dtype=np.uint8).tobytes())


def source_variant(uid: str) -> str:
    for variant in ("dinov3_vitl16_k3", "dinov3_vitl16_k3_cached"):
        if (PILOT_ROOT / uid / variant / "surface_fused_rgb_mesh.ply").is_file():
            return variant
    raise FileNotFoundError(f"{uid}: RGB mesh not found")


def pack_tree(uid: str, output: Path) -> dict:
    import open3d as o3d

    variant = source_variant(uid)
    rgb_path = PILOT_ROOT / uid / variant / "surface_fused_rgb_mesh.ply"
    material_dir = PILOT_ROOT / uid / "material_probe_pure_v2"
    material_path = material_dir / "material_probability_mesh.ply"
    metrics = json.loads((material_dir / "material_probe_metrics.json").read_text())
    with tempfile.TemporaryDirectory(prefix="que-lichen-static-") as directory:
        local = Path(directory)
        local_rgb, local_material = local / "rgb.ply", local / "material.ply"
        shutil.copyfile(rgb_path, local_rgb)
        shutil.copyfile(material_path, local_material)
        rgb_mesh = o3d.io.read_triangle_mesh(str(local_rgb))
        material_mesh = o3d.io.read_triangle_mesh(str(local_material))
    rgb_triangles = np.asarray(rgb_mesh.triangles)
    material_triangles = np.asarray(material_mesh.triangles)
    rgb_positions = np.asarray(rgb_mesh.vertices)[rgb_triangles]
    material_positions = np.asarray(material_mesh.vertices)[material_triangles]
    if rgb_positions.shape != material_positions.shape or not np.allclose(
            rgb_positions, material_positions, rtol=0, atol=1e-7):
        raise ValueError(f"{uid}: RGB and material geometry differ")
    vertices, inverse = np.unique(material_positions.reshape(-1, 3), axis=0,
                                  return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    rgb_colors = np.rint(
        np.asarray(rgb_mesh.vertex_colors)[rgb_triangles[:, 0]] * 255).clip(0, 255)
    material_colors = np.rint(
        np.asarray(material_mesh.vertex_colors)[material_triangles[:, 0]] * 255).clip(0, 255)
    write_packed(output, vertices, triangles, rgb_colors, material_colors)
    return {key: metrics[key] for key in (
        "material_triangles", "material_area_cm2", "estimated_lichen_area_cm2",
        "hard_bark_area_cm2", "review_band_area_cm2", "hard_lichen_area_cm2",
    )} | {"cross_validation": {key: metrics["cross_validation"][key] for key in (
        "mean_absolute_error", "directional_accuracy")}}


def build(output: Path, limit: int | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    with PRODUCTION_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    samples = []
    for index, row in enumerate(rows, 1):
        uid = row["scan_uid"]
        asset_name = f"{uid}.qlm.gz"
        metrics = pack_tree(uid, assets / asset_name)
        samples.append({"uid": uid, "name": row["scan"], "prior": "baseline approved",
                        "asset": f"./assets/{asset_name}", "metrics": metrics})
        print(f"[{index}/{len(rows)}] {uid}", flush=True)
    shutil.copyfile(ROOT / "surface_cluster_gallery.html",
                    output / "surface_cluster_gallery.html")
    (output / "review_catalog.json").write_text(
        json.dumps({"samples": samples, "ready": len(samples), "total": len(rows)},
                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (output / "vercel.json").write_text(json.dumps({
        "redirects": [{"source": "/", "destination":
                       "/surface_cluster_gallery.html?variant=material&static=1",
                       "permanent": False}],
        "headers": [{"source": "/assets/(.*)", "headers": [
            {"key": "Cache-Control", "value": "public, max-age=31536000, immutable"},
            {"key": "Content-Type", "value": "application/octet-stream"}]}],
    }, indent=2), encoding="utf-8")
    return {"samples": len(samples), "bytes": sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()
