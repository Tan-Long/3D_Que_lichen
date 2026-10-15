#!/usr/bin/env python3
"""Measure observed TSDF surface area globally and by terrain-facing half."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

MIN_CENTERLINE_INLIER_FRACTION = 0.30


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def submesh(mesh: o3d.geometry.TriangleMesh, keep: np.ndarray) -> o3d.geometry.TriangleMesh:
    result = copy.deepcopy(mesh)
    result.remove_triangles_by_mask(~keep)
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    return result


def circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    p0, p1, p2 = points
    matrix = 2 * np.vstack((p1 - p0, p2 - p0))
    rhs = np.array([np.dot(p1, p1) - np.dot(p0, p0), np.dot(p2, p2) - np.dot(p0, p0)])
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(p0 - center))


def robust_circle(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float] | None:
    if len(points) < 80:
        return None
    sample = points if len(points) <= 2500 else points[rng.choice(len(points), 2500, replace=False)]
    best = None
    for _ in range(500):
        candidate = circle_from_three(sample[rng.choice(len(sample), 3, replace=False)])
        if candidate is None:
            continue
        center, radius = candidate
        if not 0.02 <= radius <= 0.25:
            continue
        residual = np.abs(np.linalg.norm(sample - center, axis=1) - radius)
        mask = residual <= 0.012
        score = int(mask.sum())
        if best is None or score > best[0]:
            best = (score, center, radius, mask)
    if best is None or best[0] < 50:
        return None
    inliers = sample[best[3]]
    initial = np.r_[best[1], best[2]]
    fitted = least_squares(
        lambda value: np.linalg.norm(inliers - value[:2], axis=1) - value[2],
        initial, loss="soft_l1", f_scale=0.005, max_nfev=100,
    ).x
    if not 0.02 <= fitted[2] <= 0.25:
        return None
    residual = np.abs(np.linalg.norm(sample - fitted[:2], axis=1) - fitted[2])
    return fitted[:2], float(fitted[2]), float((residual <= 0.012).mean())


def estimate_centerline(vertices: np.ndarray, base: np.ndarray, up: np.ndarray) -> dict:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, up)) > 0.85:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = seed - np.dot(seed, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    relative = vertices - base
    xy = np.column_stack((relative @ e1, relative @ e2))
    height = relative @ up
    rng = np.random.default_rng(92041)
    slices = []
    bins = np.arange(max(0.0, height.min()), height.max() + 0.1001, 0.1)
    for low, high in zip(bins[:-1], bins[1:]):
        selected = xy[(height >= low) & (height < high)]
        fit = robust_circle(selected, rng)
        if fit is not None:
            center, radius, inlier_fraction = fit
            slices.append({"height_m": (low + high) / 2, "x": center[0], "y": center[1],
                           "radius_m": radius, "inlier_fraction": inlier_fraction})
    if len(slices) < 3:
        raise RuntimeError("fewer than three reliable trunk cross-sections")
    # Reject isolated centre jumps before interpolation.
    centers = np.array([[s["x"], s["y"]] for s in slices])
    median = np.median(centers, axis=0)
    distance = np.linalg.norm(centers - median, axis=1)
    limit = max(0.08, float(np.quantile(distance, 0.8) * 1.5))
    slices = [s for s, keep in zip(slices, distance <= limit) if keep]
    if len(slices) < 3:
        raise RuntimeError("trunk centreline became unstable after outlier rejection")
    # Forks or nearby objects can fit a plausible but much larger circle in one
    # slice. Such a radius jump is not a centreline observation.
    radii = np.asarray([s["radius_m"] for s in slices])
    radius_limit = max(0.08, float(np.median(radii) * 2.5))
    # A connected trunk component can legitimately contain buttress roots or
    # branch fragments.  RANSAC already requires at least 50 circle inliers;
    # retain that stable circle when clutter lowers its share of the slice.
    slices = [s for s in slices if s["radius_m"] <= radius_limit
              and s["inlier_fraction"] >= MIN_CENTERLINE_INLIER_FRACTION]
    if len(slices) < 3:
        raise RuntimeError("trunk centreline became unstable after radius rejection")
    return {"e1": e1, "e2": e2, "up": up, "base": base, "slices": slices}


def centers_at(centerline: dict, points: np.ndarray) -> np.ndarray:
    relative = points - centerline["base"]
    heights = relative @ centerline["up"]
    hs = np.array([s["height_m"] for s in centerline["slices"]])
    xs = np.array([s["x"] for s in centerline["slices"]])
    ys = np.array([s["y"] for s in centerline["slices"]])
    return np.column_stack((np.interp(heights, hs, xs), np.interp(heights, hs, ys)))


def surface_coverage(vertices: np.ndarray, centerline: dict, down_xy: np.ndarray) -> dict:
    relative = vertices - centerline["base"]
    heights = relative @ centerline["up"]
    xy = np.column_stack((relative @ centerline["e1"], relative @ centerline["e2"]))
    radial = xy - centers_at(centerline, vertices)
    angles = (np.arctan2(radial[:, 1], radial[:, 0]) + 2 * np.pi) % (2 * np.pi)
    signed = radial @ down_xy
    total_values, up_values, down_values = [], [], []
    bins = np.arange(heights.min(), heights.max() + 0.1001, 0.1)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (heights >= low) & (heights < high)
        if mask.sum() < 100:
            continue
        sectors = np.floor(angles[mask] / (2 * np.pi / 36)).astype(int)
        total_values.append(len(np.unique(sectors)) / 36)
        up_values.append(min(1.0, len(np.unique(sectors[signed[mask] < 0])) / 18))
        down_values.append(min(1.0, len(np.unique(sectors[signed[mask] >= 0])) / 18))
    summarize = lambda values: {
        "median": float(np.median(values)) if values else 0.0,
        "p10": float(np.quantile(values, 0.1)) if values else 0.0,
    }
    return {
        "evaluated_height_slices": len(total_values),
        "all_azimuth": summarize(total_values),
        "upslope_half": summarize(up_values),
        "downslope_half": summarize(down_values),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scan-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terrain-results", type=Path,
                        default=root / "prototypes/upslope_downslope/upslope_downslope_prototype.json")
    args = parser.parse_args()
    records = json.loads(args.terrain_results.read_text(encoding="utf-8"))
    record = next((r for r in records if r["name"] == args.scan_name), None)
    if record is None:
        raise SystemExit(f"scan missing from terrain results: {args.scan_name}")
    fit = record["variants"]["adaptive"]
    if not fit.get("quality_ok"):
        raise SystemExit("terrain direction is undefined; refusing to invent upslope/downslope areas")
    base = np.asarray(fit["base"], dtype=float)
    up = -np.asarray(record["gravity"], dtype=float)
    up /= np.linalg.norm(up)
    down = np.asarray(fit["down"], dtype=float)
    down -= np.dot(down, up) * up
    down /= np.linalg.norm(down)

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(triangles):
        raise SystemExit("mesh is empty")
    centroids = vertices[triangles].mean(axis=1)
    centerline = estimate_centerline(vertices, base, up)
    relative = centroids - base
    centroid_xy = np.column_stack((relative @ centerline["e1"], relative @ centerline["e2"]))
    radial_xy = centroid_xy - centers_at(centerline, centroids)
    down_xy = np.array([np.dot(down, centerline["e1"]), np.dot(down, centerline["e2"])])
    signed = radial_xy @ down_xy
    is_down = signed >= 0
    areas = triangle_areas(vertices, triangles)
    total = float(areas.sum())
    downslope = float(areas[is_down].sum())
    upslope = float(areas[~is_down].sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    up_mesh = submesh(mesh, ~is_down)
    down_mesh = submesh(mesh, is_down)
    up_mesh.paint_uniform_color([1.0, 0.55, 0.16])
    down_mesh.paint_uniform_color([0.18, 0.58, 1.0])
    up_path = args.output_dir / "trunk_mesh_upslope.ply"
    down_path = args.output_dir / "trunk_mesh_downslope.ply"
    sides_path = args.output_dir / "trunk_mesh_sides.ply"
    o3d.io.write_triangle_mesh(str(up_path), up_mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(down_path), down_mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(sides_path), up_mesh + down_mesh, write_vertex_colors=True)
    result = {
        "scan": args.scan_name,
        "domain": "observed valid TSDF surface only; holes and unobserved surface are excluded",
        "split_rule": "downslope-oriented half-plane through robust cross-section centreline; triangle assigned by centroid",
        "terrain_source_variant": fit.get("source_variant"),
        "surface_area_m2": {
            "total": total,
            "upslope": upslope,
            "downslope": downslope,
        },
        "surface_area_cm2": {
            "total": total * 10000,
            "upslope": upslope * 10000,
            "downslope": downslope * 10000,
        },
        "observed_surface_fraction": {
            "upslope": upslope / total,
            "downslope": downslope / total,
        },
        "triangle_count": {
            "total": int(len(triangles)),
            "upslope": int((~is_down).sum()),
            "downslope": int(is_down.sum()),
        },
        "centerline": {
            "cross_sections_used": len(centerline["slices"]),
            "median_radius_m": float(np.median([s["radius_m"] for s in centerline["slices"]])),
            "median_circle_inlier_fraction": float(np.median([s["inlier_fraction"] for s in centerline["slices"]])),
            "slices": centerline["slices"],
        },
        "observed_coverage": surface_coverage(vertices, centerline, down_xy),
        "outputs": {"upslope_mesh": str(up_path), "downslope_mesh": str(down_path),
                    "combined_sides_mesh": str(sides_path)},
    }
    (args.output_dir / "surface_area_by_side.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
