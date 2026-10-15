#!/usr/bin/env python3
"""Collect TSDF metrics into CSV/JSON/Markdown and flag unstable area."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    config, mesh, heldout = record.get("config", {}), record.get("mesh", {}), record.get("heldout_depth", {})
    side_path = path.parent / "surface_area_by_side.json"
    side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.exists() else {}
    coverage = side.get("observed_coverage", {})
    return {
        "scan": record.get("scan"),
        "run": path.parent.name,
        "method": record.get("method"),
        "voxel_mm": round(1000 * config.get("voxel_m", 0), 3),
        "confidence_min": config.get("confidence_min"),
        "frame_step": config.get("frame_step"),
        "area_m2": mesh.get("surface_area_m2"),
        "triangles": mesh.get("triangles"),
        "components": mesh.get("components"),
        "largest_component_fraction": mesh.get("largest_component_fraction"),
        "azimuth_coverage_median": mesh.get("azimuth_coverage_median"),
        "centerline_coverage_median": coverage.get("all_azimuth", {}).get("median"),
        "upslope_coverage_median": coverage.get("upslope_half", {}).get("median"),
        "downslope_coverage_median": coverage.get("downslope_half", {}).get("median"),
        "heldout_hit_fraction": heldout.get("mesh_hit_fraction"),
        "heldout_within_20mm": heldout.get("within_20mm_fraction"),
        "heldout_median_mm": None if heldout.get("absolute_residual_median_m") is None else 1000 * heldout["absolute_residual_median_m"],
        "heldout_p90_mm": None if heldout.get("absolute_residual_p90_m") is None else 1000 * heldout["absolute_residual_p90_m"],
        "elapsed_seconds": record.get("elapsed_seconds"),
        "metrics_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/meshes"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/meshes/benchmark_summary"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("**/metrics.json")):
        if path.parent.name.endswith("_probe") or path.parent.name.startswith("debug_"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        mesh_path = raw.get("outputs", {}).get("mesh")
        if mesh_path and not Path(mesh_path).exists():
            continue
        rows.append(flatten(path))
    if not rows:
        raise SystemExit("no metrics.json files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TSDF mesh benchmark",
        "",
        "| Scan | Run | Backend | Voxel | Conf | Area m² | Coverage all / up / down | Held-out median / P90 | Hit | ≤20 mm |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        method = "tensor" if "Tensor" in str(row["method"]) else "legacy"
        fmt = lambda x, digits=3: "—" if x is None else f"{x:.{digits}f}"
        cov_all = row["centerline_coverage_median"]
        if cov_all is None:
            cov_all = row["azimuth_coverage_median"]
        coverage_text = f"{fmt(cov_all)} / {fmt(row['upslope_coverage_median'])} / {fmt(row['downslope_coverage_median'])}"
        lines.append(
            f"| {row['scan']} | {row['run']} | {method} | {row['voxel_mm']:.0f} mm | "
            f"{row['confidence_min']} | {fmt(row['area_m2'])} | {coverage_text} | "
            f"{fmt(row['heldout_median_mm'],1)} / {fmt(row['heldout_p90_mm'],1)} mm | "
            f"{fmt(row['heldout_hit_fraction'])} | {fmt(row['heldout_within_20mm'])} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Collected {len(rows)} runs -> {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
