#!/usr/bin/env python3
"""Split pilot scans into geometry-QA pass and deferred manifests.

Only canonical production runs are considered. Ablations/probes are ignored.
The pass manifest is intended to be the sole input to downstream semantic work.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


LIMITS = {
    "heldout_median_mm_max": 10.0,
    "heldout_p90_mm_max": 25.0,
    "heldout_within_20mm_min": 0.80,
    "heldout_hit_fraction_min": 0.80,
    "largest_component_fraction_min": 0.90,
}


def verdict(metrics: dict) -> tuple[bool, list[str], dict]:
    mesh = metrics.get("mesh", {})
    heldout = metrics.get("heldout_depth", {})
    values = {
        "heldout_median_mm": _millimetres(heldout.get("absolute_residual_median_m")),
        "heldout_p90_mm": _millimetres(heldout.get("absolute_residual_p90_m")),
        "heldout_within_20mm": heldout.get("within_20mm_fraction"),
        "heldout_hit_fraction": heldout.get("mesh_hit_fraction"),
        "largest_component_fraction": mesh.get("largest_component_fraction"),
    }
    checks = [
        ("heldout_median_mm", "median residual > 10 mm", lambda x: x <= LIMITS["heldout_median_mm_max"]),
        ("heldout_p90_mm", "P90 residual > 25 mm", lambda x: x <= LIMITS["heldout_p90_mm_max"]),
        ("heldout_within_20mm", "less than 80% depth within 20 mm",
         lambda x: x >= LIMITS["heldout_within_20mm_min"]),
        ("heldout_hit_fraction", "held-out mesh hit rate < 80%",
         lambda x: x >= LIMITS["heldout_hit_fraction_min"]),
        ("largest_component_fraction", "largest mesh component < 90%",
         lambda x: x >= LIMITS["largest_component_fraction_min"]),
    ]
    reasons = []
    for key, message, check in checks:
        value = values[key]
        if value is None:
            reasons.append(f"missing {key}")
        elif not check(value):
            reasons.append(message)
    return not reasons, reasons, values


def _millimetres(value: float | None) -> float | None:
    return None if value is None else float(value) * 1000.0


def preferred_metrics(paths: list[Path]) -> Path | None:
    production = [path for path in paths if "_probe" not in path.parent.name
                  and "conf2" not in path.parent.name and path.parent.name != "tensor_v3mm_step12_r22"
                  and not path.parent.name.startswith("legacy_") and not path.parent.name.startswith("v6mm_")
                  and not path.parent.name.startswith("tensor_v8mm_")
                  and (path.parent / "trunk_mesh_tsdf.ply").exists()
                  and (path.parent / "surface_area_by_side.json").exists()]
    if not production:
        return None
    def priority(path: Path) -> tuple[int, str]:
        name = path.parent.name
        if "adaptive" in name:
            return 0, name
        if name == "tensor_v5mm_step12_r22":
            return 1, name
        if name == "tensor_v5mm_conf1":
            return 2, name
        return 9, name
    return sorted(production, key=priority)[0]


def exact_run_metrics(paths: list[Path], run_name: str) -> Path | None:
    matches = [path for path in paths if path.parent.name == run_name
               and (path.parent / "trunk_mesh_tsdf.ply").exists()]
    return sorted(matches)[0] if matches else None


def best_available_metrics(paths: list[Path]) -> Path | None:
    candidates = [path for path in paths
                  if "_probe" not in path.parent.name and "conf2" not in path.parent.name
                  and (path.parent / "trunk_mesh_tsdf.ply").exists()]
    if not candidates:
        return None

    def rank(path: Path) -> tuple[float, float, str]:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        ok, reasons, values = verdict(metrics)
        penalty = 0.0
        limits = (("heldout_median_mm", 10.0, "max"), ("heldout_p90_mm", 25.0, "max"),
                  ("heldout_within_20mm", 0.80, "min"),
                  ("heldout_hit_fraction", 0.80, "min"),
                  ("largest_component_fraction", 0.90, "min"))
        for key, limit, direction in limits:
            value = values.get(key)
            if value is None:
                penalty += 10.0
            elif direction == "max":
                penalty += max(0.0, value / limit - 1.0)
            else:
                penalty += max(0.0, (limit - value) / limit)
        return (0 if ok else 1 + len(reasons), penalty, path.parent.name)

    return min(candidates, key=rank)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scan_uid", "scan", "raw_path", "qa_status", "qa_reasons"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=root / "manifests/pilot.csv")
    parser.add_argument("--manifest", type=Path, default=root / "manifests/scans.csv")
    parser.add_argument("--mesh-root", type=Path, default=root / "artifacts/meshes")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=root / "manifests")
    parser.add_argument("--all-geometry", action="store_true",
                        help="Classify every usable canonical scan in scans.csv")
    parser.add_argument("--run-name", help="Require this exact production run")
    parser.add_argument("--best-available", action="store_true",
                        help="Select the QA-best completed production/recovery run per scan")
    args = parser.parse_args()

    source_manifest = args.manifest if args.all_geometry else args.pilot
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        if args.all_geometry:
            pilot = [row for row in csv.DictReader(handle)
                     if row["usable_geometry"] == "True" and row["duplicate_canonical"] == "True"]
        else:
            pilot = [row for row in csv.DictReader(handle)
                     if row["pilot_role"] == "primary_pilot"][:args.count]
    by_scan: dict[str, list[Path]] = {}
    for path in args.mesh_root.glob("**/metrics.json"):
        if path.parent.name.endswith("_probe") or path.parent.name.startswith("debug_"):
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        by_scan.setdefault(metrics.get("scan", ""), []).append(path)

    passed, deferred = [], []
    all_rows = []
    for item in pilot:
        paths = by_scan.get(item["raw_name"], [])
        path = (best_available_metrics(paths) if args.best_available else
                exact_run_metrics(paths, args.run_name) if args.run_name else
                preferred_metrics(paths))
        if path is None:
            ok, reasons, values = False, ["production mesh/metrics not built"], {}
            run = metrics_path = ""
        else:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            ok, reasons, values = verdict(metrics)
            run, metrics_path = path.parent.name, str(path)
        row = {
            "scan_uid": item["scan_uid"],
            "scan": item["raw_name"],
            "raw_path": item["raw_path"],
            "label": item["label"],
            "source": item.get("source", "new"),
            "qa_status": "pass" if ok else "deferred",
            "qa_reasons": "; ".join(reasons),
            "production_run": run,
            "heldout_median_mm": values.get("heldout_median_mm"),
            "heldout_p90_mm": values.get("heldout_p90_mm"),
            "heldout_within_20mm": values.get("heldout_within_20mm"),
            "heldout_hit_fraction": values.get("heldout_hit_fraction"),
            "largest_component_fraction": values.get("largest_component_fraction"),
            "metrics_path": metrics_path,
        }
        all_rows.append(row)
        (passed if ok else deferred).append(row)

    suffix = "_all" if args.all_geometry else ""
    write_csv(args.output_dir / f"qa_good{suffix}.csv", passed)
    write_csv(args.output_dir / f"qa_deferred{suffix}.csv", deferred)
    report = {
        "policy": LIMITS,
        "scope": "all usable canonical geometry scans" if args.all_geometry else "first primary_pilot scans",
        "counts": {"total": len(all_rows), "pass": len(passed), "deferred": len(deferred)},
        "pass": passed,
        "deferred": deferred,
    }
    (args.output_dir / f"geometry_qa_split{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Geometry QA split", "",
        f"- Total: {len(all_rows)}", f"- QA pass: {len(passed)}",
        f"- Deferred: {len(deferred)}", "",
        "## Deferred", "",
    ]
    lines.extend(f"- `{row['scan']}`: {row['qa_reasons']}" for row in deferred)
    (args.output_dir / f"geometry_qa_split{suffix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    for row in deferred:
        print(f"DEFERRED {row['scan']}: {row['qa_reasons']}")


if __name__ == "__main__":
    main()
