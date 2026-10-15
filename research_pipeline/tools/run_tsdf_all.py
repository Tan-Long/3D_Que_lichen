#!/usr/bin/env python3
"""Resumable production TSDF build for every canonical geometry scan.

Both new and legacy captures are handled identically from their raw RGB-D,
pose and calibration files.  Previous legacy meshes are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def save_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def compact_terrain(source: Path, destination: Path) -> Path:
    """Keep only production fields so each per-scan subprocess loads a small file."""
    records = json.loads(source.read_text(encoding="utf-8"))
    compact = []
    fit_fields = ("valid", "base", "normal", "down", "quality_ok", "source_variant")
    for record in records:
        fit = record.get("variants", {}).get("adaptive", {})
        compact.append({
            "name": record.get("name"), "source": record.get("source"),
            "center": record.get("center"), "gravity": record.get("gravity"),
            "variants": {"adaptive": {key: fit.get(key) for key in fit_fields}},
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "manifests/scans.csv")
    parser.add_argument("--terrain-results", type=Path,
                        default=root / "artifacts/terrain/terrain_all.json")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/meshes_all")
    parser.add_argument("--frame-step", type=int, default=12)
    parser.add_argument("--roi-search-radius", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source", choices=("all", "new", "legacy"), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["usable_geometry"] == "True" and row["duplicate_canonical"] == "True"
                and (args.source == "all" or row["source"] == args.source)]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not args.terrain_results.exists():
        raise SystemExit(f"Terrain catalog is missing: {args.terrain_results}")
    terrain_runtime = compact_terrain(
        args.terrain_results, args.terrain_results.with_name("terrain_all_runtime.json"))

    python = sys.executable
    build_script = root / "tools/build_tsdf_mesh.py"
    measure_script = root / "tools/measure_mesh_surface.py"
    run_name = f"tensor_v5mm_step{args.frame_step}_adaptive_r{round(args.roi_search_radius * 100)}"
    log_path = args.output_root / "all_run_log.json"
    records: list[dict] = []

    for index, row in enumerate(rows, 1):
        scan = Path(row["raw_path"])
        output = args.output_root / row["scan_uid"] / run_name
        mesh_path = output / "trunk_mesh_tsdf.ply"
        metrics_path = output / "metrics.json"
        area_path = output / "surface_area_by_side.json"
        started = time.time()
        print(f"[{index}/{len(rows)}] {row['source']} / {row['raw_name']}", flush=True)

        if mesh_path.exists() and metrics_path.exists() and not args.force:
            status = "skipped_existing"
            build_code = 0
        else:
            command = [
                python, str(build_script), "--scan-path", str(scan),
                "--output-dir", str(output), "--terrain-results", str(terrain_runtime),
                "--backend", "tensor", "--voxel", "0.005", "--trunc-mult", "4",
                "--frame-step", str(args.frame_step), "--confidence", "1",
                "--eval-confidence", "2", "--roi-radius", "0.22", "--adaptive-roi",
                "--roi-search-radius", str(args.roi_search_radius),
                "--min-height", "0.15", "--max-height", "2.20",
            ]
            completed = subprocess.run(command, check=False)
            build_code = completed.returncode
            status = "ok" if build_code == 0 else f"build_failed:{build_code}"

        measure_code = None
        if build_code == 0 and mesh_path.exists() and (args.force or not area_path.exists()):
            measured = subprocess.run([
                python, str(measure_script), "--mesh", str(mesh_path),
                "--scan-name", row["raw_name"], "--output-dir", str(output),
                "--terrain-results", str(terrain_runtime),
            ], check=False)
            measure_code = measured.returncode
            if measure_code != 0 and status in ("ok", "skipped_existing"):
                status = f"mesh_ok_measure_deferred:{measure_code}"

        records.append({
            "scan_uid": row["scan_uid"], "scan": row["raw_name"],
            "source": row["source"], "raw_path": str(scan), "run": run_name,
            "status": status, "build_returncode": build_code,
            "measure_returncode": measure_code, "elapsed_seconds": round(time.time() - started, 2),
            "output_dir": str(output),
        })
        save_log(log_path, records)

    subprocess.run([
        python, str(root / "tools/classify_geometry_qa.py"),
        "--manifest", str(args.manifest), "--mesh-root", str(args.output_root),
        "--run-name", run_name, "--output-dir", str(root / "manifests"), "--all-geometry",
    ], check=False)
    failures = [record for record in records if record["status"].startswith("build_failed")]
    print(f"Finished {len(records)} scans; build failures: {len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
