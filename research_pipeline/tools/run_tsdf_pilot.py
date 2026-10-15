#!/usr/bin/env python3
"""Resumable batch runner for the TSDF pilot and strict-confidence ablation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=root / "manifests/pilot.csv")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/meshes")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--frame-step", type=int, default=12)
    parser.add_argument("--strict", action="store_true", help="Also run confidence=2 ablation")
    parser.add_argument("--adaptive-roi", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a wide search ROI followed by fitted trunk-centreline cropping")
    parser.add_argument("--roi-search-radius", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--remaining-only", action="store_true",
                        help="Skip a scan when any complete production mesh + side metrics already exist")
    args = parser.parse_args()
    with args.pilot.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["pilot_role"] == "primary_pilot"
                and r["usable_geometry"] == "True" and r["has_point_cloud"] == "True"]
    selected = rows[:args.count]
    completed_scans = set()
    if args.remaining_only:
        for metrics_path in args.output_root.glob("**/metrics.json"):
            run_name = metrics_path.parent.name
            if ("_probe" in run_name or "conf2" in run_name or run_name.startswith("legacy_")
                    or run_name.startswith("tensor_v3mm_") or run_name.startswith("tensor_v8mm_")
                    or run_name.startswith("v6mm_")):
                continue
            if not (metrics_path.parent / "trunk_mesh_tsdf.ply").exists():
                continue
            if not (metrics_path.parent / "surface_area_by_side.json").exists():
                continue
            try:
                completed_scans.add(json.loads(metrics_path.read_text(encoding="utf-8"))["scan"])
            except (KeyError, json.JSONDecodeError):
                continue
    python = sys.executable
    build_script = root / "tools/build_tsdf_mesh.py"
    measure_script = root / "tools/measure_mesh_surface.py"
    results = []
    for index, row in enumerate(selected, 1):
        scan = Path(row["raw_path"])
        if args.remaining_only and scan.name in completed_scans:
            print(f"[{index}/{len(selected)}] {scan.name} / skipped_complete_scan", flush=True)
            results.append({"scan": scan.name, "run": "production", "status": "skipped_existing"})
            continue
        normal_name = (f"tensor_v5mm_step12_adaptive_r{int(args.roi_search_radius * 100)}"
                       if args.adaptive_roi else "tensor_v5mm_step12_r22")
        strict_name = (f"tensor_v5mm_step12_adaptive_r{int(args.roi_search_radius * 100)}_conf2"
                       if args.adaptive_roi else "tensor_v5mm_step12_conf2")
        modes = [(normal_name, 1)] + ([(strict_name, 2)] if args.strict else [])
        for run_name, confidence in modes:
            # Open3D on Windows cannot reliably open/write paths containing
            # private-use Unicode characters present in several raw names.
            legacy_output = args.output_root / scan.name / run_name
            uid_output = args.output_root / row["scan_uid"] / run_name
            output = (legacy_output if (legacy_output / "trunk_mesh_tsdf.ply").exists()
                      else uid_output)
            metrics = output / "metrics.json"
            print(f"[{index}/{len(selected)}] {scan.name} / {run_name}", flush=True)
            if metrics.exists() and not args.force:
                results.append({"scan": scan.name, "run": run_name, "status": "skipped_existing"})
                continue
            command = [
                python, str(build_script), "--scan-path", str(scan), "--output-dir", str(output),
                "--backend", "tensor", "--voxel", "0.005", "--trunc-mult", "4",
                "--frame-step", str(args.frame_step), "--confidence", str(confidence),
                "--eval-confidence", "2", "--roi-radius", "0.22",
                "--min-height", "0.15", "--max-height", "2.20",
            ]
            if args.adaptive_roi:
                command.extend(["--adaptive-roi", "--roi-search-radius", str(args.roi_search_radius)])
            completed = subprocess.run(command, check=False)
            status = "ok" if completed.returncode == 0 else f"failed:{completed.returncode}"
            results.append({"scan": scan.name, "run": run_name, "status": status})
            if completed.returncode == 0 and confidence == 1:
                subprocess.run([
                    python, str(measure_script), "--mesh", str(output / "trunk_mesh_tsdf.ply"),
                    "--scan-name", scan.name, "--output-dir", str(output),
                ], check=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pilot_run_log.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subprocess.run([python, str(root / "tools/summarize_mesh_benchmark.py"),
                    "--root", str(args.output_root)], check=False)
    qa_split = subprocess.run([
        python, str(root / "tools/classify_geometry_qa.py"),
        "--pilot", str(args.pilot), "--mesh-root", str(args.output_root),
        "--count", str(args.count), "--output-dir", str(root / "manifests"),
    ], check=False)
    if qa_split.returncode != 0:
        results.append({"scan": "__qa_split__", "run": "classification",
                        "status": f"failed:{qa_split.returncode}"})
    failed = [r for r in results if r["status"].startswith("failed")]
    print(f"Done: {len(results) - len(failed)}/{len(results)} runs without process failure")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
