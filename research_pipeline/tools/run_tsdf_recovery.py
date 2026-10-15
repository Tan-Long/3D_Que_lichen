#!/usr/bin/env python3
"""Build denser TSDF scans with resumable workers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

RAW_INSUFFICIENT = {
    "new-cay-0029-2205-bk6-906a5e85": "only 6 useful frames; reconstructed height 0.13 m",
    "new-cay-0032-2205-bk8-fccecbbc": "only 21 useful frames; reconstructed height 0.06 m",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_deferred_and_low_support(
    manifest_rows: list[dict], deferred_uids: set[str], observed_root: Path,
    threshold: float,
) -> tuple[list[dict], dict]:
    canonical = [row for row in manifest_rows
                 if row["usable_geometry"] == "True" and row["duplicate_canonical"] == "True"]
    low_support = set()
    missing_reports = 0
    for row in canonical:
        report = (observed_root / row["scan_uid"] /
                  "v5mm_step12_pixel2_conf1_r22" / "report.json")
        if not report.is_file():
            missing_reports += 1
            continue
        metrics = json.loads(report.read_text(encoding="utf-8"))
        total = metrics.get("observed_voxels", 0)
        if total and metrics.get("observed_voxels_multi_frame", 0) / total < threshold:
            low_support.add(row["scan_uid"])
    selected_uids = deferred_uids | low_support
    return ([row for row in canonical if row["scan_uid"] in selected_uids], {
        "deferred": len(deferred_uids), "low_support": len(low_support),
        "overlap": len(deferred_uids & low_support), "missing_support_reports": missing_reports,
    })


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deferred", type=Path, default=root / "manifests/qa_deferred_all.csv")
    parser.add_argument("--manifest", type=Path, default=root / "manifests/scans.csv")
    parser.add_argument("--terrain-results", type=Path,
                        default=root / "artifacts/terrain/terrain_all_runtime.json")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/meshes_all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--all-geometry", action="store_true")
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--roi-search-radius", type=float, default=0.40)
    parser.add_argument("--observed-support-root", type=Path,
                        help="Enable weak-support frame boosting from observed_points_all")
    parser.add_argument("--include-low-support-below", type=float,
                        help="Union deferred scans with observed multi-frame support below this fraction")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Do not overwrite production QA manifests after an experimental run")
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["scan_uid"]: row for row in manifest_rows}
    selection_stats = None
    if args.all_geometry:
        selected = [row for row in manifest_rows
                    if row["usable_geometry"] == "True" and row["duplicate_canonical"] == "True"]
    else:
        with args.deferred.open("r", encoding="utf-8-sig", newline="") as handle:
            deferred_uids = {item["scan_uid"] for item in csv.DictReader(handle)}
        if args.include_low_support_below is not None:
            if args.observed_support_root is None:
                parser.error("include-low-support-below requires observed-support-root")
            if not 0 < args.include_low_support_below <= 1:
                parser.error("include-low-support-below must be in (0, 1]")
            selected, selection_stats = select_deferred_and_low_support(
                manifest_rows, deferred_uids, args.observed_support_root,
                args.include_low_support_below)
        else:
            selected = [manifest[uid] for uid in deferred_uids]
    assert len(selected) == len({row["scan_uid"] for row in selected})
    run_name = (f"tensor_v5mm_step{args.frame_step}_conf1_adaptive_"
                f"r{round(args.roi_search_radius * 100)}")
    if args.observed_support_root is not None:
        run_name += "_observed_boost_v1"
    log_path = args.output_root / (f"{run_name}_all_run_log.json" if args.all_geometry else
                                   f"{run_name}_selected_run_log.json")
    lock = threading.Lock()
    records: list[dict] = (json.loads(log_path.read_text(encoding="utf-8"))
                           if log_path.exists() else [])
    completed_uids = {record["scan_uid"] for record in records
                      if record.get("status") in ("ok", "raw_insufficient")}
    completed_uids.update(
        row["scan_uid"] for row in selected
        if all(path.is_file() and path.stat().st_size > 0 for path in (
            args.output_root / row["scan_uid"] / run_name / "metrics.json",
            args.output_root / row["scan_uid"] / run_name / "trunk_mesh_tsdf.ply",
        ))
    )
    rows = [row for row in selected if row["scan_uid"] not in completed_uids]
    print(json.dumps({"selected": len(selected), "remaining": len(rows),
                      "selection": selection_stats}, ensure_ascii=False), flush=True)

    def recover(row: dict) -> dict:
        uid, scan = row["scan_uid"], Path(row["raw_path"])
        started = time.time()
        if uid in RAW_INSUFFICIENT:
            return {"scan_uid": uid, "scan": row["raw_name"], "source": row["source"],
                    "status": "raw_insufficient", "reason": RAW_INSUFFICIENT[uid],
                    "elapsed_seconds": 0.0}
        observed_support = None
        if args.observed_support_root is not None:
            observed_support = (args.observed_support_root / uid /
                                "v5mm_step12_pixel2_conf1_r22" / "observed_points.ply")
            if not observed_support.is_file():
                return {"scan_uid": uid, "scan": row["raw_name"], "source": row["source"],
                        "status": "skipped_no_observed_support", "elapsed_seconds": 0.0}
        output = args.output_root / uid / run_name
        metrics = output / "metrics.json"
        mesh = output / "trunk_mesh_tsdf.ply"
        area = output / "surface_area_by_side.json"
        build_code = 0
        if not (metrics.exists() and mesh.exists()):
            command = [sys.executable, str(root / "tools/build_tsdf_mesh.py"),
                       "--scan-path", str(scan), "--output-dir", str(output),
                       "--terrain-results", str(args.terrain_results), "--backend", "tensor",
                       "--voxel", "0.005", "--trunc-mult", "4", "--frame-step", str(args.frame_step),
                       "--confidence", "1", "--eval-confidence", "2", "--roi-radius", "0.22",
                       "--adaptive-roi", "--roi-search-radius", str(args.roi_search_radius),
                       "--min-height", "0.15", "--max-height", "2.20"]
            if row.get("rgb_container_complete") == "False":
                command.append("--geometry-only")
            if observed_support is not None:
                command.extend(["--observed-support", str(observed_support)])
            output.mkdir(parents=True, exist_ok=True)
            with (output / "recovery_build.log").open("w", encoding="utf-8") as log:
                built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            build_code = built.returncode
        status = "ok" if build_code == 0 and metrics.exists() and mesh.exists() else f"build_failed:{build_code}"
        measure_code = None
        if status == "ok" and not area.exists():
            with (output / "recovery_measure.log").open("w", encoding="utf-8") as log:
                measured = subprocess.run([
                    sys.executable, str(root / "tools/measure_mesh_surface.py"),
                    "--mesh", str(mesh), "--scan-name", row["raw_name"],
                    "--output-dir", str(output), "--terrain-results", str(args.terrain_results),
                ], stdout=log, stderr=subprocess.STDOUT, check=False)
            measure_code = measured.returncode
            if measure_code:
                status = f"mesh_ok_measure_deferred:{measure_code}"
        return {"scan_uid": uid, "scan": row["raw_name"], "source": row["source"],
                "status": status, "build_returncode": build_code,
                "measure_returncode": measure_code,
                "elapsed_seconds": round(time.time() - started, 2), "output_dir": str(output)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(recover, row): row for row in rows}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
            except Exception as exc:
                row = futures[future]
                record = {"scan_uid": row["scan_uid"], "scan": row["raw_name"],
                          "source": row["source"], "status": "runner_error", "reason": str(exc)}
            with lock:
                records.append(record)
                atomic_json(log_path, records)
            print(f"[{completed}/{len(rows)}] {record['scan']} / {record['status']}", flush=True)

    if not args.skip_classify:
        classify = [sys.executable, str(root / "tools/classify_geometry_qa.py"),
                    "--manifest", str(args.manifest), "--mesh-root", str(args.output_root),
                    "--output-dir", str(root / "manifests"), "--all-geometry"]
        classify.extend(["--run-name", run_name] if args.all_geometry else ["--best-available"])
        subprocess.run(classify, check=False)


if __name__ == "__main__":
    main()
