#!/usr/bin/env python3
"""Resumable observed-point diagnostic for all canonical geometry scans."""

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


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "manifests/scans.csv")
    parser.add_argument("--baseline-root", type=Path, default=root / "artifacts/meshes_all")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/observed_points_all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frame-step", type=int, default=12)
    parser.add_argument("--pixel-step", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        selected = [row for row in csv.DictReader(handle)
                    if row["usable_geometry"] == "True" and row["duplicate_canonical"] == "True"]
    assert len(selected) == len({row["scan_uid"] for row in selected})
    baseline_run = "tensor_v5mm_step3_conf1_adaptive_r35"
    run_name = f"v5mm_step{args.frame_step}_pixel{args.pixel_step}_conf1_r22"
    log_path = args.output_root / f"{run_name}_run_log.json"
    records: list[dict] = (json.loads(log_path.read_text(encoding="utf-8"))
                           if log_path.exists() else [])
    completed = {record["scan_uid"] for record in records
                 if record.get("status") in ("ok", "skipped_existing", "skipped_no_baseline")}
    rows = [row for row in selected if row["scan_uid"] not in completed]
    lock = threading.Lock()

    def observe(row: dict) -> dict:
        uid = row["scan_uid"]
        baseline = args.baseline_root / uid / baseline_run / "metrics.json"
        output = args.output_root / uid / run_name
        cloud, report = output / "observed_points.ply", output / "report.json"
        started = time.time()
        if not baseline.is_file() or baseline.stat().st_size == 0:
            return {"scan_uid": uid, "scan": row["raw_name"], "status": "skipped_no_baseline",
                    "reason": "production baseline metrics unavailable"}
        if cloud.is_file() and cloud.stat().st_size > 0 and report.is_file() and report.stat().st_size > 0:
            return {"scan_uid": uid, "scan": row["raw_name"], "status": "skipped_existing",
                    "output_dir": str(output)}
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(root / "tools/build_observed_cylinder_pilot.py"),
            "--scan-path", row["raw_path"], "--baseline-metrics", str(baseline),
            "--output-dir", str(output), "--frame-step", str(args.frame_step),
            "--pixel-step", str(args.pixel_step), "--points-only",
        ]
        with (output / "run.log").open("w", encoding="utf-8") as log:
            completed_process = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False)
        status = "ok" if (completed_process.returncode == 0 and cloud.is_file()
                          and cloud.stat().st_size > 0 and report.is_file()
                          and report.stat().st_size > 0) else f"failed:{completed_process.returncode}"
        return {"scan_uid": uid, "scan": row["raw_name"], "status": status,
                "elapsed_seconds": round(time.time() - started, 2), "output_dir": str(output)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(observe, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"scan_uid": row["scan_uid"], "scan": row["raw_name"],
                          "status": "runner_error", "reason": str(exc)}
            with lock:
                records.append(record)
                atomic_json(log_path, records)
            print(f"[{index}/{len(rows)}] {record['scan']} / {record['status']}", flush=True)

    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(json.dumps({"selected": len(selected), "records": len(records),
                      "status_counts": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
