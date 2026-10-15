#!/usr/bin/env python3
"""Run a fixed DINO surface-cluster pilot sequentially with per-scan checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_selected_rows(selection_path: Path, production_path: Path) -> list[dict]:
    with selection_path.open(encoding="utf-8-sig", newline="") as handle:
        selection = list(csv.DictReader(handle))
    with production_path.open(encoding="utf-8-sig", newline="") as handle:
        production = {row["scan_uid"]: row for row in csv.DictReader(handle)}
    rows = []
    for selected in selection:
        uid = selected["scan_uid"]
        if uid not in production:
            raise ValueError(f"selected scan is not in production manifest: {uid}")
        row = dict(production[uid])
        row["label"] = selected["label"]
        rows.append(row)
    return rows


def load_all_production_rows(production_path: Path) -> list[dict]:
    with production_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = "unknown"
    return rows


def write_log(path: Path, results: list[dict]) -> None:
    value = {"updated_at": datetime.now(timezone.utc).isoformat(), "results": results}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=root / "artifacts/surface_cluster_pilot/dinov2_expansion_selection.csv")
    parser.add_argument("--all-production", action="store_true",
                        help="Process every row in the approved production manifest")
    parser.add_argument("--production", type=Path,
                        default=root / "manifests/production_baseline_v1_approved.csv")
    parser.add_argument("--output-root", type=Path,
                        default=root / "artifacts/surface_cluster_pilot")
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--model-weights", type=Path)
    parser.add_argument("--model", choices=("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
                                            "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"),
                        default="dinov2_vits14")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--max-scans", type=int, help="Only process the first N selected scans")
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=48)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = (load_all_production_rows(args.production) if args.all_production
            else load_selected_rows(args.selection, args.production))
    if args.max_scans is not None:
        rows = rows[:args.max_scans]
    results = []
    log_path = args.log or args.output_root / "dinov2_expansion_run_log.json"
    for index, row in enumerate(rows, 1):
        output = args.output_root / row["scan_uid"] / f"{args.model}_k{args.clusters}"
        candidates = [output]
        if args.clusters == 3:
            candidates.append(args.output_root / row["scan_uid"] /
                              f"{args.model}_k3_cached")
        existing = next((path for path in candidates
                         if (path / "surface_cluster_metrics.json").is_file()), None)
        print(f"[{index}/{len(rows)}] {row['scan']} ({row['label']})", flush=True)
        if existing is not None and not args.force:
            status = "skipped_existing"
            output = existing
        else:
            command = [
                sys.executable, str(root / "tools/cluster_lichen_dinov2.py"),
                "--scan", row["raw_path"], "--mesh", row["mesh_path"],
                "--output-dir", str(output), "--model-repo", str(args.model_repo),
                "--model", args.model,
                "--frame-step", str(args.frame_step), "--min-views", "2",
                "--clusters", str(args.clusters),
            ]
            if args.model_weights is not None:
                command.extend(("--model-weights", str(args.model_weights)))
            completed = subprocess.run(command, check=False)
            status = "ok" if completed.returncode == 0 else f"failed:{completed.returncode}"
        results.append({
            "scan_uid": row["scan_uid"], "scan": row["scan"], "label": row["label"],
            "status": status, "output_dir": str(output),
        })
        write_log(log_path, results)
        if status.startswith("failed") and not args.continue_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
