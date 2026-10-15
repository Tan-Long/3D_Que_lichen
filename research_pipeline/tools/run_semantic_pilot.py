#!/usr/bin/env python3
"""Run the multi-view HSV semantic baseline on representative QA-good scans."""

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
    parser.add_argument("--qa-good", type=Path, default=root / "manifests/qa_good.csv")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/semantic_pilot")
    parser.add_argument("--all", action="store_true", help="Run all QA-good scans instead of three representatives")
    parser.add_argument("--frame-step", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with args.qa_good.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.all:
        selected = rows
    else:
        selected = []
        for desired in ("lichen_present", "lichen_absent", "unknown"):
            match = next((row for row in rows if row["label"] == desired), None)
            if match is not None:
                selected.append(match)
    results = []
    for index, row in enumerate(selected, 1):
        metrics_path = Path(row["metrics_path"])
        mesh_dir = metrics_path.parent
        side = mesh_dir / "surface_area_by_side.json"
        mesh = mesh_dir / "trunk_mesh_tsdf.ply"
        output = args.output_root / row["scan_uid"]
        print(f"[{index}/{len(selected)}] {row['scan']} ({row['label']})", flush=True)
        if (output / "lichen_hsv_metrics.json").exists() and not args.force:
            results.append({"scan_uid": row["scan_uid"], "scan": row["scan"], "label": row["label"],
                            "status": "skipped_existing", "output_dir": str(output)})
            continue
        command = [sys.executable, str(root / "tools/detect_lichen_multiview.py"),
                   "--scan", row["raw_path"], "--mesh", str(mesh),
                   "--side-metrics", str(side), "--output-dir", str(output),
                   "--frame-step", str(args.frame_step)]
        completed = subprocess.run(command, check=False)
        results.append({"scan_uid": row["scan_uid"], "scan": row["scan"], "label": row["label"],
                        "status": "ok" if completed.returncode == 0 else f"failed:{completed.returncode}",
                        "output_dir": str(output)})
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pilot_run_log.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(item["status"].startswith("failed") for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
