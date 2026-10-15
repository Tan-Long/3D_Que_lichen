#!/usr/bin/env python3
"""Summarize semantic pilot outputs and gate the HSV baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root / "artifacts/semantic_pilot")
    args = parser.parse_args()
    log = json.loads((args.root / "pilot_run_log.json").read_text(encoding="utf-8"))
    rows = []
    for item in log:
        path = Path(item["output_dir"]) / "lichen_hsv_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        area = metrics["area"]
        rows.append({
            "scan": item["scan"], "tree_label": item["label"],
            "observed_min_views_fraction": metrics["triangle_observation"]["observed_min_views_fraction"],
            "median_views": metrics["triangle_observation"]["median_views"],
            "uncertain_pct": (100 * area["overall"]["class_area_cm2"]["uncertain"]
                              / area["overall"]["observed_surface_cm2"]),
            "lichen_candidate_pct": area["overall"]["lichen_coverage_observed_pct"],
            "upslope_lichen_candidate_pct": area["upslope"]["lichen_coverage_observed_pct"],
            "downslope_lichen_candidate_pct": area["downslope"]["lichen_coverage_observed_pct"],
            "green_biota_candidate_pct": (100 * area["overall"]["class_area_cm2"]["green_biota_candidate"]
                                           / area["overall"]["observed_surface_cm2"]),
            "metrics_path": str(path),
        })
    positives = [row["lichen_candidate_pct"] for row in rows if row["tree_label"] == "lichen_present"]
    negatives = [row["lichen_candidate_pct"] for row in rows if row["tree_label"] == "lichen_absent"]
    negative_mean = sum(negatives) / len(negatives) if negatives else None
    positive_mean = sum(positives) / len(positives) if positives else None
    accepted = bool(positives and negatives and negative_mean <= 2.0 and positive_mean >= negative_mean + 5.0)
    gate = {
        "status": "accepted_for_measurement" if accepted else "rejected_for_measurement",
        "rule": "negative mean <=2% and positive-negative separation >=5 percentage points",
        "positive_mean_pct": positive_mean,
        "negative_mean_pct": negative_mean,
        "separation_percentage_points": (positive_mean - negative_mean
                                         if positive_mean is not None and negative_mean is not None else None),
        "reason": None if accepted else "HSV confuses pale/bright bark with lichen; supervised reference labels required",
    }
    report = {"gate": gate, "rows": rows}
    (args.root / "semantic_pilot_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.root / "semantic_pilot_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Semantic pilot — multi-view HSV baseline", "", f"**Gate: `{gate['status']}`**", "",
             f"- Positive mean: {positive_mean:.2f}%" if positive_mean is not None else "- Positive mean: n/a",
             f"- Negative mean: {negative_mean:.2f}%" if negative_mean is not None else "- Negative mean: n/a",
             f"- Separation: {gate['separation_percentage_points']:.2f} percentage points"
             if gate["separation_percentage_points"] is not None else "- Separation: n/a", "",
             "| Scan | Tree label | ≥3 views | Uncertain | Pale lichen candidate | Up | Down | Green biota |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['scan']} | {row['tree_label']} | {100*row['observed_min_views_fraction']:.1f}% | "
                     f"{row['uncertain_pct']:.1f}% | {row['lichen_candidate_pct']:.1f}% | "
                     f"{row['upslope_lichen_candidate_pct']:.1f}% | {row['downslope_lichen_candidate_pct']:.1f}% | "
                     f"{row['green_biota_candidate_pct']:.1f}% |")
    lines.extend(["", "HSV is retained only as a regression baseline. These candidate percentages are not biological measurements."])
    (args.root / "semantic_pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
