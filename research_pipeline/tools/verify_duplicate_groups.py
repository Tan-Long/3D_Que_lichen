#!/usr/bin/env python3
"""Quick-check duplicate candidates by file names, counts, and sizes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


IGNORED_FILES = {"sample_metadata.json"}


def payload_inventory(root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for directory, _, filenames in os.walk(root):
        base = Path(directory)
        for filename in sorted(filenames):
            if filename in IGNORED_FILES:
                continue
            path = base / filename
            relative = path.relative_to(root).as_posix()
            result[relative] = path.stat().st_size
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/scans.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("manifests/duplicate_verification.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["duplicate_group"]:
            groups[row["duplicate_group"]].append(row)

    report = []
    for group_id in sorted(groups):
        group = sorted(groups[group_id], key=lambda row: row["duplicate_canonical"] != "True")
        canonical = next(row for row in group if row["duplicate_canonical"] == "True")
        canonical_inventory = payload_inventory(Path(canonical["raw_path"]))
        comparisons = []
        for candidate in group:
            if candidate is canonical:
                continue
            candidate_inventory = payload_inventory(Path(candidate["raw_path"]))
            comparisons.append(
                {
                    "candidate": candidate["raw_path"],
                    "same_file_layout_and_sizes": candidate_inventory == canonical_inventory,
                    "canonical_file_count": len(canonical_inventory),
                    "candidate_file_count": len(candidate_inventory),
                    "only_in_canonical": sorted(canonical_inventory.keys() - candidate_inventory.keys()),
                    "only_in_candidate": sorted(candidate_inventory.keys() - canonical_inventory.keys()),
                    "different_files": sorted(
                        name
                        for name in canonical_inventory.keys() & candidate_inventory.keys()
                        if canonical_inventory[name] != candidate_inventory[name]
                    ),
                }
            )
        report.append(
            {
                "group": group_id,
                "canonical": canonical["raw_path"],
                "comparisons": comparisons,
            }
        )
        print(group_id, all(item["same_file_layout_and_sizes"] for item in comparisons))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
