#!/usr/bin/env python3
"""Show resumable TSDF recovery progress from its manifest and run log."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deferred", type=Path, default=root / "manifests/qa_deferred_all.csv")
    parser.add_argument("--log", type=Path,
                        default=root / "artifacts/meshes_all/recovery_run_log.json")
    args = parser.parse_args()

    with args.deferred.open("r", encoding="utf-8-sig", newline="") as handle:
        targets = list(csv.DictReader(handle))
    records = json.loads(args.log.read_text(encoding="utf-8")) if args.log.exists() else []
    latest = {record["scan_uid"]: record for record in records}
    complete = {uid for uid, record in latest.items()
                if record.get("status") in ("ok", "raw_insufficient")}
    target_uids = {row["scan_uid"] for row in targets}
    statuses = Counter(record.get("status", "unknown") for record in latest.values()
                       if record["scan_uid"] in target_uids)
    remaining = [row for row in targets if row["scan_uid"] not in complete]

    print(f"targets={len(targets)} complete={len(target_uids & complete)} remaining={len(remaining)}")
    print("statuses=" + json.dumps(dict(statuses), ensure_ascii=False, sort_keys=True))
    if records:
        last = records[-1]
        print(f"last={last.get('scan', last['scan_uid'])} status={last.get('status', 'unknown')}")
    if remaining:
        print(f"next={remaining[0]['scan']} ({remaining[0]['scan_uid']})")


if __name__ == "__main__":
    main()
