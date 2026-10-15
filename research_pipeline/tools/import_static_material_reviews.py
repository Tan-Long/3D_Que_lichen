#!/usr/bin/env python3
"""Validate and atomically merge an exported static-review JSON file."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from serve_deferred_gallery import MATERIAL_REVIEW_PATH, record_material_review


def import_reviews(source: Path, destination: Path = MATERIAL_REVIEW_PATH) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("reviews"), dict):
        raise ValueError("review export must use schema_version 1")
    count = 0
    with tempfile.TemporaryDirectory(prefix="que-lichen-review-import-") as directory:
        staged = Path(directory) / "material_reviews.json"
        if destination.is_file():
            shutil.copyfile(destination, staged)
        for scan_uid, bands in payload["reviews"].items():
            if not isinstance(bands, dict):
                raise ValueError(f"{scan_uid}: bands must be an object")
            for band, review in bands.items():
                if not isinstance(review, dict) or review.get("scan_uid") != scan_uid:
                    raise ValueError(f"{scan_uid}/{band}: invalid review")
                saved = record_material_review(
                    staged, scan_uid, band, review.get("action"),
                    review.get("decision"), review.get("updated_at"))
                for key in ("original_class", "original_name", "new_name"):
                    if review.get(key) != saved[key]:
                        raise ValueError(f"{scan_uid}/{band}: {key} does not match")
                count += 1
        data = json.loads(staged.read_text(encoding="utf-8"))
        data.setdefault("imports", []).append({
            "reviewer": payload.get("reviewer", ""),
            "exported_at": payload.get("exported_at"),
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source": source.name,
            "records": count,
        })
        staged.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(staged, temporary)
        temporary.replace(destination)
    return {"records": count, "trees": len(payload["reviews"]),
            "reviewer": payload.get("reviewer", "")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_json", type=Path)
    parser.add_argument("--destination", type=Path, default=MATERIAL_REVIEW_PATH)
    args = parser.parse_args()
    print(json.dumps(import_reviews(args.review_json, args.destination),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
