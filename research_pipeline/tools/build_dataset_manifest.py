#!/usr/bin/env python3
"""Build a non-destructive, normalized index for the cinnamon-tree scans.

The script never renames, moves, or edits raw scan files. It writes small
manifests that provide stable logical identifiers and leakage groups across
the new and legacy datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FILES = ("rgb.mp4", "camera_matrix.csv", "odometry.csv")
REQUIRED_DIRS = ("depth", "confidence")
MANIFEST_FIELDS = (
    "scan_uid",
    "source",
    "raw_name",
    "raw_path",
    "tree_key",
    "site",
    "recorded_at",
    "label",
    "is_important",
    "depth_frames",
    "confidence_frames",
    "rgb_bytes",
    "rgb_container_complete",
    "has_intrinsics",
    "has_odometry",
    "has_imu",
    "has_location",
    "has_point_cloud",
    "usable_geometry",
    "usable_semantic",
    "missing_required",
    "content_fingerprint",
    "duplicate_group",
    "duplicate_canonical",
    "same_tree_count",
    "pilot_role",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Co")
    return re.sub(r"\s+", " ", text).strip(" _-*!")


def slugify(value: Any) -> str:
    text = clean_text(value).lower().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def count_frames(path: Path) -> int:
    if not path.is_dir():
        return 0
    try:
        return sum(
            1
            for entry in os.scandir(path)
            if entry.is_file() and entry.name.lower().endswith(".png")
        )
    except OSError:
        return 0


def mp4_has_moov(path: Path) -> bool:
    """Return whether an MP4 has a complete top-level moov atom."""
    if not path.is_file():
        return False
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            offset = 0
            while offset + 8 <= file_size:
                handle.seek(offset)
                header = handle.read(8)
                if len(header) != 8:
                    return False
                atom_size = int.from_bytes(header[:4], "big")
                atom_type = header[4:8]
                header_size = 8
                if atom_size == 1:
                    extended = handle.read(8)
                    if len(extended) != 8:
                        return False
                    atom_size = int.from_bytes(extended, "big")
                    header_size = 16
                elif atom_size == 0:
                    atom_size = file_size - offset
                if atom_type == b"moov":
                    return True
                if atom_size < header_size:
                    return False
                offset += atom_size
    except OSError:
        return False
    return False


def digest_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8", errors="surrogatepass"))
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    label = slugify(value)
    if label in {"dia-y", "lichen"}:
        return "lichen_present"
    if label in {"khong-dia-y", "no-lichen", "lichen-absent"}:
        return "lichen_absent"
    return "unknown"


def normalize_site(value: Any) -> str:
    text = clean_text(value)
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", text)
    if match:
        return f"gps-{float(match.group(1)):.3f}-{float(match.group(2)):.3f}"
    return slugify(text) if text else "unknown"


def infer_tree_name(raw_name: str, metadata: dict[str, Any], source: str) -> str:
    if metadata.get("sample_id"):
        return clean_text(metadata["sample_id"])
    if source == "legacy":
        return raw_name

    cleaned = clean_text(raw_name)
    cleaned = re.split(r"__(?:StrayScanner|Thucdia)", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"_+video_\d+$", "", cleaned, flags=re.IGNORECASE)
    match = re.match(r"^cay_\d+_\d+_(.+)$", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = match.group(1)
    return cleaned.rstrip("_") or raw_name


def make_scan_uid(source: str, raw_name: str) -> str:
    suffix = hashlib.sha1(raw_name.encode("utf-8", errors="surrogatepass")).hexdigest()[:8]
    return f"{source}-{slugify(raw_name)}-{suffix}"


def scan_record(root: Path, scan_dir: Path, source: str) -> dict[str, Any]:
    metadata = read_json(scan_dir / "sample_metadata.json")
    raw_name = scan_dir.name
    site = normalize_site(metadata.get("sample_site"))
    tree_name = infer_tree_name(raw_name, metadata, source)
    tree_key = f"{source}:{site}:{slugify(tree_name)}"

    depth_frames = count_frames(scan_dir / "depth")
    confidence_frames = count_frames(scan_dir / "confidence")
    missing = [name for name in REQUIRED_FILES if not (scan_dir / name).is_file()]
    missing += [name for name in REQUIRED_DIRS if not (scan_dir / name).is_dir()]
    rgb_path = scan_dir / "rgb.mp4"
    rgb_bytes = rgb_path.stat().st_size if rgb_path.is_file() else 0
    rgb_container_complete = mp4_has_moov(rgb_path)

    identity_digest = digest_files((scan_dir / "camera_matrix.csv", scan_dir / "odometry.csv"))
    fingerprint_payload = {
        "identity_digest": identity_digest,
        "rgb_bytes": rgb_bytes,
        "rgb_container_complete": rgb_container_complete,
        "depth_frames": depth_frames,
        "confidence_frames": confidence_frames,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "scan_uid": make_scan_uid(source, raw_name),
        "source": source,
        "raw_name": raw_name,
        "raw_path": str(scan_dir.resolve()),
        "tree_key": tree_key,
        "site": site,
        "recorded_at": clean_text(metadata.get("recorded_at")),
        "label": normalize_label(metadata.get("sample_loai_mau")),
        "is_important": bool(metadata.get("is_important", False)),
        "depth_frames": depth_frames,
        "confidence_frames": confidence_frames,
        "rgb_bytes": rgb_bytes,
        "rgb_container_complete": rgb_container_complete,
        "has_intrinsics": (scan_dir / "camera_matrix.csv").is_file(),
        "has_odometry": (scan_dir / "odometry.csv").is_file(),
        "has_imu": (scan_dir / "imu.csv").is_file(),
        "has_location": (scan_dir / "location.csv").is_file(),
        "has_point_cloud": (scan_dir / "point_cloud_raw.ply").is_file(),
        "usable_geometry": not missing and depth_frames > 0 and confidence_frames > 0,
        "usable_semantic": (
            not missing
            and depth_frames > 0
            and confidence_frames > 0
            and rgb_container_complete
        ),
        "missing_required": ";".join(missing),
        "content_fingerprint": fingerprint,
        "duplicate_group": "",
        "duplicate_canonical": True,
        "same_tree_count": 1,
        "pilot_role": "not_selected",
    }


def inventory_source(root: Path, source: str) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return [
        scan_record(root, path, source)
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold())
        if path.is_dir()
    ]


def mark_groups(records: list[dict[str, Any]]) -> None:
    fingerprint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tree_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        fingerprint_groups[record["content_fingerprint"]].append(record)
        tree_groups[record["tree_key"]].append(record)

    duplicate_number = 0
    for group in fingerprint_groups.values():
        if len(group) < 2:
            continue
        duplicate_number += 1
        group.sort(key=lambda record: record["scan_uid"])
        group_id = f"dup-{duplicate_number:03d}"
        for index, record in enumerate(group):
            record["duplicate_group"] = group_id
            record["duplicate_canonical"] = index == 0

    for group in tree_groups.values():
        for record in group:
            record["same_tree_count"] = len(group)


def round_robin_sites(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_site[record["site"]].append(record)
    for site_records in by_site.values():
        site_records.sort(
            key=lambda record: (
                not record["is_important"],
                record["same_tree_count"] > 1,
                record["scan_uid"],
            )
        )

    selected: list[dict[str, Any]] = []
    sites = sorted(by_site)
    while sites and len(selected) < count:
        next_sites: list[str] = []
        for site in sites:
            if by_site[site] and len(selected) < count:
                selected.append(by_site[site].pop(0))
            if by_site[site]:
                next_sites.append(site)
        sites = next_sites
    return selected


def assign_pilot_roles(records: list[dict[str, Any]]) -> None:
    eligible_new = [
        record
        for record in records
        if record["source"] == "new"
        and record["usable_semantic"]
        and record["has_point_cloud"]
        and record["has_location"]
        and record["duplicate_canonical"]
    ]

    targets = {"lichen_present": 15, "lichen_absent": 10, "unknown": 5}
    for label, target in targets.items():
        candidates = [record for record in eligible_new if record["label"] == label]
        for record in round_robin_sites(candidates, target):
            record["pilot_role"] = "primary_pilot"

    legacy = sorted(
        (
            record
            for record in records
            if record["source"] == "legacy" and record["usable_geometry"]
        ),
        key=lambda record: record["scan_uid"],
    )
    if legacy:
        indices = sorted({round(index * (len(legacy) - 1) / 7) for index in range(8)})
        for index in indices:
            legacy[index]["pilot_role"] = "legacy_hsv_regression"


def write_csv(path: Path, records: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_summary(records: list[dict[str, Any]], archive: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "raw_data_policy": "immutable; manifests contain logical normalization only",
        "archive": {
            "path": str(archive.resolve()),
            "exists": archive.is_file(),
            "bytes": archive.stat().st_size if archive.is_file() else 0,
            "action": "retain_as_archive; do_not_extract_while_legacy_raw_root_exists",
        },
        "counts": {
            "total": len(records),
            "by_source": dict(Counter(record["source"] for record in records)),
            "by_label": dict(Counter(record["label"] for record in records)),
            "usable_geometry": sum(bool(record["usable_geometry"]) for record in records),
            "usable_semantic": sum(bool(record["usable_semantic"]) for record in records),
            "incomplete_rgb_containers": sum(
                not bool(record["rgb_container_complete"]) for record in records
            ),
            "with_point_cloud": sum(bool(record["has_point_cloud"]) for record in records),
            "strong_duplicate_candidates": sum(
                bool(record["duplicate_group"]) for record in records
            ),
            "duplicate_groups": len(
                {record["duplicate_group"] for record in records if record["duplicate_group"]}
            ),
            "primary_pilot": sum(record["pilot_role"] == "primary_pilot" for record in records),
            "legacy_hsv_regression": sum(
                record["pilot_role"] == "legacy_hsv_regression" for record in records
            ),
        },
        "pilot_by_label": dict(
            Counter(
                record["label"]
                for record in records
                if record["pilot_role"] == "primary_pilot"
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, default=ROOT / "data raw")
    parser.add_argument("--legacy-root", type=Path, default=ROOT.parent / "3D_Que/Raw_data")
    parser.add_argument("--archive", type=Path, default=ROOT / "Raw_data.rar")
    parser.add_argument("--output", type=Path, default=Path("manifests"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = inventory_source(args.new_root, "new")
    records.extend(inventory_source(args.legacy_root, "legacy"))
    mark_groups(records)
    assign_pilot_roles(records)
    records.sort(key=lambda record: record["scan_uid"])

    output = args.output.resolve()
    write_csv(output / "scans.csv", records, MANIFEST_FIELDS)
    write_csv(
        output / "duplicate_candidates.csv",
        [
            record
            for record in records
            if record["duplicate_group"] or record["same_tree_count"] > 1
        ],
        MANIFEST_FIELDS,
    )
    write_csv(
        output / "pilot.csv",
        [record for record in records if record["pilot_role"] != "not_selected"],
        MANIFEST_FIELDS,
    )
    write_csv(
        output / "incomplete_captures.csv",
        [
            record
            for record in records
            if not record["usable_geometry"] or not record["usable_semantic"]
        ],
        MANIFEST_FIELDS,
    )
    summary = build_summary(records, args.archive)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
