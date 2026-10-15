#!/usr/bin/env python3
"""Audit all legacy Stray Scanner captures without changing raw data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - reported as an audit limitation
    cv2 = None


REQUIRED = ("camera_matrix.csv", "odometry.csv", "imu.csv", "rgb.mp4", "depth", "confidence")
ESSENTIAL_OUTPUTS = (
    "pointcloud.ply",
    "trunk_pointcloud.ply",
    "trunk_mesh_final.ply",
    "trunk_mesh_detected.ply",
    "lichen_stats.json",
)


def png_map(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        return {}
    return {item.stem: item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return [name.strip() for name in (reader.fieldnames or [])], list(reader)


def read_matrix(path: Path) -> list[list[float]] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [[float(value.strip()) for value in row] for row in csv.reader(handle)]
        return rows if len(rows) == 3 and all(len(row) == 3 for row in rows) else None
    except (OSError, ValueError):
        return None


def sampled_image_issues(paths: list[Path], expected_shape: tuple[int, int] | None = None) -> tuple[list[str], tuple[int, int] | None]:
    if cv2 is None:
        return ["opencv_unavailable"], expected_shape
    issues: list[str] = []
    shape = expected_shape
    if not paths:
        return ["no_images"], shape
    sample_indices = sorted({0, len(paths) // 2, len(paths) - 1})
    for index in sample_indices:
        image = cv2.imread(str(paths[index]), cv2.IMREAD_UNCHANGED)
        if image is None:
            issues.append(f"unreadable:{paths[index].name}")
            continue
        current = tuple(int(value) for value in image.shape[:2])
        if shape is None:
            shape = current
        elif current != shape:
            issues.append(f"shape_mismatch:{paths[index].name}:{current}")
    return issues, shape


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def finite_float(value: str) -> float:
    parsed = float(value.strip())
    if not math.isfinite(parsed):
        raise ValueError("non-finite")
    return parsed


def audit_scan(scan: Path) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    missing = [name for name in REQUIRED if not (scan / name).exists()]
    if missing:
        issues.append("missing_required")

    depth = png_map(scan / "depth")
    confidence = png_map(scan / "confidence")
    if set(depth) != set(confidence):
        issues.append("depth_confidence_frame_mismatch")

    odom_fields, odom_rows = read_csv(scan / "odometry.csv")
    odom_frames = [(row.get("frame") or "").strip() for row in odom_rows]
    if set(odom_frames) != set(depth):
        issues.append("odometry_depth_frame_mismatch")
    if len(odom_frames) != len(set(odom_frames)):
        issues.append("duplicate_odometry_frame")

    odom_timestamps: list[float] = []
    quaternion_deviation: list[float] = []
    try:
        for row in odom_rows:
            odom_timestamps.append(finite_float(row["timestamp"]))
            quaternion = [finite_float(row[name]) for name in ("qx", "qy", "qz", "qw")]
            quaternion_deviation.append(abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0))
    except (KeyError, TypeError, ValueError):
        issues.append("invalid_odometry_numeric")
    if any(right <= left for left, right in zip(odom_timestamps, odom_timestamps[1:])):
        issues.append("non_monotonic_odometry_timestamp")
    max_quaternion_deviation = max(quaternion_deviation, default=math.nan)
    if quaternion_deviation and max_quaternion_deviation > 0.02:
        issues.append("invalid_quaternion_norm")

    _, imu_rows = read_csv(scan / "imu.csv")
    imu_timestamps: list[float] = []
    try:
        imu_timestamps = [finite_float(row["timestamp"]) for row in imu_rows]
    except (KeyError, TypeError, ValueError):
        issues.append("invalid_imu_numeric")
    if any(right <= left for left, right in zip(imu_timestamps, imu_timestamps[1:])):
        issues.append("non_monotonic_imu_timestamp")

    matrix = read_matrix(scan / "camera_matrix.csv")
    matrix_valid = False
    fx = fy = cx = cy = math.nan
    if matrix is None:
        issues.append("invalid_camera_matrix")
    else:
        fx, fy, cx, cy = matrix[0][0], matrix[1][1], matrix[0][2], matrix[1][2]
        matrix_valid = (
            all(math.isfinite(value) for row in matrix for value in row)
            and fx > 0
            and fy > 0
            and abs(matrix[2][2] - 1.0) < 1e-6
            and abs(matrix[0][1]) < 1e-6
            and abs(matrix[1][0]) < 1e-6
        )
        if not matrix_valid:
            issues.append("invalid_camera_matrix")

    video_open = False
    video_width = video_height = video_frames = 0
    video_fps = 0.0
    if cv2 is None:
        warnings.append("video_not_checked_opencv_unavailable")
    else:
        capture = cv2.VideoCapture(str(scan / "rgb.mp4"))
        video_open = bool(capture.isOpened())
        if video_open:
            video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if not video_open:
            issues.append("rgb_video_unreadable")
        elif abs(video_frames - len(depth)) > 2:
            issues.append("rgb_depth_frame_count_mismatch")

    if matrix_valid and video_open and not (0 <= cx < video_width and 0 <= cy < video_height):
        issues.append("principal_point_outside_rgb")

    depth_paths = [depth[key] for key in sorted(depth)]
    confidence_paths = [confidence[key] for key in sorted(confidence)]
    depth_image_issues, depth_shape = sampled_image_issues(depth_paths)
    confidence_image_issues, confidence_shape = sampled_image_issues(confidence_paths)
    if depth_image_issues:
        issues.extend(f"depth_{item}" for item in depth_image_issues)
    if confidence_image_issues:
        issues.extend(f"confidence_{item}" for item in confidence_image_issues)
    if depth_shape and confidence_shape and depth_shape != confidence_shape:
        issues.append("depth_confidence_shape_mismatch")

    rgb_dir = scan / "rgb"
    rgb_extracted = len([item for item in rgb_dir.iterdir() if item.is_file()]) if rgb_dir.is_dir() else 0
    if rgb_extracted and abs(rgb_extracted - len(depth)) > 2:
        warnings.append("extracted_rgb_depth_count_mismatch")

    output = scan / "output"
    missing_outputs = [name for name in ESSENTIAL_OUTPUTS if not (output / name).is_file()]
    if missing_outputs:
        issues.append("missing_essential_output")

    stats_path = output / "lichen_stats.json"
    coverage = lichen_area = surface_area = green_area = math.nan
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))["stats"]
        overall = stats["overall"]
        coverage = float(overall["lichen_ratio_pct"])
        lichen_area = float(overall["total_lichen_cm2"])
        surface_area = float(overall["total_cm2"])
        green_area = float(stats["upslope"]["green_lichen_cm2"]) + float(
            stats["downslope"]["green_lichen_cm2"]
        )
        if not (0 <= coverage <= 100 and 0 <= lichen_area <= surface_area):
            issues.append("invalid_lichen_stats")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        issues.append("invalid_lichen_stats")

    fingerprint = ""
    if (scan / "camera_matrix.csv").is_file() and (scan / "odometry.csv").is_file():
        fingerprint = sha256_files([scan / "camera_matrix.csv", scan / "odometry.csv"])

    return {
        "scan": scan.name,
        "path": str(scan.resolve()),
        "status": "ok" if not issues else "error",
        "issues": sorted(set(issues)),
        "warnings": sorted(set(warnings)),
        "missing": missing,
        "depth_frames": len(depth),
        "confidence_frames": len(confidence),
        "odometry_rows": len(odom_rows),
        "imu_rows": len(imu_rows),
        "rgb_video_open": video_open,
        "rgb_width": video_width,
        "rgb_height": video_height,
        "rgb_frames": video_frames,
        "rgb_fps": video_fps,
        "rgb_extracted_frames": rgb_extracted,
        "depth_shape": list(depth_shape) if depth_shape else None,
        "confidence_shape": list(confidence_shape) if confidence_shape else None,
        "camera_fx": fx,
        "camera_fy": fy,
        "camera_cx": cx,
        "camera_cy": cy,
        "max_quaternion_norm_deviation": max_quaternion_deviation,
        "missing_outputs": missing_outputs,
        "surface_area_cm2": surface_area,
        "lichen_area_cm2": lichen_area,
        "lichen_ratio_pct": coverage,
        "green_lichen_cm2": green_area,
        "identity_fingerprint": fingerprint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"E:\3D_Que\Raw_data"))
    parser.add_argument("--output", type=Path, default=Path("manifests/legacy_audit.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scans = sorted((path for path in args.root.iterdir() if path.is_dir()), key=lambda path: path.name)
    details = [audit_scan(scan) for scan in scans]

    fingerprints: dict[str, list[str]] = defaultdict(list)
    for detail in details:
        if detail["identity_fingerprint"]:
            fingerprints[detail["identity_fingerprint"]].append(detail["scan"])
    duplicate_groups = [group for group in fingerprints.values() if len(group) > 1]

    issue_counts = Counter(issue for detail in details for issue in detail["issues"])
    warning_counts = Counter(warning for detail in details for warning in detail["warnings"])
    coverages = [detail["lichen_ratio_pct"] for detail in details if math.isfinite(detail["lichen_ratio_pct"])]
    frame_counts = [detail["depth_frames"] for detail in details]
    summary = {
        "root": str(args.root.resolve()),
        "total_scans": len(details),
        "ok_scans": sum(detail["status"] == "ok" for detail in details),
        "error_scans": sum(detail["status"] == "error" for detail in details),
        "issue_counts": dict(issue_counts),
        "warning_counts": dict(warning_counts),
        "duplicate_identity_groups": duplicate_groups,
        "frame_count": {
            "min": min(frame_counts, default=0),
            "median": statistics.median(frame_counts) if frame_counts else 0,
            "max": max(frame_counts, default=0),
        },
        "coverage_pct": {
            "min": min(coverages, default=None),
            "mean": statistics.mean(coverages) if coverages else None,
            "max": max(coverages, default=None),
        },
        "all_green_lichen_zero": all(detail["green_lichen_cm2"] == 0 for detail in details),
        "opencv_available": cv2 is not None,
    }
    report = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    fields = [
        "scan",
        "status",
        "issues",
        "warnings",
        "depth_frames",
        "rgb_frames",
        "rgb_extracted_frames",
        "camera_fx",
        "surface_area_cm2",
        "lichen_area_cm2",
        "lichen_ratio_pct",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for detail in details:
            row = dict(detail)
            row["issues"] = ";".join(detail["issues"])
            row["warnings"] = ";".join(detail["warnings"])
            writer.writerow(row)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["error_scans"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
