import os, csv, json
from pathlib import Path

root = Path(r"C:\Users\Admin\Downloads\3D_Que\Raw_data")
required_files = ["camera_matrix.csv", "odometry.csv", "rgb.mp4"]
required_dirs = ["depth", "confidence"]
required_odom_cols = {"timestamp","frame","x","y","z","qx","qy","qz","qw"}

rows = []
for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
    issues = []
    info = {
        "folder": d.name,
        "path": str(d),
        "missing": [],
        "depth_png": 0,
        "conf_png": 0,
        "odom_rows": 0,
        "odom_has_required_cols": None,
        "common_frame_count": 0,
    }

    for f in required_files:
        if not (d / f).is_file():
            info["missing"].append(f)
    for sub in required_dirs:
        if not (d / sub).is_dir():
            info["missing"].append(sub + "/")

    depth_ids, conf_ids = set(), set()

    depth_dir = d / "depth"
    if depth_dir.is_dir():
        for p in depth_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".png":
                info["depth_png"] += 1
                depth_ids.add(p.stem)
        if info["depth_png"] == 0:
            issues.append("depth_rong")

    conf_dir = d / "confidence"
    if conf_dir.is_dir():
        for p in conf_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".png":
                info["conf_png"] += 1
                conf_ids.add(p.stem)
        if info["conf_png"] == 0:
            issues.append("confidence_rong")

    odom_path = d / "odometry.csv"
    odom_ids = set()
    if odom_path.is_file():
        try:
            with odom_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                cols = set(reader.fieldnames or [])
                info["odom_has_required_cols"] = required_odom_cols.issubset(cols)
                if not info["odom_has_required_cols"]:
                    issues.append("odometry_thieu_cot")
                for row in reader:
                    info["odom_rows"] += 1
                    fr = (row.get("frame") or "").strip()
                    if fr:
                        odom_ids.add(fr)
            if info["odom_rows"] == 0:
                issues.append("odometry_rong")
        except Exception as e:
            issues.append(f"odometry_doc_loi:{type(e).__name__}")

    if depth_ids and conf_ids:
        if depth_ids != conf_ids:
            issues.append("depth_conf_khong_khop_frame")

    if odom_ids and depth_ids and conf_ids:
        common = odom_ids & depth_ids & conf_ids
        info["common_frame_count"] = len(common)
        if len(common) == 0:
            issues.append("khong_co_frame_chung_odom_depth_conf")

    if info["missing"]:
        issues.append("thieu_file_bat_buoc")

    info["issues"] = issues
    rows.append(info)

summary = {
    "total_folders": len(rows),
    "ok_folders": sum(1 for r in rows if not r["issues"]),
    "error_folders": sum(1 for r in rows if r["issues"]),
    "issue_counts": {},
}
for r in rows:
    for it in r["issues"]:
        summary["issue_counts"][it] = summary["issue_counts"].get(it, 0) + 1

report = {"summary": summary, "details": rows}
out = root.parent / "raw_data_validation_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"REPORT={out}")
print(json.dumps(summary, ensure_ascii=False))
