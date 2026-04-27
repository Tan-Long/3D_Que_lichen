#!/usr/bin/env python3
"""
Step 1: Clean point cloud — lọc điểm theo camera trajectory.

Ý tưởng: Stray Scanner quay sát vào vỏ cây → điểm trunk luôn nằm
trong vòng R cm từ camera. Cây khác / nền ở xa hơn → loại.

Thuật toán:
  1. Load odometry.csv → lấy tất cả camera positions (x,y,z)
  2. Build KDTree từ camera positions
  3. Mỗi điểm PCD: tìm camera gần nhất
  4. Giữ nếu min_dist <= radius (default 0.30m)

Usage:
    python 01_clean_pointcloud.py --scan 26Q1
    python 01_clean_pointcloud.py --all
    python 01_clean_pointcloud.py --all --radius 0.30
    python 01_clean_pointcloud.py --all --radius 0.25 --skip-existing
"""

import argparse, json, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")


def load_camera_positions(scan_dir: str) -> np.ndarray:
    """Load odometry.csv → array (N, 3) camera positions."""
    odo_path = os.path.join(scan_dir, "odometry.csv")
    if not os.path.exists(odo_path):
        return None

    data = np.genfromtxt(odo_path, delimiter=",", skip_header=1)
    # columns: timestamp, frame, x, y, z, qx, qy, qz, qw
    cam_pos = data[:, 2:5].astype(np.float32)   # (N, 3)
    return cam_pos


def filter_by_camera_distance(pcd: o3d.geometry.PointCloud,
                               cam_pos: np.ndarray,
                               radius: float) -> o3d.geometry.PointCloud:
    """
    Giữ chỉ các điểm có khoảng cách đến camera gần nhất <= radius.
    Dùng KDTree trên camera positions → query toàn bộ points.
    """
    pts = np.asarray(pcd.points, dtype=np.float32)

    # Safety: Open3D có thể đọc PLY lỗi -> pts chứa NaN/Inf.
    # SciPy cKDTree sẽ crash nếu có NaN/Inf, nên lọc trước.
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Invalid point array shape: {pts.shape}")

    finite_mask = np.isfinite(pts).all(axis=1)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        # Chọn lại pcd theo index hợp lệ
        valid_idx = np.where(finite_mask)[0]
        if len(valid_idx) == 0:
            raise ValueError("No finite points available after filtering NaN/Inf.")
        pcd = pcd.select_by_index(valid_idx)
        pts = pts[finite_mask]

    if cam_pos is None or len(cam_pos) == 0:
        raise ValueError("Empty camera positions.")

    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    cam_finite_mask = np.isfinite(cam_pos).all(axis=1)
    if not cam_finite_mask.all():
        cam_pos = cam_pos[cam_finite_mask]
    if len(cam_pos) == 0:
        raise ValueError("No finite camera positions available.")

    # Build KDTree từ camera trajectory
    tree = cKDTree(cam_pos)

    # Query: với mỗi điểm, distance đến camera gần nhất
    dists, _ = tree.query(pts, k=1, workers=-1)   # workers=-1 = dùng hết CPU

    keep = dists <= radius
    return pcd.select_by_index(np.where(keep)[0])


def process_scan(scan_dir: str, radius: float, verbose: bool = True) -> dict:
    name     = os.path.basename(scan_dir)
    out_dir  = os.path.join(scan_dir, "output")
    ply_path = os.path.join(out_dir, "pointcloud.ply")
    result   = {"name": name, "ok": False, "msg": "", "n_in": 0, "n_out": 0}

    if not os.path.exists(ply_path):
        result["msg"] = "pointcloud.ply not found"
        return result

    # Load camera positions
    cam_pos = load_camera_positions(scan_dir)
    if cam_pos is None or len(cam_pos) == 0:
        result["msg"] = "odometry.csv not found or empty"
        return result

    t0 = time.time()

    # Load point cloud
    pcd   = o3d.io.read_point_cloud(ply_path)
    n_raw = len(pcd.points)
    result["n_in"] = n_raw

    if verbose:
        print(f"  {name}: {n_raw:,} pts, {len(cam_pos)} cam frames", flush=True)

    # Filter by camera distance
    pcd_clean = filter_by_camera_distance(pcd, cam_pos, radius)
    n_out     = len(pcd_clean.points)

    if verbose:
        pct = 100 * n_out / max(n_raw, 1)
        print(f"    radius={radius}m -> {n_out:,} pts ({pct:.1f}%)  [{time.time()-t0:.1f}s]",
              flush=True)

    if n_out < 50:
        result["msg"] = f"too few points after filter ({n_out})"
        return result

    # Save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trunk_pointcloud.ply")
    o3d.io.write_point_cloud(out_path, pcd_clean)

    meta = {"name": name, "radius": radius,
            "n_raw": n_raw, "n_trunk": n_out,
            "n_cam_frames": len(cam_pos)}
    with open(os.path.join(out_dir, "clean_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    result.update({"ok": True, "n_out": n_out,
                   "msg": f"{n_out:,} pts ({100*n_out/n_raw:.1f}%)"})
    return result


def run_batch(raw_data_dir, radius, scan_filter=None, skip_existing=False):
    if scan_filter:
        dirs = [os.path.join(raw_data_dir, scan_filter)]
        if not os.path.isdir(dirs[0]):
            print(f"Scan not found: {dirs[0]}"); sys.exit(1)
    else:
        dirs = sorted([
            os.path.join(raw_data_dir, d)
            for d in os.listdir(raw_data_dir)
            if os.path.isdir(os.path.join(raw_data_dir, d))
        ])

    total = len(dirs)
    print(f"\n{'='*55}")
    print(f"Camera-distance filter  |  radius = {radius}m  |  {total} scans")
    print(f"{'='*55}\n")

    ok_count = skip_count = 0
    results  = []
    failed  = []

    for i, scan_dir in enumerate(dirs):
        name     = os.path.basename(scan_dir)
        out_path = os.path.join(scan_dir, "output", "trunk_pointcloud.ply")

        if skip_existing and os.path.exists(out_path):
            print(f"[{i+1:3d}/{total}] {name} -- SKIP")
            skip_count += 1
            continue

        print(f"[{i+1:3d}/{total}] {name}")
        try:
            r = process_scan(scan_dir, radius, verbose=True)
        except Exception as e:
            # Skip lỗi để batch tiếp tục chạy
            r = {
                "name": name,
                "ok": False,
                "msg": str(e),
                "n_in": 0,
                "n_out": 0,
            }
        results.append(r)

        if r["ok"]:
            ok_count += 1
            print(f"  OK  {r['msg']}")
        else:
            print(f"  FAIL  {r['msg']}")
            failed.append(r)
        print()

    print(f"{'='*55}")
    print(f"DONE: {ok_count}/{total - skip_count} OK  ({skip_count} skipped)")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['name']}: {r['msg']}")

    with open(os.path.join(raw_data_dir, "clean_summary.json"), "w") as f:
        json.dump({"radius": radius, "results": results}, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description="Filter point cloud by camera trajectory distance")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME", help="Ten scan, vd: 26Q1")
    g.add_argument("--all",  action="store_true", help="Chay tat ca scans")

    p.add_argument("--radius",        type=float, default=0.30,
                   help="Giu diem trong R met tu camera (default: 0.30)")
    p.add_argument("--raw-data",      default=RAW_DATA_DIR)
    p.add_argument("--skip-existing", action="store_true")

    args = p.parse_args()
    run_batch(args.raw_data, args.radius,
              scan_filter=args.scan if args.scan else None,
              skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
