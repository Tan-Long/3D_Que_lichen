#!/usr/bin/env python3
"""
Step 5: Trunk Isolation — Tách thân cây quế khỏi nhiễu nền.

Vấn đề với pointcloud raw:
  - Có nhiều thực vật mặt đất (cỏ, rêu xanh)
  - Cây khác ở phía sau
  - Điểm lơ lửng (floating noise)
  - Mặt đất (ground plane)

Pipeline:
  1. RANSAC ground plane removal — loại bỏ mặt phẳng đất
  2. Ước tính trục thân cây từ vùng trên cao (ít bị nhiễu nhất)
  3. Crop theo bán kính XY xung quanh trục thân
  4. Statistical Outlier Removal — loại điểm lẻ
  5. Lọc theo màu (tuỳ chọn) — loại lá xanh còn sót
  6. Lưu trunk_pointcloud.ply

Usage:
    python 05_trunk_isolation.py E:/3D_Que/Raw_data/26Q1
    python 05_trunk_isolation.py E:/3D_Que/Raw_data/26Q1 --trunk-radius 0.3
    python 05_trunk_isolation.py E:/3D_Que/Raw_data  --batch   (xử lý tất cả 92 mẫu)
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# 1. Ground Plane Removal (RANSAC)
# ---------------------------------------------------------------------------

def remove_ground_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.08,
    ransac_n: int = 3,
    num_iterations: int = 2000,
    min_ground_normal_z: float = 0.70,
    bottom_fraction: float = 0.25,
) -> o3d.geometry.PointCloud:
    """
    Fit RANSAC plane trên vùng z thấp, loại bỏ nếu là mặt ngang.

    Chiến lược:
    - Chỉ chạy RANSAC trên bottom_fraction điểm thấp nhất
      (tránh RANSAC tìm tường/thân cây thay vì đất)
    - Chấp nhận mặt phẳng nếu normal_z >= min_ground_normal_z
    - distance_threshold cao hơn để bắt đất dốc
    """
    if len(pcd.points) < 10:
        return pcd

    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_cut = z_min + (z_max - z_min) * bottom_fraction

    low_idx = np.where(z_vals <= z_cut)[0]
    if len(low_idx) < 50:
        print("  Ground removal: too few low points, skipping")
        return pcd

    pcd_low = pcd.select_by_index(low_idx)
    plane_model, inliers_local = pcd_low.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    if abs(normal[2]) >= min_ground_normal_z:
        # Map inliers từ pcd_low → pcd gốc
        ground_idx_global = low_idx[inliers_local]
        # Thêm buffer: loại thêm điểm ngay trên mặt đất (+5cm)
        ground_z = np.mean(points[ground_idx_global, 2])
        above_ground = z_vals > ground_z + 0.05
        kept_idx = np.where(above_ground)[0]
        result = pcd.select_by_index(kept_idx)
        print(f"  Ground removed: {len(pcd.points)} -> {len(result.points)} pts "
              f"(normal_z={normal[2]:.3f}, ground_z={ground_z:.3f})")
        return result
    else:
        print(f"  Ground plane not detected (normal_z={normal[2]:.3f}), "
              f"using z_min offset fallback")
        # Fallback: loại bỏ điểm quá gần sàn (z < z_min + 10cm)
        kept = pcd.select_by_index(np.where(z_vals > z_min + 0.10)[0])
        print(f"  Fallback z-floor cut: {len(pcd.points)} -> {len(kept.points)} pts")
        return kept


# ---------------------------------------------------------------------------
# 2. Trunk Axis Estimation
# ---------------------------------------------------------------------------

def estimate_trunk_axis(
    pcd: o3d.geometry.PointCloud,
    upper_fraction: float = 0.4,
) -> tuple[float, float]:
    """
    Ước tính tâm trục thân cây (cx, cy) bằng cách:
    - Lấy 40% điểm cao nhất (ít bị nhiễu mặt đất nhất)
    - Tính trung vị XY của vùng đó

    Trả về (cx, cy) — tâm trục thân cây trên mặt phẳng XY.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return 0.0, 0.0

    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_threshold = z_max - (z_max - z_min) * upper_fraction

    upper_mask = z_vals >= z_threshold
    upper_pts = points[upper_mask]

    if len(upper_pts) < 5:
        # Fallback: dùng toàn bộ điểm
        upper_pts = points

    cx = float(np.median(upper_pts[:, 0]))
    cy = float(np.median(upper_pts[:, 1]))
    print(f"  Trunk axis estimated at XY=({cx:.3f}, {cy:.3f}) "
          f"from top {upper_fraction*100:.0f}% ({upper_mask.sum()} pts)")
    return cx, cy


# ---------------------------------------------------------------------------
# 3. Cylinder Crop (XY radius around trunk axis)
# ---------------------------------------------------------------------------

def crop_to_trunk_cylinder(
    pcd: o3d.geometry.PointCloud,
    cx: float,
    cy: float,
    radius: float = 0.30,
    z_min_offset: float = 0.05,
) -> o3d.geometry.PointCloud:
    """
    Giữ lại điểm trong bán kính `radius` tính theo XY (trụ đứng).
    Loại bỏ thêm những điểm quá gần mặt đất (z < z_min + z_min_offset).
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    # XY distance từ trục thân
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    xy_dist = np.sqrt(dx**2 + dy**2)

    # Bỏ điểm ngay sát đất (rễ cây, lá rụng)
    z_vals = points[:, 2]
    z_floor = z_vals.min() + z_min_offset

    mask = (xy_dist <= radius) & (z_vals >= z_floor)
    cropped = pcd.select_by_index(np.where(mask)[0])
    print(f"  Cylinder crop r={radius}m: {len(points)} -> {len(cropped.points)} pts")
    return cropped


# ---------------------------------------------------------------------------
# 4. Statistical Outlier Removal
# ---------------------------------------------------------------------------

def denoise(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) < nb_neighbors + 1:
        return pcd
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    cleaned = pcd.select_by_index(ind)
    print(f"  SOR denoised: {len(pcd.points)} -> {len(cleaned.points)} pts")
    return cleaned


# ---------------------------------------------------------------------------
# 5. Green Vegetation Filter (loại lá xanh còn sót)
# ---------------------------------------------------------------------------

def remove_green_vegetation(
    pcd: o3d.geometry.PointCloud,
    green_threshold: float = 0.15,
) -> o3d.geometry.PointCloud:
    """
    Loại điểm có màu xanh lá cây mạnh (lá, rêu, cỏ).
    green_threshold: mức độ "xanh hơn đỏ và xanh dương" để lọc.
    """
    if not pcd.has_colors():
        return pcd

    colors = np.asarray(pcd.colors)  # [0,1] RGB
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    # Giữ điểm KHÔNG phải xanh lá đậm
    not_green = ~((g > r + green_threshold) & (g > b + green_threshold))
    kept = pcd.select_by_index(np.where(not_green)[0])
    removed = len(pcd.points) - len(kept.points)
    print(f"  Green filter: removed {removed} pts ({100*removed/max(len(pcd.points),1):.1f}%)")
    return kept


# ---------------------------------------------------------------------------
# 6. Keep Largest Connected Cluster (DBSCAN)
# ---------------------------------------------------------------------------

def keep_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.05,
    min_points: int = 50,
) -> o3d.geometry.PointCloud:
    """
    Dùng DBSCAN để tìm cluster lớn nhất — thường là thân cây chính.
    """
    if len(pcd.points) < min_points:
        return pcd

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        print("  DBSCAN: no clusters found, keeping all")
        return pcd

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[np.argmax(counts)]

    mask = labels == largest_label
    kept = pcd.select_by_index(np.where(mask)[0])
    print(f"  DBSCAN largest cluster: {counts.max()} / {len(pcd.points)} pts "
          f"({len(unique)} clusters total)")
    return kept


# ---------------------------------------------------------------------------
# Main isolation pipeline
# ---------------------------------------------------------------------------

def isolate_trunk(
    dataset_dir: str,
    trunk_radius: float = 0.30,
    remove_green: bool = True,
    use_dbscan: bool = True,
) -> bool:
    """
    Đọc pointcloud.ply từ output/, chạy pipeline làm sạch, lưu trunk_pointcloud.ply.
    Trả về True nếu thành công.
    """
    pcd_path = os.path.join(dataset_dir, "output", "pointcloud.ply")
    if not os.path.exists(pcd_path):
        print(f"  SKIP: {pcd_path} not found")
        return False

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(dataset_dir)}")
    print(f"{'='*60}")
    print(f"  Loading {pcd_path} ...")

    pcd = o3d.io.read_point_cloud(pcd_path)
    n_orig = len(pcd.points)
    print(f"  Loaded {n_orig} points")

    if n_orig == 0:
        print("  SKIP: empty point cloud")
        return False

    # Step 1: Loại bỏ ground plane
    pcd = remove_ground_plane(pcd)

    # Step 2: Ước tính trục thân cây
    cx, cy = estimate_trunk_axis(pcd)

    # Step 3: Crop hình trụ quanh trục thân
    pcd = crop_to_trunk_cylinder(pcd, cx, cy, radius=trunk_radius)

    # Step 4: Denoising lần 1 (rộng)
    pcd = denoise(pcd, nb_neighbors=20, std_ratio=2.5)

    # Step 5: Loại thực vật xanh
    if remove_green:
        pcd = remove_green_vegetation(pcd)

    # Step 6: DBSCAN — giữ cluster lớn nhất (thân cây)
    if use_dbscan and len(pcd.points) > 100:
        pcd = keep_largest_cluster(pcd, eps=0.04, min_points=30)

    # Step 7: Denoising lần 2 (tinh)
    pcd = denoise(pcd, nb_neighbors=30, std_ratio=2.0)

    n_final = len(pcd.points)
    print(f"\n  Result: {n_orig} -> {n_final} pts "
          f"({100*n_final/n_orig:.1f}% retained)")

    if n_final < 100:
        print("  WARNING: very few points remain, check trunk_radius parameter")

    # Lưu
    out_path = os.path.join(dataset_dir, "output", "trunk_pointcloud.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  Saved: {out_path}")
    return True


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(raw_data_dir: str, trunk_radius: float = 0.30) -> None:
    """Xử lý tất cả mẫu trong thư mục Raw_data/."""
    scan_dirs = sorted([
        os.path.join(raw_data_dir, d)
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
        and os.path.exists(os.path.join(raw_data_dir, d, "output", "pointcloud.ply"))
    ])

    print(f"Found {len(scan_dirs)} scans with pointcloud.ply")

    success, skip = 0, 0
    for scan_dir in scan_dirs:
        ok = isolate_trunk(scan_dir, trunk_radius=trunk_radius)
        if ok:
            success += 1
        else:
            skip += 1

    print(f"\n{'='*60}")
    print(f"Batch done: {success} processed, {skip} skipped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Isolate tree trunk from raw point cloud"
    )
    parser.add_argument(
        "dataset",
        help="Path to single scan folder (e.g., Raw_data/26Q1) "
             "or to Raw_data/ folder with --batch"
    )
    parser.add_argument(
        "--trunk-radius", type=float, default=0.30,
        help="XY cylinder radius in meters to keep around trunk axis (default: 0.30)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all scans in the given folder"
    )
    parser.add_argument(
        "--no-green-filter", action="store_true",
        help="Disable green vegetation color filter"
    )
    parser.add_argument(
        "--no-dbscan", action="store_true",
        help="Disable DBSCAN largest-cluster selection"
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.dataset, trunk_radius=args.trunk_radius)
    else:
        ok = isolate_trunk(
            args.dataset,
            trunk_radius=args.trunk_radius,
            remove_green=not args.no_green_filter,
            use_dbscan=not args.no_dbscan,
        )
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
