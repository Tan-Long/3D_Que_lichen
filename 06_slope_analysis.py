#!/usr/bin/env python3
"""
Step 6: Slope Analysis — Xác định mặt Upslope / Downslope của thân cây.

Nguyên lý:
  - IMU accelerometer (a_x, a_y, a_z) khi thiết bị gần tĩnh ≈ gravity vector
    trong device frame (~9.8 m/s²).
  - Odometry cho camera pose (position + quaternion) ở mỗi frame.
  - Gravity trong world frame = R_cam2world @ g_device (trung bình nhiều frame)
  - World-frame gravity gồm:
      • g_vertical  = thành phần thẳng đứng (dọc theo thân cây)
      • g_slope     = thành phần nằm ngang (hướng dốc xuống)
  - Phía DOWNSLOPE = hướng g_slope (đất dốc xuống về phía đó)
  - Phía UPSLOPE   = ngược lại (-g_slope)

Output:
  - slope_info.json  : gravity vector, slope direction, góc nghiêng, trunk axis
  - trunk_labeled.ply: thêm scalar field "side" (0=upslope, 1=downslope) vào vertex colors
                        upslope  → màu đỏ
                        downslope → màu xanh dương

Usage:
    python 06_slope_analysis.py E:/3D_Que/Raw_data/26Q1
    python 06_slope_analysis.py E:/3D_Que/Raw_data  --batch
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Load sensor data
# ---------------------------------------------------------------------------

def load_imu(dataset_dir: str) -> np.ndarray:
    """Load accelerometer data. Returns (N, 3) array [a_x, a_y, a_z] m/s²."""
    path = os.path.join(dataset_dir, "imu.csv")
    data = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            data.append([float(row["a_x"]), float(row["a_y"]), float(row["a_z"])])
    return np.array(data)


def load_odometry(dataset_dir: str) -> list[dict]:
    """Load camera poses from odometry.csv."""
    path = os.path.join(dataset_dir, "odometry.csv")
    poses = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            poses.append({
                "frame": row["frame"].strip(),
                "position": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                "quaternion": np.array([
                    float(row["qx"]), float(row["qy"]),
                    float(row["qz"]), float(row["qw"])
                ]),
            })
    return poses


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),    2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])


# ---------------------------------------------------------------------------
# Gravity estimation in world frame
# ---------------------------------------------------------------------------

def estimate_gravity_world(
    imu_data: np.ndarray,
    poses: list[dict],
    low_motion_threshold: float = 0.5,
) -> np.ndarray:
    """
    Ước tính gravity vector trong world frame.

    Cách 1 (preferred): Với mỗi IMU sample, rotate từ device frame sang world frame
    bằng camera rotation gần nhất. Trung bình tất cả → gravity world.

    Cách 2 (fallback): Dùng trực tiếp trung bình IMU trong device frame
    rồi rotate bằng rotation trung bình.

    Lọc các frame có chuyển động mạnh (|a| xa 9.81) trước khi trung bình.
    """
    g_nominal = 9.80665

    # Lọc IMU: chỉ lấy sample gần tĩnh (|a| ≈ g)
    a_mag = np.linalg.norm(imu_data, axis=1)
    static_mask = np.abs(a_mag - g_nominal) < low_motion_threshold
    static_imu = imu_data[static_mask]

    if len(static_imu) < 10:
        print(f"  Warning: only {len(static_imu)} static IMU samples, using all")
        static_imu = imu_data

    # Trung bình gravity trong device frame
    g_device = static_imu.mean(axis=0)
    g_device_norm = g_device / np.linalg.norm(g_device)
    print(f"  Gravity in device frame: {g_device_norm.round(4)}")

    # Rotate sang world frame dùng rotation trung bình của camera
    # (Stray Scanner: camera frame ≈ device frame)
    rotations = [quaternion_to_rotation_matrix(p["quaternion"]) for p in poses]

    # Trung bình rotation: dùng trung bình quaternion đơn giản
    quats = np.array([p["quaternion"] for p in poses])
    # Chuẩn hóa dấu quaternion (flip nếu cần để tránh averaging cancel)
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    q_mean = quats.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    R_mean = quaternion_to_rotation_matrix(q_mean)

    # Gravity trong world frame
    g_world = R_mean @ g_device_norm
    g_world_norm = g_world / np.linalg.norm(g_world)
    print(f"  Gravity in world frame: {g_world_norm.round(4)}")

    return g_world_norm


# ---------------------------------------------------------------------------
# Trunk axis estimation
# ---------------------------------------------------------------------------

def estimate_trunk_axis_pca(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Dùng PCA trên point cloud thân cây để tìm trục chính (hướng thân cây).
    Thường là eigenvector ứng với eigenvalue lớn nhất.
    Trả về unit vector hướng lên (positive z component).
    """
    points = np.asarray(pcd.points)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Eigenvector của eigenvalue lớn nhất = trục chính thân cây
    trunk_axis = eigvecs[:, -1]
    # Đảm bảo hướng lên
    if trunk_axis[2] < 0:
        trunk_axis = -trunk_axis
    return trunk_axis


# ---------------------------------------------------------------------------
# Slope direction
# ---------------------------------------------------------------------------

def compute_slope_direction(
    g_world: np.ndarray,
    trunk_axis: np.ndarray,
) -> dict:
    """
    Tính hướng dốc (slope direction) từ gravity và trục thân cây.

    slope_down = thành phần của gravity vuông góc với trục thân
               = hướng dốc XUỐNG (downslope direction)
    slope_up   = -slope_down

    Trả về dict với các thông tin slope.
    """
    # Bỏ thành phần dọc theo trục thân cây
    g_proj_trunk = np.dot(g_world, trunk_axis) * trunk_axis
    slope_down = g_world - g_proj_trunk  # thành phần nằm ngang

    slope_magnitude = np.linalg.norm(slope_down)
    if slope_magnitude < 1e-6:
        print("  WARNING: slope direction nearly zero (flat terrain?)")
        # Chọn hướng tùy ý
        slope_down = np.array([1.0, 0.0, 0.0])
        slope_magnitude = 1.0

    slope_down_norm = slope_down / slope_magnitude
    slope_up_norm = -slope_down_norm

    # Góc nghiêng địa hình (angle between gravity and trunk axis)
    cos_angle = abs(np.dot(g_world, trunk_axis))
    slope_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    terrain_slope_deg = 90.0 - slope_angle_deg  # góc nghiêng so với nằm ngang

    print(f"  Slope down direction (world): {slope_down_norm.round(4)}")
    print(f"  Terrain slope angle: {terrain_slope_deg:.1f}°")

    return {
        "slope_down": slope_down_norm,
        "slope_up": slope_up_norm,
        "terrain_slope_deg": float(terrain_slope_deg),
        "slope_horizontal_magnitude": float(slope_magnitude),
    }


# ---------------------------------------------------------------------------
# Label trunk vertices: upslope / downslope
# ---------------------------------------------------------------------------

def label_trunk_by_slope(
    pcd: o3d.geometry.PointCloud,
    cx: float,
    cy: float,
    slope_down: np.ndarray,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Gán nhãn từng điểm là upslope (0) hoặc downslope (1).

    Nguyên tắc:
    - Project vị trí điểm từ trục thân ra hướng ngang (XY plane)
    - Dot product với slope_down:
        > 0  → downslope (phía dốc xuống)
        <= 0 → upslope   (phía dốc lên)

    Màu hóa:
        upslope   → đỏ  (1,0,0)
        downslope → xanh dương (0,0,1)
    """
    points = np.asarray(pcd.points)

    # Vector từ trục thân đến từng điểm (XY plane)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    radial = np.stack([dx, dy, np.zeros(len(dx))], axis=1)  # (N, 3)

    # Dot product với slope_down (chỉ dùng XY của slope_down)
    slope_xy = slope_down.copy()
    slope_xy[2] = 0.0
    slope_xy_mag = np.linalg.norm(slope_xy)
    if slope_xy_mag > 1e-6:
        slope_xy /= slope_xy_mag

    dot = radial @ slope_xy  # (N,) — dương = phía downslope

    # Nhãn
    labels = np.where(dot > 0, 1, 0)  # 1=downslope, 0=upslope

    # Màu hóa
    colors = np.zeros((len(points), 3))
    colors[labels == 0] = [1.0, 0.0, 0.0]   # upslope = đỏ
    colors[labels == 1] = [0.0, 0.0, 1.0]   # downslope = xanh dương

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = pcd.points
    labeled_pcd.colors = o3d.utility.Vector3dVector(colors)

    n_up = (labels == 0).sum()
    n_down = (labels == 1).sum()
    print(f"  Upslope: {n_up} pts ({100*n_up/len(points):.1f}%)")
    print(f"  Downslope: {n_down} pts ({100*n_down/len(points):.1f}%)")

    return labeled_pcd, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_slope(dataset_dir: str) -> dict | None:
    """
    Chạy toàn bộ slope analysis cho một mẫu.
    Trả về dict thông tin slope, lưu trunk_labeled.ply và slope_info.json.
    """
    trunk_pcd_path = os.path.join(dataset_dir, "output", "trunk_pointcloud.ply")
    if not os.path.exists(trunk_pcd_path):
        print(f"  SKIP: {trunk_pcd_path} not found. Run 05_trunk_isolation.py first.")
        return None

    print(f"\n{'='*60}")
    print(f"Slope analysis: {os.path.basename(dataset_dir)}")
    print(f"{'='*60}")

    # Load dữ liệu
    imu_data = load_imu(dataset_dir)
    poses = load_odometry(dataset_dir)
    pcd = o3d.io.read_point_cloud(trunk_pcd_path)

    print(f"  IMU samples: {len(imu_data)}")
    print(f"  Poses: {len(poses)}")
    print(f"  Trunk points: {len(pcd.points)}")

    # Gravity trong world frame
    g_world = estimate_gravity_world(imu_data, poses)

    # Trục thân cây
    trunk_axis = estimate_trunk_axis_pca(pcd)
    print(f"  Trunk axis (PCA): {trunk_axis.round(4)}")

    # Hướng dốc
    slope_info = compute_slope_direction(g_world, trunk_axis)

    # Tâm trục (để project radial)
    pts = np.asarray(pcd.points)
    cx = float(np.median(pts[:, 0]))
    cy = float(np.median(pts[:, 1]))

    # Gán nhãn upslope/downslope
    labeled_pcd, labels = label_trunk_by_slope(
        pcd, cx, cy, slope_info["slope_down"]
    )

    # Lưu labeled point cloud
    out_labeled = os.path.join(dataset_dir, "output", "trunk_labeled.ply")
    o3d.io.write_point_cloud(out_labeled, labeled_pcd)
    print(f"  Saved labeled PCD: {out_labeled}")

    # Lưu slope info JSON
    result = {
        "sample": os.path.basename(dataset_dir),
        "gravity_world": g_world.tolist(),
        "trunk_axis": trunk_axis.tolist(),
        "slope_down_direction": slope_info["slope_down"].tolist(),
        "slope_up_direction": slope_info["slope_up"].tolist(),
        "terrain_slope_deg": slope_info["terrain_slope_deg"],
        "trunk_center_xy": [cx, cy],
        "n_upslope_pts": int((labels == 0).sum()),
        "n_downslope_pts": int((labels == 1).sum()),
    }
    json_path = os.path.join(dataset_dir, "output", "slope_info.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved slope info: {json_path}")

    return result


def run_batch(raw_data_dir: str) -> None:
    scan_dirs = sorted([
        os.path.join(raw_data_dir, d)
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
        and os.path.exists(os.path.join(raw_data_dir, d, "output", "trunk_pointcloud.ply"))
    ])
    print(f"Found {len(scan_dirs)} scans with trunk_pointcloud.ply")

    results = []
    for scan_dir in scan_dirs:
        r = analyze_slope(scan_dir)
        if r:
            results.append(r)

    # Tổng hợp
    summary_path = os.path.join(raw_data_dir, "slope_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBatch done. Summary saved: {summary_path}")

    # In thống kê
    if results:
        angles = [r["terrain_slope_deg"] for r in results]
        print(f"Terrain slope: mean={np.mean(angles):.1f}°, "
              f"min={np.min(angles):.1f}°, max={np.max(angles):.1f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Slope analysis: determine upslope/downslope sides of tree trunk"
    )
    parser.add_argument(
        "dataset",
        help="Path to single scan folder or Raw_data/ folder with --batch"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all scans with trunk_pointcloud.ply"
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.dataset)
    else:
        result = analyze_slope(args.dataset)
        if result is None:
            sys.exit(1)
        print(f"\nSlope analysis complete:")
        print(f"  Terrain slope: {result['terrain_slope_deg']:.1f}°")
        print(f"  Slope down direction: {np.array(result['slope_down_direction']).round(3)}")


if __name__ == "__main__":
    main()
