#!/usr/bin/env python3
"""
Step 3: Prepare data for the strayrobots/3d-annotation-tool.

The annotation tool expects a specific directory layout:
    scene/
    ├── rgb/          (color images as .jpg)
    ├── depth/        (depth maps as 16-bit .png, millimeters)
    ├── intrinsic.json
    ├── poses/        (4x4 camera-to-world matrices, one .txt per frame)
    └── mesh.ply      (reconstructed mesh)

This script converts Stray Scanner format into that layout.

Usage:
    python 03_prepare_for_annotation.py c9ae76fff8
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

# Reuse helpers from our mesh script

def load_intrinsics(dataset_dir: str) -> np.ndarray:
    path = os.path.join(dataset_dir, "camera_matrix.csv")
    return np.loadtxt(path, delimiter=",")


def load_odometry(dataset_dir: str) -> list[dict]:
    import csv
    path = os.path.join(dataset_dir, "odometry.csv")
    poses = []
    with open(path, "r") as f:
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
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
    ])


def prepare_annotation_data(dataset_dir: str) -> None:
    scene_dir = os.path.join(dataset_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    rgb_src = os.path.join(dataset_dir, "rgb")
    depth_src = os.path.join(dataset_dir, "depth")
    conf_src = os.path.join(dataset_dir, "confidence")
    if not os.path.exists(rgb_src):
        raise FileNotFoundError("No rgb/ folder found. Run 01_extract_rgb_frames.py first.")
    if not os.path.exists(depth_src):
        raise FileNotFoundError("No depth/ folder found in dataset.")

    def ids_in_dir(d: str, ext: str) -> set[str]:
        if not os.path.isdir(d):
            return set()
        out: set[str] = set()
        for f in os.listdir(d):
            if f.lower().endswith(ext.lower()):
                out.add(os.path.splitext(f)[0])
        return out

    rgb_ids = ids_in_dir(rgb_src, ".jpg")
    depth_ids = ids_in_dir(depth_src, ".png")
    conf_ids = ids_in_dir(conf_src, ".png")

    poses = load_odometry(dataset_dir)
    pose_ids = {p["frame"] for p in poses}

    common = rgb_ids & depth_ids & pose_ids
    if conf_ids:
        common &= conf_ids
    common_sorted = sorted(common)
    if not common_sorted:
        raise RuntimeError("No common frames across rgb/depth/(confidence)/odometry.")
    if len(common_sorted) != len(pose_ids):
        print(f"NOTE: Preparing scene for {len(common_sorted)} frames (filtered from odometry).")

    # --- 1. Copy RGB images (filtered to common frames) ---
    rgb_dst = os.path.join(scene_dir, "rgb")
    if os.path.exists(rgb_dst):
        shutil.rmtree(rgb_dst)
    os.makedirs(rgb_dst, exist_ok=True)
    for fid in common_sorted:
        shutil.copy2(os.path.join(rgb_src, f"{fid}.jpg"), os.path.join(rgb_dst, f"{fid}.jpg"))
    print(f"Copied {len(common_sorted)} RGB frames to {rgb_dst}/")

    # --- 2. Copy depth images (filtered) ---
    depth_dst = os.path.join(scene_dir, "depth")
    if os.path.exists(depth_dst):
        shutil.rmtree(depth_dst)
    os.makedirs(depth_dst, exist_ok=True)
    for fid in common_sorted:
        shutil.copy2(os.path.join(depth_src, f"{fid}.png"), os.path.join(depth_dst, f"{fid}.png"))
    print(f"Copied {len(common_sorted)} depth maps to {depth_dst}/")

    # --- 3. Create intrinsic.json ---
    K = load_intrinsics(dataset_dir)

    # Read actual image dimensions to compute correct intrinsics per resolution
    sample_rgb = cv2.imread(os.path.join(rgb_dst, f"{common_sorted[0]}.jpg"))
    rgb_h, rgb_w = sample_rgb.shape[:2]

    sample_depth = cv2.imread(
        os.path.join(depth_dst, f"{common_sorted[0]}.png"), cv2.IMREAD_UNCHANGED
    )
    depth_h, depth_w = sample_depth.shape[:2]

    # The annotation tool typically uses depth-resolution intrinsics
    sx = depth_w / rgb_w
    sy = depth_h / rgb_h

    intrinsic_data = {
        "width": depth_w,
        "height": depth_h,
        "fx": float(K[0, 0] * sx),
        "fy": float(K[1, 1] * sy),
        "cx": float(K[0, 2] * sx),
        "cy": float(K[1, 2] * sy),
        "depth_scale": 1000.0,  # depth is in mm, scale to meters = /1000
    }
    intrinsic_path = os.path.join(scene_dir, "intrinsic.json")
    with open(intrinsic_path, "w") as f:
        json.dump(intrinsic_data, f, indent=2)
    print(f"Wrote intrinsics to {intrinsic_path}")
    print(f"  {intrinsic_data}")

    # --- 4. Create per-frame pose files ---
    poses_dir = os.path.join(scene_dir, "poses")
    if os.path.exists(poses_dir):
        shutil.rmtree(poses_dir)
    os.makedirs(poses_dir)

    pose_by_id = {p["frame"]: p for p in poses}
    for fid in common_sorted:
        pose = pose_by_id[fid]
        T = np.eye(4)
        T[:3, :3] = quaternion_to_rotation_matrix(pose["quaternion"])
        T[:3, 3] = pose["position"]

        pose_path = os.path.join(poses_dir, f"{fid}.txt")
        np.savetxt(pose_path, T, fmt="%.10f")

    print(f"Wrote {len(common_sorted)} pose files to {poses_dir}/")

    # --- 5. Copy mesh if it exists ---
    mesh_src = os.path.join(dataset_dir, "output", "mesh.ply")
    mesh_dst = os.path.join(scene_dir, "mesh.ply")
    if os.path.exists(mesh_src):
        shutil.copy2(mesh_src, mesh_dst)
        print(f"Copied mesh to {mesh_dst}")
    else:
        print(f"WARNING: No mesh found at {mesh_src}")
        print("  Run 02_pointcloud_and_mesh.py first, then re-run this script.")

    print(f"\nAnnotation scene ready at: {scene_dir}/")
    print("You can now open this folder with the 3d-annotation-tool.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Stray Scanner data for 3d-annotation-tool"
    )
    parser.add_argument("dataset", help="Path to dataset folder (e.g., c9ae76fff8)")
    args = parser.parse_args()
    prepare_annotation_data(args.dataset)
