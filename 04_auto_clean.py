import os
import sys

import numpy as np
import open3d as o3d


def auto_crop_tree(ply_path, crop_radius=1.5):
    """
    Tu dong tim tam cay bang median va cat bo diem ben ngoai ban kinh.
    Sau do loc nhieu bang Statistical Outlier Removal (SOR).
    """
    if not os.path.exists(ply_path):
        print(f"BO QUA: Khong tim thay file {ply_path}")
        return

    print(f"   -> Dang doc file: {os.path.basename(ply_path)}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    if len(points) == 0:
        print("   -> Point cloud rong, bo qua.")
        return

    center_x = np.median(points[:, 0])
    center_y = np.median(points[:, 1])
    center_z = np.median(points[:, 2])
    tree_center = np.array([center_x, center_y, center_z])

    distances = np.linalg.norm(points - tree_center, axis=1)
    mask = distances < crop_radius
    cropped_pcd = pcd.select_by_index(np.where(mask)[0])
    print(f"   -> Da cat got: {len(points)} -> {len(cropped_pcd.points)} diem.")

    if len(cropped_pcd.points) == 0:
        print("   -> Sau khi cat khong con diem nao, bo qua SOR.")
        return

    print("   -> Dang don diem nhieu (SOR)...")
    _, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    clean_pcd = cropped_pcd.select_by_index(ind)

    out_path = ply_path.replace(".ply", "_clean.ply")
    o3d.io.write_point_cloud(out_path, clean_pcd)
    print(f"   -> DA LUU: {os.path.basename(out_path)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Loi: Thieu duong dan thu muc!")
        sys.exit(1)

    folder_path = sys.argv[1]
    target_ply = os.path.join(folder_path, "output", "pointcloud.ply")
    auto_crop_tree(target_ply, crop_radius=1.5)
