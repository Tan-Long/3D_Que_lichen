import os
import random
import subprocess
import sys
import argparse

raw_data_dir = r"C:\Users\Admin\Downloads\3D_Que\Raw_data"

parser = argparse.ArgumentParser(description="Batch preprocess Stray Scanner folders")
parser.add_argument("--sample-ratio", type=float, default=1.0,
                    help="Ty le folder duoc xu ly, vd 0.1 = 10%% (default: 1.0)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed khi lay mau folder (default: 42)")
parser.add_argument("--max-width", type=int, default=640,
                    help="Do rong xu ly cho mesh/pointcloud (default: 640)")
args = parser.parse_args()

extract_script = os.path.join(os.getcwd(), "01_extract_rgb_frames.py")
mesh_script = os.path.join(os.getcwd(), "02_pointcloud_and_mesh.py")
clean_script = os.path.join(os.getcwd(), "04_auto_clean.py")

if not os.path.isdir(raw_data_dir):
    print(f"Khong tim thay thu muc raw data: {raw_data_dir}")
    sys.exit(1)

folders = [f.path for f in os.scandir(raw_data_dir) if f.is_dir()]
folders.sort()

if args.sample_ratio <= 0 or args.sample_ratio > 1:
    print("--sample-ratio phai nam trong khoang (0, 1].")
    sys.exit(1)

if args.sample_ratio < 1.0:
    rng = random.Random(args.seed)
    sample_count = max(1, int(len(folders) * args.sample_ratio))
    folders = sorted(rng.sample(folders, sample_count))
    print(f"Che do lay mau: {sample_count} / {len([f.path for f in os.scandir(raw_data_dir) if f.is_dir()])} folder")

print(f"Tim thay tong cong {len(folders)} thu muc can xu ly.")

if not os.path.isfile(extract_script):
    print(f"THIEU FILE: {extract_script}")
if not os.path.isfile(mesh_script):
    print(f"THIEU FILE: {mesh_script}")
if not os.path.isfile(clean_script):
    print(f"THIEU FILE: {clean_script}")

for index, folder in enumerate(folders, 1):
    print("\n==================================================")
    print(f"[{index}/{len(folders)}] DANG XU LY: {os.path.basename(folder)}")
    print("==================================================")

    depth_dir = os.path.join(folder, "depth")
    video_path = os.path.join(folder, "rgb.mp4")
    if not os.path.isdir(depth_dir) or not os.path.isfile(video_path):
        print("--> Bo qua: thieu depth/ hoac rgb.mp4")
        continue

    print("--> Dang chay trich xuat RGB...")
    r1 = subprocess.run([sys.executable, "01_extract_rgb_frames.py", folder])
    if r1.returncode != 0:
        print("--> Loi buoc extract RGB, bo qua folder nay.")
        continue

    print("--> Dang dung Point Cloud & Mesh...")
    r2 = subprocess.run([
        sys.executable,
        "02_pointcloud_and_mesh.py",
        folder,
        "--no-visualize",
        "--max-width",
        str(args.max_width),
    ])
    if r2.returncode != 0:
        print("--> Loi buoc Point Cloud/Mesh, chuyen folder tiep theo.")
        continue

    print("--> Dang tu dong got tia va hut bui Point Cloud...")
    r3 = subprocess.run([sys.executable, "04_auto_clean.py", folder])
    if r3.returncode != 0:
        print("--> Loi buoc auto clean, chuyen folder tiep theo.")

print("\nDA HOAN THANH XU LY TOAN BO THU MUC!")
