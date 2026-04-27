#!/usr/bin/env python3
"""
Step 02 (TSDF): Raw Stray Scanner frames -> colored trunk mesh

Dung TSDF Volume Integration thay vi Poisson:
  - Tich hop truc tiep tu tung RGB-D frame goc (khong qua point cloud)
  - Mau lay tu anh RGB 1920x1440 -> chinh xac hon nhieu
  - Geometry tot hon: khong can estimate normal, khong co artifact Poisson
  - Sau do cat mesh theo camera trajectory (giong step 01)

Pipeline:
  1. Load camera_matrix.csv + odometry.csv
  2. TSDF Volume tich hop tat ca frame RGB-D
  3. Extract full scene mesh
  4. Filter vertex theo khoang cach camera trajectory (giu trunk)
  5. Save trunk_mesh_tsdf.ply

Usage:
    python 02_mesh_tsdf.py --scan 26Q1
    python 02_mesh_tsdf.py --scan 26Q1 --step 2   # nhanh hon, bo qua 1 frame
    python 02_mesh_tsdf.py --all --skip-existing
    python 02_mesh_tsdf.py --scan 26Q1 --voxel 0.003   # 3mm, chi tiet hon (cham)
    python 02_mesh_tsdf.py --scan 26Q1 --voxel 0.008   # 8mm, nhanh hon
"""

import argparse, copy, json, os, sys, time
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")

# Stray Scanner output resolution
RGB_W,   RGB_H   = 1920, 1440   # anh mau goc
DEPTH_W, DEPTH_H =  256,  192   # depth map / confidence map


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_intrinsics(scan_dir: str) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Doc camera_matrix.csv (3x3, dung cho RGB 1920x1440).
    Scale xuong DEPTH_W x DEPTH_H de dung voi depth map.
    """
    K  = np.loadtxt(os.path.join(scan_dir, "camera_matrix.csv"), delimiter=",")
    sx = DEPTH_W / RGB_W    # 256/1920
    sy = DEPTH_H / RGB_H    # 192/1440
    fx = float(K[0, 0]) * sx
    fy = float(K[1, 1]) * sy
    cx = float(K[0, 2]) * sx
    cy = float(K[1, 2]) * sy
    return o3d.camera.PinholeCameraIntrinsic(DEPTH_W, DEPTH_H, fx, fy, cx, cy)


def load_odometry(scan_dir: str):
    """
    Doc odometry.csv -> dict {frame_idx: T_WC (4x4 ndarray)}.
    Format: timestamp, frame, x, y, z, qx, qy, qz, qw
    """
    data  = np.loadtxt(os.path.join(scan_dir, "odometry.csv"),
                       delimiter=",", skiprows=1)
    poses = {}
    for row in data:
        fid  = int(row[1])
        pos  = row[2:5]
        quat = row[5:9]   # qx qy qz qw (scipy convention)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3]  = pos
        poses[fid] = T
    return poses


def load_depth_img(path: str,
                   confidence: np.ndarray = None,
                   conf_thresh: int = 1) -> o3d.geometry.Image:
    """
    Doc depth PNG (gia tri = mm) -> float32 (m).
    Tat cac pixel co confidence < conf_thresh (0=thap nhat, 2=cao nhat).
    """
    depth_mm = np.array(Image.open(path), dtype=np.float32)
    depth_m  = depth_mm / 1000.0
    if confidence is not None:
        depth_m[confidence < conf_thresh] = 0.0
    return o3d.geometry.Image(depth_m)


def load_rgb_img(path: str) -> o3d.geometry.Image:
    """
    Doc RGB JPG -> resize ve DEPTH_W x DEPTH_H -> open3d Image uint8.
    Resize ve depth resolution de khop voi RGBD image.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((DEPTH_W, DEPTH_H), Image.BILINEAR)
    return o3d.geometry.Image(np.asarray(img, dtype=np.uint8))


def filter_mesh_trunk(mesh: o3d.geometry.TriangleMesh,
                      cam_positions: np.ndarray,
                      radius: float = 0.65) -> o3d.geometry.TriangleMesh:
    """
    Giu cac vertex trong radius met tinh tu bat ky vi tri camera nao.
    radius=0.65: camera ~40cm + duong kinh than ~15cm + buffer = ~65cm
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tree  = cKDTree(cam_positions)
    dists, _ = tree.query(verts, k=1, workers=-1)

    remove_mask = dists > radius
    mesh_out    = copy.deepcopy(mesh)
    mesh_out.remove_vertices_by_mask(remove_mask)
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()
    mesh_out.remove_duplicated_triangles()
    return mesh_out


def filter_fragments(mesh: o3d.geometry.TriangleMesh,
                     min_triangles: int = 500) -> o3d.geometry.TriangleMesh:
    """
    Xoa cac manh roi rac nho (floating fragments).
    Chi giu cac connected component co >= min_triangles tam giac.
    """
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters  = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)

    # Mask: xoa tam giac thuoc cluster nho
    remove_mask = cluster_n_tris[tri_clusters] < min_triangles
    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_triangles_by_mask(remove_mask)
    mesh_out.remove_unreferenced_vertices()
    return mesh_out


# ── Per-scan pipeline ─────────────────────────────────────────────────────────

def process_scan(scan_dir: str,
                 voxel_size: float   = 0.004,
                 sdf_trunc: float    = 0.05,
                 conf_thresh: int    = 1,
                 max_depth: float    = 2.0,
                 trunk_radius: float = 0.65,
                 min_triangles: int  = 500,
                 frame_step: int     = 1,
                 verbose: bool       = True) -> dict:

    name    = os.path.basename(scan_dir)
    out_dir = os.path.join(scan_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    result  = {"name": name, "ok": False, "msg": ""}

    depth_dir = os.path.join(scan_dir, "depth")
    rgb_dir   = os.path.join(scan_dir, "rgb")
    conf_dir  = os.path.join(scan_dir, "confidence")

    if not os.path.isdir(depth_dir) or not os.path.isdir(rgb_dir):
        result["msg"] = "depth/ hoac rgb/ folder khong ton tai"
        return result

    t0 = time.time()

    # ── 1. Intrinsics + odometry ──────────────────────────────────────────────
    try:
        intrinsics = load_intrinsics(scan_dir)
        poses      = load_odometry(scan_dir)
    except Exception as e:
        result["msg"] = f"load intrinsics/odometry loi: {e}"
        return result

    all_fids      = sorted(poses.keys())
    fids          = all_fids[::frame_step]
    cam_positions = np.array([poses[i][:3, 3] for i in all_fids], dtype=np.float32)

    if verbose:
        print(f"  Frames: {len(fids)}/{len(all_fids)}  "
              f"voxel={voxel_size*100:.1f}cm  conf>={conf_thresh}  "
              f"R={trunk_radius}m", flush=True)

    # ── 2. TSDF Volume ────────────────────────────────────────────────────────
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length = voxel_size,
        sdf_trunc    = sdf_trunc,
        color_type   = o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # ── 3. Integration loop ───────────────────────────────────────────────────
    n_ok = n_skip = 0
    for k, fid in enumerate(fids):
        depth_path = os.path.join(depth_dir, f"{fid:06d}.png")
        rgb_path   = os.path.join(rgb_dir,   f"{fid:06d}.jpg")
        conf_path  = os.path.join(conf_dir,  f"{fid:06d}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            n_skip += 1
            continue

        # confidence map (optional)
        conf = None
        if os.path.isdir(conf_dir) and os.path.exists(conf_path):
            conf = np.array(Image.open(conf_path))

        depth_img = load_depth_img(depth_path, conf, conf_thresh)
        rgb_img   = load_rgb_img(rgb_path)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img, depth_img,
            depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)

        T_CW = np.linalg.inv(poses[fid])
        volume.integrate(rgbd, intrinsics, T_CW)
        n_ok += 1

        if verbose and (k + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{k+1}/{len(fids)}] {elapsed:.0f}s", flush=True)

    if verbose:
        print(f"  Integrated {n_ok} frames ({n_skip} skip)", flush=True)

    if n_ok == 0:
        result["msg"] = "khong co frame nao duoc tich hop"
        return result

    # ── 4. Extract full scene mesh ────────────────────────────────────────────
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    n_full = len(mesh.vertices)

    if verbose:
        print(f"  Full scene: {n_full:,} verts  "
              f"{len(mesh.triangles):,} tri", flush=True)

    if n_full == 0:
        result["msg"] = "TSDF ra mesh rong — thu giam voxel hoac conf"
        return result

    # ── 5. Filter trunk region (camera distance) ─────────────────────────────
    mesh_trunk = filter_mesh_trunk(mesh, cam_positions, trunk_radius)
    if verbose:
        print(f"  After radius filter: {len(mesh_trunk.vertices):,} verts  "
              f"{len(mesh_trunk.triangles):,} tri", flush=True)

    # ── 6. Remove floating fragments ─────────────────────────────────────────
    mesh_trunk = filter_fragments(mesh_trunk, min_triangles)
    n_trunk    = len(mesh_trunk.vertices)
    if verbose:
        print(f"  After fragment removal: {n_trunk:,} verts  "
              f"{len(mesh_trunk.triangles):,} tri", flush=True)

    if n_trunk == 0:
        result["msg"] = f"trunk mesh rong sau filter R={trunk_radius}m — thu tang --radius"
        return result

    # ── 6. Save ───────────────────────────────────────────────────────────────
    mesh_path = os.path.join(out_dir, "trunk_mesh_tsdf.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh_trunk, write_vertex_colors=True)

    elapsed = time.time() - t0
    result.update({
        "ok":  True,
        "msg": (f"verts:{n_trunk:,} tris:{len(mesh_trunk.triangles):,} "
                f"[{elapsed:.0f}s]"),
    })
    if verbose:
        print(f"  Saved {mesh_path}  [{elapsed:.0f}s]", flush=True)
    return result


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_batch(raw_data_dir, scan_filter=None, skip_existing=False,
              voxel_size=0.004, conf_thresh=1, max_depth=2.0,
              trunk_radius=0.65, min_triangles=500, frame_step=1):

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
    print(f"\n{'='*62}")
    print(f"TSDF Mesh  |  voxel={voxel_size*100:.1f}cm  "
          f"conf>={conf_thresh}  step={frame_step}  R={trunk_radius}m"
          f"  |  {total} scans")
    print(f"{'='*62}\n")

    ok_count = skip_count = 0
    results  = []

    for i, scan_dir in enumerate(dirs):
        name      = os.path.basename(scan_dir)
        mesh_path = os.path.join(scan_dir, "output", "trunk_mesh_tsdf.ply")

        if skip_existing and os.path.exists(mesh_path):
            print(f"[{i+1:3d}/{total}] {name} -- SKIP")
            skip_count += 1
            continue

        print(f"[{i+1:3d}/{total}] {name}")
        r = process_scan(scan_dir,
                         voxel_size    = voxel_size,
                         conf_thresh   = conf_thresh,
                         max_depth     = max_depth,
                         trunk_radius  = trunk_radius,
                         min_triangles = min_triangles,
                         frame_step    = frame_step,
                         verbose       = True)
        results.append(r)

        status = "OK  " if r["ok"] else "FAIL"
        print(f"  {status} {r['msg']}\n")
        if r["ok"]:
            ok_count += 1

    print(f"{'='*62}")
    print(f"DONE: {ok_count}/{total - skip_count} OK  ({skip_count} skipped)")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['name']}: {r['msg']}")

    summary_path = os.path.join(raw_data_dir, "tsdf_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"voxel": voxel_size, "frame_step": frame_step,
                   "results": results}, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Stray Scanner RGB-D frames -> TSDF colored trunk mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  python 02_mesh_tsdf.py --scan 26Q1                   # default: 5mm voxel
  python 02_mesh_tsdf.py --scan 26Q1 --voxel 0.003     # 3mm, dep hon, cham hon
  python 02_mesh_tsdf.py --scan 26Q1 --step 3          # bo qua 2/3 frame, nhanh 3x
  python 02_mesh_tsdf.py --all --step 2 --skip-existing
""")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan",  metavar="NAME",  help="Ten scan, vd: 26Q1")
    g.add_argument("--all",   action="store_true", help="Chay toan bo scan")

    p.add_argument("--voxel",     type=float, default=0.005,
                   help="TSDF voxel size (m). 0.003=dep/cham  0.005=default  0.008=nhanh/tho")
    p.add_argument("--confidence",type=int,   default=1, choices=[0, 1, 2],
                   help="Nguong confidence (0=giu het, 1=trung binh, 2=chi giu tot nhat)")
    p.add_argument("--max-depth",      type=float, default=2.0,
                   help="Max depth (m) (default 2.0)")
    p.add_argument("--radius",         type=float, default=0.65,
                   help="Radius camera-to-trunk filter (m, default 0.65)")
    p.add_argument("--min-component",  type=int,   default=500,
                   help="Xoa fragment co < N triangles (default 500)")
    p.add_argument("--step",      type=int,   default=1,
                   help="Lay 1 trong N frame (1=all, 2=nhanh 2x, 3=nhanh 3x)")
    p.add_argument("--raw-data",  default=RAW_DATA_DIR)
    p.add_argument("--skip-existing", action="store_true")

    args = p.parse_args()
    run_batch(
        args.raw_data,
        scan_filter   = args.scan if args.scan else None,
        skip_existing = args.skip_existing,
        voxel_size    = args.voxel,
        conf_thresh   = args.confidence,
        max_depth     = args.max_depth,
        trunk_radius  = args.radius,
        min_triangles = args.min_component,
        frame_step    = args.step,
    )


if __name__ == "__main__":
    main()
