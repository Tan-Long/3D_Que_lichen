#!/usr/bin/env python3
"""
Step 2: Clean trunk PCD → colored 3D mesh.

Pipeline cho mỗi scan:
  1. Load trunk_pointcloud.ply (output của 01_clean_pointcloud.py)
  2. Remove large flat planes (RANSAC) — loại tường/nền phẳng còn sót
  3. Statistical Outlier Removal — loại điểm rác lơ lửng
  4. Keep largest cluster (DBSCAN) — chỉ giữ thân cây chính
  5. Estimate normals (orient toward camera centroid)
  6. Poisson Surface Reconstruction → mesh
  7. Trim low-density mesh faces
  8. Transfer color từ PCD sang mesh vertices (nearest-neighbor)
  9. Save trunk_mesh.ply + trunk_clean.ply

Usage:
    python 02_mesh.py --scan 26Q1
    python 02_mesh.py --all
    python 02_mesh.py --all --skip-existing
    python 02_mesh.py --all --depth 9          # Poisson depth (8-11, default 9)
    python 02_mesh.py --all --no-cluster       # bo qua buoc DBSCAN
"""

import argparse, json, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def remove_planes(pcd: o3d.geometry.PointCloud,
                  min_ratio: float = 0.05,
                  max_planes: int = 3,
                  dist_thresh: float = 0.015) -> o3d.geometry.PointCloud:
    """
    Lặp RANSAC để loại các mặt phẳng lớn (tường, sàn, nền).
    Chỉ loại nếu mặt phẳng chiếm >= min_ratio tổng điểm.
    """
    remaining = pcd
    n_orig = len(pcd.points)
    for _ in range(max_planes):
        if len(remaining.points) < 500:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=500,
        )
        ratio = len(inliers) / max(len(remaining.points), 1)
        if ratio < min_ratio:
            break   # mặt phẳng quá nhỏ → không loại nữa
        remaining = remaining.select_by_index(inliers, invert=True)

    removed = n_orig - len(remaining.points)
    return remaining, removed


def remove_outliers(pcd: o3d.geometry.PointCloud,
                    nb_neighbors: int = 20,
                    std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """Statistical Outlier Removal."""
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl, len(pcd.points) - len(ind)


def keep_largest_cluster(pcd: o3d.geometry.PointCloud,
                         eps: float = 0.05,
                         min_samples: int = 10) -> o3d.geometry.PointCloud:
    """DBSCAN → giữ cluster lớn nhất (thân cây)."""
    pts = np.asarray(pcd.points)
    # Voxel down trước để DBSCAN nhanh hơn
    vsize = 0.02
    pcd_ds = pcd.voxel_down_sample(vsize)
    labels = np.array(pcd_ds.cluster_dbscan(
        eps=eps, min_points=min_samples, print_progress=False))

    if labels.max() < 0:
        return pcd, 0  # không tìm được cluster nào

    # Cluster lớn nhất
    counts = np.bincount(labels[labels >= 0])
    best   = int(np.argmax(counts))

    # Build KDTree từ downsampled points của cluster đó
    ds_pts    = np.asarray(pcd_ds.points)
    best_pts  = ds_pts[labels == best]
    tree      = cKDTree(best_pts)

    # Map ngược về full-res PCD: giữ điểm trong eps của cluster
    full_pts  = np.asarray(pcd.points)
    dists, _  = tree.query(full_pts, k=1, workers=-1)
    keep      = np.where(dists <= eps * 1.5)[0]

    removed = len(pcd.points) - len(keep)
    return pcd.select_by_index(keep), removed


# ── Mesh reconstruction ───────────────────────────────────────────────────────

def estimate_normals(pcd: o3d.geometry.PointCloud,
                     radius: float = 0.05,
                     max_nn: int = 30) -> o3d.geometry.PointCloud:
    """Tính normal và orient về phía centroid."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
    # Orient: nhìn về centroid (camera thường xung quanh thân cây)
    centroid = np.asarray(pcd.points).mean(axis=0)
    pcd.orient_normals_towards_camera_location(centroid)
    return pcd


def poisson_mesh(pcd: o3d.geometry.PointCloud,
                 depth: int = 9) -> o3d.geometry.TriangleMesh:
    """Poisson Surface Reconstruction."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False)
    return mesh, np.asarray(densities)


def trim_low_density(mesh: o3d.geometry.TriangleMesh,
                     densities: np.ndarray,
                     quantile: float = 0.05) -> o3d.geometry.TriangleMesh:
    """Loại các mặt tam giác ở vùng mật độ thấp (viền ngoài không tin cậy)."""
    thresh  = np.quantile(densities, quantile)
    keep_v  = densities > thresh
    mesh.remove_vertices_by_mask(~keep_v)   # in-place, returns None in Open3D >= 0.16
    return mesh


def transfer_colors_pcd_to_mesh(
        mesh: o3d.geometry.TriangleMesh,
        pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """
    Gán màu cho mỗi vertex của mesh bằng màu của điểm PCD gần nhất.
    """
    if not pcd.has_colors():
        return mesh

    pcd_pts  = np.asarray(pcd.points,  dtype=np.float32)
    pcd_cols = np.asarray(pcd.colors,  dtype=np.float32)   # [0,1]
    mesh_pts = np.asarray(mesh.vertices, dtype=np.float32)

    tree   = cKDTree(pcd_pts)
    _, idx = tree.query(mesh_pts, k=1, workers=-1)

    mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_cols[idx])
    return mesh


# ── Per-scan pipeline ─────────────────────────────────────────────────────────

def process_scan(scan_dir: str,
                 poisson_depth: int = 9,
                 use_planes: bool = False,
                 use_cluster: bool = False,
                 max_poisson_pts: int = 500_000,
                 verbose: bool = True) -> dict:
    name    = os.path.basename(scan_dir)
    out_dir = os.path.join(scan_dir, "output")
    trunk_in = os.path.join(out_dir, "trunk_pointcloud.ply")

    result  = {"name": name, "ok": False, "msg": ""}

    if not os.path.exists(trunk_in):
        result["msg"] = "trunk_pointcloud.ply not found — chay 01 truoc"
        return result

    t0 = time.time()

    # 1. Load
    pcd = o3d.io.read_point_cloud(trunk_in)
    n0  = len(pcd.points)
    if verbose:
        print(f"  Load: {n0:,} pts", flush=True)

    if n0 < 100:
        result["msg"] = f"too few points ({n0})"
        return result

    # 2. (optional) Remove large flat planes — TẮT mặc định vì step 01 đã clean
    #    Bật bằng --planes nếu vẫn còn tường / sàn dính vào
    if use_planes:
        pcd, n_plane = remove_planes(pcd, min_ratio=0.25, max_planes=2)
        if verbose and n_plane:
            print(f"  Plane removal: -{n_plane:,} pts -> {len(pcd.points):,}", flush=True)

    # 3. Statistical Outlier Removal — loại điểm lơ lửng cô lập
    pcd, n_sor = remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
    if verbose:
        print(f"  SOR: -{n_sor:,} pts -> {len(pcd.points):,}", flush=True)

    # 4. (optional) Keep largest cluster — TẮT mặc định vì data đã sạch từ step 01
    if use_cluster and len(pcd.points) > 200:
        pcd, n_clust = keep_largest_cluster(pcd, eps=0.08, min_samples=10)
        if verbose:
            print(f"  Cluster: -{n_clust:,} pts -> {len(pcd.points):,}", flush=True)

    n_clean = len(pcd.points)
    if n_clean < 100:
        result["msg"] = f"too few after cleaning ({n_clean})"
        return result

    # Save clean PCD
    clean_path = os.path.join(out_dir, "trunk_clean.ply")
    o3d.io.write_point_cloud(clean_path, pcd)

    # 5. Cap số điểm trước Poisson — quá nhiều sẽ tốn RAM + chậm không cần thiết
    #    Poisson depth=9 → grid 512³, độ phân giải ~4mm, không cần >500K input
    if n_clean > max_poisson_pts:
        vol   = max(np.prod(pcd.get_axis_aligned_bounding_box().get_extent()), 1e-6)
        vsize = max((vol / max_poisson_pts) ** (1/3), 0.003)
        pcd   = pcd.voxel_down_sample(vsize)
        if verbose:
            print(f"  Downsample for Poisson: {n_clean:,} -> {len(pcd.points):,} pts "
                  f"(voxel={vsize:.4f}m)", flush=True)

    n_poisson = len(pcd.points)

    # 6. Estimate normals
    pcd = estimate_normals(pcd)

    # 7. Poisson reconstruction
    if verbose:
        print(f"  Poisson depth={poisson_depth} ({n_poisson:,} pts) ...", flush=True)
    mesh, densities = poisson_mesh(pcd, depth=poisson_depth)
    if verbose:
        print(f"  Mesh raw: {len(mesh.vertices):,} verts {len(mesh.triangles):,} tri",
              flush=True)

    # 8. Trim low-density faces
    mesh = trim_low_density(mesh, densities, quantile=0.05)
    if verbose:
        print(f"  Mesh trimmed: {len(mesh.vertices):,} verts {len(mesh.triangles):,} tri",
              flush=True)

    # 9. Transfer màu từ PCD sang mesh (dùng full clean PCD trước khi cap)
    pcd_full = o3d.io.read_point_cloud(clean_path)
    mesh = transfer_colors_pcd_to_mesh(mesh, pcd_full)

    # 10. Post-process mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Save mesh
    mesh_path = os.path.join(out_dir, "trunk_mesh.ply")
    o3d.io.write_point_cloud(clean_path, pcd)   # update clean PCD
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    # Meta
    elapsed = time.time() - t0
    meta = {
        "name": name,
        "n_raw": n0, "n_clean": n_clean,
        "n_verts": len(mesh.vertices),
        "n_tris":  len(mesh.triangles),
        "poisson_depth": poisson_depth,
        "elapsed_s": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "mesh_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    result.update({
        "ok":  True,
        "msg": (f"clean:{n_clean:,} verts:{len(mesh.vertices):,} "
                f"tris:{len(mesh.triangles):,} [{elapsed:.0f}s]"),
    })
    if verbose:
        print(f"  Saved {mesh_path}  [{elapsed:.0f}s]", flush=True)
    return result


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_batch(raw_data_dir, poisson_depth, scan_filter=None,
              skip_existing=False, use_planes=False, use_cluster=False,
              max_poisson_pts=500_000):
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
    flags = []
    if use_planes:  flags.append("planes")
    if use_cluster: flags.append("cluster")
    flag_str = f"  [{', '.join(flags)}]" if flags else "  [SOR only]"
    print(f"\n{'='*60}")
    print(f"Trunk -> Mesh  |  depth={poisson_depth}  cap={max_poisson_pts//1000}K{flag_str}  |  {total} scans")
    print(f"{'='*60}\n")

    ok_count = skip_count = 0
    results  = []

    for i, scan_dir in enumerate(dirs):
        name      = os.path.basename(scan_dir)
        mesh_path = os.path.join(scan_dir, "output", "trunk_mesh.ply")

        if skip_existing and os.path.exists(mesh_path):
            print(f"[{i+1:3d}/{total}] {name} -- SKIP")
            skip_count += 1
            continue

        print(f"[{i+1:3d}/{total}] {name}")
        r = process_scan(scan_dir, poisson_depth,
                         use_planes=use_planes, use_cluster=use_cluster,
                         max_poisson_pts=max_poisson_pts, verbose=True)
        results.append(r)

        if r["ok"]:
            ok_count += 1
            print(f"  OK  {r['msg']}")
        else:
            print(f"  FAIL  {r['msg']}")
        print()

    print(f"{'='*60}")
    print(f"DONE: {ok_count}/{total - skip_count} OK  ({skip_count} skipped)")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['name']}: {r['msg']}")

    with open(os.path.join(raw_data_dir, "mesh_summary.json"), "w") as f:
        json.dump({"depth": poisson_depth, "results": results}, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Trunk PCD -> colored mesh")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME", help="Ten scan, vd: 26Q1")
    g.add_argument("--all",  action="store_true")

    p.add_argument("--depth",        type=int,   default=9,
                   help="Poisson depth (8=nhanh/tho, 9=default, 10-11=chi tiet/cham)")
    p.add_argument("--raw-data",     default=RAW_DATA_DIR)
    p.add_argument("--skip-existing",action="store_true")
    p.add_argument("--planes",       action="store_true",
                   help="Bat plane removal (RANSAC) — mac dinh TAT vi step 01 da clean")
    p.add_argument("--cluster",      action="store_true",
                   help="Bat DBSCAN cluster — mac dinh TAT")
    p.add_argument("--max-pts",      type=int,   default=500_000,
                   help="So diem toi da truoc Poisson (default 500K)")

    args = p.parse_args()
    run_batch(args.raw_data, args.depth,
              scan_filter=args.scan if args.scan else None,
              skip_existing=args.skip_existing,
              use_planes=args.planes,
              use_cluster=args.cluster,
              max_poisson_pts=args.max_pts)


if __name__ == "__main__":
    main()
