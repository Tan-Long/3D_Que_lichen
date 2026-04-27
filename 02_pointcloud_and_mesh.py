#!/usr/bin/env python3
"""
Step 2: Point Cloud Visualization, Camera Trajectory, and TSDF Mesh Generation.

Reads Stray Scanner data (camera_matrix.csv, odometry.csv, rgb/, depth/, confidence/)
and produces:
  - A combined point cloud (saved as pointcloud.ply)
  - A camera trajectory visualization
  - A TSDF-fused mesh (saved as mesh.ply)

Based on the approach used by kekeblom/StrayVisualizer.

Usage:
    python 02_pointcloud_and_mesh.py c9ae76fff8 [--skip-every N] [--voxel-size V]
"""

import argparse
import csv
import gc
import os

import cv2
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _frame_ids_in_dir(dir_path: str, ext: str) -> set[str]:
    if not os.path.isdir(dir_path):
        return set()
    out: set[str] = set()
    for f in os.listdir(dir_path):
        if f.lower().endswith(ext.lower()):
            out.add(os.path.splitext(f)[0])
    return out


def _filter_poses_to_available_frames(dataset_dir: str, poses: list[dict]) -> list[dict]:
    rgb_ids = _frame_ids_in_dir(os.path.join(dataset_dir, "rgb"), ".jpg")
    depth_ids = _frame_ids_in_dir(os.path.join(dataset_dir, "depth"), ".png")
    conf_ids = _frame_ids_in_dir(os.path.join(dataset_dir, "confidence"), ".png")
    common = rgb_ids & depth_ids
    if conf_ids:
        common &= conf_ids

    filtered = [p for p in poses if p["frame"] in common]
    if len(filtered) != len(poses):
        missing = sorted({p["frame"] for p in poses} - common)
        print(
            f"NOTE: Filtering poses to frames with data present. "
            f"Using {len(filtered)}/{len(poses)} frames. "
            f"Missing examples: {missing[:5]}"
        )
    return filtered


def load_intrinsics(dataset_dir: str) -> np.ndarray:
    """Load the 3x3 camera intrinsic matrix from camera_matrix.csv."""
    path = os.path.join(dataset_dir, "camera_matrix.csv")
    K = np.loadtxt(path, delimiter=",")
    assert K.shape == (3, 3), f"Expected 3x3 intrinsic matrix, got {K.shape}"
    return K


def load_odometry(dataset_dir: str) -> list[dict]:
    """Load per-frame camera poses from odometry.csv.

    Returns a list of dicts with keys: timestamp, frame, position (3,), quaternion (4,).
    Quaternion order from file: qx, qy, qz, qw.
    """
    path = os.path.join(dataset_dir, "odometry.csv")
    poses = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            poses.append({
                "timestamp": float(row["timestamp"]),
                "frame": row["frame"].strip(),
                "position": np.array([
                    float(row["x"]), float(row["y"]), float(row["z"])
                ]),
                "quaternion": np.array([
                    float(row["qx"]), float(row["qy"]),
                    float(row["qz"]), float(row["qw"])
                ]),
            })
    return poses


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (qx, qy, qz, qw) to a 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
    ])


def pose_to_extrinsic(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Convert position + quaternion to a 4x4 camera-to-world transform."""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    T[:3, 3] = position
    return T


def load_depth(path: str) -> np.ndarray:
    """Load a 16-bit depth map and convert to meters.

    Stray Scanner stores depth in millimeters as 16-bit PNG.
    """
    depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    return depth_raw.astype(np.float64) / 1000.0  # mm -> meters


def load_confidence(path: str) -> np.ndarray:
    """Load confidence map. Values: 0 (low), 1 (medium), 2 (high)."""
    conf = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if conf is None:
        raise FileNotFoundError(f"Cannot read confidence image: {path}")
    return conf


# ---------------------------------------------------------------------------
# Intrinsics Helper — scale to depth resolution
# ---------------------------------------------------------------------------

def scale_intrinsics(K: np.ndarray, from_w: int, from_h: int,
                     to_w: int, to_h: int) -> np.ndarray:
    """Scale intrinsics from one resolution to another."""
    sx = to_w / from_w
    sy = to_h / from_h
    K_scaled = K.copy()
    K_scaled[0, :] *= sx  # fx, cx
    K_scaled[1, :] *= sy  # fy, cy
    return K_scaled


def make_o3d_intrinsic(K: np.ndarray, width: int, height: int):
    """Create Open3D PinholeCameraIntrinsic from a 3x3 matrix."""
    return o3d.camera.PinholeCameraIntrinsic(
        width, height,
        K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    )


def _pick_work_resolution(
    rgb_w: int, rgb_h: int, depth_w: int, depth_h: int,
    max_width: int = 640,
) -> tuple[int, int]:
    """Choose an intermediate resolution for RGBD fusion.

    Upscale depth to RGB resolution (capped at max_width) so that
    colour detail from the original frames is preserved while keeping
    memory usage manageable.
    """
    max_w = min(rgb_w, max_width)
    scale = max_w / rgb_w
    work_w = int(rgb_w * scale)
    work_h = int(rgb_h * scale)
    # Make sure they are even (some codecs need it)
    work_w = work_w if work_w % 2 == 0 else work_w + 1
    work_h = work_h if work_h % 2 == 0 else work_h + 1
    return work_w, work_h


# ---------------------------------------------------------------------------
# Point Cloud Generation
# ---------------------------------------------------------------------------

def create_rgbd_pointcloud(
    dataset_dir: str,
    skip_every: int = 5,
    confidence_threshold: int = 1,
    max_depth: float = 3.0,
    max_width: int = 640,
) -> tuple[o3d.geometry.PointCloud, list[np.ndarray]]:
    """Create a combined point cloud with high-resolution RGB colours.

    Instead of down-scaling RGB to the tiny depth resolution (256x192),
    we up-scale depth to a working resolution close to the original RGB
    so that each 3-D point keeps sharp colour from the camera frame.
    """
    K = load_intrinsics(dataset_dir)
    poses = load_odometry(dataset_dir)
    poses = _filter_poses_to_available_frames(dataset_dir, poses)
    if not poses:
        raise RuntimeError("No frames available with matching rgb/depth/(confidence) files.")

    # Determine native resolutions
    sample_depth = load_depth(os.path.join(dataset_dir, "depth", f"{poses[0]['frame']}.png"))
    depth_h, depth_w = sample_depth.shape[:2]

    sample_rgb_path = os.path.join(dataset_dir, "rgb", f"{poses[0]['frame']}.jpg")
    sample_rgb = cv2.imread(sample_rgb_path)
    if sample_rgb is None:
        raise FileNotFoundError(
            f"No RGB frame found at {sample_rgb_path}. "
            "Did you run 01_extract_rgb_frames.py first?"
        )
    rgb_h, rgb_w = sample_rgb.shape[:2]

    # Pick a working resolution that preserves colour detail
    work_w, work_h = _pick_work_resolution(
        rgb_w, rgb_h, depth_w, depth_h, max_width=max_width
    )
    K_work = scale_intrinsics(K, rgb_w, rgb_h, work_w, work_h)
    intrinsic = make_o3d_intrinsic(K_work, work_w, work_h)

    print(f"RGB resolution   : {rgb_w}x{rgb_h}")
    print(f"Depth resolution : {depth_w}x{depth_h}")
    print(f"Work resolution  : {work_w}x{work_h}")
    print(f"Scaled intrinsics:\n{K_work}\n")

    combined_pcd = o3d.geometry.PointCloud()
    camera_positions = []

    for i, pose in enumerate(poses):
        if i % skip_every != 0:
            continue

        frame_id = pose["frame"]
        depth_path = os.path.join(dataset_dir, "depth", f"{frame_id}.png")
        rgb_path = os.path.join(dataset_dir, "rgb", f"{frame_id}.jpg")
        conf_path = os.path.join(dataset_dir, "confidence", f"{frame_id}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        # Load data
        depth = load_depth(depth_path)
        confidence = load_confidence(conf_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Mask out low-confidence and far pixels at native depth res
        mask = (confidence >= confidence_threshold) & (depth > 0) & (depth < max_depth)
        depth[~mask] = 0

        # Upscale depth & mask to working resolution (NEAREST to avoid
        # blending depth values across edges)
        depth_up = cv2.resize(depth, (work_w, work_h),
                              interpolation=cv2.INTER_NEAREST)
        # Downscale RGB to working resolution (LANCZOS for sharp colour)
        rgb_work = cv2.resize(rgb, (work_w, work_h),
                              interpolation=cv2.INTER_LANCZOS4)

        # Create RGBD image at working resolution
        color_o3d = o3d.geometry.Image(rgb_work.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_up.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        T_cam2world = pose_to_extrinsic(pose["position"], pose["quaternion"])
        pcd.transform(T_cam2world)

        combined_pcd += pcd
        camera_positions.append(pose["position"])

        del depth, confidence, rgb, depth_up, rgb_work
        del color_o3d, depth_o3d, rgbd, pcd

        if (i // skip_every) % 20 == 0:
            print(f"  Processed frame {frame_id} ({i+1}/{len(poses)})")
            gc.collect()

    # Downsample — use a finer voxel to keep more detail
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.003)
    print(f"\nCombined point cloud: {len(combined_pcd.points)} points")

    return combined_pcd, camera_positions


# ---------------------------------------------------------------------------
# Camera Trajectory Visualization
# ---------------------------------------------------------------------------

def create_trajectory_lineset(camera_positions: list[np.ndarray]) -> o3d.geometry.LineSet:
    """Create a LineSet showing the camera path."""
    points = np.array(camera_positions)
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [[1, 0, 0] for _ in lines]  # red trajectory

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# ---------------------------------------------------------------------------
# TSDF Mesh Generation
# ---------------------------------------------------------------------------

def generate_mesh_tsdf(
    dataset_dir: str,
    voxel_size: float = 0.005,
    sdf_trunc: float = 0.02,
    skip_every: int = 3,
    confidence_threshold: int = 1,
    max_depth: float = 3.0,
    max_width: int = 640,
) -> o3d.geometry.TriangleMesh:
    """Fuse RGBD frames into a TSDF volume, extract mesh, then refine
    vertex colours by projecting back into the original high-res RGB frames.
    """
    K = load_intrinsics(dataset_dir)
    poses = load_odometry(dataset_dir)
    poses = _filter_poses_to_available_frames(dataset_dir, poses)
    if not poses:
        raise RuntimeError("No frames available with matching rgb/depth/(confidence) files.")

    sample_depth = load_depth(os.path.join(dataset_dir, "depth", f"{poses[0]['frame']}.png"))
    depth_h, depth_w = sample_depth.shape[:2]

    sample_rgb = cv2.imread(os.path.join(dataset_dir, "rgb", f"{poses[0]['frame']}.jpg"))
    rgb_h, rgb_w = sample_rgb.shape[:2]

    # TSDF integration uses working resolution for better colour
    work_w, work_h = _pick_work_resolution(
        rgb_w, rgb_h, depth_w, depth_h, max_width=max_width
    )
    K_work = scale_intrinsics(K, rgb_w, rgb_h, work_w, work_h)
    intrinsic = make_o3d_intrinsic(K_work, work_w, work_h)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    total_to_process = len([p for i, p in enumerate(poses) if i % skip_every == 0])
    print(f"TSDF volume : voxel={voxel_size}, trunc={sdf_trunc}")
    print(f"Work resolution: {work_w}x{work_h}")
    print(f"Processing {total_to_process} frames...\n", flush=True)

    processed = 0
    for i, pose in enumerate(poses):
        if i % skip_every != 0:
            continue

        frame_id = pose["frame"]
        depth_path = os.path.join(dataset_dir, "depth", f"{frame_id}.png")
        rgb_path = os.path.join(dataset_dir, "rgb", f"{frame_id}.jpg")
        conf_path = os.path.join(dataset_dir, "confidence", f"{frame_id}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        depth = load_depth(depth_path)
        confidence = load_confidence(conf_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        mask = (confidence >= confidence_threshold) & (depth > 0) & (depth < max_depth)
        depth[~mask] = 0

        # Upscale depth + downscale RGB to working resolution
        depth_up = cv2.resize(depth, (work_w, work_h),
                              interpolation=cv2.INTER_NEAREST)
        rgb_work = cv2.resize(rgb, (work_w, work_h),
                              interpolation=cv2.INTER_LANCZOS4)

        color_o3d = o3d.geometry.Image(rgb_work.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_up.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        T_cam2world = pose_to_extrinsic(pose["position"], pose["quaternion"])
        T_world2cam = np.linalg.inv(T_cam2world)
        volume.integrate(rgbd, intrinsic, T_world2cam)

        # Free per-frame memory immediately
        del depth, confidence, rgb, depth_up, rgb_work
        del color_o3d, depth_o3d, rgbd

        processed += 1
        if processed % 5 == 0 or processed == 1:
            print(f"  Integrated frame {frame_id} ({processed}/{total_to_process})",
                  flush=True)
        # Periodic GC to keep resident memory in check
        if processed % 20 == 0:
            gc.collect()

    print("\nExtracting mesh from TSDF volume...", flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles",
          flush=True)

    # ------------------------------------------------------------------
    # Refine vertex colours using original full-res RGB frames
    # ------------------------------------------------------------------
    print("\nRefining vertex colours from high-res RGB...", flush=True)
    mesh = _refine_vertex_colors(mesh, dataset_dir, K, poses,
                                 rgb_w, rgb_h, skip_every)

    return mesh


def _refine_vertex_colors(
    mesh: o3d.geometry.TriangleMesh,
    dataset_dir: str,
    K: np.ndarray,
    poses: list[dict],
    rgb_w: int,
    rgb_h: int,
    skip_every: int = 2,
) -> o3d.geometry.TriangleMesh:
    """Re-colour every mesh vertex by projecting it into the original
    full-resolution RGB frames and averaging the sampled colours.

    For each vertex we:
      1. Transform it from world → camera coordinates.
      2. Project onto the image plane using full-res intrinsics.
      3. Check it lands inside the image and is in front of the camera.
      4. Bi-linearly sample the colour from the high-res RGB.
      5. Average across all frames that see the vertex.

    This replaces the blurry TSDF vertex colours with sharp originals.
    """
    vertices = np.asarray(mesh.vertices)          # (N, 3)
    n_verts = len(vertices)

    # Accumulators: sum of RGB and count per vertex
    color_sum = np.zeros((n_verts, 3), dtype=np.float64)
    color_cnt = np.zeros(n_verts, dtype=np.float64)

    # Homogeneous vertices (N, 4)
    verts_h = np.hstack([vertices, np.ones((n_verts, 1))])

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    frames_used = 0
    for i, pose in enumerate(poses):
        if i % skip_every != 0:
            continue

        frame_id = pose["frame"]
        rgb_path = os.path.join(dataset_dir, "rgb", f"{frame_id}.jpg")
        if not os.path.exists(rgb_path):
            continue

        rgb = cv2.imread(rgb_path)              # BGR, full-res
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # World-to-camera transform
        T_c2w = pose_to_extrinsic(pose["position"], pose["quaternion"])
        T_w2c = np.linalg.inv(T_c2w)

        # Project all vertices at once  (4x4 @ 4xN -> 4xN -> 3xN)
        cam = (T_w2c @ verts_h.T)[:3, :]       # (3, N)
        z = cam[2, :]

        # Only keep points in front of the camera
        valid = z > 0.05
        u = fx * cam[0, :] / z + cx
        v = fy * cam[1, :] / z + cy

        # Must land inside image (with 1px border for bilinear)
        valid &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        idx = np.where(valid)[0]

        if len(idx) == 0:
            continue

        # Bilinear sampling (stay in float64 to avoid rounding past bounds)
        uf = u[idx]
        vf = v[idx]
        u0 = np.floor(uf).astype(int)
        v0 = np.floor(vf).astype(int)
        u1 = np.minimum(u0 + 1, w - 1)
        v1 = np.minimum(v0 + 1, h - 1)
        du = (uf - u0).reshape(-1, 1)
        dv = (vf - v0).reshape(-1, 1)

        c00 = rgb[v0, u0].astype(np.float64)
        c01 = rgb[v0, u1].astype(np.float64)
        c10 = rgb[v1, u0].astype(np.float64)
        c11 = rgb[v1, u1].astype(np.float64)
        sampled = (c00 * (1 - du) * (1 - dv) +
                   c01 * du * (1 - dv) +
                   c10 * (1 - du) * dv +
                   c11 * du * dv)

        color_sum[idx] += sampled
        color_cnt[idx] += 1.0
        frames_used += 1

        if frames_used % 10 == 0:
            print(f"  Colour-sampled {frames_used} frames...", flush=True)

    print(f"  Used {frames_used} frames for colour refinement", flush=True)

    # Average and assign
    painted = color_cnt > 0
    colors = np.asarray(mesh.vertex_colors).copy()  # fallback: keep TSDF colour
    colors[painted] = color_sum[painted] / color_cnt[painted, np.newaxis] / 255.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    unpainted_pct = 100.0 * (~painted).sum() / n_verts
    if unpainted_pct > 0:
        print(f"  {unpainted_pct:.1f}% vertices kept TSDF fallback colour")

    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _crop_mesh_to_mask(
    mesh: o3d.geometry.TriangleMesh,
    mask_path: str,
    distance_threshold: float = 0.03,
) -> o3d.geometry.TriangleMesh:
    """Remove vertices from *mesh* that are far from the cleaned mask geometry.

    Loads *mask_path* (PLY — mesh or point cloud), builds a KD-tree from its
    vertices, and keeps only mesh vertices within *distance_threshold* of at
    least one mask vertex.
    """
    # Load mask geometry (could be mesh or point cloud)
    mask_mesh = o3d.io.read_triangle_mesh(mask_path)
    mask_pts = np.asarray(mask_mesh.vertices)
    if len(mask_pts) == 0:
        # Try as point cloud
        mask_pcd = o3d.io.read_point_cloud(mask_path)
        mask_pts = np.asarray(mask_pcd.points)
    if len(mask_pts) == 0:
        print("WARNING: Cleaned mask has no vertices, skipping crop.")
        return mesh

    print(f"Mask has {len(mask_pts)} vertices, threshold={distance_threshold}m",
          flush=True)

    # Build KD-tree
    mask_pcd = o3d.geometry.PointCloud()
    mask_pcd.points = o3d.utility.Vector3dVector(mask_pts)
    tree = o3d.geometry.KDTreeFlann(mask_pcd)

    verts = np.asarray(mesh.vertices)
    keep = np.zeros(len(verts), dtype=bool)
    thresh_sq = distance_threshold ** 2
    n_total = len(verts)

    for i in range(n_total):
        _, _, dists = tree.search_knn_vector_3d(verts[i], 1)
        if dists[0] < thresh_sq:
            keep[i] = True
        if (i + 1) % 10000 == 0:
            print(f"  Spatial filter: ({i+1}/{n_total})", flush=True)

    remove_mask = ~keep
    n_before = len(mesh.vertices)
    mesh.remove_vertices_by_mask(remove_mask.tolist())
    n_after = len(mesh.vertices)
    print(f"Cropped mesh: {n_before} -> {n_after} vertices "
          f"({n_before - n_after} removed)", flush=True)

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Generate point cloud and mesh from Stray Scanner data"
    )
    parser.add_argument("dataset", help="Path to dataset folder (e.g., c9ae76fff8)")
    parser.add_argument("--skip-every", type=int, default=5,
                        help="Process every N-th frame for point cloud (default: 5)")
    parser.add_argument("--voxel-size", type=float, default=0.005,
                        help="TSDF voxel size in meters (default: 0.005)")
    parser.add_argument("--max-depth", type=float, default=3.0,
                        help="Max depth in meters (default: 3.0)")
    parser.add_argument("--max-width", type=int, default=640,
                        help="Working RGB width for processing speed/quality tradeoff (default: 640)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip interactive visualization")
    parser.add_argument("--refine-from", type=str, default=None,
                        help="Path to cleaned PLY for spatial-mask refinement")
    args = parser.parse_args()

    dataset_dir = args.dataset
    output_dir = os.path.join(dataset_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Refine mode: skip pointcloud, high-quality TSDF + crop + RGB ----
    if args.refine_from:
        print("=" * 60, flush=True)
        print("REFINE MODE: High-quality mesh from cleaned mask", flush=True)
        print("=" * 60, flush=True)

        mesh = generate_mesh_tsdf(
            dataset_dir,
            voxel_size=args.voxel_size,
            sdf_trunc=args.voxel_size * 4,
            skip_every=args.skip_every,
            max_depth=args.max_depth,
            max_width=args.max_width,
        )

        # Crop to cleaned mask region
        print("\nCropping to cleaned region...", flush=True)
        mesh = _crop_mesh_to_mask(mesh, args.refine_from)

        refined_path = os.path.join(output_dir, "refined_mesh.ply")
        o3d.io.write_triangle_mesh(refined_path, mesh)
        print(f"\nSaved refined mesh to {refined_path}", flush=True)
        print(f"  Vertices:  {len(mesh.vertices)}", flush=True)
        print(f"  Triangles: {len(mesh.triangles)}", flush=True)
        return

    # ---- Normal mode ----

    # --- Point Cloud ---
    print("=" * 60)
    print("STEP 1: Generating Point Cloud")
    print("=" * 60)
    pcd, cam_positions = create_rgbd_pointcloud(
        dataset_dir,
        skip_every=args.skip_every,
        max_depth=args.max_depth,
        max_width=args.max_width,
    )
    pcd_path = os.path.join(output_dir, "pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud to {pcd_path}")

    # --- Mesh ---
    print("\n" + "=" * 60)
    print("STEP 2: Generating TSDF Mesh")
    print("=" * 60)
    mesh = generate_mesh_tsdf(
        dataset_dir,
        voxel_size=args.voxel_size,
        sdf_trunc=args.voxel_size * 4,
        skip_every=args.skip_every,
        max_depth=args.max_depth,
        max_width=args.max_width,
    )
    mesh_path = os.path.join(output_dir, "mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Saved mesh to {mesh_path}")

    # --- Visualization ---
    if not args.no_visualize:
        print("\n" + "=" * 60)
        print("STEP 3: Launching Visualization")
        print("=" * 60)
        print("Close the window to proceed to the next view.\n")

        trajectory = create_trajectory_lineset(cam_positions)

        # Show point cloud + trajectory
        print("Showing: Point Cloud + Camera Trajectory")
        o3d.visualization.draw_geometries(
            [pcd, trajectory],
            window_name="Point Cloud + Trajectory",
            width=1280, height=720,
        )

        # Show mesh
        print("Showing: TSDF Mesh")
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="TSDF Mesh",
            width=1280, height=720,
        )

    print("\nDone! Output files:")
    print(f"  Point cloud: {pcd_path}")
    print(f"  Mesh:        {mesh_path}")


if __name__ == "__main__":
    main()
