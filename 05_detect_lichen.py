#!/usr/bin/env python3
"""
Step 05: Detect lichen + split upslope/downslope + tinh dien tich.

Pipeline:
  1. Load trunk_mesh_trimmed.ply
  2. Uoc tinh huong trong luc (gravity) tu camera poses
  3. Tim truc than cay (PCA)
  4. Chia mesh: upslope / downslope (theo gravity perp to trunk)
  5. Classify tung face: green_lichen / white_lichen / bark
  6. Tinh dien tich (cm2) moi phan
  7. Luu visualization mesh (mau ro rang) + JSON ket qua

Mau hien thi:
  - Lichen xanh  (upslope)   = #00FF00 xanh neon
  - Lichen xanh  (downslope) = #00CC88 xanh ngoc
  - Lichen trang (upslope)   = #FFFF00 vang neon
  - Lichen trang (downslope) = #FFA500 cam
  - Vo cay (upslope)         = #6495ED xanh duong nhat (tint)
  - Vo cay (downslope)       = #CD853F nau (tint)

Usage:
    python 05_detect_lichen.py --scan 26Q1
    python 05_detect_lichen.py --scan 26Q1 --show-sides   # chi hien thi upslope/downslope
    python 05_detect_lichen.py --all --skip-existing
"""

import argparse, json, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")

# Phan tram moi dau/cuoi truc cay se bi force = bark (tranh rac o ngon/goc)
# 0.05 = cat 5% moi dau (vd: cay dai 1m -> cat 5 cm moi dau)
END_ZONE_FRAC = 0.05


# ── Color helpers ─────────────────────────────────────────────────────────────

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N,3) float [0,1] -> hsv: (N,3) H[0,360] S[0,100] V[0,100]"""
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-9

    h = np.zeros(len(rgb))
    mask_r = (cmax == r)
    mask_g = (cmax == g)
    mask_b = (cmax == b)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    s = np.where(cmax > 0, delta / (cmax + 1e-9) * 100, 0)
    v = cmax * 100
    return np.stack([h, s, v], axis=1)


def classify_lichen(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (N,3) float [0,1]
    Returns label (N,): 0=bark  1=green_lichen  2=white_lichen

    Calibrated tu phan tich video thuc te (frame_03.jpg):
      White lichen (crustose tron): V=87% mean, S=16% mean, range S<47%, V>73%
      Bark:                         V=34%, S=45%
      Key: lichen SANG hon (V>65%) va IT BAO HOA hon (S<35%) so voi bark
    """
    hsv = rgb_to_hsv(rgb)
    H, S, V = hsv[:,0], hsv[:,1], hsv[:,2]

    # --- White/cream lichen (crustose tron) ---
    # Calibrated: PCD analysis cho "S<25, V>70" = 16.0% pts = dung voi video (16.7%)
    # Video thuc te: lichen V=87% mean, S=16% mean vs bark V=34%, S=45%
    # Key: V>70% (sang hon bark nhieu) VA S<25% (it bao hoa hon bark)
    is_vegetation = (H >= 65) & (H <= 150) & (S > 25)  # la cay xanh that su
    white = (
        (V >= 70) & (V <= 97) &
        (S < 25)  &
        ~is_vegetation
    )

    # --- Green lichen (algae + green crustose) ---
    # H=90±14, S=43±17, V=59±17 (tu PCD analysis) — ~1% of pts
    green = (
        (H >= 60)  & (H <= 155) &
        (S >= 25)  &
        (V >= 30)  & (V <= 92) &
        ~white   # green khong overlap white
    )

    labels = np.zeros(len(rgb), dtype=np.uint8)
    labels[white] = 2
    labels[green] = 1
    return labels


# ── Geometry helpers ──────────────────────────────────────────────────────────

def pca_axis(pts: np.ndarray) -> np.ndarray:
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[0]   # largest singular vector = trunk axis


def estimate_gravity(scan_dir: str) -> np.ndarray:
    """
    Tra ve huong 'len' (world up) trong Stray Scanner coordinate system.
    Xac nhan qua PCA cac mesh sach: World Y-axis = huong len.
    scan_dir: giu de tuong thich API cu.
    """
    return np.array([0.0, 1.0, 0.0])


def classify_lichen_local_contrast(pts: np.ndarray,
                                    cols: np.ndarray,
                                    k_local: int = 40) -> np.ndarray:
    """
    Adaptive Local Contrast Detection — ngưỡng tự động theo màu của từng cây.

    Vấn đề của ngưỡng tuyệt đối:
      - Cây tối (median_V~0.30): ngưỡng V>=0.68 quá cao → miss địa y
      - Cây sáng (median_V~0.55): ngưỡng V>=0.68 quá thấp → cả cây = địa y

    Giải pháp: địa y = điểm SÁNG HƠN VÀ NHẠT MÀU HƠN so với lân cận,
    với độ nhạy điều chỉnh theo median_V của toàn bộ scan.
    """
    from scipy.spatial import cKDTree

    # Compute HSV
    r, g, b = cols[:,0], cols[:,1], cols[:,2]
    cmax = np.maximum(np.maximum(r,g),b)
    cmin = np.minimum(np.minimum(r,g),b)
    delta_c = cmax - cmin + 1e-9
    h = np.zeros(len(cols), dtype=np.float32)
    mr=cmax==r; mg=cmax==g; mb=cmax==b
    h[mr] = 60*(((g[mr]-b[mr])/delta_c[mr])%6)
    h[mg] = 60*(((b[mg]-r[mg])/delta_c[mg])+2)
    h[mb] = 60*(((r[mb]-g[mb])/delta_c[mb])+4)
    s = np.where(cmax>0, delta_c/(cmax+1e-9), 0).astype(np.float32)
    v = cmax.astype(np.float32)

    # Global stats — dùng percentile 20 để estimate màu VỎ CÂY (không bị kéo bởi lichen sáng)
    global_med_V = float(np.percentile(v, 20))   # ~bark color (lichen là minority → p20 ≈ bark)
    global_std_V = float(np.std(v))
    print(f"      [adapt] bark_p20={global_med_V:.3f}  std_V={global_std_V:.3f}", flush=True)

    # KDTree cho local neighbor lookup
    tree = cKDTree(pts)
    _, idx_nbr = tree.query(pts, k=k_local+1, workers=-1)
    idx_nbr = idx_nbr[:, 1:]   # bỏ self

    # Local statistics
    local_mean_V = v[idx_nbr].mean(axis=1)
    local_std_V  = v[idx_nbr].std(axis=1)
    local_mean_S = s[idx_nbr].mean(axis=1)

    v_contrast = v - local_mean_V   # dương = sáng hơn lân cận
    s_drop     = local_mean_S - s   # dương = ít bão hoà hơn lân cận

    is_veg = (h >= 65) & (h <= 150) & (s > 0.25)

    # ── Adaptive thresholds theo loại cây ────────────────────────────────────
    # Phân loại cây theo màu vỏ (p20 = bark estimate)
    if global_med_V < 0.32:
        # Cây tối: nới lỏng V, s_drop thấp
        abs_v, abs_s = float(max(0.55, global_med_V + 1.8 * global_std_V)), 0.28
        v_sigma_k, v_sigma_base, req_s_drop = 0.8, 0.03, 0.03
    elif global_med_V < 0.45:
        # Cây trung bình (vỏ nâu trung): lichen phải nhạt HẲN lân cận (s_drop >= 0.08)
        abs_v, abs_s = float(max(0.63, global_med_V + 2.1 * global_std_V)), 0.24
        v_sigma_k, v_sigma_base, req_s_drop = 1.0, 0.04, 0.08
    else:
        # Cây sáng (vỏ nâu nhạt/vàng): s_drop là discriminator chính
        abs_v, abs_s = float(max(0.70, global_med_V + 2.5 * global_std_V)), 0.20
        v_sigma_k, v_sigma_base, req_s_drop = 1.2, 0.05, 0.08

    # Method A: Local contrast — sáng hơn lân cận VÀ nhạt hơn lân cận
    v_thr = np.maximum(v_sigma_base, v_sigma_k * local_std_V)
    local_white = (
        (v_contrast > v_thr) &
        (s          < local_mean_S - req_s_drop) &  # nhạt HƠN lân cận (không phải ≈)
        (s          < 0.42) &
        ~is_veg
    )

    # Method B: Absolute — địa y trắng rõ ràng (adaptive theo loại cây)
    abs_white = (v >= abs_v) & (s < abs_s) & ~is_veg

    white = local_white | abs_white

    labels = np.zeros(len(cols), dtype=np.uint8)
    labels[white] = 2
    return labels


def detect_on_pcd_transfer_to_mesh(pcd_path: str,
                                    verts: np.ndarray,
                                    k: int = 15) -> np.ndarray:
    """
    1. Detect lichen tren PCD bang local contrast (khong dung nguong V tuyet doi)
    2. Transfer sang mesh vertex: k nearest PCD neighbors, majority vote
       co uu tien diem sang nhat (goc chup tot nhat)
    Returns: vertex_labels (V,) uint8 — 0=bark 1=green 2=white_lichen
    """
    from scipy.spatial import cKDTree

    pcd  = o3d.io.read_point_cloud(pcd_path)
    pts  = np.asarray(pcd.points,  dtype=np.float32)
    cols = np.asarray(pcd.colors,  dtype=np.float32)

    # Cap PCD size — giới hạn 500K điểm để KDTree k=40 hoàn thành trong ~30s
    MAX_PTS = 500_000
    if len(pts) > MAX_PTS:
        idx_sub = np.random.choice(len(pts), MAX_PTS, replace=False)
        idx_sub.sort()
        pts  = pts[idx_sub]
        cols = cols[idx_sub]
        print(f"    PCD subsampled: {len(idx_sub):,} / {len(np.asarray(pcd.points)):,} pts", flush=True)

    pcd_v = np.max(cols, axis=1)

    # Local contrast classification
    print("    Local contrast analysis...", flush=True)
    pcd_labels = classify_lichen_local_contrast(pts, cols)
    n = len(pcd_labels)
    print(f"    PCD: white={( pcd_labels==2).sum():,} ({(pcd_labels==2).sum()/n*100:.1f}%)"
          f"  green={(pcd_labels==1).sum():,} ({(pcd_labels==1).sum()/n*100:.1f}%)",
          flush=True)

    # Transfer to mesh: k nearest + uu tien bright lichen
    tree = cKDTree(pts)
    _, idx = tree.query(verts.astype(np.float32), k=k, workers=-1)

    neighbor_labels = pcd_labels[idx]   # (N_verts, k)
    neighbor_V      = pcd_v[idx]        # (N_verts, k)

    # Lichen neighbor hop le: phai co V tuong doi cao (>= 60% cua max neighbor V)
    max_V_per_vert  = neighbor_V.max(axis=1, keepdims=True)
    bright_enough   = neighbor_V >= (max_V_per_vert * 0.75)  # nới lỏng 0.85→0.75

    bright_white = (neighbor_labels == 2) & bright_enough

    vert_labels = np.zeros(len(verts), dtype=np.uint8)
    vert_labels[bright_white.sum(axis=1) >= 2] = 2   # >= 2/15 neighbors (nới lỏng từ 3)
    # Green lichen (label=1) khong su dung — chi giu white lichen (label=2)

    return vert_labels


def triangle_areas(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Dien tich tung tam giac (m2) dung cross product."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


# ── Visualization colors ──────────────────────────────────────────────────────

VIZ_COLORS = {
    # (side, label) -> RGB float [0,1]
    # Chi dia y TRANG (label=2):  upslope = CAM,  downslope = VANG
    ("up",   2): np.array([1.00, 0.50, 0.00]),   # CAM  – upslope lichen
    ("down", 2): np.array([1.00, 0.95, 0.00]),   # VANG – downslope lichen
    # label=1 (dia y xanh) khong dung nua, giu bark goc
}


# ── Per-scan ──────────────────────────────────────────────────────────────────

def process_scan(scan_dir: str,
                 show_sides: bool = False,
                 verbose: bool = True) -> dict:
    name    = os.path.basename(scan_dir)
    out_dir = os.path.join(scan_dir, "output")
    result  = {"name": name, "ok": False}

    # Load mesh (uu tien final > trimmed > recolored > tsdf)
    for fname in ["trunk_mesh_final.ply",
                  "trunk_mesh_trimmed.ply",
                  "trunk_mesh_recolored.ply",
                  "trunk_mesh_tsdf.ply"]:
        mesh_path = os.path.join(out_dir, fname)
        if os.path.exists(mesh_path):
            break
    else:
        result["msg"] = "Khong co mesh — chay 02/03/04 truoc"
        return result

    t0   = time.time()
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices,      dtype=np.float64)   # (V,3)
    tris  = np.asarray(mesh.triangles,     dtype=np.int64)     # (F,3)
    cols  = np.asarray(mesh.vertex_colors, dtype=np.float32)   # (V,3) [0,1]

    V, F = len(verts), len(tris)
    if verbose:
        print(f"  Mesh: {V:,} verts  {F:,} faces", flush=True)

    # ── 1. Trunk axis (PCA) ───────────────────────────────────────────────────
    trunk_axis = pca_axis(verts)
    centroid   = verts.mean(axis=0)

    # ── 2. Gravity direction tu camera poses ──────────────────────────────────
    try:
        gravity_up = estimate_gravity(scan_dir)
    except Exception:
        gravity_up = np.array([0.0, 1.0, 0.0])   # fallback: Y len
    if verbose:
        print(f"  Gravity 'up' : {gravity_up.round(3)}", flush=True)

    # ── 3. Upslope direction = gravity projected perp to trunk ────────────────
    # Upslope = phia doi dien voi huong xuong doc (mat nhin len doc)
    up_perp = gravity_up - np.dot(gravity_up, trunk_axis) * trunk_axis
    norm_up = np.linalg.norm(up_perp)
    if norm_up < 0.1:   # trunk gan nhu thang dung -> dung y ngang
        up_perp = np.array([1.0, 0.0, 0.0])
    else:
        up_perp /= norm_up

    # ── 4. Classify lichen — uu tien PCD (mau chinh xac), fallback mesh ─────────
    pcd_path = os.path.join(out_dir, "trunk_pointcloud.ply")
    if os.path.exists(pcd_path):
        if verbose:
            print(f"  Detect tren PCD ({os.path.basename(pcd_path)}) ...", flush=True)
        vert_labels = detect_on_pcd_transfer_to_mesh(pcd_path, verts)
    else:
        if verbose:
            print(f"  PCD khong co, detect tren mesh colors ...", flush=True)
        vert_labels = classify_lichen(cols)

    # Per-FACE label = majority vote tu 3 vertex
    face_label_votes = np.stack([vert_labels[tris[:,0]],
                                  vert_labels[tris[:,1]],
                                  vert_labels[tris[:,2]]], axis=1)   # (F,3)
    lichen_labels = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=3).argmax(), 1,
        face_label_votes).astype(np.uint8)   # (F,)

    # ── Mask end-cap faces (mat cat phang o dau/cuoi than) ────────────────────
    # End-cap face co normal song song voi trunk_axis (|cos| > 0.7)
    # -> day la mat cat gia tao, KHONG phai vo cay -> force label = bark
    mesh.compute_triangle_normals()
    face_normals  = np.asarray(mesh.triangle_normals, dtype=np.float64)
    cos_with_axis = np.abs(face_normals @ trunk_axis)   # (F,)
    is_endcap     = cos_with_axis > 0.7                 # ~45° threshold
    lichen_labels[is_endcap] = 0   # force bark tren end-cap
    if verbose:
        print(f"  End-cap faces masked: {is_endcap.sum():,} / {F:,}", flush=True)

    # ── 5. Per-FACE side (upslope / downslope) ────────────────────────────────
    # Centroid cua tung face
    face_centers  = (verts[tris[:,0]] + verts[tris[:,1]] + verts[tris[:,2]]) / 3.0

    # ── Mask end-ZONE faces (vung dau/cuoi doc theo truc) ─────────────────────
    # Force bark cho faces o END_ZONE_FRAC dau va cuoi truc cay
    # Manh hon end-cap mask: bat tat ca mesh rac o ngon/goc du normal nao
    face_t   = face_centers @ trunk_axis          # projection len trunk axis (m)
    t_lo     = face_t.min()
    t_hi     = face_t.max()
    t_range  = max(t_hi - t_lo, 0.01)
    ez       = END_ZONE_FRAC * t_range            # khoang cat (m)
    is_end_zone = (face_t < t_lo + ez) | (face_t > t_hi - ez)
    lichen_labels[is_end_zone] = 0
    if verbose:
        print(f"  End-zone masked ({END_ZONE_FRAC*100:.0f}% each end = {ez*100:.1f} cm): "
              f"{is_end_zone.sum():,} faces", flush=True)
    face_rel      = face_centers - centroid
    face_up_comp  = face_rel @ up_perp   # positive = upslope

    is_upslope = face_up_comp >= 0   # (F,) bool

    # ── 6. Triangle areas ─────────────────────────────────────────────────────
    areas = triangle_areas(verts, tris)   # (F,) m2

    # ── 7. Compute statistics ─────────────────────────────────────────────────
    def area_cm2(mask): return float(areas[mask].sum()) * 10000

    total_up   = area_cm2(is_upslope)
    total_down = area_cm2(~is_upslope)
    total_all  = area_cm2(np.ones(F, bool))

    stats = {}
    for side_name, side_mask in [("upslope", is_upslope),
                                   ("downslope", ~is_upslope)]:
        g_mask  = side_mask & (lichen_labels == 1)
        w_mask  = side_mask & (lichen_labels == 2)
        lich_m  = side_mask & (lichen_labels  > 0)
        side_a  = area_cm2(side_mask)

        stats[side_name] = {
            "total_cm2":       round(side_a,       1),
            "green_lichen_cm2":round(area_cm2(g_mask), 1),
            "white_lichen_cm2":round(area_cm2(w_mask), 1),
            "total_lichen_cm2":round(area_cm2(lich_m), 1),
            "lichen_ratio_pct": round(area_cm2(lich_m) / max(side_a,1) * 100, 2),
        }

    lichen_total = area_cm2(lichen_labels > 0)
    stats["overall"] = {
        "total_cm2":        round(total_all, 1),
        "total_lichen_cm2": round(lichen_total, 1),
        "lichen_ratio_pct": round(lichen_total / max(total_all,1) * 100, 2),
    }

    # ── 8. Print results ──────────────────────────────────────────────────────
    if verbose:
        sep = "=" * 50
        print(f"\n{sep}", flush=True)
        print(f"  LICHEN DETECTION — {name}", flush=True)
        print(sep, flush=True)
        for side in ["upslope", "downslope"]:
            s = stats[side]
            print(f"\n  {side.upper()} ({s['total_cm2']:,.0f} cm²):", flush=True)
            print(f"    Dia y xanh  : {s['green_lichen_cm2']:6.0f} cm²"
                  f"  ({s['green_lichen_cm2']/max(s['total_cm2'],1)*100:.1f}%)", flush=True)
            print(f"    Dia y trang : {s['white_lichen_cm2']:6.0f} cm²"
                  f"  ({s['white_lichen_cm2']/max(s['total_cm2'],1)*100:.1f}%)", flush=True)
            print(f"    TONG dia y  : {s['total_lichen_cm2']:6.0f} cm²"
                  f"  ({s['lichen_ratio_pct']:.1f}%)", flush=True)
        s = stats["overall"]
        print(f"\n  OVERALL:", flush=True)
        print(f"    Tong be mat : {s['total_cm2']:,.0f} cm²", flush=True)
        print(f"    Tong dia y  : {s['total_lichen_cm2']:,.0f} cm²"
              f"  ({s['lichen_ratio_pct']:.1f}%)", flush=True)
        print(sep, flush=True)

    # ── 9. Build visualization vertex colors ──────────────────────────────────
    # vert_labels da co tu PCD detection (step 4)
    # Side score: tich luy tu cac face lien ket
    side_score = np.zeros(V, dtype=np.float32)
    for vi_arr, fup in zip(tris.T, [face_up_comp]*3):
        np.add.at(side_score, vi_arr, fup)

    vert_label = vert_labels   # tu PCD detect (chinh xac hon)
    vert_is_up = side_score >= 0

    # Mode show_sides: chi to 2 mat khong ve lichen
    if show_sides:
        new_cols = np.where(vert_is_up[:, None],
                            np.array([[0.4, 0.6, 1.0]]),    # upslope = xanh nhat
                            np.array([[1.0, 0.6, 0.3]]))    # downslope = cam nhat
    else:
        # Bat dau tu mau goc cua mesh (bark giu nguyen)
        new_cols = cols.copy()
        for side_key, side_mask in [("up", vert_is_up), ("down", ~vert_is_up)]:
            for lbl in [2]:   # chi dia y TRANG, bark giu mau goc
                m = side_mask & (vert_label == lbl)
                if m.any():
                    c = VIZ_COLORS[(side_key, lbl)]
                    new_cols[m] = c

    mesh.vertex_colors = o3d.utility.Vector3dVector(new_cols.clip(0,1))

    # ── 10. Save ──────────────────────────────────────────────────────────────
    viz_path  = os.path.join(out_dir, "trunk_mesh_detected.ply")
    json_path = os.path.join(out_dir, "lichen_stats.json")
    o3d.io.write_triangle_mesh(viz_path, mesh, write_vertex_colors=True)
    with open(json_path, "w") as f:
        json.dump({"scan": name, "stats": stats}, f, indent=2)

    elapsed = time.time() - t0
    result.update({
        "ok":   True,
        "msg":  f"lichen {stats['overall']['lichen_ratio_pct']:.1f}% [{elapsed:.0f}s]",
        "stats": stats,
    })
    if verbose:
        print(f"\n  Saved: {viz_path}", flush=True)
        print(f"  Saved: {json_path}", flush=True)
    return result


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_batch(raw_data_dir, scan_filter=None, skip_existing=False,
              show_sides=False):
    if scan_filter:
        dirs = [os.path.join(raw_data_dir, scan_filter)]
        if not os.path.isdir(dirs[0]):
            print(f"Scan not found: {dirs[0]}"); sys.exit(1)
    else:
        dirs = sorted([os.path.join(raw_data_dir, d)
                       for d in os.listdir(raw_data_dir)
                       if os.path.isdir(os.path.join(raw_data_dir, d))])

    total = len(dirs)
    print(f"\n{'='*55}")
    print(f"Lichen Detection  |  {total} scans")
    print(f"{'='*55}\n")

    ok_count = skip_count = 0
    all_stats = []

    for i, scan_dir in enumerate(dirs):
        name = os.path.basename(scan_dir)
        out_path = os.path.join(scan_dir, "output", "lichen_stats.json")
        if skip_existing and os.path.exists(out_path):
            print(f"[{i+1:3d}/{total}] {name} -- SKIP"); skip_count += 1; continue
        print(f"[{i+1:3d}/{total}] {name}")
        r = process_scan(scan_dir, show_sides=show_sides, verbose=True)
        if r["ok"]:
            ok_count += 1
            all_stats.append(r["stats"])
            print(f"  OK  {r['msg']}")
        else:
            print(f"  FAIL  {r.get('msg','')}")
        print()

    print(f"{'='*55}")
    print(f"DONE: {ok_count}/{total - skip_count} OK  ({skip_count} skipped)")

    # Summary CSV-like
    if all_stats:
        print("\n--- SUMMARY ---")
        print(f"{'Scan':<12} {'Total_cm2':>10} "
              f"{'UP_lich%':>9} {'DOWN_lich%':>10} {'TOTAL_lich%':>11}")
        for i, (scan_dir, st) in enumerate(zip(dirs, all_stats)):
            name = os.path.basename(scan_dir)
            print(f"{name:<12} {st['overall']['total_cm2']:>10.0f} "
                  f"{st['upslope']['lichen_ratio_pct']:>9.1f} "
                  f"{st['downslope']['lichen_ratio_pct']:>10.1f} "
                  f"{st['overall']['lichen_ratio_pct']:>11.1f}")

    with open(os.path.join(raw_data_dir, "lichen_summary.json"), "w") as f:
        json.dump(all_stats, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global END_ZONE_FRAC
    p = argparse.ArgumentParser(description="Detect lichen + upslope/downslope area")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME")
    g.add_argument("--all",  action="store_true")
    p.add_argument("--show-sides",    action="store_true",
                   help="Chi to upslope/downslope (xanh/cam), khong ve lichen")
    p.add_argument("--raw-data",      default=RAW_DATA_DIR)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--end-zone",      type=float, default=None,
                   help=f"Phan tram dau/cuoi truc cay force = bark (default {END_ZONE_FRAC})")
    args = p.parse_args()
    if args.end_zone is not None:
        END_ZONE_FRAC = args.end_zone
    run_batch(args.raw_data,
              scan_filter   = args.scan if args.scan else None,
              skip_existing = args.skip_existing,
              show_sides    = args.show_sides)


if __name__ == "__main__":
    main()
