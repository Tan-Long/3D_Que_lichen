#!/usr/bin/env python3
"""
Step 04: Trim mesh - xoa dat/ngon, don sach fragments.

Pipeline:
  1. Load trunk_mesh_recolored.ply
  2. Xoa fragments nho (min_frag)
  3. Tim truc chinh cua than cay (PCA axis)
  4. Tinh radius theo tung lat cat doc truc
  5. Tu dong phat hien cho phình ra (dat/ngon) -> cat
  6. Xoa vertex qua xa truc (outlier radial)
  7. Luu trunk_mesh_trimmed.ply

Usage:
    python 04_trim_mesh.py --scan 26Q1              # auto trim
    python 04_trim_mesh.py --scan 26Q1 --info       # chi hien thi thong tin, khong cat
    python 04_trim_mesh.py --scan 26Q1 --min-frag 1000 --max-radius 0.15
    python 04_trim_mesh.py --scan 26Q1 --trim-start 0.05 --trim-end 0.05
    python 04_trim_mesh.py --all --skip-existing
"""

import argparse, copy, json, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")


# ── Helpers ───────────────────────────────────────────────────────────────────

def remove_small_fragments(mesh: o3d.geometry.TriangleMesh,
                           min_triangles: int) -> tuple:
    """Xoa connected component co < min_triangles tam giac. Tra ve (mesh, n_removed)."""
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters   = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)
    remove_mask    = cluster_n_tris[tri_clusters] < min_triangles
    n_removed      = remove_mask.sum()
    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_triangles_by_mask(remove_mask)
    mesh_out.remove_unreferenced_vertices()
    return mesh_out, int(n_removed)


def keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> tuple:
    """Chi giu component lon nhat — loai SACH toan bo fragment con lai."""
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters   = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)
    best_cluster   = int(np.argmax(cluster_n_tris))
    remove_mask    = tri_clusters != best_cluster
    n_removed      = remove_mask.sum()
    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_triangles_by_mask(remove_mask)
    mesh_out.remove_unreferenced_vertices()
    return mesh_out, int(n_removed)


def pca_axis(pts: np.ndarray) -> np.ndarray:
    """Tra ve vector truc chinh (largest eigenvector) cua tap diem."""
    centered = pts - pts.mean(axis=0)
    cov      = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, np.argmax(vals)]   # unit vector doc than cay


def estimate_gravity(scan_dir: str):
    """
    Tra ve huong 'xuong' (gravity down) trong world space.
    Stray Scanner su dung he toa do world Y=up (xac nhan qua PCA cac mesh sach).
    -> gravity_down = [0, -1, 0] luon dung voi dataset nay.
    scan_dir: giu de tuong thich API cu.
    """
    return np.array([0.0, -1.0, 0.0])


def gravity_axis_trim(mesh: o3d.geometry.TriangleMesh,
                      verts: np.ndarray,
                      grav_down: np.ndarray,
                      n_slices: int = 80,
                      radial_factor: float = 2.5,
                      verbose: bool = True):
    """
    Xoa blob dat/goc theo 2 buoc:
      1. Per-slice radial filter doc truc trong luong:
         Moi lat cat doc truc UP/DOWN, loai diem xa hon radial_factor * ref_r
         (phan biet than cay nho vs blob dat rong)
      2. End trim: cat 2 dau qua thua/blob
    Tra ve (mesh_out, verts_out).
    """
    grav_up  = -grav_down
    centroid = verts.mean(axis=0)

    # Chieu len truc trong luong (height trong the gioi, centered)
    h_all = (verts - centroid) @ grav_up

    # Khoang cach vuong goc tu truc trong luong
    h_perp = (verts - centroid) - h_all[:, None] * grav_up
    h_rad  = np.linalg.norm(h_perp, axis=1)

    # ── Iterative trunk center estimation ────────────────────────────────────
    # Blob keo centroid lech khoi truc than -> lap lai de tim truc dung
    for _it in range(5):
        delta = np.median(h_perp[h_rad <= np.percentile(h_rad, 30)], axis=0)
        centroid = centroid + delta
        h_all  = (verts - centroid) @ grav_up
        h_perp = (verts - centroid) - h_all[:, None] * grav_up
        h_rad  = np.linalg.norm(h_perp, axis=1)
        if np.linalg.norm(delta) < 0.005:   # hoi tu
            break

    # Uoc tinh ref_r tu p15 sau khi truc da can chinh
    # p15 sau khi center dung se rat gan radius than that
    ref_r = float(np.percentile(h_rad, 15))
    ref_r = max(ref_r, 0.025)   # toi thieu 2.5 cm

    r_max_radial = ref_r * radial_factor

    if verbose:
        print(f"  Gravity trim: ref_r={ref_r*100:.1f}cm  r_max={r_max_radial*100:.1f}cm", flush=True)

    # ── Per-slice radial filter (chi filter radial, khong cat dau/cuoi) ───────
    # Viec cat dau/cuoi do PCA auto_trim_ends xu ly (chinh xac hon theo truc than)
    keep_radial = h_rad <= r_max_radial

    n_remove = int((~keep_radial).sum())
    if n_remove == 0:
        if verbose:
            print("  Gravity trim: nothing removed", flush=True)
        return mesh, verts

    if verbose:
        blob_pct = n_remove / max(len(verts), 1) * 100
        print(f"  Gravity trim: remove blob {blob_pct:.1f}%  "
              f"-> keep {keep_radial.sum():,}", flush=True)

    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_vertices_by_mask(~keep_radial)
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()
    mesh_out, n_frag = keep_largest_component(mesh_out)
    if verbose and n_frag:
        print(f"  Gravity trim: kept largest, removed {n_frag:,} frag tris", flush=True)

    verts_out = np.asarray(mesh_out.vertices, dtype=np.float64)
    return mesh_out, verts_out


def normal_ground_filter(mesh: o3d.geometry.TriangleMesh,
                         gravity_down: np.ndarray,
                         normal_thr: float = 0.65,
                         verbose: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Loai bo tam giac co normal THANG DUNG (= mat phang dat/nen).
    Vo cay co normal NGANG -> |normal . grav_down| ~ 0.
    Mat dat co normal THANG DUNG -> |normal . grav_down| ~ 1.
    normal_thr: nguong loi |dot| (default 0.75).
    """
    mesh.compute_triangle_normals()
    face_normals = np.asarray(mesh.triangle_normals)
    dots         = face_normals @ gravity_down          # +1 = ngang xuong, -1 = ngang len

    # Chi cat cac mat co normal thang dung
    ground_mask = np.abs(dots) > normal_thr

    n_ground = ground_mask.sum()
    if verbose:
        print(f"  Normal ground filter: remove {n_ground:,} / {len(face_normals):,} tris "
              f"({n_ground/max(len(face_normals),1)*100:.1f}%, thr={normal_thr})", flush=True)

    if n_ground == 0:
        return mesh

    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_triangles_by_mask(ground_mask)
    mesh_out.remove_unreferenced_vertices()
    return mesh_out


def project_onto_axis(pts: np.ndarray, axis: np.ndarray,
                      origin: np.ndarray) -> np.ndarray:
    """Chieu diem xuong truc, tra ve toa do doc truc (scalar)."""
    return (pts - origin) @ axis


def radial_distance(pts: np.ndarray, axis: np.ndarray,
                    origin: np.ndarray) -> np.ndarray:
    """Khoang cach vuong goc tu moi diem den truc."""
    proj   = project_onto_axis(pts, axis, origin)[:, None] * axis
    perp   = (pts - origin) - proj
    return np.linalg.norm(perp, axis=1)


def auto_trim_ends(t_coords: np.ndarray, radii: np.ndarray,
                   n_slices: int = 80,
                   radius_jump_factor: float = 1.3,
                   min_density_ratio: float = 0.15,
                   min_good_run: int = 3,
                   verbose: bool = True) -> tuple:
    """
    Chia than cay thanh n_slices lat cat doc truc.
    Cat hai dau theo 2 tieu chi:
      1. Radius dot ngot tang (blob dat/goc)
      2. Mat do diem qua thap (ngon thua, sparse)

    min_good_run: so lat cat tot LIEN TIEP toi thieu truoc khi chap nhan t_start/t_end.
    Tranh truong hop blob nam ben trong (vd: slice 1 tot, slice 2-6 xau) bi bo sot.
    """
    t_min, t_max = t_coords.min(), t_coords.max()
    edges   = np.linspace(t_min, t_max, n_slices + 1)
    r_p90   = np.zeros(n_slices)
    density = np.zeros(n_slices)   # so vert / slice

    for i in range(n_slices):
        mask = (t_coords >= edges[i]) & (t_coords < edges[i + 1])
        cnt = mask.sum()
        density[i] = cnt
        if cnt > 5:
            r_p90[i] = np.percentile(radii[mask], 90)

    # Reference: median cua 50% giua (than cay on dinh nhat)
    mid_lo  = int(n_slices * 0.25)
    mid_hi  = int(n_slices * 0.75)
    ref_r   = np.median(r_p90[mid_lo:mid_hi][r_p90[mid_lo:mid_hi] > 0])
    ref_r   = max(ref_r, 0.02)
    ref_den = np.median(density[mid_lo:mid_hi])

    if verbose:
        print(f"  Trunk ref radius  : {ref_r*100:.1f} cm", flush=True)
        print(f"  Axis range        : {t_min:.3f} -> {t_max:.3f} m  "
              f"(len={t_max-t_min:.2f}m)", flush=True)

    def _slice_ok(i):
        r_ok = (r_p90[i] > 0) and (r_p90[i] < ref_r * radius_jump_factor)
        d_ok = density[i] > ref_den * min_density_ratio
        return r_ok and d_ok

    # --- Cat dau NGON (trai): tim DOAN LIEN TIEP dau tien gom min_good_run slice tot
    # Tranh dung lai o slice tot don le nam truoc blob ben trong
    t_start = t_min
    for i in range(n_slices - min_good_run + 1):
        if all(_slice_ok(j) for j in range(i, i + min_good_run)):
            t_start = edges[i]
            break

    # --- Cat dau GOC/DAT (phai): tuong tu tu phai di vao
    t_end = t_max
    for i in range(n_slices - 1, min_good_run - 2, -1):
        if all(_slice_ok(j) for j in range(i - min_good_run + 1, i + 1)):
            t_end = edges[i + 1]
            break

    if verbose:
        print(f"  Auto cut          : [{t_start:.3f}, {t_end:.3f}] m  "
              f"(keep {t_end-t_start:.2f}m)", flush=True)

    return t_start, t_end, ref_r


# ── Per-scan ──────────────────────────────────────────────────────────────────

def process_scan(scan_dir: str,
                 min_frag: int       = 1500,
                 max_radius: float   = None,    # None = tu dong
                 trim_start: float   = 0.0,     # them crop tu dau ngon (m)
                 trim_end: float     = 0.0,     # them crop tu goc (m)
                 info_only: bool     = False,
                 verbose: bool       = True) -> dict:
    name    = os.path.basename(scan_dir)
    out_dir = os.path.join(scan_dir, "output")
    result  = {"name": name, "ok": False, "msg": ""}

    # Uu tien: recolored > tsdf
    for fname in ["trunk_mesh_recolored.ply", "trunk_mesh_tsdf.ply"]:
        mesh_path = os.path.join(out_dir, fname)
        if os.path.exists(mesh_path):
            break
    else:
        result["msg"] = "Khong tim thay mesh — chay 02/03 truoc"
        return result

    t0   = time.time()
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts_orig = np.asarray(mesh.vertices, dtype=np.float32)
    N0 = len(verts_orig)
    if verbose:
        print(f"  Input  : {N0:,} verts  {len(mesh.triangles):,} tris  "
              f"({os.path.basename(mesh_path)})", flush=True)

    # ── Step 1: Fragment removal ──────────────────────────────────────────────
    mesh, n_frag_tri = remove_small_fragments(mesh, min_frag)
    if verbose and n_frag_tri:
        print(f"  Fragments removed: {n_frag_tri:,} tris  "
              f"-> {len(mesh.vertices):,} verts", flush=True)

    verts  = np.asarray(mesh.vertices, dtype=np.float64)
    N      = len(verts)

    # ── Step 1b: Gravity-axis trim (xoa blob dat TRUOC khi PCA) ──────────────
    grav_down = estimate_gravity(scan_dir)
    if grav_down is not None:
        mesh, verts = gravity_axis_trim(mesh, verts, grav_down,
                                        n_slices=80,
                                        verbose=verbose)
        N = len(verts)
        # Normal-based filter: loai bo mat phang nam ngang (dat/la/muc gan mat dat)
        # Normal cua than cay ~ vuong goc Y (|ny| ~ 0), mat phang dat ~ |ny| ~ 1
        mesh = normal_ground_filter(mesh, grav_down, normal_thr=0.75, verbose=verbose)
        mesh, _ = keep_largest_component(mesh)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        N = len(verts)
    else:
        if verbose:
            print("  Gravity trim: skip (no odometry.csv)", flush=True)

    # ── Step 2: PCA trunk axis (sau gravity trim -> sach hon) ────────────────
    axis   = pca_axis(verts)
    origin = verts.mean(axis=0)
    t_all  = project_onto_axis(verts, axis, origin)
    r_all  = radial_distance(verts,   axis, origin)

    if info_only:
        print(f"  Axis vector: {axis.round(3)}")
        print(f"  Axis range : {t_all.min():.3f} -> {t_all.max():.3f} m")
        print(f"  Radius p50 : {np.percentile(r_all,50)*100:.1f} cm")
        print(f"  Radius p90 : {np.percentile(r_all,90)*100:.1f} cm")
        print(f"  Radius max : {r_all.max()*100:.1f} cm")
        result.update({"ok": True, "msg": "info only"})
        return result

    # ── Step 3: Auto detect cut points ───────────────────────────────────────
    t_start, t_end, ref_r = auto_trim_ends(t_all, r_all, verbose=verbose)
    t_start += trim_start   # cat them tu dau ngon (tang t_start = bo bo ngon)
    t_end   -= trim_end     # cat them tu goc (giam t_end = bo bo goc)

    # ── Step 4: Radial outlier removal ───────────────────────────────────────
    # ref_r tu auto_trim_ends co the bi tang len boi blob o giua scan
    # -> dung p25 cua r_p90 tren cac lat cat trong vung [t_start, t_end]
    #    de co nguong chat hon, it bi anh huong boi blob
    _n_sl = 40
    _edges = np.linspace(t_start, t_end, _n_sl + 1)
    _r90s = []
    for _i in range(_n_sl):
        _m = (t_all >= _edges[_i]) & (t_all < _edges[_i + 1])
        if _m.sum() > 5:
            _r90s.append(float(np.percentile(r_all[_m], 90)))
    if len(_r90s) >= 4:
        # Dung p15 (thay vi p25) de it bi anh huong hon boi cac lat cat blob rong
        ref_r_tight = float(np.percentile(_r90s, 15))
    else:
        ref_r_tight = ref_r
    ref_r_tight = max(ref_r_tight, 0.02)

    r_max = max_radius if max_radius else ref_r_tight * 1.7
    if verbose:
        print(f"  ref_r tight (p15) : {ref_r_tight*100:.1f} cm", flush=True)
        print(f"  Max radius cut    : {r_max*100:.1f} cm", flush=True)

    # Per-slice extra trim: lat cat co r_p90 >> ref_r_tight (blob o giua scan)
    # ap dung nguong chat hon cho chinh lat cat do
    per_slice_cut = np.zeros(len(verts), dtype=bool)   # True = nen cat
    _blob_thr  = ref_r_tight * 1.8   # lat cat co r_p90 > 1.8x ref_r_tight bi coi la blob
    _blob_rmax = ref_r_tight * 1.3   # cat den 1.3x (chat hon 1.5x cu)
    if not max_radius:   # chi ap dung khi khong override thu cong
        for _i in range(_n_sl):
            _m = (t_all >= _edges[_i]) & (t_all < _edges[_i + 1])
            if _m.sum() < 10:
                continue
            _r90 = float(np.percentile(r_all[_m], 90))
            if _r90 > _blob_thr:
                per_slice_cut[_m & (r_all > _blob_rmax)] = True
    n_ps_cut = per_slice_cut.sum()
    if verbose and n_ps_cut:
        print(f"  Per-slice blob cut: {n_ps_cut:,} verts extra", flush=True)

    # ── Step 5: Apply all masks ───────────────────────────────────────────────
    keep = (t_all >= t_start) & (t_all <= t_end) & (r_all <= r_max) & ~per_slice_cut
    n_remove = (~keep).sum()
    if verbose:
        print(f"  Remove {n_remove:,} verts -> keep {keep.sum():,}", flush=True)

    remove_mask = ~keep
    mesh_out    = copy.deepcopy(mesh)
    mesh_out.remove_vertices_by_mask(remove_mask)
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()

    # Final: chi giu component lon nhat -> SACH HOÀN TOÀN fragment
    mesh_out, n_frag2 = keep_largest_component(mesh_out)
    if verbose and n_frag2:
        print(f"  Keep largest: removed {n_frag2:,} fragment tris", flush=True)

    n_final = len(mesh_out.vertices)
    if verbose:
        print(f"  Final  : {n_final:,} verts  "
              f"{len(mesh_out.triangles):,} tris", flush=True)

    # ── Step 6: Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "trunk_mesh_trimmed.ply")
    o3d.io.write_triangle_mesh(out_path, mesh_out, write_vertex_colors=True)

    elapsed = time.time() - t0
    result.update({
        "ok":  True,
        "msg": f"verts:{n_final:,} tris:{len(mesh_out.triangles):,} [{elapsed:.0f}s]",
    })
    if verbose:
        print(f"  Saved {out_path}  [{elapsed:.0f}s]", flush=True)
    return result


# ── Batch ─────────────────────────────────────────────────────────────────────

def run_batch(raw_data_dir, scan_filter=None, skip_existing=False, **kwargs):
    if scan_filter:
        dirs = [os.path.join(raw_data_dir, scan_filter)]
        if not os.path.isdir(dirs[0]):
            print(f"Scan not found: {dirs[0]}"); sys.exit(1)
    else:
        dirs = sorted([os.path.join(raw_data_dir, d)
                       for d in os.listdir(raw_data_dir)
                       if os.path.isdir(os.path.join(raw_data_dir, d))])

    # Per-scan overrides: Raw_data/per_scan_overrides.json
    # Format: {"26Q15": {"trim_end": 1.3}, "26Q7": {"max_radius": 0.12}, ...}
    overrides_path = os.path.join(raw_data_dir, "per_scan_overrides.json")
    per_scan_ov = {}
    if os.path.exists(overrides_path):
        with open(overrides_path) as f:
            per_scan_ov = json.load(f)
        print(f"Per-scan overrides loaded: {list(per_scan_ov.keys())}")

    total = len(dirs)
    print(f"\n{'='*55}")
    print(f"Trim Mesh  |  {total} scans  min_frag={kwargs.get('min_frag',1500)}")
    print(f"{'='*55}\n")

    ok_count = skip_count = 0
    results  = []

    for i, scan_dir in enumerate(dirs):
        name     = os.path.basename(scan_dir)
        out_path = os.path.join(scan_dir, "output", "trunk_mesh_trimmed.ply")
        if skip_existing and os.path.exists(out_path):
            print(f"[{i+1:3d}/{total}] {name} -- SKIP"); skip_count += 1; continue

        # Merge global kwargs with per-scan overrides (per-scan takes priority)
        scan_kwargs = dict(kwargs)
        if name in per_scan_ov:
            scan_kwargs.update(per_scan_ov[name])
            print(f"[{i+1:3d}/{total}] {name}  [override: {per_scan_ov[name]}]")
        else:
            print(f"[{i+1:3d}/{total}] {name}")
        r = process_scan(scan_dir, verbose=True, **scan_kwargs)
        results.append(r)
        print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['msg']}\n")
        if r["ok"]: ok_count += 1

    print(f"{'='*55}")
    print(f"DONE: {ok_count}/{total - skip_count} OK  ({skip_count} skipped)")
    with open(os.path.join(raw_data_dir, "trim_summary.json"), "w") as f:
        json.dump({"results": results}, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Trim mesh: xoa dat/ngon/fragments")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", metavar="NAME")
    g.add_argument("--all",  action="store_true")

    p.add_argument("--info",         action="store_true",
                   help="Chi hien thi thong tin bounds, khong cat")
    p.add_argument("--min-frag",     type=int,   default=1500,
                   help="Xoa component < N tris (default 1500)")
    p.add_argument("--max-radius",   type=float, default=None,
                   help="Xoa vert xa truc hon R met (auto neu bo trong)")
    p.add_argument("--trim-start",   type=float, default=0.0,
                   help="Cat them tu dau ngon (m, default 0)")
    p.add_argument("--trim-end",     type=float, default=0.0,
                   help="Cat them tu goc/dat (m, default 0)")
    p.add_argument("--raw-data",     default=RAW_DATA_DIR)
    p.add_argument("--skip-existing",action="store_true")

    args = p.parse_args()
    run_batch(args.raw_data,
              scan_filter   = args.scan if args.scan else None,
              skip_existing = args.skip_existing,
              min_frag      = args.min_frag,
              max_radius    = args.max_radius,
              trim_start    = args.trim_start,
              trim_end      = args.trim_end,
              info_only     = args.info)


if __name__ == "__main__":
    main()
