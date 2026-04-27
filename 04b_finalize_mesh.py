#!/usr/bin/env python3
"""
Step 04b: Finalize mesh - lam min be mat + cap 2 dau thanh flat polygon.

Input : trunk_mesh_trimmed.ply (output cua 04_trim_mesh.py)
Output: trunk_mesh_final.ply

Pipeline:
  1. Load + giu component lon nhat (xoa sach fragments con lai)
  2. Taubin smoothing (lam min be mat than cay, giu volume)
  3. Detect boundary edge loops tren mesh
  4. Phan loai loops: top end / bottom end / holes giua
  5. Cap 2 dau lon nhat bang flat triangle fan
  6. Fill holes nho o giua (tuy chon)
  7. Recompute vertex normals
  8. Save trunk_mesh_final.ply

Usage:
    python 04b_finalize_mesh.py --scan 26Q1
    python 04b_finalize_mesh.py --scan 26Q1 --smooth 20
    python 04b_finalize_mesh.py --scan 26Q1 --no-cap
    python 04b_finalize_mesh.py --scan 26Q1 --no-smooth
    python 04b_finalize_mesh.py --all --skip-existing
"""

import argparse, copy, json, os, subprocess, sys, time, webbrowser
from collections import defaultdict
import numpy as np
import open3d as o3d

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")

# Fallback neu khong doc duoc IMU
GRAVITY_UP = np.array([0.0, 1.0, 0.0])
GRAVITY_UP_DEFAULT = GRAVITY_UP


def read_gravity_up(scan_dir: str) -> np.ndarray:
    """
    Doc vector trong truong thuc tu imu.csv (cot a_x, a_y, a_z).
    Tra ve unit vector huong LEN (nguoc chieu gia toc trong truong).
    Fallback: [0, 1, 0] neu file khong ton tai.
    """
    imu_path = os.path.join(scan_dir, "imu.csv")
    if not os.path.exists(imu_path):
        return GRAVITY_UP_DEFAULT.copy()
    try:
        data = np.genfromtxt(imu_path, delimiter=",", skip_header=1,
                             usecols=(1, 2, 3))  # a_x, a_y, a_z
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) == 0:
            return GRAVITY_UP_DEFAULT.copy()
        # Median loai outlier do chuyen dong
        g_down = np.median(data, axis=0)
        norm   = np.linalg.norm(g_down)
        if norm < 1e-6:
            return GRAVITY_UP_DEFAULT.copy()
        return -g_down / norm   # gravity UP = nguoc chieu keo xuong dat
    except Exception:
        return GRAVITY_UP_DEFAULT.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def keep_largest_component(mesh):
    """Chi giu connected component lon nhat."""
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters   = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)
    best = int(np.argmax(cluster_n_tris))
    remove_mask = tri_clusters != best
    mesh_out = copy.deepcopy(mesh)
    mesh_out.remove_triangles_by_mask(remove_mask)
    mesh_out.remove_unreferenced_vertices()
    return mesh_out, int(remove_mask.sum())


def taubin_smooth(mesh, n_iter=10, lam=0.5, mu=-0.53):
    """
    Taubin smoothing: lam min be mat nhung giu the tich (khong co rut nhu Laplacian).
    Xen ke 2 pass: shrink (lambda > 0) va inflate (mu < 0).
    """
    try:
        return mesh.filter_smooth_taubin(
            number_of_iterations=n_iter,
            lambda_filter=lam,
            mu=mu,
        )
    except TypeError:
        # open3d cu hon khong co tham so filter_scope
        return mesh.filter_smooth_taubin(
            number_of_iterations=n_iter,
            lambda_filter=lam,
            mu=mu,
        )


def find_boundary_loops(triangles):
    """
    Tim tat ca boundary edge loops trong mesh.

    Tra ve list[list[int]]: moi phan tu la mot loop theo thu tu vertex indices.
    Huong loop: nguoc chieu winding cua triangles lan can
    (de normal cua flat cap huong ra ngoai mesh).
    """
    tris = np.asarray(triangles, dtype=np.int64)

    # Buoc 1: dem so triangle dung moi undirected edge
    edge_count    = defaultdict(int)
    edge_directed = {}   # undirected key -> directed (a, b) theo winding cua tri

    for tri in tris:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_count[key] += 1
            if key not in edge_directed:
                edge_directed[key] = (a, b)

    # Buoc 2: boundary directed edges (chi dung boi 1 triangle)
    # Loop boundary theo huong nguoc lai voi winding triangle
    # -> b -> a (nguoc directed a->b cua triangle) de normal flat cap ra ngoai
    nxt = {}   # v -> v theo huong boundary loop
    for key, cnt in edge_count.items():
        if cnt == 1:
            a, b = edge_directed[key]
            nxt[b] = a   # reversed: loop di b -> a

    # Buoc 3: traverse loops
    loops = []
    visited = set()
    for start in sorted(nxt.keys()):
        if start in visited:
            continue
        loop = []
        curr = start
        for _ in range(len(nxt) + 1):
            if curr in visited:
                break
            loop.append(curr)
            visited.add(curr)
            curr = nxt.get(curr)
            if curr is None:
                break
        if len(loop) >= 3:
            loops.append(loop)

    return loops


def make_flat_cap(loop_idx, verts, colors, outward_normal, verbose_label=""):
    """
    Tao flat triangle fan cho mot boundary loop.

    Tra ve:
        new_verts  : (N+1, 3) float64  - vertices goc + centroid
        new_colors : (N+1, 3) float64 | None
        new_tris   : (N, 3) int64      - chi cac triangles cua cap
                                          (indices relative to 0 = start of new_verts block)
    """
    lp = list(loop_idx)
    N = len(lp)
    lv = verts[lp]           # (N, 3)
    centroid = lv.mean(0)    # (3,)

    # Centroid color = mean cua boundary loop
    if colors is not None:
        lc = colors[lp]
        centroid_color = lc.mean(0)
    else:
        centroid_color = None

    # Triangle fan: (centroid, lp[i], lp[i+1])
    # Kiem tra winding qua normal cua tam giac dau tien
    v0 = lv[0] - centroid
    v1 = lv[1] - centroid
    n_check = np.cross(v0, v1)
    if np.dot(n_check, outward_normal) < 0:
        lp = lp[::-1]   # flip winding

    # Vertices cua cap: giu y bong vertex goc, them centroid
    new_verts = np.vstack([verts[lp], centroid[None, :]])   # (N+1, 3)
    if colors is not None:
        new_colors = np.vstack([colors[lp], centroid_color[None, :]])
    else:
        new_colors = None

    c_local = N   # index cua centroid trong new_verts
    tris = []
    for i in range(N):
        a = i
        b = (i + 1) % N
        tris.append([c_local, a, b])

    return new_verts, new_colors, np.array(tris, dtype=np.int64)


def trim_ragged_top(mesh, grav_up=GRAVITY_UP,
                    top_fraction: float  = 0.65,
                    ragged_density: float = 0.05,
                    min_slice_verts: int  = 20,
                    sig_hole_verts: int   = 10,
                    required_clean_slices: int = 6,
                    verbose: bool = True):
    """
    Cat bo phan ngon bi tua do thieu du lieu scan.

    Hai tieu chi dua tren boundary vertex analysis:
      A) Density: % boundary verts trong lat cat > ragged_density -> rach
      B) Hole:    bat ky loop >= sig_hole_verts trong lat cat -> co lo dang ke
      Sparse:     lat cat < min_slice_verts -> qua thua -> rach

    Quet tu TREN XUONG, tim required_clean_slices lat cat sach lien tiep
    (khong rach theo ca 2 tieu chi) -> cat o diem do.

    top_fraction          : phan chieu cao xem xet tu tren (default 65%)
    ragged_density        : nguong % boundary verts (default 5%)
    min_slice_verts       : lat cat < N verts = thua = rach (default 20)
    sig_hole_verts        : loop >= N verts bi coi la lo dang ke (default 10)
    required_clean_slices : so lat cat sach lien tiep can thiet (default 5)
    """
    verts  = np.asarray(mesh.vertices,  dtype=np.float64)
    tris   = np.asarray(mesh.triangles, dtype=np.int64)

    h       = (verts - verts.mean(0)) @ grav_up
    h_min   = float(h.min())
    h_max   = float(h.max())
    h_range = h_max - h_min

    top_start = h_max - top_fraction * h_range

    # Boundary vertex mask
    all_loops = find_boundary_loops(tris)
    bnd_mask  = np.zeros(len(verts), dtype=bool)
    for lp in all_loops:
        bnd_mask[np.array(lp, dtype=np.int64)] = True

    # Loop h_center lookup (de check sig holes per slice)
    loop_info = []   # (loop, h_center)
    for lp in all_loops:
        lp_arr = np.array(lp, dtype=np.int64)
        h_ctr  = float(h[lp_arr].mean())
        if h_ctr > top_start:
            loop_info.append((lp, h_ctr))

    # Per-slice density
    n_slices = 25
    edges    = np.linspace(top_start, h_max, n_slices + 1)

    slice_total   = np.zeros(n_slices, dtype=int)
    slice_density = np.ones(n_slices)

    for i in range(n_slices):
        in_sl  = (h >= edges[i]) & (h < edges[i + 1])
        n_tot  = int(in_sl.sum())
        n_bnd  = int((in_sl & bnd_mask).sum())
        slice_total[i]   = n_tot
        slice_density[i] = n_bnd / n_tot if n_tot > 0 else 1.0

    def _is_ragged(i):
        lo, hi = edges[i], edges[i + 1]
        if slice_total[i] < min_slice_verts:
            return True   # thua
        if slice_density[i] > ragged_density:
            return True   # density cao
        # Co loop dang ke (>= sig_hole_verts) trong slice nay
        for lp, ch in loop_info:
            if lo <= ch < hi and len(lp) >= sig_hole_verts:
                return True
        return False

    if verbose:
        n_sig = sum(1 for lp, _ in loop_info if len(lp) >= sig_hole_verts)
        print(f"  Top trim: top {top_fraction*100:.0f}%  "
              f"{len(loop_info)} loops  {n_sig} significant (>={sig_hole_verts}v)",
              flush=True)

    # Quet tu TREN XUONG
    h_cut = h_max
    consecutive_clean = 0

    for i in range(n_slices - 1, -1, -1):
        if _is_ragged(i):
            h_cut = float(edges[i])
            consecutive_clean = 0
        else:
            consecutive_clean += 1
            if consecutive_clean >= required_clean_slices:
                break

    # Bao ve: giu it nhat 50% than cay
    min_keep_h = h_min + 0.50 * h_range
    if h_cut < min_keep_h:
        h_cut = min_keep_h
        if verbose:
            print(f"  Top trim: clamped (keep >= 50% height)", flush=True)

    if h_max - h_cut < 0.03:
        if verbose:
            print(f"  Top trim: top is clean (<3cm to cut), skip", flush=True)
        return mesh

    n_above = sum(1 for lp, ch in loop_info if ch > h_cut)
    if verbose:
        print(f"  Top trim: cut h={h_cut:.3f}m  "
              f"remove {(h_max - h_cut)*100:.1f}cm  "
              f"({n_above} loops removed)", flush=True)

    keep_mask = h <= h_cut
    mesh_out  = copy.deepcopy(mesh)
    mesh_out.remove_vertices_by_mask(~keep_mask)
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()
    mesh_out, _ = keep_largest_component(mesh_out)

    return mesh_out


def add_end_caps(mesh, grav_up=GRAVITY_UP,
                 min_end_loop_verts=15,
                 fill_middle_holes=True,
                 max_middle_hole_verts=150,
                 verbose=True):
    """
    Them flat caps vao 2 dau ho cua mesh (top + bottom theo grav_up).
    Tuy chon: fill holes nho o giua.

    Tra ve mesh moi voi caps duoc them vao.
    """
    verts  = np.asarray(mesh.vertices,      dtype=np.float64)
    tris   = np.asarray(mesh.triangles,     dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) \
             if mesh.has_vertex_colors() else None

    # Chieu len truc trong luong
    origin = verts.mean(0)
    h      = (verts - origin) @ grav_up
    h_min, h_max = float(h.min()), float(h.max())
    h_range = h_max - h_min

    loops = find_boundary_loops(tris)

    if verbose:
        print(f"  Boundary loops: {len(loops)} found", flush=True)

    if not loops:
        if verbose:
            print("  Mesh already closed — no caps needed", flush=True)
        return mesh

    # Phan loai loops theo vi tri tren truc trong luong
    TOP_THRESH    = h_min + 0.70 * h_range
    BOTTOM_THRESH = h_min + 0.30 * h_range

    top_loops    = []
    bottom_loops = []
    middle_loops = []

    for lp in loops:
        if len(lp) < 3:
            continue
        c_h = float(h[lp].mean())
        if verbose:
            region = ("top"    if c_h > TOP_THRESH else
                      "bottom" if c_h < BOTTOM_THRESH else "middle")
            print(f"    loop {len(lp):4d} verts  h_center={c_h:.3f}m  [{region}]",
                  flush=True)
        if c_h > TOP_THRESH:
            top_loops.append(lp)
        elif c_h < BOTTOM_THRESH:
            bottom_loops.append(lp)
        else:
            middle_loops.append(lp)

    # Chon loop lon nhat cho top va bottom
    top_loop    = max(top_loops,    key=len) if top_loops    else None
    bottom_loop = max(bottom_loops, key=len) if bottom_loops else None

    # Chuan bi buffer de gop them vertices va triangles
    all_verts  = verts.copy()
    all_colors = colors.copy() if colors is not None else None
    all_tris   = list(tris)

    def _add(loop_idx, outward_normal, label):
        nonlocal all_verts, all_colors, all_tris
        if len(loop_idx) < min_end_loop_verts:
            if verbose:
                print(f"  Cap {label}: skip (loop too small: {len(loop_idx)} verts)",
                      flush=True)
            return
        nv, nc, nt = make_flat_cap(loop_idx, all_verts, all_colors,
                                   outward_normal, label)
        offset = len(all_verts)
        # Remap triangle indices de khop voi all_verts moi
        nt_global = nt.copy()
        nt_global[:, 1:] += offset      # indices a, b cua loop verts
        nt_global[:, 0]  += offset      # index cua centroid
        # Nhung vertices trong nv la: [lp[0]..lp[N-1], centroid]
        # Luc nay nv chua vertices TAI VI TRI MOI (duplicate), can map dung
        # => dung cach khac: chi them centroid, dung indices goc cho loop
        # Recompute: chi them centroid vertex
        loop_idx_arr = np.array(loop_idx, dtype=np.int64)
        lv = all_verts[loop_idx_arr]
        centroid = lv.mean(0)
        if all_colors is not None:
            centroid_color = all_colors[loop_idx_arr].mean(0)

        c_idx = len(all_verts)
        all_verts  = np.vstack([all_verts,  centroid[None, :]])
        if all_colors is not None:
            all_colors = np.vstack([all_colors, centroid_color[None, :]])

        # Tao triangle fan su dung indices goc + c_idx
        lp = list(loop_idx)
        N  = len(lp)
        # Kiem tra winding
        v0 = all_verts[lp[0]] - centroid
        v1 = all_verts[lp[1]] - centroid
        n_check = np.cross(v0, v1)
        if np.dot(n_check, outward_normal) < 0:
            lp = lp[::-1]

        cap_tris = []
        for i in range(N):
            a = lp[i]
            b = lp[(i + 1) % N]
            cap_tris.append([c_idx, a, b])

        all_tris.extend(cap_tris)
        if verbose:
            print(f"  Cap {label}: {N} boundary verts -> {N} triangles added",
                  flush=True)

    if top_loop:
        _add(top_loop, grav_up, "top")
    else:
        if verbose:
            print("  Cap top: no suitable loop found", flush=True)

    if bottom_loop:
        _add(bottom_loop, -grav_up, "bottom")
    else:
        if verbose:
            print("  Cap bottom: no suitable loop found", flush=True)

    # Fill holes giua (nho) neu duoc yeu cau
    if fill_middle_holes:
        for lp in middle_loops:
            if min_end_loop_verts <= len(lp) <= max_middle_hole_verts:
                # Uoc tinh normal cua hole tu verts xung quanh
                # Su dung gravity_up lam fallback (flat horizontal cap)
                _add(lp, grav_up, f"hole({len(lp)}v)")

    # Rebuild mesh
    out = o3d.geometry.TriangleMesh()
    out.vertices  = o3d.utility.Vector3dVector(all_verts)
    out.triangles = o3d.utility.Vector3iVector(np.array(all_tris, dtype=np.int64))
    if all_colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(all_colors)

    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_unreferenced_vertices()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Preview launcher
# ─────────────────────────────────────────────────────────────────────────────

PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".preview.pid")


def _kill_existing_preview():
    """
    Doc PID file va kill process cu neu con song.
    Tra ve True neu da kill duoc, False neu khong co process cu.
    """
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE) as f:
            lines = f.read().strip().splitlines()
        pid = int(lines[0])
        if sys.platform == "win32":
            ret = subprocess.call(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            killed = ret == 0
        else:
            import signal as _sig
            os.kill(pid, _sig.SIGTERM)
            killed = True
        if killed:
            print(f"  Kill preview cu (PID {pid})", flush=True)
        time.sleep(0.5)
    except Exception:
        pass
    try:
        os.remove(PID_FILE)
    except Exception:
        pass
    return True


def launch_preview(port: int = 8050, force_restart: bool = False):
    """
    Khoi dong 07_preview.py trong nen va mo browser.

    - Neu co PID file va process cu con song: kill truoc, start moi.
    - Neu port da co process khac (non-preview): chi mo browser.
    - force_restart=True: luon kill va restart du sao.
    """
    import socket
    preview_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "07_preview.py")

    # Kill process cu theo PID file
    had_old = _kill_existing_preview()
    if had_old or force_restart:
        time.sleep(1)   # cho port duoc giai phong

    # Kiem tra port
    port_busy = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        port_busy = s.connect_ex(("127.0.0.1", port)) == 0

    if port_busy and not had_old:
        # Co process khac chiem port (khong phai preview cua chung ta)
        print(f"\n  Port {port} dang bi chiem boi process khac.", flush=True)
        print(f"  Mo browser nhung co the khong phai preview dung.", flush=True)
    else:
        print(f"\n  Khoi dong preview server (port {port}) ...", flush=True)
        subprocess.Popen(
            [sys.executable, preview_script, "--port", str(port)],
            creationflags=subprocess.CREATE_NEW_CONSOLE
            if sys.platform == "win32" else 0,
        )
        time.sleep(3)   # cho server san sang va ghi PID file

    webbrowser.open(f"http://localhost:{port}")
    print(f"  Browser: http://localhost:{port}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-scan pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fill_middle_holes_only(mesh, grav_up=GRAVITY_UP,
                           min_hole_verts: int = 5,
                           max_hole_verts: int = 300,
                           verbose: bool = True):
    """
    Fill cac holes o GIUA than cay (khong phai 2 dau) bang triangle fan.
    Phan loai hole theo chieu cao: [30%..70%] la giua, con lai la 2 dau -> bo qua.
    """
    verts  = np.asarray(mesh.vertices,      dtype=np.float64)
    tris   = np.asarray(mesh.triangles,     dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) \
             if mesh.has_vertex_colors() else None

    origin = verts.mean(0)
    h      = (verts - origin) @ grav_up
    h_min, h_max = float(h.min()), float(h.max())
    h_range = h_max - h_min
    BOT_THR = h_min + 0.25 * h_range
    TOP_THR = h_min + 0.75 * h_range

    loops = find_boundary_loops(tris)
    middle_loops = [
        lp for lp in loops
        if (min_hole_verts <= len(lp) <= max_hole_verts and
            BOT_THR <= float(h[np.array(lp, dtype=np.int64)].mean()) <= TOP_THR)
    ]

    if not middle_loops:
        if verbose:
            print(f"  Fill middle holes: none found in [{BOT_THR:.2f}m..{TOP_THR:.2f}m]",
                  flush=True)
        return mesh

    all_verts  = verts.copy()
    all_colors = colors.copy() if colors is not None else None
    all_tris   = list(tris)

    n_filled = 0
    for lp in middle_loops:
        lp_arr = np.array(lp, dtype=np.int64)
        lv = all_verts[lp_arr]
        centroid = lv.mean(0)
        c_idx = len(all_verts)
        all_verts = np.vstack([all_verts, centroid[None, :]])
        if all_colors is not None:
            all_colors = np.vstack([all_colors, all_colors[lp_arr].mean(0)[None, :]])
        # Winding check
        v0 = all_verts[lp[0]] - centroid
        v1 = all_verts[lp[1]] - centroid
        if np.dot(np.cross(v0, v1), grav_up) < 0:
            lp = lp[::-1]
        for i in range(len(lp)):
            all_tris.append([c_idx, lp[i], lp[(i + 1) % len(lp)]])
        n_filled += 1

    out = o3d.geometry.TriangleMesh()
    out.vertices  = o3d.utility.Vector3dVector(all_verts)
    out.triangles = o3d.utility.Vector3iVector(np.array(all_tris, dtype=np.int64))
    if all_colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(all_colors)
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_unreferenced_vertices()

    if verbose:
        sizes = sorted([len(lp) for lp in middle_loops], reverse=True)
        print(f"  Fill middle holes: {n_filled} filled  sizes={sizes}", flush=True)
    return out


def process_scan(scan_dir: str,
                 smooth_iter: int    = 20,
                 do_smooth: bool     = True,
                 do_trim_top: bool   = True,
                 do_cap: bool        = False,   # mac dinh: de ho 2 dau (khong anh huong DT)
                 fill_middle: bool   = False,
                 min_end_verts: int  = 15,
                 verbose: bool       = True) -> dict:

    name    = os.path.basename(scan_dir)
    out_dir = os.path.join(scan_dir, "output")
    result  = {"name": name, "ok": False, "msg": ""}

    in_path = os.path.join(out_dir, "trunk_mesh_trimmed.ply")
    if not os.path.exists(in_path):
        result["msg"] = "trunk_mesh_trimmed.ply not found — chay 04_trim_mesh.py truoc"
        return result

    t0   = time.time()
    mesh = o3d.io.read_triangle_mesh(in_path)
    N0   = len(mesh.vertices)
    T0   = len(mesh.triangles)
    if verbose:
        print(f"  Input  : {N0:,} verts  {T0:,} tris", flush=True)

    if N0 == 0:
        result["msg"] = "mesh rong"
        return result

    # ── 1. Keep largest component ─────────────────────────────────────────────
    mesh, n_frag = keep_largest_component(mesh)
    if verbose and n_frag:
        print(f"  Keep largest: removed {n_frag:,} fragment tris  "
              f"-> {len(mesh.vertices):,} verts", flush=True)

    # ── 2. Taubin smoothing ───────────────────────────────────────────────────
    if do_smooth and smooth_iter > 0:
        mesh = taubin_smooth(mesh, n_iter=smooth_iter)
        # Smooth xoa vertex colors trong mot so version open3d
        # -> re-assign colors tu input sau smooth
        if not mesh.has_vertex_colors():
            orig = o3d.io.read_triangle_mesh(in_path)
            if orig.has_vertex_colors():
                from scipy.spatial import cKDTree
                orig_v = np.asarray(orig.vertices,      dtype=np.float32)
                orig_c = np.asarray(orig.vertex_colors, dtype=np.float32)
                new_v  = np.asarray(mesh.vertices,      dtype=np.float32)
                tree   = cKDTree(orig_v)
                _, idx = tree.query(new_v, k=1, workers=-1)
                mesh.vertex_colors = o3d.utility.Vector3dVector(orig_c[idx])
                if verbose:
                    print("  Colors re-assigned after smooth (fallback)", flush=True)
        if verbose:
            print(f"  Smooth: Taubin {smooth_iter} iter  "
                  f"-> {len(mesh.vertices):,} verts", flush=True)

    # Doc gravity vector thuc tu IMU cua scan nay
    grav_up = read_gravity_up(scan_dir)
    if verbose:
        print(f"  Gravity up: [{grav_up[0]:.3f}, {grav_up[1]:.3f}, {grav_up[2]:.3f}]",
              flush=True)

    # ── 3. Trim ragged top (cat ngon thieu data) ─────────────────────────────
    if do_trim_top:
        mesh = trim_ragged_top(mesh, grav_up=grav_up, verbose=verbose)

    # ── 3b. Fill middle holes (doc lap voi end caps) ─────────────────────────
    if fill_middle:
        mesh = fill_middle_holes_only(mesh, grav_up=grav_up, verbose=verbose)

    # ── 4. Flat caps (mac dinh tat — de ho 2 dau) ────────────────────────────
    if do_cap:
        mesh = add_end_caps(mesh,
                            grav_up=grav_up,
                            min_end_loop_verts=min_end_verts,
                            fill_middle_holes=False,   # da fill o buoc 3b
                            verbose=verbose)

    # ── 5. Recompute normals ──────────────────────────────────────────────────
    mesh.compute_vertex_normals()

    # ── 6. Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "trunk_mesh_final.ply")
    o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)

    elapsed = time.time() - t0
    N1 = len(mesh.vertices)
    T1 = len(mesh.triangles)
    result.update({
        "ok":  True,
        "msg": f"verts:{N1:,} tris:{T1:,} [{elapsed:.0f}s]",
    })
    if verbose:
        print(f"  Saved {out_path}  [{elapsed:.0f}s]", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(raw_data_dir, scan_filter=None, skip_existing=False, **kwargs):
    if scan_filter:
        dirs = [os.path.join(raw_data_dir, scan_filter)]
        if not os.path.isdir(dirs[0]):
            print(f"Scan not found: {dirs[0]}"); sys.exit(1)
    else:
        import re
        def _nat_key(s):
            parts = re.split(r'(\d+)', os.path.basename(s))
            return [int(p) if p.isdigit() else p.lower() for p in parts]
        dirs = sorted([
            os.path.join(raw_data_dir, d)
            for d in os.listdir(raw_data_dir)
            if os.path.isdir(os.path.join(raw_data_dir, d))
        ], key=_nat_key)

    # Per-scan overrides: Raw_data/finalize_overrides.json
    # Format: {"26Q13": {"fill_middle": true}, "26Q7": {"smooth_iter": 30}, ...}
    overrides_path = os.path.join(raw_data_dir, "finalize_overrides.json")
    per_scan_ov = {}
    if os.path.exists(overrides_path):
        import json as _json
        with open(overrides_path) as f:
            per_scan_ov = _json.load(f)
        print(f"Per-scan overrides loaded: {list(per_scan_ov.keys())}")

    total = len(dirs)
    print(f"\n{'='*60}")
    print(f"Finalize Mesh  |  smooth={kwargs.get('smooth_iter',10)}iter  "
          f"cap={kwargs.get('do_cap',True)}  |  {total} scans")
    print(f"{'='*60}\n")

    ok_count = skip_count = 0
    results  = []

    for i, scan_dir in enumerate(dirs):
        name     = os.path.basename(scan_dir)
        out_path = os.path.join(scan_dir, "output", "trunk_mesh_final.ply")

        if skip_existing and os.path.exists(out_path):
            print(f"[{i+1:3d}/{total}] {name} -- SKIP"); skip_count += 1; continue

        scan_kwargs = dict(kwargs)
        if name in per_scan_ov:
            scan_kwargs.update(per_scan_ov[name])
            print(f"[{i+1:3d}/{total}] {name}  [override: {per_scan_ov[name]}]")
        else:
            print(f"[{i+1:3d}/{total}] {name}")
        r = process_scan(scan_dir, verbose=True, **scan_kwargs)
        results.append(r)
        print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['msg']}\n")
        if r["ok"]:
            ok_count += 1

    print(f"{'='*60}")
    print(f"DONE: {ok_count}/{total - skip_count} OK  ({skip_count} skipped)")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['name']}: {r['msg']}")

    summary_path = os.path.join(raw_data_dir, "finalize_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"smooth_iter": kwargs.get("smooth_iter", 10),
                   "do_cap":      kwargs.get("do_cap",      True),
                   "results":     results}, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Finalize mesh: lam min + flat cap 2 dau",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  python 04b_finalize_mesh.py --scan 26Q1
  python 04b_finalize_mesh.py --scan 26Q1 --smooth 20
  python 04b_finalize_mesh.py --scan 26Q1 --no-smooth --no-cap  (chi clean fragments)
  python 04b_finalize_mesh.py --all --skip-existing
  python 04b_finalize_mesh.py --scan 26Q1 --preview    (mo preview sau khi chay)
  python 04b_finalize_mesh.py --all --preview           (batch + mo preview)
""")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan",         metavar="NAME", help="Ten scan, vd: 26Q1")
    g.add_argument("--all",          action="store_true", help="Chay toan bo scan")
    g.add_argument("--preview-only", action="store_true",
                   help="Chi mo/restart preview server, khong xu ly scan nao")

    p.add_argument("--smooth",      type=int,  default=20,
                   help="So iteration Taubin smoothing (default 20, 0=tat)")
    p.add_argument("--no-smooth",   action="store_true",
                   help="Tat smoothing")
    p.add_argument("--no-trim-top", action="store_true",
                   help="Khong cat phan ngon bi tua (giu nguyen)")
    p.add_argument("--cap",         action="store_true",
                   help="Them flat cap 2 dau (mac dinh: tat, de ho)")
    p.add_argument("--fill",        action="store_true",
                   help="Fill holes nho o giua")
    p.add_argument("--min-end-verts", type=int, default=15,
                   help="So vertices toi thieu cua loop de cap (default 15)")
    p.add_argument("--preview",        action="store_true",
                   help="Mo 07_preview.py + browser sau khi chay xong")
    p.add_argument("--preview-port",   type=int, default=8050,
                   help="Port cua preview server (default 8050)")
    p.add_argument("--raw-data",      default=RAW_DATA_DIR)
    p.add_argument("--skip-existing", action="store_true")

    args = p.parse_args()

    if args.preview_only:
        launch_preview(port=args.preview_port, force_restart=True)
        return

    run_batch(
        args.raw_data,
        scan_filter    = args.scan if args.scan else None,
        skip_existing  = args.skip_existing,
        smooth_iter    = 0 if args.no_smooth else args.smooth,
        do_smooth      = not args.no_smooth,
        do_trim_top    = not args.no_trim_top,
        do_cap         = args.cap,
        fill_middle    = args.fill,
        min_end_verts  = args.min_end_verts,
    )

    if args.preview:
        launch_preview(port=args.preview_port)


if __name__ == "__main__":
    main()
