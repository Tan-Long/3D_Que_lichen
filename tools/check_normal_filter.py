"""
Kiem tra tac dong cua normal_ground_filter(thr=0.75) tren tat ca 92 mau.
Chay tren trunk_mesh_trimmed.ply hien tai (da qua gravity trim).
"""
import open3d as o3d, numpy as np, os, copy, sys

RAW = os.path.join(os.path.dirname(__file__), '..', 'Raw_data')
GRAV_DOWN = np.array([0.0, -1.0, 0.0])
NORMAL_THR = 0.75

results = []

scan_dirs = sorted([d for d in os.listdir(RAW)
                    if os.path.isdir(os.path.join(RAW, d)) and d.startswith('26Q')])

print(f"{'ID':<7} {'verts':>7} {'faces':>7}  {'removed_f':>9}  {'rm%':>5}  {'after_v':>7}  {'status'}")
print("-" * 65)

for name in scan_dirs:
    path = os.path.join(RAW, name, "output", "trunk_mesh_trimmed.ply")
    if not os.path.exists(path):
        continue
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_triangle_normals()
    face_normals = np.asarray(mesh.triangle_normals)
    dots = face_normals @ GRAV_DOWN          # +1 = ngang xuong, -1 = ngang len
    ground_mask = np.abs(dots) > NORMAL_THR
    n_faces = len(face_normals)
    n_remove = int(ground_mask.sum())
    pct = n_remove / max(n_faces, 1) * 100

    # Simulate removal
    mesh2 = copy.deepcopy(mesh)
    mesh2.remove_triangles_by_mask(ground_mask)
    mesh2.remove_unreferenced_vertices()
    after_v = len(mesh2.vertices)

    if pct > 20:
        status = "<< HIGH"
    elif pct > 10:
        status = "< warn"
    else:
        status = "ok"

    print(f"{name:<7} {len(np.asarray(mesh.vertices)):>7,} {n_faces:>7,}  "
          f"{n_remove:>8,}({pct:4.1f}%)  {after_v:>7,}  {status}")
    results.append((name, pct))

high = [(n, p) for n, p in results if p > 20]
warn = [(n, p) for n, p in results if 10 < p <= 20]
ok   = [(n, p) for n, p in results if p <= 10]
print()
print(f"OK   ({len(ok)}): {[n for n,_ in ok]}")
print(f"Warn ({len(warn)}): {[(n,f'{p:.0f}%') for n,p in warn]}")
print(f"HIGH ({len(high)}): {[(n,f'{p:.0f}%') for n,p in high]}")
