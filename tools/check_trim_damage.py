"""
Kiem tra tac dong cua 2-pass gravity_axis_trim tren cac mau random.
So sanh: verts hien tai vs du kien sau pass-2.
"""
import open3d as o3d, numpy as np, os, sys

RAW = os.path.join(os.path.dirname(__file__), '..', 'Raw_data')

def analyze_trim_passes(verts):
    """
    Tinh toan: neu chay gravity_axis_trim lan 2 tren verts nay thi se cat bao nhieu.
    Tra ve: (p15, p50, p85, r_max, n_removed, n_remaining)
    """
    grav_up = np.array([0.0, 1.0, 0.0])
    centroid = verts.mean(axis=0)
    h_all = (verts - centroid) @ grav_up
    h_perp = (verts - centroid) - h_all[:, None] * grav_up
    h_rad = np.linalg.norm(h_perp, axis=1)
    for _ in range(5):
        delta = np.median(h_perp[h_rad <= np.percentile(h_rad, 30)], axis=0)
        centroid = centroid + delta
        h_all = (verts - centroid) @ grav_up
        h_perp = (verts - centroid) - h_all[:, None] * grav_up
        h_rad = np.linalg.norm(h_perp, axis=1)
        if np.linalg.norm(delta) < 0.005:
            break
    p15 = float(np.percentile(h_rad, 15))
    p50 = float(np.percentile(h_rad, 50))
    p85 = float(np.percentile(h_rad, 85))
    p99 = float(np.percentile(h_rad, 99))
    r_max = max(p15, 0.025) * 2.5
    removed = int((h_rad > r_max).sum())
    return p15*100, p50*100, p85*100, p99*100, r_max*100, removed, len(verts)


# Sample IDs to test: mix of clean + problematic
samples = [1, 2, 3, 5, 6, 8, 10, 12, 15, 17, 22, 25, 32, 36,
           40, 45, 50, 57, 58, 64, 66, 70, 72, 76, 80, 84, 86, 92]

header = "ID     verts    p15   p50   p85   p99  rmax  removed   after  status"
print(header)
print("-" * 75)

damaged = []
ok = []

for qid in samples:
    path = os.path.join(RAW, f"26Q{qid}", "output", "trunk_mesh_trimmed.ply")
    if not os.path.exists(path):
        print(f"26Q{qid:<3} -- FILE NOT FOUND")
        continue
    mesh = o3d.io.read_triangle_mesh(path)
    verts = np.asarray(mesh.vertices)
    if len(verts) < 100:
        print(f"26Q{qid:<3} -- EMPTY ({len(verts)} verts)")
        continue

    p15, p50, p85, p99, rmax, rem, n = analyze_trim_passes(verts)
    after = n - rem
    pct = rem / n * 100

    if pct > 25:
        status = "<< DAMAGE"
        damaged.append((qid, pct, after))
    elif pct > 10:
        status = "< warn"
        damaged.append((qid, pct, after))
    else:
        status = "ok"
        ok.append(qid)

    print(f"26Q{qid:<3} {n:7,}  {p15:5.1f} {p50:5.1f} {p85:5.1f} {p99:5.1f} "
          f"{rmax:5.1f}  {rem:6,}({pct:4.1f}%)  {after:6,}  {status}")

print()
print(f"OK     ({len(ok)}): {ok}")
print(f"Damaged ({len(damaged)}): {[(q,f'{p:.0f}%') for q,p,_ in damaged]}")
