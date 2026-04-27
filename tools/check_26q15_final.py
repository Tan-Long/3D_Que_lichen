import open3d as o3d, numpy as np
mesh = o3d.io.read_triangle_mesh(r'E:\3D_Que\Raw_data\26Q15\output\trunk_mesh_trimmed.ply')
v = np.asarray(mesh.vertices)
print(f'Verts: {len(v):,}  Tris: {len(mesh.triangles):,}')
print(f'Abs Y: {v[:,1].min()*100:.1f} to {v[:,1].max()*100:.1f} cm')

grav_up = np.array([0., 1., 0.])
c = v.mean(axis=0)
h = (v - c) @ grav_up
hp = (v - c) - h[:, None] * grav_up
hr = np.linalg.norm(hp, axis=1)
for _ in range(5):
    d = np.median(hp[hr <= np.percentile(hr, 30)], axis=0)
    c += d
    h = (v - c) @ grav_up
    hp = (v - c) - h[:, None] * grav_up
    hr = np.linalg.norm(hp, axis=1)
    if np.linalg.norm(d) < 0.005:
        break

print(f'Height range (centroid-relative): {h.min()*100:.1f} to {h.max()*100:.1f} cm')
edges = np.linspace(h.min(), h.max(), 16)
print('Slice  h_range(cm)       r50(cm)  r90(cm)  n')
for i in range(15):
    m = (h >= edges[i]) & (h < edges[i+1])
    if m.sum() < 5:
        continue
    r50 = np.percentile(hr[m], 50) * 100
    r90 = np.percentile(hr[m], 90) * 100
    flag = ' <<BLOB' if r90 > 10 else ''
    print(f'  {i:2d}  {edges[i]*100:5.1f}..{edges[i+1]*100:5.1f}  r50={r50:4.1f}  r90={r90:4.1f}  n={m.sum()}{flag}')
