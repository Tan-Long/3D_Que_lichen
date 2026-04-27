"""Debug script for 26Q15 PCA pipeline tracing."""
import open3d as o3d, numpy as np, copy, sys
sys.path.insert(0, r'E:\3D_Que')

def pca_axis(pts):
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, np.argmax(vals)]

# Load
mesh = o3d.io.read_triangle_mesh(r'E:\3D_Que\Raw_data\26Q15\output\trunk_mesh_recolored.ply')
verts = np.asarray(mesh.vertices)
print(f'Recolored: {len(verts):,} verts, abs Y {verts[:,1].min()*100:.1f} to {verts[:,1].max()*100:.1f} cm, mean Y={verts[:,1].mean()*100:.1f}')

# Gravity trim simulation
grav_up = np.array([0., 1., 0.])
centroid = verts.mean(axis=0)
h_all = (verts - centroid) @ grav_up
h_perp = (verts - centroid) - h_all[:, None] * grav_up
h_rad = np.linalg.norm(h_perp, axis=1)
for _ in range(5):
    delta = np.median(h_perp[h_rad <= np.percentile(h_rad, 30)], axis=0)
    centroid += delta
    h_all = (verts - centroid) @ grav_up
    h_perp = (verts - centroid) - h_all[:, None] * grav_up
    h_rad = np.linalg.norm(h_perp, axis=1)
    if np.linalg.norm(delta) < 0.005: break

ref_r = max(float(np.percentile(h_rad, 15)), 0.025)
r_max = ref_r * 2.5
print(f'Gravity trim: ref_r={ref_r*100:.1f}cm, r_max={r_max*100:.1f}cm')

keep_radial = h_rad <= r_max
print(f'After gravity radial cut: keep {keep_radial.sum():,} verts')

# Simulate removal
mesh2 = copy.deepcopy(mesh)
mesh2.remove_vertices_by_mask(~keep_radial)
mesh2.remove_unreferenced_vertices()
mesh2.remove_degenerate_triangles()
verts2 = np.asarray(mesh2.vertices)
print(f'After gravity trim: {len(verts2):,} verts, abs Y {verts2[:,1].min()*100:.1f} to {verts2[:,1].max()*100:.1f} cm, mean Y={verts2[:,1].mean()*100:.1f}')

# PCA
axis = pca_axis(verts2)
origin = verts2.mean(axis=0)
print(f'PCA axis: {axis.round(3)}, origin Y={origin[1]*100:.1f}cm')

t_all = (verts2 - origin) @ axis
print(f'PCA t range: {t_all.min():.3f} to {t_all.max():.3f} m')

# Check: where are low-Y vertices in PCA space?
low_y_mask = verts2[:, 1] < -0.2  # below -20cm absolute
if low_y_mask.sum() > 0:
    t_low = t_all[low_y_mask]
    print(f'Low-Y verts (Y<-20cm): {low_y_mask.sum():,}, PCA t range {t_low.min():.3f} to {t_low.max():.3f}')

high_y_mask = verts2[:, 1] > 0.3  # above +30cm absolute
if high_y_mask.sum() > 0:
    t_high = t_all[high_y_mask]
    print(f'High-Y verts (Y>+30cm): {high_y_mask.sum():,}, PCA t range {t_high.min():.3f} to {t_high.max():.3f}')
