import json
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\Admin\Downloads\3D_Que\Raw_data")
folders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

mats = {}
missing = []
for d in folders:
    p = d / 'camera_matrix.csv'
    if p.is_file():
        try:
            k = np.loadtxt(p, delimiter=',')
            mats[d.name] = k
        except Exception:
            missing.append(d.name)
    else:
        missing.append(d.name)

# group unique matrices by approximate equality
keys = list(mats.keys())
used = set()
groups = []
for k in keys:
    if k in used:
        continue
    base = mats[k]
    group = [k]
    used.add(k)
    for kk in keys:
        if kk in used:
            continue
        if np.allclose(base, mats[kk], rtol=1e-9, atol=1e-9):
            group.append(kk)
            used.add(kk)
    groups.append(group)

# sort biggest group first
groups.sort(key=len, reverse=True)

print('TOTAL_WITH_MATRIX', len(mats))
print('MISSING', ','.join(missing))
print('UNIQUE_GROUPS', len(groups))
for i,g in enumerate(groups[:5],1):
    print(f'GROUP_{i}_SIZE', len(g), 'SAMPLE', g[:5])

if groups:
    ref = groups[0][0]
    ref_mat = mats[ref]
    print('REF_FOLDER', ref)
    print('REF_MATRIX')
    for row in ref_mat:
        print(','.join(f'{x:.12f}' for x in row))

# decide safe fill if dominant group has all existing matrices
safe_fill = len(groups) == 1
print('SAFE_FILL', safe_fill)
