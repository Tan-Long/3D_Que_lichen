from pathlib import Path
import numpy as np
import re

root = Path(r"C:\Users\Admin\Downloads\3D_Que\Raw_data")
vals = []
for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.name):
    p = d/'camera_matrix.csv'
    if not p.is_file():
        continue
    K = np.loadtxt(p, delimiter=',')
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    idx = int(re.sub(r'[^0-9]','', d.name)) if re.search(r'\d+', d.name) else 0
    vals.append((d.name, idx, fx, fy, cx, cy))

fxs = np.array([v[2] for v in vals])
fys = np.array([v[3] for v in vals])
cxs = np.array([v[4] for v in vals])
cys = np.array([v[5] for v in vals])
print('FX_RANGE', float(fxs.min()), float(fxs.max()))
print('FY_RANGE', float(fys.min()), float(fys.max()))
print('CX_RANGE', float(cxs.min()), float(cxs.max()))
print('CY_RANGE', float(cys.min()), float(cys.max()))

# print neighbor candidates around missing folders
by_idx = {v[1]:v for v in vals}
for miss in [74,82]:
    print(f'-- CANDIDATES_FOR_26Q{miss} --')
    for j in [miss-2, miss-1, miss+1, miss+2]:
        if j in by_idx:
            n,_,fx,fy,cx,cy = by_idx[j]
            print(n, fx, fy, cx, cy)
