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
    m = re.search(r'Q(\d+)$', d.name)
    if not m:
        continue
    idx = int(m.group(1))
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    vals.append((d.name, idx, fx, fy, cx, cy))

by_idx = {v[1]:v for v in vals}
for miss in [74,82]:
    print(f'-- CANDIDATES_FOR_26Q{miss} --')
    for j in [miss-2, miss-1, miss+1, miss+2]:
        if j in by_idx:
            n,_,fx,fy,cx,cy = by_idx[j]
            print(n, f'fx={fx}', f'fy={fy}', f'cx={cx}', f'cy={cy}')
