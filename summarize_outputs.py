from pathlib import Path
import json

root=Path(r"C:\Users\Admin\Downloads\3D_Que\Raw_data")
folders=sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.name)

success=[]
fail=[]
for d in folders:
    mesh=d/'output'/'mesh.ply'
    pcd=d/'output'/'pointcloud.ply'
    if mesh.is_file() and pcd.is_file():
        success.append(d.name)
    else:
        reason=[]
        if not pcd.is_file():
            reason.append('missing_pointcloud')
        if not mesh.is_file():
            reason.append('missing_mesh')
        fail.append((d.name, ','.join(reason)))

print('TOTAL', len(folders))
print('SUCCESS', len(success))
print('FAIL', len(fail))
print('---SUCCESS_LIST---')
for s in success:
    print(s)
print('---FAIL_LIST---')
for n,r in fail:
    print(f'{n}: {r}')
