import re
from pathlib import Path

log=Path(r"C:\Users\Admin\.cursor\projects\c-Users-Admin-Downloads-3D-Que\terminals\3.txt")
text=log.read_text(encoding='utf-8', errors='replace').splitlines()
cur=None
status={}
order=[]
for line in text:
    m=re.search(r"\[(\d+)/(\d+)\] DANG XU LY: (.+)$", line)
    if m:
        cur=m.group(3).strip()
        if cur not in status:
            status[cur]='running'
            order.append(cur)
        continue
    if cur is None:
        continue
    if 'Saved mesh to ' in line:
        status[cur]='success'
    if 'Loi buoc Point Cloud/Mesh' in line or 'RuntimeError:' in line or 'Traceback' in line:
        if status.get(cur)!='success':
            status[cur]='fail'

succ=[k for k in order if status.get(k)=='success']
fail=[k for k in order if status.get(k)=='fail']
running=[k for k in order if status.get(k)=='running']
print('TOTAL_SEEN',len(order))
print('SUCCESS',len(succ))
print('FAIL',len(fail))
print('RUNNING',len(running))
print('LAST', order[-1] if order else 'NONE')
print('SUCCESS_LIST', ','.join(succ))
print('FAIL_LIST', ','.join(fail))
