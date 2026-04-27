import json
import os
import urllib.request

arr = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/samples').read().decode())
pick = None
for s in arr:
    p = s.get('source_path')
    if p and os.path.isdir(os.path.join(p, 'output')):
        pick = s
        break

print('PICK', pick['id'], pick['name'])
url = f"http://127.0.0.1:8000/api/samples/{pick['id']}/reset"
req = urllib.request.Request(url, data=b'', method='POST')
r = urllib.request.urlopen(req)
print('RESET_STATUS', r.status)
print(r.read().decode()[:220])
