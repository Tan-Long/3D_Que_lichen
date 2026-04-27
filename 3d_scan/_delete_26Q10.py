import json
import urllib.request

base='http://127.0.0.1:8000'
samples = json.loads(urllib.request.urlopen(base + '/api/samples').read().decode())
target = next((s for s in samples if s.get('name') == '26Q10'), None)
if not target:
    print('NOT_FOUND_26Q10')
    raise SystemExit(0)

sid = target['id']
req = urllib.request.Request(base + f'/api/samples/{sid}', method='DELETE')
resp = urllib.request.urlopen(req)
print('DELETE_STATUS', resp.status)

samples2 = json.loads(urllib.request.urlopen(base + '/api/samples').read().decode())
exists = any(s.get('name') == '26Q10' for s in samples2)
print('EXISTS_AFTER_DELETE', exists)
