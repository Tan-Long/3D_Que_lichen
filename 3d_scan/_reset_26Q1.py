import json
import urllib.request

samples = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/samples').read().decode())
target = next((s for s in samples if s.get('name') == '26Q1'), None)
if not target:
    print('NOT_FOUND_SAMPLE_26Q1')
    raise SystemExit(1)

sid = target['id']
url = f"http://127.0.0.1:8000/api/samples/{sid}/reset"
req = urllib.request.Request(url, data=b'', method='POST')
resp = urllib.request.urlopen(req)
body = json.loads(resp.read().decode())
print('RESET_OK', sid)
print('STATUS', body['sample'].get('status'))
print('FILES', len(body['sample'].get('files', [])))
print('FILE_NAMES', ', '.join(f['filename'] for f in body['sample'].get('files', [])))
