import json, urllib.request
base='http://127.0.0.1:8000'
data=json.dumps({'path': r'C:\Users\Admin\Downloads\3D_Que\Raw_data\26Q10', 'name':'26Q10'}).encode('utf-8')
req=urllib.request.Request(base+'/api/import/directory', data=data, headers={'Content-Type':'application/json'}, method='POST')
urllib.request.urlopen(req)
arr=json.loads(urllib.request.urlopen(base+'/api/samples').read().decode())
item=next((s for s in arr if s.get('name')=='26Q10'), None)
print('FINAL_STATUS', item.get('status') if item else 'MISSING')
print('FINAL_FILES', len(item.get('files',[])) if item else 0)
