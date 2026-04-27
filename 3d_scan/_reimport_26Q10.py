import json, urllib.request
base='http://127.0.0.1:8000'
arr=json.loads(urllib.request.urlopen(base+'/api/samples').read().decode())
exists=any(s.get('name')=='26Q10' for s in arr)
print('EXISTS_BEFORE', exists)
if not exists:
    data=json.dumps({'path': r'C:\Users\Admin\Downloads\3D_Que\Raw_data\26Q10', 'name':'26Q10'}).encode('utf-8')
    req=urllib.request.Request(base+'/api/import/directory', data=data, headers={'Content-Type':'application/json'}, method='POST')
    r=urllib.request.urlopen(req)
    print('IMPORT_STATUS', r.status)
arr2=json.loads(urllib.request.urlopen(base+'/api/samples').read().decode())
item=next((s for s in arr2 if s.get('name')=='26Q10'), None)
print('EXISTS_AFTER', item is not None)
if item:
    print('SAMPLE_ID', item.get('id'))
    print('STATUS', item.get('status'))
