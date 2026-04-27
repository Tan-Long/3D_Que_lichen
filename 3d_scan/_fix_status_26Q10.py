import asyncio
import json
import urllib.request
from pathlib import Path
from app.services import database as db

base='http://127.0.0.1:8000'
arr=json.loads(urllib.request.urlopen(base+'/api/samples').read().decode())
item=next((s for s in arr if s.get('name')=='26Q10'), None)
if not item:
    print('MISSING_26Q10')
    raise SystemExit(1)

source=Path(item['source_path'])
status='unprocessed'
if (source/'output'/'mesh.ply').exists():
    status='meshed'
elif (source/'rgb').is_dir() and any((source/'rgb').iterdir()):
    status='rgb_extracted'

async def main():
    updated = await db.update_sample(item['id'], status=status)
    print('UPDATED_STATUS', updated.get('status'))

asyncio.run(main())
