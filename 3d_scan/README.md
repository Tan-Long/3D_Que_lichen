# 3D Scan Processor

Web app (FastAPI) + script pipeline xử lý dữ liệu Stray Scanner: extract RGB, point cloud / mesh (Open3D), chuẩn bị scene cho annotation.

## Chạy local

```bash
cd /path/to/3d_scan
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt: `http://localhost:8000` (không dùng `http://0.0.0.0:8000`).

## Pipeline (CLI)

Xem các file `01_extract_rgb_frames.py`, `02_pointcloud_and_mesh.py`, `03_prepare_for_annotation.py`.

## Dữ liệu lớn

Thư mục `venv311/`, `uploads/`, dataset (`c9ae76fff8/`, `*.zip`) được `.gitignore` — không đẩy lên Git.
