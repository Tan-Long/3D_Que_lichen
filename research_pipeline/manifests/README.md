# Dataset manifests

Các file trong thư mục này là lớp chuẩn hóa logic. Dữ liệu raw được xem là bất biến: không đổi tên, di chuyển hoặc chỉnh sửa scan gốc.

## Nguồn

- `new`: `<project>/data raw`, dùng cho pipeline 3D, mặt đất và upslope/downslope.
- `legacy`: `<project-parent>/3D_Que/Raw_data`, dùng làm dữ liệu tham chiếu và regression cho HSV cũ.
- `<project>/Raw_data.rar`: archive của bộ legacy; giữ nguyên, không giải nén lần nữa khi thư mục legacy đã tồn tại.

## Artifacts

- `scans.csv`: một dòng cho mỗi scan, gồm logical ID, tree leakage key, provenance, nhãn, QA và fingerprint.
- `duplicate_candidates.csv`: scan trùng nội dung hoặc có cùng `tree_key`, cần kiểm tra trước khi chia train/test.
- `pilot.csv`: tập pilot đề xuất và bộ legacy dùng regression HSV.
- `incomplete_captures.csv`: scan thiếu đầu vào hình học hoặc có MP4 chưa hoàn tất.
- `summary.json`: tổng hợp số lượng và chính sách archive.
- `production_baseline_v1.json` và `.csv`: snapshot bất biến của toàn bộ 299 mẫu,
  quyết định thủ công mới nhất và đường dẫn baseline production đã chốt.
- `production_baseline_v1_approved.csv`: đầu vào duy nhất cho khâu downstream;
  chỉ chứa mẫu `approved` có baseline mesh và metrics tồn tại.
- `production_baseline_v1.lock.json`: SHA-256 của ba file snapshot để phát hiện
  manifest bị thay đổi ngoài ý muốn.
- `legacy_audit.md`, `legacy_audit.json`, `legacy_audit.csv`: kết quả rà soát 92 scan legacy.

Các bản export lặp đã được chuyển khỏi tập active sang `<project>/quarantine/duplicate_scans`. File `<project>/quarantine/duplicate_scans.csv` ghi đường dẫn gốc, bản canonical và vị trí phục hồi.

Tái tạo manifest:

```powershell
python tools/build_dataset_manifest.py
```

Chỉ tạo lại snapshot production khi chủ dự án yêu cầu chốt một phiên bản mới:

```bash
.venv/bin/python tools/finalize_baseline_manifest.py
```

Theo dõi và tiếp tục TSDF recovery trên macOS:

```bash
.venv/bin/python tools/show_recovery_status.py
.venv/bin/python tools/run_tsdf_recovery.py --workers 2
```

`content_fingerprint` băm toàn bộ `camera_matrix.csv` và `odometry.csv`, kết hợp kích thước video cùng số frame depth/confidence. Đây là fingerprint phát hiện bản sao scan, không phải checksum lưu trữ của toàn bộ dữ liệu.
