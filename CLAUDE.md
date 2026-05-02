# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## Project Overview

End-to-end pipeline for detecting and quantifying lichen (*địa y*) on cinnamon tree trunks (*Cinnamomum cassia*, quế) using iPhone LiDAR scans. Input: raw Stray Scanner data (RGB video, depth maps, IMU, odometry, GPS metadata). Output: colored 3D mesh with lichen labeled by type (white/green) and spatial orientation (upslope/downslope), with coverage statistics in cm² and %.

**Institution:** VinUniversity, Hanoi  
**Scale:** 500–1000 trees across 2–3 field sites  
**Current accuracy:** ~70% on clean samples (HSV adaptive thresholding)

---

## Repository Structure

```
3D_Que_lichen/
├── 01_clean_pointcloud.py       # Filter trunk from background noise
├── 01_extract_rgb_frames.py     # Extract JPEG frames from rgb.mp4 via ffmpeg
├── 02_mesh_tsdf.py              # TSDF volumetric fusion → trunk_mesh_tsdf.ply
├── 02_mesh.py                   # Poisson surface reconstruction (alternative)
├── 02_pointcloud_and_mesh.py    # Server-friendly combined stage
├── 03_recolor_mesh.py           # Project RGB texture onto mesh vertices
├── 04_trim_mesh.py              # Remove non-trunk artifacts (reads gravity from location.csv)
├── 04_auto_clean.py             # Automated trim helper
├── 04b_finalize_mesh.py         # Taubin smoothing + end caps using IMU gravity
├── 05_detect_lichen.py          # HSV adaptive detection → lichen_stats.json
├── 05_trunk_isolation.py        # Trunk-only point cloud extraction
├── 06_slope_analysis.py         # Slope/aspect from IMU + odometry → slope_info.json
├── 07_solar_exposure.py         # Solar exposure from GPS + pysolar → solar_exposure.json
├── requirements_pipeline.txt    # Pipeline dependencies (open3d, scipy, pysolar, etc.)
├── Dockerfile                   # Multi-stage build for Railway deployment
├── railway.toml                 # Railway.app deployment config
└── 3d_scan/                     # FastAPI web application (separate .git)
    ├── app/
    │   ├── main.py              # FastAPI entry point + /api/health
    │   ├── config.py            # SCAN_DATA_DIR, TEAM_MEMBERS, TSDF_VOXEL_SIZE env vars
    │   ├── services/
    │   │   ├── pipeline.py      # Orchestrates all 10 pipeline stages as subprocesses
    │   │   ├── job_manager.py   # Async job queue with WebSocket progress push
    │   │   ├── database.py      # SQLite via aiosqlite (labels: bark/white_lichen/green_lichen/artifact)
    │   │   └── sample_store.py  # Sample CRUD
    │   └── routers/
    │       ├── upload.py        # POST /api/upload — accepts ZIP
    │       ├── jobs.py          # GET /api/jobs, WebSocket /ws/jobs/{id}
    │       ├── samples.py       # CRUD + /stats /slope /solar /location /wind endpoints
    │       ├── viewer.py        # PLY viewer
    │       ├── refine.py        # Re-run TSDF with uploaded cleaned PLY
    │       └── import_router.py # Bulk import from SCAN_DATA_DIR
    └── requirements.txt         # Server-only deps (fastapi, uvicorn, aiosqlite, httpx)
```

---

## Pipeline — Stage by Stage

All scripts support `--scan NAME`, `--all`, `--skip-existing` flags.

| Stage | Script | Input | Output | Key notes |
|-------|--------|-------|--------|-----------|
| 0 | `01_extract_rgb_frames.py` | `rgb.mp4`, `depth/` | `rgb/*.jpg` | ffmpeg; frame count = depth PNG count |
| 1 | `01_clean_pointcloud.py` | `output/pointcloud.ply` | `output/trunk_pointcloud.ply` | Distance filter 0.30m from camera trajectory |
| 2 | `02_mesh_tsdf.py` | `depth/`, `rgb/`, `camera_matrix.csv`, `odometry.csv`, `confidence/` | `output/trunk_mesh_tsdf.ply` | voxel=0.005m local / 0.008m server; sdf_trunc=0.05m |
| 3 | `03_recolor_mesh.py` | `trunk_mesh_tsdf.ply`, `trunk_pointcloud.ply` | `trunk_mesh_recolored.ply` | KDTree color transfer |
| 4 | `04_trim_mesh.py` | `trunk_mesh_recolored.ply` | `trunk_mesh_trimmed.ply` | **Reads gravity from `location.csv`** (gravity_x/y/z); fallback [0,-1,0] |
| 5 | `04b_finalize_mesh.py` | `trunk_mesh_trimmed.ply`, `imu.csv` | `trunk_mesh_final.ply` | Taubin smoothing + IMU gravity caps |
| 6 | `05_detect_lichen.py` | `trunk_mesh_final.ply`, `trunk_pointcloud.ply` | `trunk_mesh_detected.ply`, `lichen_stats.json` | **Reads gravity from `location.csv`**; adaptive HSV on PCD, transfer to mesh |
| 7 | `06_slope_analysis.py` | `trunk_pointcloud.ply`, `imu.csv`, `odometry.csv` | `slope_info.json` | IMU gravity rotated to world frame via mean quaternion |
| 8 | `07_solar_exposure.py` | `location.csv`, `output/slope_info.json` | `solar_exposure.json` | pysolar monthly solar budget; fallback approximation |

---

## Input Data Format (ZIP upload)

```
<scan_name>.zip
└── <scan_name>/
    ├── camera_matrix.csv        # 3×3, comma-separated, calibrated for 1920×1440
    ├── odometry.csv             # timestamp, frame, x, y, z, qx, qy, qz, qw
    ├── imu.csv                  # timestamp, a_x, a_y, a_z (m/s²), [gyro cols]
    ├── rgb.mp4                  # H.264, 1920×1440 @ ~30fps
    ├── depth/                   # 16-bit PNG, uint16, values in mm
    │   ├── 000000.png
    │   └── ...
    ├── confidence/              # 8-bit PNG, 0=low 1=med 2=high (optional)
    │   └── ...
    │
    # --- Added by patched Stray Scanner app ---
    ├── location.csv             # Per-ARFrame GPS + IMU metadata (see schema below)
    ├── location.json            # Same data in JSON format
    ├── tree_gps_anchor.json     # GPS fix at session start (world coordinate origin)
    ├── frame_transforms.csv     # Full 4×4 camera transform per frame
    └── point_cloud_raw.ply      # Sparse ARKit feature points (not depth-derived)
```

### location.csv schema

```
timestamp_iso, timestamp_unix, latitude, longitude, altitude_asl_m,
gps_accuracy_m, heading_degrees, heading_cardinal, place_name, locality,
cam_yaw, cam_pitch, cam_roll,
slope_degrees, gravity_x, gravity_y, gravity_z
```

- `gravity_x/y/z` — CMMotionManager device-frame gravity (used by stages 4 and 6)
- `heading_degrees` — CLHeading.trueHeading (not magnetic)
- `timestamp_unix` — ARFrame.timestamp (not wall-clock)

### tree_gps_anchor.json schema

```json
{
  "scan_start_iso": "2026-04-16T09:06:33Z",
  "anchor_lat": 20.988605,
  "anchor_lon": 105.945381,
  "anchor_alt": 13.1,
  "anchor_accuracy_m": 10.3
}
```

---

## Key Implementation Details

### Gravity vector (critical)

- `04_trim_mesh.py` and `05_detect_lichen.py` both call `estimate_gravity(dataset_path)`.
- The function reads `location.csv`, takes the **median** of `gravity_x/y/z` across all frames, normalizes, and returns gravity-down (trim) or gravity-up (detect).
- **Fallback:** if `location.csv` is missing → `[0, -1, 0]` (trim) / `[0, 1, 0]` (detect).
- `06_slope_analysis.py` reads `imu.csv` directly and rotates to world frame using `odometry.csv` quaternions — this is the more accurate slope calculation.

### TSDF parameters

- Library: `open3d.pipelines.integration.ScalableTSDFVolume`
- `voxel_size`: 0.005m local, 0.008m server (`TSDF_VOXEL_SIZE` env var)
- `sdf_trunc`: 0.05m (hardcoded)
- Depth: uint16 PNG mm → float32 meters; `depth_scale=1.0` to Open3D
- `max_depth`: 2.0m
- Confidence filter: pixels with conf < 1 zeroed

### HSV lichen detection (adaptive)

Bark type determined by `bark_p20 = np.percentile(V, 20)`:

| Bark type | Condition | abs_v threshold | abs_s |
|-----------|-----------|-----------------|-------|
| Dark | bark_p20 < 0.32 | max(0.55, bark_p20 + 1.8×std_V) | 0.28 |
| Medium | 0.32–0.45 | max(0.63, bark_p20 + 2.1×std_V) | 0.24 |
| Light | ≥ 0.45 | max(0.70, bark_p20 + 2.5×std_V) | 0.20 |

Vegetation exclusion: H ∈ [65°, 150°] AND S > 0.25

---

## Web Server (3d_scan/app/)

### Running locally

```bash
cd 3d_scan
./venv311/bin/python -m uvicorn app.main:app --reload --port 8000
```

**Important:** The server uses `venv311` (not `.venv`). The `pipeline.py` subprocess calls use `venv311/bin/python` explicitly.

### Environment variables

```bash
SCAN_DATA_DIR=/path/to/raw/scans   # For bulk import
TEAM_MEMBERS=alice,bob,charlie      # Comma-separated
TSDF_VOXEL_SIZE=0.008               # 0.008 for server (RAM), 0.005 for local (quality)
```

### Pipeline stages (server)

`pipeline.py:run_pipeline()` runs 10 stages in sequence:
1. Extract ZIP (0→5%)
2. Extract RGB frames (5→10%)
3. Build quick mesh — `02_pointcloud_and_mesh.py` (10→50%)
4. TSDF fusion — `02_mesh_tsdf.py` (50→75%) — non-fatal if fails
5. Clean pointcloud (75→78%)
6. Recolor mesh (78→82%)
7. Trim mesh (82→85%)
8. Finalize mesh (85→88%)
9. Detect lichen (88→93%)
10. Slope analysis (93→96%)
11. Solar exposure (96→99%)

Progress parsed from `(N/total)` stdout pattern. WebSocket pushes to client.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Accept ZIP ≤500MB, launch pipeline |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/jobs/{id}` | Job status + progress |
| WS | `/ws/jobs/{id}` | Real-time progress |
| POST | `/api/refine` | Re-run TSDF with uploaded cleaned PLY |
| GET | `/api/samples/{id}/stats` | `lichen_stats.json` |
| GET | `/api/samples/{id}/slope` | `slope_info.json` |
| GET | `/api/samples/{id}/solar` | `solar_exposure.json` |
| GET | `/api/samples/{id}/location` | `tree_gps_anchor.json` |
| GET | `/api/samples/{id}/wind` | Open-Meteo wind data (async fetch) |
| GET | `/api/health` | `{"status":"ok"}` |

### Dashboard panels (frontend)

1. 3D mesh viewer — `trunk_mesh_detected.ply` via Three.js
2. Lichen stats table — upslope/downslope cm² and %
3. Upslope/downslope bar chart — Chart.js
4. Solar exposure monthly chart — Chart.js (12-month bar)
5. Wind rose — Chart.js polar (from Open-Meteo ERA5, 9km)
6. GPS map — Leaflet + OpenStreetMap
7. Slope info panel — terrain_slope_deg, trunk axis direction

---

## Deployment (Railway.app)

```bash
git push origin main   # Railway auto-deploys on push
```

- Builder: Dockerfile
- Health check: `GET /api/health`
- RAM requirement: min 8GB (Hobby plan) for TSDF at 8mm voxel
- System deps in Docker: `libgl1-mesa-glx`, `libgomp1`, `ffmpeg`

### Local Docker test

```bash
docker build -t lichen-server .
docker run -p 8000:8000 -e TSDF_VOXEL_SIZE=0.008 lichen-server
```

---

## Output Files

All outputs written to `<dataset_path>/output/`:

| File | Description |
|------|-------------|
| `trunk_mesh_detected.ply` | Colored mesh: white=lichen, green=green lichen, gray=bark |
| `lichen_stats.json` | Coverage by type and slope side (cm², %) |
| `slope_info.json` | gravity_world, trunk_axis, slope_deg, upslope/downslope point counts |
| `solar_exposure.json` | sun_facing_fraction, monthly_facing[12], peak_solar_month |
| `trunk_pointcloud.ply` | Cleaned trunk point cloud |
| `trunk_mesh_final.ply` | Finalized mesh before lichen labeling |

---

## Common Issues

**ffmpeg not found:** `sudo apt install ffmpeg` (server) or `brew install ffmpeg` (Mac)

**open3d TSDF OOM:** Increase `TSDF_VOXEL_SIZE` (0.008→0.010) or reduce `max_depth` (2.0→1.5)

**DBH = null:** Fewer than 20 points at z=1.2–1.4m — scan too sparse at breast height; re-scan

**Gravity fallback used:** `location.csv` missing or empty — means patched Stray Scanner app not installed; gravity defaults to hardcoded axis

**Wind data unavailable:** Open-Meteo API unreachable from Railway — check network egress settings

---

## Notes

- Code comments and variable names are primarily in Vietnamese
- `3d_scan/` has its own `.git` — treat as separate repo when committing
- CloudCompare (`tools/`) is optional for manual inspection only
- `point_cloud_raw.ply` (ARKit sparse feature points) is NOT used for TSDF — only `depth/*.png` are used