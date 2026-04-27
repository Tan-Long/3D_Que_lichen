#!/usr/bin/env python3
"""
09_demo.py - Demo pipeline step-by-step:
  Video quay  ->  RGB+Depth frames  ->  Point Cloud  ->  TSDF Mesh
  ->  Tô màu  ->  Cắt lọc  ->  Làm mịn  ->  Phát hiện địa y

Chạy tại: http://localhost:8052
"""

import base64, cv2, json, os, re, sys, time
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(ROOT, "Raw_data")
PORT         = 8052

# ─── Step definitions ─────────────────────────────────────────────────────────
STEPS = [
    {"id": "video",   "icon": "🎬", "label": "Video quay"},
    {"id": "depth",   "icon": "📸", "label": "RGB + Depth"},
    {"id": "pcd",     "icon": "☁",  "label": "Point Cloud"},
    {"id": "tsdf",    "icon": "▦",  "label": "TSDF Mesh"},
    {"id": "recolor", "icon": "🎨", "label": "Tô màu RGB"},
    {"id": "trim",    "icon": "✂",  "label": "Cắt & lọc"},
    {"id": "lichen",  "icon": "🌿", "label": "Địa y"},
]

DESCS = {
    "video": {
        "title": "Video quay – Stray Scanner (iPhone LiDAR)",
        "body": (
            "Quét vỏ cây quế bằng iPhone tích hợp cảm biến LiDAR (ứng dụng Stray Scanner). "
            "Camera di chuyển từ gốc lên ngọn, đồng thời ghi lại video RGB, depth map theo "
            "từng frame, tín hiệu gia tốc kế IMU và pose trajectory (odometry) từ ARKit."
        ),
        "tech": "rgb.mp4  ·  odometry.csv  ·  imu.csv  ·  camera_matrix.csv",
    },
    "depth": {
        "title": "RGB + Depth frames đồng bộ",
        "body": (
            "Mỗi frame đồng bộ gồm ảnh RGB độ phân giải cao (1920×1440 px) và depth map "
            "từ cảm biến LiDAR (256×192 px, uint16 – đơn vị mm). "
            "Confidence map (0/1/2) lọc bỏ các điểm đo kém tin cậy trước khi dựng 3D."
        ),
        "tech": "rgb/*.jpg  ·  depth/*.png (uint16, mm)  ·  confidence/*.png",
    },
    "pcd": {
        "title": "Point Cloud 3D thô",
        "body": (
            "Chiếu từng pixel depth ra không gian 3D dùng ma trận intrinsic camera "
            "và pose trajectory (extrinsic). Lọc theo bán kính camera để chỉ giữ "
            "vùng thân cây, loại bỏ nền và mặt đất."
        ),
        "tech": "01_clean_pointcloud.py  →  trunk_pointcloud.ply",
    },
    "tsdf": {
        "title": "TSDF Mesh – bề mặt liên tục từ depth fusion",
        "body": (
            "TSDF (Truncated Signed Distance Function) fusion: tích hợp toàn bộ depth "
            "frames thành một bề mặt mesh 3D liên tục bằng Marching Cubes. "
            "Phương pháp này tự động lấp đầy khoảng trống giữa các frame và khử nhiễu đo lường."
        ),
        "tech": "02_mesh_tsdf.py  ·  voxel_size ≈ 4mm  →  trunk_mesh_tsdf.ply",
    },
    "recolor": {
        "title": "Tô màu RGB lên mesh",
        "body": (
            "Ray-casting từ mỗi vertex về các frame camera: tìm frame nhìn thấy vertex "
            "rõ nhất (góc nhìn tốt, không bị che khuất), lấy màu pixel tương ứng từ "
            "ảnh RGB độ phân giải cao. Kết quả là mesh có màu sắc thực tế của vỏ cây."
        ),
        "tech": "03_recolor_mesh.py  →  trunk_mesh_recolored.ply",
    },
    "trim": {
        "title": "Cắt và lọc mesh",
        "body": (
            "Phân tích PCA để tìm trục dọc thân cây. Lọc radial loại bỏ đất, gốc rễ, "
            "cành nhỏ và các fragment thừa (min 1500 triangles). Gravity-axis trim "
            "loại bỏ blob đất. Cắt 2 đầu bị lởm chởm bằng boundary density analysis."
        ),
        "tech": "04_trim_mesh.py  →  trunk_mesh_trimmed.ply",
    },
    "lichen": {
        "title": "Phát hiện địa y (Lichen detection)",
        "body": (
            "Phân loại màu vertex sang HSV: địa y trắng/crustose có V > 70% và S < 25% "
            "(sáng và nhạt màu hơn vỏ cây). Chia upslope / downslope theo vector trọng lực "
            "từ IMU. Tính diện tích bằng tổng diện tích các triangle thuộc vùng địa y."
        ),
        "tech": "05_detect_lichen.py  →  trunk_mesh_detected.ply  +  lichen_stats.json",
    },
}

# ─── Scan list ────────────────────────────────────────────────────────────────
def nat_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

SCANS = sorted(
    [d for d in os.listdir(RAW_DATA_DIR)
     if os.path.isdir(os.path.join(RAW_DATA_DIR, d))],
    key=nat_key,
)

# ─── Data cache ───────────────────────────────────────────────────────────────
CACHE = {}   # (scan_id, step_id) -> {"type": ..., "content": ..., "stats": ...}

# ─── Viz helpers ──────────────────────────────────────────────────────────────
DARK_BG = '#0f1117'

def _scene():
    return dict(
        bgcolor=DARK_BG,
        xaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
        zaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
        camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.5, y=0.4, z=0.5)),
        aspectmode='data',
    )

def _layout():
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color='#e5e7eb'),
        margin=dict(l=0, r=0, t=0, b=0),
    )

def _img_b64(bgr, fmt='.jpg', q=85):
    if fmt == '.jpg':
        _, buf = cv2.imencode(fmt, bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    else:
        _, buf = cv2.imencode(fmt, bgr)
    mime = 'image/jpeg' if fmt == '.jpg' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(buf).decode()

def _resize(img, max_w=960, max_h=540):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def _mesh_to_fig(mesh, max_tris=70000):
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices,  dtype=np.float32)
    t = np.asarray(mesh.triangles, dtype=np.int32)

    # Random triangle sampling — O(n), nhanh hơn quadric decimation nhiều
    if len(t) > max_tris:
        idx = np.random.default_rng(42).choice(len(t), max_tris, replace=False)
        idx.sort()
        t = t[idx]

    vc = None
    if mesh.has_vertex_colors():
        c  = (np.asarray(mesh.vertex_colors) * 255).astype(int).clip(0, 255)
        vc = ['rgb({},{},{})'.format(r, g, b) for r, g, b in c]
    trace = go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=t[:, 0], j=t[:, 1], k=t[:, 2],
        vertexcolor=vc,
        color='#8B7355' if vc is None else None,
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1, roughness=0.6),
        lightposition=dict(x=2, y=4, z=2),
        flatshading=False,
    )
    fig = go.Figure(trace)
    fig.update_layout(scene=_scene(), **_layout())
    stats = dict(
        verts=f"{len(v):,}",
        tris=f"{len(t):,}",
        height=f"{float(v[:,1].max()-v[:,1].min()):.2f} m",
    )
    return fig, stats

# ─── Per-step data loaders ────────────────────────────────────────────────────

def load_video(scan_dir):
    video_path = os.path.join(scan_dir, "rgb.mp4")
    if not os.path.exists(video_path):
        return {"type": "error", "msg": "rgb.mp4 not found"}

    cap = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    dur_s   = total / fps

    # 5 evenly spaced frames
    positions = [int(total * p) for p in (0.05, 0.2, 0.45, 0.7, 0.9)]
    imgs = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _resize(frame, max_w=700, max_h=394)
        imgs.append(_img_b64(frame))
    cap.release()

    stats = dict(
        frames=f"{total:,}",
        fps=f"{fps:.0f}",
        duration=f"{dur_s:.1f} s",
    )
    return {"type": "frames", "imgs": imgs, "stats": stats}


def load_depth(scan_dir):
    rgb_dir   = os.path.join(scan_dir, "rgb")
    depth_dir = os.path.join(scan_dir, "depth")
    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
        return {"type": "error", "msg": "rgb/ or depth/ not found"}

    frames = sorted(os.listdir(rgb_dir))
    if not frames:
        return {"type": "error", "msg": "no frames"}

    # Use frame at ~30% through scan for a clear trunk view
    idx    = int(len(frames) * 0.30)
    fname  = os.path.splitext(frames[idx])[0]

    rgb   = cv2.imread(os.path.join(rgb_dir,   f"{fname}.jpg"))
    depth = cv2.imread(os.path.join(depth_dir, f"{fname}.png"), cv2.IMREAD_ANYDEPTH)

    if rgb is None or depth is None:
        return {"type": "error", "msg": "frame read failed"}

    rgb_small = _resize(rgb, max_w=700, max_h=400)

    # Depth colormap: mask zeros, apply PLASMA
    d = depth.astype(np.float32)
    valid = d > 0
    if valid.any():
        d_norm = np.zeros_like(d)
        dmin, dmax = float(d[valid].min()), float(d[valid].max())
        d_norm[valid] = (d[valid] - dmin) / (dmax - dmin + 1e-6)
    else:
        d_norm = d
        dmin, dmax = 0, 0

    depth_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    # Black out invalid pixels
    depth_color[~valid] = [20, 20, 20]
    depth_small = _resize(depth_color, max_w=700, max_h=400)

    # Add labels
    for img, text in [(rgb_small, "RGB (1920x1440)"), (depth_small, f"Depth map ({dmin/10:.0f}–{dmax/10:.0f} cm)")]:
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30),   1, cv2.LINE_AA)

    stats = dict(
        frame=fname,
        depth_range=f"{dmin/10:.0f}–{dmax/10:.0f} cm",
        rgb_res="1920×1440 px",
        depth_res="256×192 px",
    )
    return {
        "type":  "dual_image",
        "img1":  _img_b64(rgb_small),
        "img2":  _img_b64(depth_small),
        "stats": stats,
    }


def load_pcd(out_dir, scan_dir):
    ply_path = os.path.join(out_dir, "trunk_pointcloud.ply")
    if not os.path.exists(ply_path):
        # Fallback: sample vertices from TSDF mesh
        mesh_path = os.path.join(out_dir, "trunk_mesh_tsdf.ply")
        if not os.path.exists(mesh_path):
            return {"type": "error", "msg": "trunk_pointcloud.ply not found"}
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pcd  = mesh.sample_points_uniformly(number_of_points=25000)
    else:
        pcd = o3d.io.read_point_cloud(ply_path)

    pts  = np.asarray(pcd.points,  dtype=np.float32)
    cols = np.asarray(pcd.colors,  dtype=np.float32) if pcd.has_colors() else None

    # Subsample
    max_pts = 25000
    if len(pts) > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]

    if cols is not None:
        c     = (cols * 255).astype(int).clip(0, 255)
        color = ['rgb({},{},{})'.format(r, g, b) for r, g, b in c]
    else:
        color = '#9ca3af'

    fig = go.Figure(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=color, opacity=0.85),
    ))
    fig.update_layout(scene=_scene(), **_layout())

    stats = dict(
        pts=f"{len(pts):,}",
        height=f"{float(pts[:,1].max()-pts[:,1].min()):.2f} m",
        note="(hiển thị 25k / tổng)",
    )
    return {"type": "3d", "fig": fig, "stats": stats}


def load_mesh(out_dir, filename, step_id):
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        return {"type": "error", "msg": f"{filename} not found"}
    mesh = o3d.io.read_triangle_mesh(path)
    fig, stats = _mesh_to_fig(mesh)
    return {"type": "3d", "fig": fig, "stats": stats}


def load_lichen(out_dir, scan_id):
    path  = os.path.join(out_dir, "trunk_mesh_detected.ply")
    jpath = os.path.join(out_dir, "lichen_stats.json")
    if not os.path.exists(path):
        return {"type": "error", "msg": "trunk_mesh_detected.ply not found"}

    mesh = o3d.io.read_triangle_mesh(path)
    fig, mesh_stats = _mesh_to_fig(mesh)

    lichen_stats = {}
    if os.path.exists(jpath):
        with open(jpath) as f:
            d = json.load(f)
        s = d.get("stats", {})
        lichen_stats = dict(
            pct_total=f"{s['overall']['lichen_ratio_pct']:.1f} %",
            area_total=f"{s['overall']['total_cm2']:.0f} cm²",
            lichen_total=f"{s['overall']['total_lichen_cm2']:.0f} cm²",
            upslope_pct=f"{s['upslope']['lichen_ratio_pct']:.1f} %",
            downslope_pct=f"{s['downslope']['lichen_ratio_pct']:.1f} %",
        )

    return {"type": "3d", "fig": fig, "stats": {**mesh_stats, **lichen_stats}}


# ─── Unified loader ───────────────────────────────────────────────────────────

def get_data(scan_id, step_idx):
    step = STEPS[step_idx]
    key  = (scan_id, step["id"])
    if key in CACHE:
        return CACHE[key]

    scan_dir = os.path.join(RAW_DATA_DIR, scan_id)
    out_dir  = os.path.join(scan_dir, "output")

    loaders = {
        "video":   lambda: load_video(scan_dir),
        "depth":   lambda: load_depth(scan_dir),
        "pcd":     lambda: load_pcd(out_dir, scan_dir),
        "tsdf":    lambda: load_mesh(out_dir, "trunk_mesh_tsdf.ply",      "tsdf"),
        "recolor": lambda: load_mesh(out_dir, "trunk_mesh_recolored.ply", "recolor"),
        "trim":    lambda: load_mesh(out_dir, "trunk_mesh_trimmed.ply",   "trim"),
        "lichen":  lambda: load_lichen(out_dir, scan_id),
    }

    t0 = time.time()
    data = loaders[step["id"]]()
    data["_load_ms"] = int((time.time() - t0) * 1000)
    CACHE[key] = data
    return data


# ─── Dash app ─────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Demo Pipeline Scan → Địa y")

# ── Styles ────────────────────────────────────────────────────────────────────
S = {
    "app": {
        "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        "background": DARK_BG,
        "color": "#e5e7eb",
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "overflow": "hidden",
    },
    "header": {
        "background": "#1f2937",
        "borderBottom": "1px solid #374151",
        "padding": "0 20px",
        "height": "56px",
        "display": "flex",
        "alignItems": "center",
        "gap": "20px",
        "flexShrink": "0",
    },
    "body": {
        "display": "flex",
        "flex": "1",
        "overflow": "hidden",
    },
    "sidebar": {
        "width": "220px",
        "minWidth": "220px",
        "background": "#111827",
        "borderRight": "1px solid #374151",
        "padding": "16px 12px",
        "display": "flex",
        "flexDirection": "column",
        "overflow": "hidden",
    },
    "main": {
        "flex": "1",
        "display": "flex",
        "flexDirection": "column",
        "overflow": "hidden",
    },
    "viz_area": {
        "flex": "1",
        "position": "relative",
        "overflow": "hidden",
        "minHeight": "0",
    },
    "info_bar": {
        "background": "#1f2937",
        "borderTop": "1px solid #374151",
        "padding": "14px 20px",
        "flexShrink": "0",
        "maxHeight": "200px",
        "overflowY": "auto",
    },
    "btn": {
        "background": "#374151",
        "color": "#e5e7eb",
        "border": "none",
        "borderRadius": "6px",
        "padding": "7px 16px",
        "cursor": "pointer",
        "fontSize": "13px",
        "fontWeight": "500",
    },
    "btn_primary": {
        "background": "#4f46e5",
        "color": "white",
        "border": "none",
        "borderRadius": "6px",
        "padding": "7px 18px",
        "cursor": "pointer",
        "fontSize": "13px",
        "fontWeight": "500",
    },
}

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    # Header
    html.Div([
        html.Span("🌿", style={"fontSize": "22px"}),
        html.Span("Demo Pipeline 3D Scan → Phát hiện Địa y",
                  style={"fontWeight": "600", "fontSize": "15px"}),
        html.Div(style={"flex": "1"}),
        html.Span("Chọn mẫu:", style={"fontSize": "13px", "color": "#9ca3af"}),
        dcc.Dropdown(
            id="dd-scan",
            options=[{"label": s, "value": s} for s in SCANS],
            value="26Q1",
            clearable=False,
            style={"width": "130px", "fontSize": "13px"},
        ),
    ], style=S["header"]),

    # Body
    html.Div([
        # ── Sidebar ───────────────────────────────────────────────────────────
        html.Div([
            html.Div(
                id=f"step-btn-{i}",
                children=[
                    html.Span(s["icon"], style={"marginRight": "8px", "fontSize": "14px"}),
                    html.Span(s["label"], style={"fontSize": "13px"}),
                ],
                n_clicks=0,
                style={"padding": "9px 12px", "borderRadius": "8px", "cursor": "pointer",
                       "marginBottom": "4px", "display": "flex", "alignItems": "center",
                       "transition": "background 0.15s"},
            )
            for i, s in enumerate(STEPS)
        ] + [
            html.Div(style={"flex": "1"}),
            html.Div(id="progress-text",
                     style={"fontSize": "12px", "color": "#6b7280",
                            "marginBottom": "12px", "textAlign": "center"}),
            html.Div([
                html.Button("← Trước", id="btn-prev", n_clicks=0, style=S["btn"]),
                html.Button("Tiếp →",  id="btn-next", n_clicks=0, style=S["btn_primary"]),
            ], style={"display": "flex", "gap": "8px"}),
        ], style=S["sidebar"]),

        # ── Main ──────────────────────────────────────────────────────────────
        html.Div([
            dcc.Loading(
                html.Div(id="viz-area", style=S["viz_area"]),
                type="circle", color="#4f46e5",
                style={"height": "100%"},
            ),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(id="info-title",
                                  style={"fontWeight": "600", "fontSize": "14px",
                                         "color": "#e5e7eb", "display": "block",
                                         "marginBottom": "4px"}),
                        html.Span(id="info-body",
                                  style={"fontSize": "13px", "color": "#9ca3af",
                                         "lineHeight": "1.5"}),
                        html.Div(id="info-tech",
                                 style={"fontFamily": "monospace", "fontSize": "11px",
                                        "color": "#6b7280", "marginTop": "4px"}),
                    ], style={"flex": "1", "marginRight": "24px"}),
                    html.Div(id="stats-panel"),
                ], style={"display": "flex", "alignItems": "flex-start"}),
            ], style=S["info_bar"]),
        ], style=S["main"]),

    ], style=S["body"]),

    # State store
    dcc.Store(id="cur-step", data=0),

], style=S["app"])


# ─── Callbacks ────────────────────────────────────────────────────────────────

# Navigation
@app.callback(
    Output("cur-step", "data"),
    [Input("btn-prev",       "n_clicks"),
     Input("btn-next",       "n_clicks")] +
    [Input(f"step-btn-{i}", "n_clicks") for i in range(len(STEPS))],
    State("cur-step", "data"),
    prevent_initial_call=True,
)
def navigate(n_prev, n_next, *args):
    step_clicks = args[:len(STEPS)]
    current     = args[len(STEPS)]
    ctx         = callback_context
    if not ctx.triggered:
        return current
    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    if tid == "btn-prev":
        return max(0, current - 1)
    if tid == "btn-next":
        return min(len(STEPS) - 1, current + 1)
    for i in range(len(STEPS)):
        if tid == f"step-btn-{i}":
            return i
    return current


# Sidebar step highlighting
@app.callback(
    [Output(f"step-btn-{i}", "style") for i in range(len(STEPS))],
    Input("cur-step", "data"),
)
def highlight_step(current):
    styles = []
    for i in range(len(STEPS)):
        base = {"padding": "9px 12px", "borderRadius": "8px", "cursor": "pointer",
                "marginBottom": "4px", "display": "flex", "alignItems": "center",
                "transition": "background 0.15s"}
        if i == current:
            styles.append({**base, "background": "#4f46e5", "color": "white"})
        elif i < current:
            styles.append({**base, "background": "#1a3a2e", "color": "#4ade80"})
        else:
            styles.append({**base, "background": "transparent", "color": "#6b7280"})
    return styles


# Progress text
@app.callback(
    Output("progress-text", "children"),
    Input("cur-step", "data"),
)
def update_progress(current):
    return f"Bước {current + 1} / {len(STEPS)}"


# Main render
@app.callback(
    Output("viz-area",    "children"),
    Output("info-title",  "children"),
    Output("info-body",   "children"),
    Output("info-tech",   "children"),
    Output("stats-panel", "children"),
    Input("cur-step",  "data"),
    Input("dd-scan",   "value"),
)
def render(step_idx, scan_id):
    step = STEPS[step_idx]
    desc = DESCS[step["id"]]
    data = get_data(scan_id, step_idx)

    # ── Visualization ─────────────────────────────────────────────────────────
    if data["type"] == "3d":
        viz = dcc.Graph(
            figure=data["fig"],
            style={"width": "100%", "height": "100%"},
            config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"],
                    "scrollZoom": True},
        )

    elif data["type"] == "frames":
        viz = html.Div([
            html.Div([
                html.Img(src=src,
                         style={"height": "100%", "borderRadius": "6px",
                                "border": "1px solid #374151", "cursor": "pointer"})
                for src in data["imgs"]
            ], style={"display": "flex", "gap": "10px", "height": "90%",
                      "overflowX": "auto", "padding": "10px 16px"}),
            html.Div(
                f"▶ {data['stats'].get('frames','?')} frames  ·  "
                f"{data['stats'].get('fps','?')} fps  ·  "
                f"Thời lượng: {data['stats'].get('duration','?')}",
                style={"textAlign": "center", "fontSize": "12px",
                       "color": "#6b7280", "paddingBottom": "8px"},
            ),
        ], style={"height": "100%", "display": "flex", "flexDirection": "column",
                  "justifyContent": "center", "background": "#111827"})

    elif data["type"] == "dual_image":
        viz = html.Div([
            html.Div([
                html.Div([
                    html.Img(src=data["img1"],
                             style={"width": "100%", "borderRadius": "6px",
                                    "border": "1px solid #374151"}),
                    html.Div("▲ RGB Frame",
                             style={"textAlign": "center", "fontSize": "12px",
                                    "color": "#6b7280", "marginTop": "4px"}),
                ], style={"flex": "1"}),
                html.Div([
                    html.Img(src=data["img2"],
                             style={"width": "100%", "borderRadius": "6px",
                                    "border": "1px solid #374151"}),
                    html.Div("▲ Depth map (colorized PLASMA)",
                             style={"textAlign": "center", "fontSize": "12px",
                                    "color": "#6b7280", "marginTop": "4px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "padding": "16px",
                      "height": "100%", "boxSizing": "border-box"}),
        ], style={"height": "100%", "background": "#111827", "overflow": "hidden"})

    elif data["type"] == "error":
        viz = html.Div(
            f"⚠ {data.get('msg', 'Không tìm thấy dữ liệu')}",
            style={"display": "flex", "alignItems": "center", "justifyContent": "center",
                   "height": "100%", "color": "#f87171", "fontSize": "14px"},
        )
    else:
        viz = html.Div("Đang tải...",
                       style={"display": "flex", "alignItems": "center",
                              "justifyContent": "center", "height": "100%",
                              "color": "#9ca3af"})

    # ── Stats panel ───────────────────────────────────────────────────────────
    stats = data.get("stats", {})
    stat_labels = {
        # generic mesh
        "verts":           ("Vertices", "#818cf8"),
        "tris":            ("Triangles", "#818cf8"),
        "height":          ("Chiều cao", "#34d399"),
        "pts":             ("Điểm 3D",   "#818cf8"),
        # video
        "frames":          ("Frames",    "#f59e0b"),
        "fps":             ("FPS",       "#f59e0b"),
        "duration":        ("Thời lượng","#f59e0b"),
        # depth
        "frame":           ("Frame",     "#9ca3af"),
        "depth_range":     ("Độ sâu",    "#60a5fa"),
        "rgb_res":         ("RGB",       "#9ca3af"),
        "depth_res":       ("Depth",     "#9ca3af"),
        # lichen
        "pct_total":       ("Địa y tổng",    "#f97316"),
        "area_total":      ("DT cây",         "#34d399"),
        "lichen_total":    ("DT địa y",       "#f97316"),
        "upslope_pct":     ("Upslope",        "#60a5fa"),
        "downslope_pct":   ("Downslope",      "#f97316"),
        # misc
        "note":            ("Ghi chú",   "#6b7280"),
    }

    stat_cards = []
    for key, val in stats.items():
        if key.startswith("_"):
            continue
        label, color = stat_labels.get(key, (key, "#9ca3af"))
        stat_cards.append(html.Div([
            html.Div(str(val),   style={"fontSize": "15px", "fontWeight": "700",
                                        "color": color}),
            html.Div(label,      style={"fontSize": "11px", "color": "#6b7280",
                                        "marginTop": "1px"}),
        ], style={"background": "#111827", "border": "1px solid #374151",
                  "borderRadius": "8px", "padding": "8px 14px",
                  "minWidth": "90px", "textAlign": "center"}))

    stats_el = html.Div(stat_cards,
                        style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})

    # ── Lichen legend ─────────────────────────────────────────────────────────
    if step["id"] == "lichen":
        legend = html.Div([
            _legend_dot("#ff8000", "Địa y Upslope"),
            _legend_dot("#fff200", "Địa y Downslope"),
            _legend_dot("#7ca3a0", "Vỏ cây Upslope"),
            _legend_dot("#aa7c4e", "Vỏ cây Downslope"),
        ], style={"display": "flex", "gap": "14px", "marginTop": "8px",
                  "flexWrap": "wrap"})
        stats_el = html.Div([stats_el, legend])

    return (viz,
            desc["title"],
            desc["body"],
            f"[ {desc['tech']} ]",
            stats_el)


def _legend_dot(color, label):
    return html.Div([
        html.Div(style={"width": "12px", "height": "12px", "borderRadius": "50%",
                        "background": color, "flexShrink": "0"}),
        html.Span(label, style={"fontSize": "11px", "color": "#9ca3af"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "5px"})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, webbrowser, threading

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()

    def _open():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{args.port}")
    threading.Thread(target=_open, daemon=True).start()

    print(f"\nDemo pipeline -> http://localhost:{args.port}\n", flush=True)
    app.run(host="0.0.0.0", port=args.port, debug=False)
