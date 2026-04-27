#!/usr/bin/env python3
# MARKER_NO_VXY_2024
"""
Preview + Editor — xem và chỉnh sửa point cloud 92 mẫu quế.

Usage:
    python 07_preview.py
    http://localhost:8050
"""

import json, os, re
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
from scipy.spatial import cKDTree

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")
MAX_EDIT     = 120_000   # điểm tối đa khi Edit (raw PLY)
MAX_TRUNK    = 200_000   # trunk đã clean — voxel-cap để load nhanh

# Màu dùng để vẽ địa y khi user tự thêm nhãn thủ công
_ANNOT_ADD_COLOR = np.array([255, 220,  40], dtype=np.uint8)   # vàng sáng
_ANNOT_ADD_UP    = np.array([255, 255,   0], dtype=np.uint8)   # vàng neon (giống detect)
_ANNOT_ADD_DOWN  = np.array([255, 166,   0], dtype=np.uint8)   # cam (giống detect)

def _rgb_to_hsv(rgb_norm: np.ndarray) -> np.ndarray:
    """rgb_norm: (N,3) float [0,1] -> (N,3): H[0,360], S[0,1], V[0,1]"""
    r, g, b = rgb_norm[:,0], rgb_norm[:,1], rgb_norm[:,2]
    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-9
    h = np.zeros(len(rgb_norm), dtype=np.float32)
    mr, mg, mb = cmax == r, cmax == g, cmax == b
    h[mr] = 60 * (((g[mr] - b[mr]) / delta[mr]) % 6)
    h[mg] = 60 * (((b[mg] - r[mg]) / delta[mg]) + 2)
    h[mb] = 60 * (((r[mb] - g[mb]) / delta[mb]) + 4)
    s = np.where(cmax > 0, delta / (cmax + 1e-9), 0).astype(np.float32)
    v = cmax.astype(np.float32)
    return np.stack([h, s, v], axis=1)

# ── Server-side cache ────────────────────────────────────────────────────────
# key: "name:mode" → {"pts": ndarray, "cols": ndarray|None, "n_raw": int}
CACHE: dict = {}

def cache_key(name, mode):
    return f"{name}:{mode}"

def load_to_cache(name, ply_path, is_trunk=False):
    key = cache_key(name, "trunk" if is_trunk else "raw")
    if key in CACHE:
        return CACHE[key]

    pcd   = o3d.io.read_point_cloud(ply_path)
    n_raw = len(pcd.points)
    limit = MAX_TRUNK if is_trunk else MAX_EDIT

    if limit and n_raw > limit:
        vol   = max(np.prod(pcd.get_axis_aligned_bounding_box().get_extent()), 1e-6)
        vsize = max((vol / limit) ** (1/3), 0.01)
        pcd   = pcd.voxel_down_sample(vsize)

    pts  = np.asarray(pcd.points, dtype=np.float32)
    cols = (np.asarray(pcd.colors)*255).astype(np.uint8) if pcd.has_colors() else None
    CACHE[key] = {"pts": pts, "cols": cols, "n_raw": n_raw, "ply": ply_path}
    return CACHE[key]

def load_mesh_to_cache(name, mesh_path):
    key = cache_key(name, "mesh")
    if key in CACHE:
        return CACHE[key]
    mesh  = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices,  dtype=np.float32)
    tris  = np.asarray(mesh.triangles, dtype=np.int32)
    cols  = None
    if mesh.has_vertex_colors():
        cols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)

    # Load mau goc (trimmed, khong co label) de toggle
    cols_orig = cols
    out_dir   = os.path.dirname(mesh_path)
    for orig_name in ["trunk_mesh_final.ply", "trunk_mesh_trimmed.ply",
                      "trunk_mesh_recolored.ply", "trunk_mesh_tsdf.ply"]:
        orig_path = os.path.join(out_dir, orig_name)
        if os.path.exists(orig_path) and orig_path != mesh_path:
            om = o3d.io.read_triangle_mesh(orig_path)
            if om.has_vertex_colors() and len(om.vertices) == len(verts):
                cols_orig = (np.asarray(om.vertex_colors) * 255).astype(np.uint8)
            break

    # Decode per-vertex lichen labels by comparing detected vs original colors.
    # Vertices that differ significantly from original = painted by detection script = lichen.
    labels = np.zeros(len(verts), dtype=np.int8)  # 0=bark, 2=lichen
    if cols is not None and cols_orig is not None and not np.array_equal(cols, cols_orig):
        diff = np.linalg.norm(cols.astype(np.float32) - cols_orig.astype(np.float32), axis=1)
        labels[diff > 25] = 2

    CACHE[key] = {"pts": verts, "cols": cols.copy() if cols is not None else None,
                  "cols_orig": cols_orig, "labels": labels,
                  "tris": tris, "n_raw": len(verts)}
    return CACHE[key]

def evict(name):
    for k in [k for k in CACHE if k.startswith(name+":")]:
        del CACHE[k]

# ── Scan discovery ───────────────────────────────────────────────────────────

def discover_scans(d):
    out = []
    for name in sorted(os.listdir(d), key=natural_scan_key):
        sd = os.path.join(d, name)
        if not os.path.isdir(sd): continue
        od    = os.path.join(sd, "output")
        raw   = os.path.join(od, "pointcloud.ply")
        trunk = os.path.join(od, "trunk_pointcloud.ply")
        # Uu tien: final > trimmed > recolored > tsdf > poisson
        # (bo qua trunk_mesh_detected.ply — tap trung xem mesh sach)
        for _mf in ["trunk_mesh_final.ply",
                    "trunk_mesh_trimmed.ply",
                    "trunk_mesh_recolored.ply", "trunk_mesh_tsdf.ply", "trunk_mesh.ply"]:
            _mp = os.path.join(od, _mf)
            if os.path.exists(_mp):
                mesh = _mp; break
        else:
            mesh = None
        if not (os.path.exists(raw) or os.path.exists(trunk)): continue
        stats_path = os.path.join(od, "lichen_stats.json")
        lichen_pct = None
        if os.path.exists(stats_path):
            try:
                with open(stats_path) as f:
                    _s = json.load(f)
                lichen_pct = _s.get("stats",{}).get("overall",{}).get("lichen_ratio_pct")
            except Exception:
                pass
        out.append({"name": name, "out": od,
                    "raw":   raw   if os.path.exists(raw)   else None,
                    "trunk": trunk if os.path.exists(trunk) else None,
                    "mesh":  mesh,
                    "lichen_pct": lichen_pct})
    return out


def natural_scan_key(name: str):
    s = str(name).strip()
    m = re.match(r"^(.*?)(\d+)\s*$", s)
    if not m:
        return (s.lower(), float("inf"), s.lower())
    return (m.group(1).strip().lower(), int(m.group(2)), s.lower())

# ── Figures ──────────────────────────────────────────────────────────────────

def _rgb_colors(pts, cols, deleted, pending):
    """Per-point color strings — chỉ dùng khi edit (chậm hơn)."""
    c = []
    for i in range(len(pts)):
        if i in pending:
            c.append("rgba(255,60,60,1)")
        elif i in deleted:
            c.append("rgba(80,80,80,0.1)")
        elif cols is not None:
            r,g,b = int(cols[i,0]), int(cols[i,1]), int(cols[i,2])
            c.append(f"rgb({r},{g},{b})")
        else:
            c.append("rgba(100,200,255,0.8)")
    return c

def _fast_colors(pts, cols):
    """Màu nhanh cho VIEW mode: pack RGB vào int32 → plotly hiểu."""
    if cols is None:
        return pts[:,2].tolist()   # fallback: color by Z
    # plotly chấp nhận "rgb(r,g,b)" list nhưng quá chậm khi nhiều
    # Dùng hex strings — nhanh hơn ~30% so với rgb()
    r = cols[:,0].astype(np.uint8)
    g = cols[:,1].astype(np.uint8)
    b = cols[:,2].astype(np.uint8)
    # Vectorized hex encoding
    hex_arr = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    return [f"#{v:06x}" for v in hex_arr]

def fig3d(pts, cols, deleted, pending, title="", edit=False):
    c = _rgb_colors(pts, cols, deleted, pending) if edit \
        else _fast_colors(pts, cols)
    fig = go.Figure(go.Scatter3d(
        x=pts[:,0].tolist(), y=pts[:,1].tolist(), z=pts[:,2].tolist(),
        mode="markers", marker=dict(size=2, color=c, opacity=0.9),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#ccc")),
        paper_bgcolor="#0d0d1e", margin=dict(l=0,r=0,t=30,b=0),
        scene=dict(bgcolor="#0d0d1e",
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            zaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            aspectmode="data"),
        uirevision="keep3d",
    )
    return fig

def fig_mesh3d(verts, cols, tris, title="", clickable=False):
    """Render mesh 3D với vertex colors dùng go.Mesh3d.
    clickable=True: bật hover/click để dùng với annotation mode."""
    if cols is not None:
        r = cols[:,0].astype(np.uint32)
        g = cols[:,1].astype(np.uint32)
        b = cols[:,2].astype(np.uint32)
        vc = [f"#{(rv<<16|gv<<8|bv):06x}" for rv,gv,bv in zip(r,g,b)]
    else:
        vc = verts[:,2].tolist()   # fallback: color by Z

    fig = go.Figure(go.Mesh3d(
        x=verts[:,0].tolist(), y=verts[:,1].tolist(), z=verts[:,2].tolist(),
        i=tris[:,0].tolist(),  j=tris[:,1].tolist(),  k=tris[:,2].tolist(),
        vertexcolor=vc,
        opacity=1.0,
        # hoverinfo="none" cho phép click event; "skip" tắt hoàn toàn
        hoverinfo="x+y+z" if clickable else "skip",
        hovertemplate="Click để thêm/xóa nhãn<extra></extra>" if clickable else None,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1,
                      roughness=0.8, fresnel=0.1),
        lightposition=dict(x=100, y=200, z=100),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#ccc")),
        paper_bgcolor="#0d0d1e", margin=dict(l=0,r=0,t=30,b=0),
        scene=dict(bgcolor="#0d0d1e",
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            zaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            aspectmode="data"),
        uirevision="keep3d",
    )
    return fig

def fig2d(pts, cols, deleted, pending, xi, yi, label,
          edit_mode=False, orig_indices=None):
    c = _rgb_colors(pts, cols, deleted, pending) if edit_mode \
        else _fast_colors(pts, cols)
    cdata = orig_indices if orig_indices is not None else list(range(len(pts)))
    # Scattergl dùng WebGL — nhanh hơn Scatter thường ~5x khi nhiều điểm
    trace = go.Scattergl(
        x=pts[:,xi].tolist(), y=pts[:,yi].tolist(),
        mode="markers",
        marker=dict(size=3, color=c),
        customdata=cdata,
        hoverinfo="skip",
        selected=dict(marker=dict(color="red", size=5)),
        unselected=dict(marker=dict(opacity=0.3)),
    )
    fig = go.Figure(trace)
    fig.update_layout(
        title=dict(text=label, font=dict(size=10, color="#aaa")),
        paper_bgcolor="#111122", plot_bgcolor="#111122",
        margin=dict(l=5,r=5,t=22,b=5),
        dragmode="lasso" if edit_mode else "pan",
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,color="#444"),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,color="#444",
                   scaleanchor="x"),
        font=dict(color="#777"),
        uirevision=label,
    )
    return fig

def empty_fig(bg="#111122"):
    f = go.Figure()
    f.update_layout(paper_bgcolor=bg, plot_bgcolor=bg,
                    margin=dict(l=0,r=0,t=0,b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f

# ── Layout ───────────────────────────────────────────────────────────────────

def _btn(label, bid, bg="#252540", fg="#99aaff", **kw):
    return html.Button(label, id=bid, n_clicks=0, style={
        "background": bg, "color": fg,
        "border": f"1px solid {fg}55", "borderRadius":"5px",
        "padding":"4px 11px", "cursor":"pointer",
        "fontSize":"12px", "marginLeft":"5px",
        "whiteSpace":"nowrap", **kw,
    })

def sidebar_item(scan, selected):
    has_trunk  = scan.get("trunk") and os.path.exists(scan["trunk"])
    has_mesh   = scan.get("mesh")  and os.path.exists(scan["mesh"])
    lichen_pct = scan.get("lichen_pct")
    badges = []
    if has_trunk: badges.append(html.Span(" T", style={"color":"#4c9","fontSize":"10px"}))
    if has_mesh:  badges.append(html.Span(" M", style={"color":"#f90","fontSize":"10px"}))
    if lichen_pct is not None:
        # Màu theo tỉ lệ: xanh <10%, vàng 10-25%, đỏ >25%
        pct_col = "#66dd66" if lichen_pct < 10 else ("#ffcc44" if lichen_pct < 25 else "#ff6644")
        badges.append(html.Span(
            f" {lichen_pct:.0f}%",
            style={"color": pct_col, "fontSize":"10px", "fontWeight":"bold"}
        ))
    return html.Div(
        id={"type":"si","index":scan["name"]}, n_clicks=0,
        children=[
            html.Span(scan["name"], style={
                "fontSize":"12px","fontWeight":"bold" if selected else "normal",
                "color":"#eee" if selected else "#999",
            }),
            *badges,
        ],
        style={
            "padding":"5px 8px","cursor":"pointer","borderRadius":"3px",
            "marginBottom":"1px",
            "background":"#2a2a4a" if selected else "transparent",
        },
    )

def _stats_panel():
    """Bảng thống kê địa y — nội dung cập nhật qua callback cb_stats."""
    th = {"padding":"3px 12px","textAlign":"center","color":"#88aacc",
          "fontSize":"11px","fontWeight":"bold","borderBottom":"1px solid #2a2a4a",
          "whiteSpace":"nowrap"}
    return html.Div(id="stats-panel", style={
        "background":"#0d0d20","borderBottom":"1px solid #1a1a3a",
        "padding":"4px 8px","display":"none",   # ẩn khi chưa có stats
    }, children=[
        html.Table(style={"width":"100%","borderCollapse":"collapse"}, children=[
            html.Thead(html.Tr([
                html.Th("",             style={**th,"textAlign":"left","width":"110px"}),
                html.Th("⬆ Upslope",   style=th),
                html.Th("⬇ Downslope", style=th),
                html.Th("🌿 Cả cây",   style=th),
            ])),
            html.Tbody(id="stats-body"),
        ]),
    ])


def build_layout(scans):
    names  = [s["name"] for s in scans]
    first  = names[0] if names else ""

    sidebar = html.Div(style={
        "width":"185px","minWidth":"185px","background":"#161628",
        "borderRight":"1px solid #2a2a4a",
        "display":"flex","flexDirection":"column","overflow":"hidden",
    }, children=[
        html.Div(f"3D Que — {len(scans)} scans", style={
            "padding":"10px 8px","fontWeight":"bold","color":"#aad4ff",
            "fontSize":"13px","borderBottom":"1px solid #2a2a4a",
        }),
        html.Div(id="sidebar", style={"overflowY":"auto","flex":"1","padding":"3px"}),
    ])

    topbar = html.Div(style={
        "display":"flex","alignItems":"center","flexWrap":"wrap",
        "padding":"5px 12px","background":"#161628",
        "borderBottom":"1px solid #2a2a4a","gap":"4px",
    }, children=[
        html.Div(id="title", style={
            "fontWeight":"bold","color":"#ddd","fontSize":"14px","marginRight":"6px",
        }),
        _btn("◀", "btn-prev"), _btn("▶", "btn-next"),
        html.Div(style={"width":"8px"}),
        # Nguồn
        html.Span("Nguon:", style={"color":"#666","fontSize":"11px"}),
        dcc.RadioItems(id="src-mode",
            options=[{"label":" Raw","value":"raw"},
                     {"label":" Trunk","value":"trunk"},
                     {"label":" Mesh","value":"mesh"}],
            value="mesh", inline=True,
            labelStyle={"color":"#bbb","marginRight":"8px","fontSize":"12px"},
            inputStyle={"marginRight":"3px"}),
        html.Div(style={"width":"8px"}),
        # Points detail
        html.Span("Pts:", style={"color":"#666","fontSize":"11px"}),
        dcc.Dropdown(id="detail",
            options=[{"label":k,"value":k}
                     for k in ["10K","30K","50K","100K","200K"]],
            value="10K", clearable=False,
            style={"width":"70px","fontSize":"12px","background":"#1e1e3a",
                   "color":"#ccc","border":"1px solid #3a3a7a"}),
        html.Div(style={"flex":"1"}),
        # Mode toggle
        _btn("👁 View", "btn-view", bg="#1a2a1a", fg="#88ff88"),
        _btn("✏ Edit", "btn-edit", bg="#2a1a2a", fg="#ff88ff"),
        html.Div(style={"width":"8px"}),
        _btn("🏷 Label", "btn-label", bg="#2a2a1a", fg="#ffcc44"),
        html.Div(style={"width":"8px"}),
        # Edit actions (hidden when not editing)
        html.Div(id="edit-actions", style={"display":"none"}, children=[
            _btn("🗑 Delete", "btn-del",  bg="#5a1a1a", fg="#ff9999"),
            _btn("↩ Undo",   "btn-undo", bg="#1a3a1a", fg="#99ff99"),
            _btn("💾 Save",  "btn-save", bg="#1a3a5a", fg="#99ccff"),
        ]),
        # Annotation toolbar (chỉ hiện khi Mesh mode)
        html.Div(id="annot-toolbar", style={"display":"none"}, children=[
            html.Div(style={"width":"8px"}),
            html.Span("✏ Nhãn:", style={"color":"#aaa","fontSize":"11px","whiteSpace":"nowrap"}),
            _btn("➕ Thêm địa y", "btn-annot-add", bg="#1a3a1a", fg="#88ff88"),
            _btn("➖ Xóa địa y",  "btn-annot-rem", bg="#3a1a1a", fg="#ff8888"),
            html.Span("R:", style={"color":"#888","fontSize":"11px",
                                   "marginLeft":"8px","whiteSpace":"nowrap"}),
            dcc.RadioItems(id="annot-radius",
                options=[{"label":" 2","value":0.02},
                         {"label":" 4","value":0.04},
                         {"label":" 7","value":0.07},
                         {"label":" 12","value":0.12}],
                value=0.04, inline=True,
                labelStyle={"color":"#bbb","marginRight":"5px","fontSize":"11px"},
                inputStyle={"marginRight":"2px"}),
            html.Span("cm", style={"color":"#888","fontSize":"10px"}),
            _btn("💾 Lưu nhãn", "btn-annot-save", bg="#1a3a5a", fg="#99ccff"),
        ]),
    ])

    status = html.Div(id="status", style={
        "padding":"2px 12px","fontSize":"11px","color":"#666",
        "background":"#0a0a1a","borderBottom":"1px solid #1a1a3a","minHeight":"17px",
    })

    view3d = html.Div(style={"flex":"1","minHeight":"0"}, children=[
        dcc.Graph(id="v3d", style={"height":"100%"},
                  config={"displayModeBar":True,"scrollZoom":True}),
    ])

    return html.Div(style={
        "display":"flex","height":"100vh","background":"#111122",
        "color":"#ccc","fontFamily":"monospace","overflow":"hidden",
    }, children=[
        sidebar,
        html.Div(style={
            "flex":"1","display":"flex","flexDirection":"column","overflow":"hidden",
        }, children=[topbar, status, _stats_panel(), view3d]),

        # Stores (nhỏ — không chứa point data)
        dcc.Store(id="s-names",   data=names),
        dcc.Store(id="s-idx",     data=0),
        dcc.Store(id="s-cur",     data=first),
        # state: {name, mode(raw/trunk), edit(bool), n_pts, deleted:[[],[]], pending:[]}
        dcc.Store(id="s-state",   data=None),
        dcc.Store(id="s-label",      data=False, storage_type="memory"),
        dcc.Store(id="s-annot-mode", data=None,  storage_type="memory"),
    ])

# ── App ──────────────────────────────────────────────────────────────────────

def create_app(scans):
    scan_map = {s["name"]: s for s in scans}

    app = Dash(__name__, title="3D Que Preview", suppress_callback_exceptions=True)
    app.layout = build_layout(scans)

    # ── Sidebar ───────────────────────────────────────────────
    @app.callback(Output("sidebar","children"),
                  Input("s-cur","data"), State("s-names","data"))
    def cb_sidebar(cur, names):
        return [sidebar_item(scan_map[n], n==cur) for n in names]

    # ── Navigate ──────────────────────────────────────────────
    @app.callback(
        Output("s-cur","data"), Output("s-idx","data"),
        Input({"type":"si","index":ALL},"n_clicks"),
        Input("btn-prev","n_clicks"), Input("btn-next","n_clicks"),
        State("s-idx","data"), State("s-names","data"),
        prevent_initial_call=True,
    )
    def cb_navigate(*args):
        ctx = callback_context
        if not ctx.triggered: return no_update, no_update
        tid   = ctx.triggered[0]["prop_id"]
        names, idx = args[-1], args[-2]
        if "btn-prev" in tid:  ni = max(0, idx-1)
        elif "btn-next" in tid: ni = min(len(names)-1, idx+1)
        else:
            try:
                name = json.loads(tid.rsplit(".",1)[0])["index"]
                ni   = names.index(name)
            except Exception: return no_update, no_update
        return names[ni], ni

    # ── Load scan → populate state + cache ────────────────────
    @app.callback(
        Output("s-state","data"),
        Output("title","children"),
        Input("s-cur","data"),
        Input("src-mode","value"),
    )
    def cb_load(name, src_mode):
        if not name or name not in scan_map:
            return None, "—"
        s = scan_map[name]

        if src_mode == "mesh":
            # Uu tien: final > trimmed > recolored > tsdf > poisson
            for candidate in ["trunk_mesh_final.ply",
                               "trunk_mesh_trimmed.ply",
                               "trunk_mesh_recolored.ply",
                               "trunk_mesh_tsdf.ply",
                               "trunk_mesh.ply"]:
                mp = os.path.join(s["out"], candidate)
                if os.path.exists(mp):
                    break
            else:
                return None, f"{name} — chua co mesh (chay 02_mesh_tsdf.py truoc)"
            scan_map[name]["mesh"] = mp
            mesh_type = ("final"     if "final"     in mp else
                         "trimmed"   if "trimmed"   in mp else
                         "recolored" if "recolored" in mp else
                         "tsdf"      if "tsdf"      in mp else "poisson")
            data  = load_mesh_to_cache(name, mp)
            n_pts = len(data["pts"])
            n_tri = len(data["tris"])
            label = f"{name}  [{mesh_type}  {n_pts:,} verts  {n_tri:,} tris]"
            state = {"name":name, "mode":"mesh", "edit":False,
                     "n_pts":n_pts, "deleted":[], "pending":[]}
            return state, label

        is_trunk = False
        if src_mode == "trunk":
            tp = os.path.join(s["out"], "trunk_pointcloud.ply")
            scan_map[name]["trunk"] = tp
            if os.path.exists(tp):
                ply, is_trunk = tp, True
            elif s["raw"]:
                ply = s["raw"]
            else:
                return None, f"{name} — no file"
        else:
            ply = s["raw"]
            if not ply or not os.path.exists(ply):
                return None, f"{name} — no raw PLY"

        data  = load_to_cache(name, ply, is_trunk)
        n_pts = len(data["pts"])
        note  = f" (ds {n_pts:,}/{data['n_raw']:,})" if data["n_raw"] != n_pts else f" ({n_pts:,} pts)"
        label = f"{name}  [{('trunk' if is_trunk else 'raw')}{note}]"

        state = {"name":name, "mode":src_mode, "edit":False,
                 "n_pts":n_pts, "deleted":[], "pending":[]}
        return state, label

    # ── Toggle view/edit mode ─────────────────────────────────
    @app.callback(
        Output("s-state","data",       allow_duplicate=True),
        Output("edit-actions","style", allow_duplicate=True),
        Output("btn-view","style",     allow_duplicate=True),
        Output("btn-edit","style",     allow_duplicate=True),
        Input("btn-view","n_clicks"),
        Input("btn-edit","n_clicks"),
        State("s-state","data"),
        prevent_initial_call=True,
    )
    def cb_toggle_mode(_, __, state):
        ctx  = callback_context
        if not ctx.triggered or not state: return no_update, no_update, no_update, no_update
        tid  = ctx.triggered[0]["prop_id"]
        edit = "btn-edit" in tid
        ns   = dict(state, edit=edit, pending=[])

        ea_style  = {"display":"flex","alignItems":"center"} if edit else {"display":"none"}
        bv_style  = _btn_active_style(not edit, "#88ff88")
        be_style  = _btn_active_style(edit,     "#ff88ff")
        return ns, ea_style, bv_style, be_style

    def _btn_active_style(active, fg):
        bg = "#223322" if fg=="#88ff88" else "#332233"
        return {
            "background": bg if active else "#252540",
            "color": fg, "border": f"1px solid {fg}{'aa' if active else '33'}",
            "borderRadius":"5px","padding":"4px 11px","cursor":"pointer",
            "fontSize":"12px","marginLeft":"5px","fontWeight":"bold" if active else "normal",
        }

    # ── Toggle label hien thi dia y ───────────────────────────
    @app.callback(
        Output("s-label",    "data"),
        Output("btn-label",  "style"),
        Input("btn-label",   "n_clicks"),
        State("s-label",     "data"),
        prevent_initial_call=True,
    )
    def cb_toggle_label(_, show):
        new_show = not show
        style = {
            "background": "#3a3a1a" if new_show else "#252540",
            "color": "#ffcc44",
            "border": f"1px solid #ffcc44{'aa' if new_show else '33'}",
            "borderRadius":"5px","padding":"4px 11px","cursor":"pointer",
            "fontSize":"12px","marginLeft":"5px",
            "fontWeight": "bold" if new_show else "normal",
        }
        return new_show, style

    # ── Bảng thống kê địa y ───────────────────────────────────
    @app.callback(
        Output("stats-panel", "style"),
        Output("stats-body",  "children"),
        Input("s-state", "data"),
    )
    def cb_stats(state):
        hidden = {"background":"#0d0d20","borderBottom":"1px solid #1a1a3a",
                  "padding":"4px 8px","display":"none"}
        visible = {**hidden, "display":"block"}

        # Tam thoi an stats panel — tap trung kiem tra mesh truoc
        return hidden, []

        if not state:  # noqa: unreachable
            return hidden, []

        name    = state["name"]
        sd      = scan_map.get(name, {})
        jpath   = os.path.join(sd.get("out",""), "lichen_stats.json")
        if not os.path.exists(jpath):
            return hidden, []

        with open(jpath) as f:
            data = json.load(f)
        st = data.get("stats", {})
        up   = st.get("upslope",   {})
        dn   = st.get("downslope", {})
        ov   = st.get("overall",   {})

        td_label = {"padding":"2px 8px","fontSize":"11px","color":"#889",
                    "whiteSpace":"nowrap"}
        def td(txt, bold=False, color="#ccc"):
            return html.Td(txt, style={"padding":"2px 12px","fontSize":"11px",
                                       "color":color,"textAlign":"center",
                                       "fontWeight":"bold" if bold else "normal"})

        def fmt_area(side):
            total  = side.get("total_cm2", 0)
            lichen = side.get("total_lichen_cm2", 0)
            pct    = side.get("lichen_ratio_pct", 0)
            return total, lichen, pct

        up_t,  up_l,  up_p  = fmt_area(up)
        dn_t,  dn_l,  dn_p  = fmt_area(dn)
        ov_t,  ov_l,  ov_p  = fmt_area(ov)

        # Màu cho tỉ lệ: xanh thấp → vàng cao
        def pct_color(p):
            if p < 10: return "#88dd88"
            if p < 20: return "#ffcc44"
            return "#ff8866"

        rows = [
            html.Tr([
                html.Td("Diện tích cây", style=td_label),
                td(f"{up_t:,.0f} cm²"),
                td(f"{dn_t:,.0f} cm²"),
                td(f"{ov_t:,.0f} cm²"),
            ]),
            html.Tr([
                html.Td("Địa y",         style=td_label),
                td(f"{up_l:,.0f} cm²"),
                td(f"{dn_l:,.0f} cm²"),
                td(f"{ov_l:,.0f} cm²"),
            ]),
            html.Tr([
                html.Td("Tỉ lệ địa y",  style=td_label),
                td(f"{up_p:.1f}%",  bold=True, color=pct_color(up_p)),
                td(f"{dn_p:.1f}%",  bold=True, color=pct_color(dn_p)),
                td(f"{ov_p:.1f}%",  bold=True, color=pct_color(ov_p)),
            ]),
        ]
        return visible, rows

    # ── Hiện annotation toolbar khi Mesh mode ─────────────────
    @app.callback(
        Output("annot-toolbar", "style"),
        Input("s-state", "data"),
    )
    def cb_annot_toolbar_vis(state):
        if state and state.get("mode") == "mesh":
            return {"display":"flex","alignItems":"center","flexWrap":"wrap","gap":"2px"}
        return {"display":"none"}

    # ── Toggle annotation mode (add / remove) ─────────────────
    @app.callback(
        Output("s-annot-mode",   "data"),
        Output("btn-annot-add",  "style"),
        Output("btn-annot-rem",  "style"),
        Input("btn-annot-add",   "n_clicks"),
        Input("btn-annot-rem",   "n_clicks"),
        State("s-annot-mode",    "data"),
        prevent_initial_call=True,
    )
    def cb_toggle_annot_mode(_, __, cur_mode):
        ctx = callback_context
        if not ctx.triggered: return no_update, no_update, no_update
        tid = ctx.triggered[0]["prop_id"]
        if "btn-annot-add" in tid:
            new_mode = "add" if cur_mode != "add" else None
        else:
            new_mode = "remove" if cur_mode != "remove" else None
        add_style = _btn_active_style(new_mode == "add",    "#88ff88")
        rem_style = _btn_active_style(new_mode == "remove", "#ff8888")
        return new_mode, add_style, rem_style

    # ── Click 3D mesh → thêm / xóa địa y ─────────────────────
    @app.callback(
        Output("s-state",  "data",     allow_duplicate=True),
        Output("status",   "children", allow_duplicate=True),
        Input("v3d",       "clickData"),
        State("s-annot-mode", "data"),
        State("s-state",   "data"),
        State("annot-radius", "value"),
        prevent_initial_call=True,
    )
    def cb_annot_click(click_data, annot_mode, state, radius):
        if not click_data or not annot_mode or not state:
            return no_update, no_update
        if state.get("mode") != "mesh":
            return no_update, no_update

        name = state["name"]
        data = CACHE.get(cache_key(name, "mesh"))
        if data is None:
            return no_update, "Cache miss — chon lai scan"

        pt   = click_data["points"][0]
        cx   = float(pt["x"]); cy = float(pt["y"]); cz = float(pt["z"])

        verts     = data["pts"]          # (V,3) float32
        cols_orig = data["cols_orig"]    # (V,3) uint8, original bark colors
        labels    = data.get("labels", np.zeros(len(verts), dtype=np.int8))
        cols      = data["cols"].copy()  # mutable copy

        # Tìm đỉnh gần nhất với điểm click
        tree = cKDTree(verts)
        _, seed_idx = tree.query([cx, cy, cz])

        # Tìm tất cả đỉnh trong bán kính
        radius    = float(radius) if radius else 0.04
        neighbors = np.array(tree.query_ball_point(verts[seed_idx], r=radius),
                             dtype=np.int32)

        if annot_mode == "add":
            # Region grow: giữ lại các đỉnh có màu tương tự seed (độ sáng ± 0.25)
            if cols_orig is not None and len(neighbors) > 0:
                seed_col  = cols_orig[seed_idx].astype(np.float32) / 255.0
                seed_hsv  = _rgb_to_hsv(seed_col.reshape(1, 3))[0]  # H, S, V
                cand_norm = cols_orig[neighbors].astype(np.float32) / 255.0
                cand_hsv  = _rgb_to_hsv(cand_norm)
                v_diff    = np.abs(cand_hsv[:, 2] - seed_hsv[2])
                # Giữ: V gần seed và S thấp (không quá bão hòa — đặc điểm địa y)
                keep      = (v_diff < 0.25) & (cand_hsv[:, 1] < 0.50)
                add_idxs  = neighbors[keep]
            else:
                add_idxs  = neighbors

            labels[add_idxs] = 2
            cols[add_idxs]   = _ANNOT_ADD_COLOR
            n_changed = len(add_idxs)
            msg = f"[+] Thêm {n_changed} đỉnh địa y | R={radius*100:.0f}cm"

        else:  # remove
            lichen_mask = labels[neighbors] > 0
            rem_idxs    = neighbors[lichen_mask]
            labels[rem_idxs] = 0
            if cols_orig is not None:
                cols[rem_idxs] = cols_orig[rem_idxs]
            n_changed = len(rem_idxs)
            msg = f"[-] Xóa {n_changed} đỉnh địa y | R={radius*100:.0f}cm"

        # Cập nhật CACHE in-place
        data["labels"] = labels
        data["cols"]   = cols

        # Bump annot_rev để trigger re-render
        return dict(state, annot_rev=state.get("annot_rev", 0) + 1), msg

    # ── Lưu nhãn annotation về file PLY ───────────────────────
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("btn-annot-save", "n_clicks"),
        State("s-state", "data"),
        prevent_initial_call=True,
    )
    def cb_annot_save(_, state):
        if not state or state.get("mode") != "mesh":
            return "Chi luu duoc trong Mesh mode."
        name = state["name"]
        data = CACHE.get(cache_key(name, "mesh"))
        if data is None:
            return "Cache miss — chon lai scan."

        mesh_out = o3d.geometry.TriangleMesh()
        mesh_out.vertices  = o3d.utility.Vector3dVector(
            data["pts"].astype(np.float64))
        mesh_out.triangles = o3d.utility.Vector3iVector(
            data["tris"].astype(np.int32))
        mesh_out.vertex_colors = o3d.utility.Vector3dVector(
            data["cols"].astype(np.float64) / 255.0)

        save_path = scan_map[name].get("mesh") or os.path.join(
            scan_map[name]["out"], "trunk_mesh_detected.ply")
        o3d.io.write_triangle_mesh(save_path, mesh_out, write_vertex_colors=True)
        scan_map[name]["mesh"] = save_path

        labels    = data.get("labels", np.array([]))
        n_lichen  = int((labels > 0).sum()) if len(labels) else 0
        n_total   = len(data["pts"])
        pct       = n_lichen / n_total * 100 if n_total else 0
        return (f"Saved → {os.path.basename(save_path)} | "
                f"lichen: {n_lichen:,}/{n_total:,} ({pct:.1f}%)")

    # ── Lasso select → pending (v3d only) ────────────────────
    @app.callback(
        Output("s-state","data", allow_duplicate=True),
        Input("v3d","selectedData"),
        State("s-state","data"),
        prevent_initial_call=True,
    )
    def cb_select(sel3d, state):
        if not state or not state.get("edit"): return no_update
        idx = set()
        if sel3d and "points" in sel3d:
            for p in sel3d["points"]:
                if "customdata" in p:
                    idx.add(int(p["customdata"]))
        return dict(state, pending=list(idx))

    # ── Delete ────────────────────────────────────────────────
    @app.callback(
        Output("s-state","data", allow_duplicate=True),
        Input("btn-del","n_clicks"),
        State("s-state","data"),
        prevent_initial_call=True,
    )
    def cb_delete(_, state):
        if not state or not state.get("pending"): return no_update
        new_deleted = state.get("deleted",[]) + [state["pending"]]
        return dict(state, deleted=new_deleted, pending=[])

    # ── Undo ──────────────────────────────────────────────────
    @app.callback(
        Output("s-state","data", allow_duplicate=True),
        Input("btn-undo","n_clicks"),
        State("s-state","data"),
        prevent_initial_call=True,
    )
    def cb_undo(_, state):
        if not state or not state.get("deleted"): return no_update
        return dict(state, deleted=state["deleted"][:-1])

    # ── Save ──────────────────────────────────────────────────
    @app.callback(
        Output("status","children", allow_duplicate=True),
        Input("btn-save","n_clicks"),
        State("s-state","data"),
        prevent_initial_call=True,
    )
    def cb_save(_, state):
        if not state: return "Nothing to save."
        name = state["name"]
        mode = state["mode"]
        data = CACHE.get(cache_key(name, "trunk" if mode=="trunk" else "raw"))
        if data is None: return "Reload scan truoc khi save."

        pts, cols = data["pts"], data["cols"]
        deleted   = set()
        for batch in state.get("deleted",[]): deleted.update(batch)

        keep     = [i for i in range(len(pts)) if i not in deleted]
        pts_out  = pts[keep]
        cols_out = cols[keep] if cols is not None else None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_out.astype(np.float64))
        if cols_out is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                cols_out.astype(np.float64)/255.0)

        save_path = os.path.join(scan_map[name]["out"], "trunk_pointcloud.ply")
        os.makedirs(scan_map[name]["out"], exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
        scan_map[name]["trunk"] = save_path
        evict(name)   # clear cache sau khi save

        return (f"Saved {save_path} | kept {len(pts_out):,} | "
                f"deleted {len(deleted):,}")

    # ── Render 3D view ───────────────────────────────────────
    @app.callback(
        Output("v3d","figure"),
        Output("status","children"),
        Input("s-state","data"),
        Input("detail","value"),
        Input("s-label","data"),
        Input("s-annot-mode","data"),
    )
    def cb_render(state, detail, show_label, annot_mode):
        ef3 = empty_fig("#0d0d1e")

        if not state:
            return ef3, "Chon scan ben trai..."

        name = state["name"]
        mode = state["mode"]
        edit = state.get("edit", False)

        # ── MESH mode ────────────────────────────────────────────
        if mode == "mesh":
            data = CACHE.get(cache_key(name, "mesh"))
            if data is None:
                return ef3, "Cache miss mesh — chon lai scan"
            cols_use = data["cols"] if show_label else data.get("cols_orig", data["cols"])
            verts, cols, tris = data["pts"], cols_use, data["tris"]
            clickable = annot_mode in ("add", "remove")
            annot_hint = f"  [✏ {'➕ thêm' if annot_mode=='add' else '➖ xóa'} địa y — click vào mesh]" if clickable else ""
            title3d = f"{name} | mesh  {len(verts):,} verts  {len(tris):,} tris{annot_hint}"
            f3 = fig_mesh3d(verts, cols, tris, title3d, clickable=clickable)
            status = f"MESH VIEW | {len(verts):,} verts | {len(tris):,} tris"
            return f3, status

        # ── PCD mode (raw / trunk) ────────────────────────────────
        data = CACHE.get(cache_key(name, "trunk" if mode=="trunk" else "raw"))
        if data is None:
            return ef3, "Cache miss — click sang scan khac roi quay lai"

        pts_full, cols_full = data["pts"], data["cols"]

        # Tính deleted + pending sets
        deleted = set()
        for batch in state.get("deleted",[]): deleted.update(batch)
        pending = set(state.get("pending",[]))

        N_DETAIL = {"10K":10_000,"30K":30_000,"50K":50_000,
                    "100K":100_000,"200K":200_000}
        n_3d = N_DETAIL.get(detail, 10_000)

        def subsample(n_max):
            n = len(pts_full)
            if n <= n_max:
                return pts_full, cols_full, list(range(n))
            idx = np.random.choice(n, n_max, replace=False)
            c   = cols_full[idx] if cols_full is not None else None
            return pts_full[idx], c, idx.tolist()

        pts3, cols3, idx3 = subsample(n_3d)
        deleted3 = {i for i,orig in enumerate(idx3) if orig in deleted}
        pending3 = {i for i,orig in enumerate(idx3) if orig in pending}

        n_alive = len(pts_full) - len(deleted)
        title3d = (f"{name} | total:{n_alive:,} shown:{len(pts3):,}"
                   + (f" | del:{len(deleted):,} sel:{len(pending):,}" if edit else ""))
        f3 = fig3d(pts3, cols3, deleted3, pending3, title3d)

        mode_txt = "EDIT — Lasso/Box chon diem → Delete → Save" if edit \
                   else "VIEW — nhan Edit de chinh sua"
        status = (f"{mode_txt} | 3D:{len(pts3):,} total:{len(pts_full):,} "
                  f"del:{len(deleted):,} sel:{len(pending):,}")

        return f3, status

    return app

# ── Main ─────────────────────────────────────────────────────────────────────

PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".preview.pid")

def _write_pid(port):
    """Ghi PID + port hien tai vao file de co the kill chinh xac sau."""
    with open(PID_FILE, "w") as f:
        f.write(f"{os.getpid()}\n{port}\n")

def _cleanup_pid():
    """Xoa PID file khi server thoat."""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass


def main():
    import argparse, atexit, signal
    p = argparse.ArgumentParser()
    p.add_argument("--raw-data", default=RAW_DATA_DIR)
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()

    # Ghi PID ngay khi start de launch_preview() co the kill sau
    _write_pid(args.port)
    atexit.register(_cleanup_pid)

    # Xu ly SIGTERM (bi kill tu ben ngoai) -> chay cleanup truoc khi thoat
    def _handle_sigterm(sig, frame):
        _cleanup_pid()
        raise SystemExit(0)
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (OSError, ValueError):
        pass   # SIGTERM khong kha dung tren mot so moi truong Windows

    scans = discover_scans(args.raw_data)
    if not scans:
        print(f"No scans in {args.raw_data}"); return

    trunk_count = sum(1 for s in scans if s["trunk"] and os.path.exists(s["trunk"]))
    print(f"Found {len(scans)} scans ({trunk_count} with trunk)")
    print(f"App: http://127.0.0.1:{args.port}\n")

    app = create_app(scans)

    @app.server.route("/version")
    def version():
        return "NO_VXY_VERSION_2024"

    app.run(host="127.0.0.1", port=args.port, debug=False)

if __name__ == "__main__":
    main()
