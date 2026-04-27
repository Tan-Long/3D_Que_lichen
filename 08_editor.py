#!/usr/bin/env python3
"""
Step 8: Interactive Point Cloud Editor
- Server-side cache: PCD data ở trên server, browser chỉ giữ indices
- Lasso/Box select trên 3 view 2D → Delete → Save
- Undo từng batch

Usage:  python 08_editor.py
        http://localhost:8051
"""

import json, os
import re
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "Raw_data")
MAX_DISPLAY  = 120_000   # tối đa điểm hiển thị (raw PLY)

# ── Server-side cache ────────────────────────────────────────────────────────
# Không đi qua browser — chỉ lưu trên server
PCD_CACHE: dict = {}   # key: "scan:mode" → {"pts": np, "cols": np, "n_raw": int}

def cache_key(name, mode):
    return f"{name}:{mode}"

def load_to_cache(name, ply_path, is_trunk):
    key = cache_key(name, "trunk" if is_trunk else "raw")
    if key in PCD_CACHE:
        return PCD_CACHE[key]   # đã load rồi

    pcd   = o3d.io.read_point_cloud(ply_path)
    n_raw = len(pcd.points)

    if not is_trunk and n_raw > MAX_DISPLAY:
        bbox  = pcd.get_axis_aligned_bounding_box()
        vol   = max(np.prod(bbox.get_extent()), 1e-6)
        vsize = max((vol / MAX_DISPLAY) ** (1/3), 0.01)
        pcd   = pcd.voxel_down_sample(vsize)

    pts  = np.asarray(pcd.points,  dtype=np.float32)
    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) \
           if pcd.has_colors() else None

    data = {"pts": pts, "cols": cols, "n_raw": n_raw, "ply": ply_path}
    PCD_CACHE[key] = data
    return data

def get_cache(name, mode):
    return PCD_CACHE.get(cache_key(name, mode))


def natural_scan_key(name: str):
    s = str(name).strip()
    m = re.match(r"^(.*?)(\d+)\s*$", s)
    if not m:
        return (s.lower(), float("inf"), s.lower())
    return (m.group(1).strip().lower(), int(m.group(2)), s.lower())

# ── Scan discovery ───────────────────────────────────────────────────────────

def discover_scans(d):
    out = []
    for name in sorted(os.listdir(d), key=natural_scan_key):
        sd = os.path.join(d, name)
        if not os.path.isdir(sd):
            continue
        od   = os.path.join(sd, "output")
        raw  = os.path.join(od, "pointcloud.ply")
        trunk = os.path.join(od, "trunk_pointcloud.ply")
        if not (os.path.exists(raw) or os.path.exists(trunk)):
            continue
        out.append({"name": name, "out": od,
                    "raw":   raw   if os.path.exists(raw)   else None,
                    "trunk": trunk if os.path.exists(trunk) else None})
    return out

# ── Plotly figures ───────────────────────────────────────────────────────────

def scatter2d(pts, cols, deleted, pending, xi, yi, title):
    n = len(pts)
    c = []
    for i in range(n):
        if i in pending:   c.append("rgba(255,60,60,0.95)")
        elif i in deleted: c.append("rgba(120,120,120,0.18)")
        else:
            c.append(f"rgba({cols[i,0]},{cols[i,1]},{cols[i,2]},0.85)"
                     if cols is not None else "rgba(100,200,255,0.7)")

    fig = go.Figure(go.Scatter(
        x=pts[:, xi].tolist(), y=pts[:, yi].tolist(),
        mode="markers",
        marker=dict(size=3, color=c),
        customdata=list(range(n)),
        hoverinfo="skip",
        selected=dict(marker=dict(color="red", size=5)),
        unselected=dict(marker=dict(opacity=0.3)),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#aaa")),
        paper_bgcolor="#111122", plot_bgcolor="#111122",
        margin=dict(l=5, r=5, t=25, b=5),
        dragmode="lasso",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   color="#444"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   color="#444", scaleanchor="x"),
        font=dict(color="#888"),
        uirevision=title,
    )
    return fig

def scatter3d(pts, cols, deleted, pending):
    c = []
    for i in range(len(pts)):
        if i in pending:   c.append("rgba(255,60,60,1)")
        elif i in deleted: c.append("rgba(80,80,80,0.08)")
        else:
            c.append(f"rgb({cols[i,0]},{cols[i,1]},{cols[i,2]})"
                     if cols is not None else "rgb(80,180,255)")
    n_alive = len(pts) - len(deleted)
    fig = go.Figure(go.Scatter3d(
        x=pts[:,0].tolist(), y=pts[:,1].tolist(), z=pts[:,2].tolist(),
        mode="markers",
        marker=dict(size=2, color=c, opacity=0.9),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(
            text=f"3D — {n_alive:,} pts kept  |  {len(deleted):,} deleted  |  {len(pending):,} selected",
            font=dict(size=11, color="#ccc")),
        paper_bgcolor="#0d0d1e",
        margin=dict(l=0, r=0, t=28, b=0),
        scene=dict(bgcolor="#0d0d1e",
                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   aspectmode="data"),
        uirevision="3d",
    )
    return fig

def empty_fig(bg="#111122"):
    f = go.Figure()
    f.update_layout(paper_bgcolor=bg, plot_bgcolor=bg,
                    margin=dict(l=0,r=0,t=0,b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f

# ── Layout ───────────────────────────────────────────────────────────────────

def btn(label, id_, bg="#2a2a5a", fg="#aad4ff"):
    return html.Button(label, id=id_, n_clicks=0, style={
        "background": bg, "color": fg, "border": f"1px solid {fg}44",
        "borderRadius": "5px", "padding": "5px 14px", "cursor": "pointer",
        "fontSize": "13px", "marginRight": "6px", "whiteSpace": "nowrap",
    })

def build_layout(scans):
    names = [s["name"] for s in scans]

    sidebar = html.Div(style={
        "width": "185px", "minWidth": "185px", "background": "#161628",
        "borderRight": "1px solid #2a2a4a",
        "display": "flex", "flexDirection": "column", "overflow": "hidden",
    }, children=[
        html.Div("✏️ PCD Editor", style={
            "padding": "10px", "fontWeight": "bold", "color": "#aad4ff",
            "fontSize": "13px", "borderBottom": "1px solid #2a2a4a",
        }),
        html.Div(id="sidebar", style={
            "overflowY": "auto", "flex": "1", "padding": "3px",
        }),
    ])

    topbar = html.Div(style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "padding": "5px 12px", "background": "#161628",
        "borderBottom": "1px solid #2a2a4a", "gap": "4px",
    }, children=[
        html.Div(id="ed-title", style={
            "fontWeight": "bold", "color": "#ddd",
            "fontSize": "14px", "marginRight": "8px",
        }),
        btn("◀", "ed-prev"), btn("▶", "ed-next"),
        html.Div(style={"flex": "1"}),
        html.Span("Nguồn: ", style={"color":"#666","fontSize":"12px"}),
        dcc.RadioItems(id="src-mode",
            options=[{"label":" Raw ","value":"raw"},
                     {"label":" Trunk","value":"trunk"}],
            value="raw", inline=True,
            labelStyle={"color":"#bbb","marginRight":"8px","fontSize":"12px"},
            inputStyle={"marginRight":"3px"}),
        html.Div(style={"width":"8px"}),
        btn("🗑 Delete", "btn-del", "#5a1a1a", "#ff9999"),
        btn("↩ Undo",   "btn-undo","#1a3a1a", "#99ff99"),
        btn("💾 Save",  "btn-save","#1a3a5a", "#99ccff"),
    ])

    status = html.Div(id="status", style={
        "padding": "2px 12px", "fontSize": "11px", "color": "#666",
        "background": "#0d0d1e", "borderBottom": "1px solid #1a1a3a",
        "minHeight": "18px",
    })

    views2d = html.Div(style={
        "display": "flex", "gap": "3px", "padding": "3px",
        "background": "#0a0a1a", "flex": "1", "minHeight": "0",
    }, children=[
        dcc.Graph(id="vxy", style={"flex":"1","height":"100%"},
                  config={"displayModeBar":True,
                          "modeBarButtonsToAdd":["lasso2d","select2d"],
                          "modeBarButtonsToRemove":["autoScale2d","resetScale2d"]}),
        dcc.Graph(id="vxz", style={"flex":"1","height":"100%"},
                  config={"displayModeBar":True,
                          "modeBarButtonsToAdd":["lasso2d","select2d"],
                          "modeBarButtonsToRemove":["autoScale2d","resetScale2d"]}),
        dcc.Graph(id="vyz", style={"flex":"1","height":"100%"},
                  config={"displayModeBar":True,
                          "modeBarButtonsToAdd":["lasso2d","select2d"],
                          "modeBarButtonsToRemove":["autoScale2d","resetScale2d"]}),
    ])

    view3d = html.Div(style={
        "height": "240px", "minHeight": "240px",
        "borderTop": "1px solid #2a2a4a",
    }, children=[
        dcc.Graph(id="v3d", style={"height":"100%"},
                  config={"displayModeBar":False,"scrollZoom":True}),
    ])

    return html.Div(style={
        "display":"flex","height":"100vh","background":"#111122",
        "color":"#ccc","fontFamily":"monospace","overflow":"hidden",
    }, children=[
        sidebar,
        html.Div(style={
            "flex":"1","display":"flex","flexDirection":"column","overflow":"hidden",
        }, children=[topbar, status, views2d, view3d]),

        # Stores — chỉ lưu metadata nhỏ, KHÔNG lưu point data
        dcc.Store(id="s-names",   data=names),
        dcc.Store(id="s-idx",     data=0),
        dcc.Store(id="s-cur",     data=names[0] if names else ""),
        dcc.Store(id="s-mode",    data="raw"),
        # s-state: {"name", "mode", "n_pts", "deleted": [[batch1], [batch2]...]}
        dcc.Store(id="s-state",   data=None),
        dcc.Store(id="s-pending", data=[]),
    ])

# ── App ──────────────────────────────────────────────────────────────────────

def create_app(scans):
    scan_map = {s["name"]: s for s in scans}
    app = Dash(__name__, title="PCD Editor")
    app.layout = build_layout(scans)

    # ── Sidebar ────────────────────────────────────────────────
    @app.callback(Output("sidebar", "children"),
                  Input("s-cur", "data"),
                  State("s-names", "data"))
    def render_sidebar(cur, names):
        items = []
        for name in names:
            s   = scan_map[name]
            sel = (name == cur)
            has_trunk = s["trunk"] and os.path.exists(s["trunk"])
            items.append(html.Div(
                id={"type": "si", "index": name},
                n_clicks=0,
                children=[
                    html.Span(name, style={
                        "fontSize":"12px","fontWeight":"bold" if sel else "normal",
                        "color":"#eee" if sel else "#999",
                    }),
                    html.Span(" T" if has_trunk else "",
                              style={"color":"#4c9","fontSize":"10px","marginLeft":"4px"}),
                ],
                style={
                    "padding":"4px 8px","cursor":"pointer","borderRadius":"3px",
                    "marginBottom":"1px",
                    "background":"#2a2a4a" if sel else "transparent",
                },
            ))
        return items

    # ── Navigate ───────────────────────────────────────────────
    @app.callback(
        Output("s-cur", "data"),
        Output("s-idx", "data"),
        Input({"type":"si","index":ALL}, "n_clicks"),
        Input("ed-prev", "n_clicks"),
        Input("ed-next", "n_clicks"),
        State("s-idx",   "data"),
        State("s-names", "data"),
        prevent_initial_call=True,
    )
    def navigate(*args):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        tid = ctx.triggered[0]["prop_id"]
        names, idx = args[-1], args[-2]
        if "ed-prev" in tid:
            ni = max(0, idx - 1)
        elif "ed-next" in tid:
            ni = min(len(names)-1, idx + 1)
        else:
            try:
                name = json.loads(tid.rsplit(".",1)[0])["index"]
                ni   = names.index(name)
            except Exception:
                return no_update, no_update
        return names[ni], ni

    # ── Load scan → populate s-state ──────────────────────────
    @app.callback(
        Output("s-state",   "data"),
        Output("s-pending", "data"),
        Output("ed-title",  "children"),
        Input("s-cur",  "data"),
        Input("src-mode","value"),
    )
    def load_scan(name, mode):
        if not name or name not in scan_map:
            return None, [], "—"
        s = scan_map[name]

        if mode == "trunk":
            trunk_path = os.path.join(s["out"], "trunk_pointcloud.ply")
            scan_map[name]["trunk"] = trunk_path   # refresh
            ply       = trunk_path if os.path.exists(trunk_path) else s["raw"]
            is_trunk  = os.path.exists(trunk_path)
        else:
            ply      = s["raw"]
            is_trunk = False

        if not ply or not os.path.exists(ply):
            return None, [], f"{name} — file not found"

        data  = load_to_cache(name, ply, is_trunk)
        n_pts = len(data["pts"])
        label = (f"{name}  [{('trunk' if is_trunk else 'raw')}  "
                 f"{n_pts:,} pts{'  (downsampled)' if data['n_raw']!=n_pts else ''}]")

        state = {"name": name, "mode": mode,
                 "n_pts": n_pts, "deleted": []}
        return state, [], label

    # ── Lasso select → pending ────────────────────────────────
    @app.callback(
        Output("s-pending", "data"),
        Input("vxy", "selectedData"),
        Input("vxz", "selectedData"),
        Input("vyz", "selectedData"),
        prevent_initial_call=True,
    )
    def update_pending(sxy, sxz, syz):
        idx = set()
        for sel in [sxy, sxz, syz]:
            if sel and "points" in sel:
                for p in sel["points"]:
                    if "customdata" in p:
                        idx.add(int(p["customdata"]))
        return list(idx)

    # ── Delete ────────────────────────────────────────────────
    @app.callback(
        Output("s-state",   "data", allow_duplicate=True),
        Output("s-pending", "data", allow_duplicate=True),
        Input("btn-del", "n_clicks"),
        State("s-pending", "data"),
        State("s-state",   "data"),
        prevent_initial_call=True,
    )
    def do_delete(_, pending, state):
        if not pending or not state:
            return no_update, no_update
        new_state = dict(state)
        new_state["deleted"] = state.get("deleted", []) + [pending]
        return new_state, []

    # ── Undo ──────────────────────────────────────────────────
    @app.callback(
        Output("s-state", "data", allow_duplicate=True),
        Input("btn-undo", "n_clicks"),
        State("s-state",  "data"),
        prevent_initial_call=True,
    )
    def do_undo(_, state):
        if not state or not state.get("deleted"):
            return no_update
        new_state = dict(state)
        new_state["deleted"] = state["deleted"][:-1]
        return new_state

    # ── Save ──────────────────────────────────────────────────
    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("btn-save", "n_clicks"),
        State("s-state",  "data"),
        State("src-mode", "value"),
        prevent_initial_call=True,
    )
    def do_save(_, state, mode):
        if not state:
            return "Nothing to save."
        name = state["name"]
        data = get_cache(name, mode)
        if data is None:
            return "Cache miss — reload scan first."

        pts, cols = data["pts"], data["cols"]
        all_del = set()
        for batch in state.get("deleted", []):
            all_del.update(batch)

        keep     = [i for i in range(len(pts)) if i not in all_del]
        pts_out  = pts[keep]
        cols_out = cols[keep] if cols is not None else None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_out.astype(np.float64))
        if cols_out is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                cols_out.astype(np.float64) / 255.0)

        out_dir  = scan_map[name]["out"]
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "trunk_pointcloud.ply")
        o3d.io.write_point_cloud(save_path, pcd)
        scan_map[name]["trunk"] = save_path

        # Evict cache so next load reads fresh file
        for k in list(PCD_CACHE.keys()):
            if k.startswith(name):
                del PCD_CACHE[k]

        return (f"Saved: {save_path}  |  "
                f"{len(pts_out):,} pts kept  |  {len(all_del):,} deleted")

    # ── Render all views ──────────────────────────────────────
    @app.callback(
        Output("vxy",    "figure"),
        Output("vxz",    "figure"),
        Output("vyz",    "figure"),
        Output("v3d",    "figure"),
        Output("status", "children"),
        Input("s-state",   "data"),
        Input("s-pending", "data"),
    )
    def render_all(state, pending):
        ef = empty_fig()
        if not state:
            return ef, ef, ef, empty_fig("#0d0d1e"), "Loading..."

        name = state["name"]
        mode = state["mode"]
        data = get_cache(name, mode)
        if data is None:
            return ef, ef, ef, empty_fig("#0d0d1e"), "Cache miss — đổi scan để reload"

        pts, cols = data["pts"], data["cols"]

        deleted = set()
        for batch in state.get("deleted", []):
            deleted.update(batch)
        pending_set = set(pending or [])

        fxy = scatter2d(pts, cols, deleted, pending_set, 0, 1, "Top  (X – Y)")
        fxz = scatter2d(pts, cols, deleted, pending_set, 0, 2, "Front (X – Z)")
        fyz = scatter2d(pts, cols, deleted, pending_set, 1, 2, "Side  (Y – Z)")
        f3d = scatter3d(pts, cols, deleted, pending_set)

        n_shown = len(pts)
        n_alive = n_shown - len(deleted)
        status  = (f"Hiển thị: {n_shown:,} pts  |  "
                   f"Còn lại: {n_alive:,}  |  "
                   f"Đã xóa: {len(deleted):,}  |  "
                   f"Đang chọn: {len(pending_set):,}  "
                   f"— Lasso/Box trên view 2D → Delete → Save PLY")
        return fxy, fxz, fyz, f3d, status

    return app

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-data", default=RAW_DATA_DIR)
    p.add_argument("--port", type=int, default=8051)
    args = p.parse_args()

    scans = discover_scans(args.raw_data)
    if not scans:
        print(f"No scans found in {args.raw_data}"); return

    print(f"Found {len(scans)} scans")
    print(f"Editor: http://127.0.0.1:{args.port}\n")
    app = create_app(scans)
    app.run(host="127.0.0.1", port=args.port, debug=False)

if __name__ == "__main__":
    main()
