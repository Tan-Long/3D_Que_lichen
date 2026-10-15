#!/usr/bin/env python3
"""Throwaway prototype: infer terrain aspect and split a trunk upslope/downslope.

The output is one self-contained HTML file.  It deliberately uses only the sparse
ARKit point cloud, camera trajectory, and location gravity; it is not production
segmentation code.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


VARIANTS = {
    "near": (0.30, 1.20),
    "mid": (0.35, 2.00),
    "wide": (0.25, 3.00),
    "far": (0.25, 4.50),
}

PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_xyz(path: Path) -> np.ndarray:
    """Read scalar vertex properties without handing a private-use path to Open3D."""
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"truncated PLY header: {path}")
            text = line.decode("ascii", errors="strict").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        fmt = next(x.split()[1] for x in header_lines if x.startswith("format "))
        count = 0
        props = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count, in_vertex = int(parts[2]), True
            elif parts and parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                if parts[1] == "list":
                    raise ValueError("list property inside vertex element is unsupported")
                props.append((parts[2], PLY_TYPES[parts[1]]))
        if fmt == "binary_little_endian":
            arr = np.fromfile(f, dtype=np.dtype([(n, "<" + t) for n, t in props]), count=count)
            return np.column_stack((arr["x"], arr["y"], arr["z"])).astype(float)
        if fmt == "ascii":
            arr = np.loadtxt(f, max_rows=count)
            names = [p[0] for p in props]
            return arr[:, [names.index("x"), names.index("y"), names.index("z")]].astype(float)
        raise ValueError(f"unsupported PLY format: {fmt}")


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def read_xyz_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])


def read_gravity(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([0.0, -1.0, 0.0])
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                values.append([float(row["gravity_x"]), float(row["gravity_y"]), float(row["gravity_z"])])
            except (KeyError, TypeError, ValueError):
                pass
    if not values:
        return np.array([0.0, -1.0, 0.0])
    values = np.asarray(values)
    values = values[np.all(np.isfinite(values), axis=1)]
    return unit(np.median(values, axis=0))


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, up)) > 0.85:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = unit(seed - np.dot(seed, up) * up)
    return e1, unit(np.cross(up, e1))


def circle_center(camera: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float]:
    e1, e2 = horizontal_basis(up)
    origin = np.median(camera, axis=0)
    xy = np.column_stack(((camera - origin) @ e1, (camera - origin) @ e2))
    # Trim isolated trajectory jumps, then algebraic least-squares circle.
    med = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - med, axis=1)
    keep = dist <= np.quantile(dist, 0.98)
    x, y = xy[keep].T
    A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    center = origin + cx * e1 + cy * e2
    # Preserve the median trajectory height; plane fitting later sets the base.
    center += np.dot(np.median(camera, axis=0) - center, up) * up
    return center, radius


def fit_plane_ransac(points: np.ndarray, up: np.ndarray, seed: int) -> tuple[np.ndarray, float, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = None
    threshold = 0.045
    if len(points) < 30:
        raise ValueError("too few ground candidates")
    for _ in range(900):
        tri = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(normal) < 1e-8:
            continue
        normal = unit(normal)
        if np.dot(normal, up) < 0:
            normal = -normal
        if np.dot(normal, up) < math.cos(math.radians(38)):
            continue
        d = -float(np.dot(normal, tri[0]))
        residual = np.abs(points @ normal + d)
        mask = residual < threshold
        score = int(mask.sum())
        if best is None or score > best[0]:
            best = (score, mask)
    if best is None or best[0] < 20:
        raise ValueError("no gravity-compatible plane")
    mask = best[1]
    # Two robust orthogonal refits.
    for _ in range(2):
        fit = points[mask]
        centroid = np.mean(fit, axis=0)
        _, _, vh = np.linalg.svd(fit - centroid, full_matrices=False)
        normal = vh[-1]
        if np.dot(normal, up) < 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        residual = np.abs(points @ normal + d)
        mad = np.median(np.abs(residual - np.median(residual)))
        cutoff = min(0.075, max(0.03, 2.8 * 1.4826 * mad))
        mask = residual < cutoff
    rmse = float(np.sqrt(np.mean((points[mask] @ normal + d) ** 2)))
    return normal, d, mask, rmse


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


def analyze_points(scan_name: str, pts: np.ndarray, cameras: np.ndarray,
                   gravity: np.ndarray, seed: int) -> dict:
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    up = -gravity
    center, camera_radius = circle_center(cameras, up)
    e1, e2 = horizontal_basis(up)
    delta = pts - center
    px, py, ph = delta @ e1, delta @ e2, delta @ up
    radial = np.hypot(px, py)

    results = {}
    for index, (name, (r0, r1)) in enumerate(VARIANTS.items()):
        annulus = (radial >= r0) & (radial <= r1)
        local_h = ph[annulus]
        if len(local_h) < 50:
            results[name] = {"valid": False, "reason": "too_few_annulus_points"}
            continue
        # Ground is expected in the lower 58% along gravity-up; RANSAC rejects clutter.
        ceiling = float(np.quantile(local_h, 0.58))
        candidate_idx = np.flatnonzero(annulus & (ph <= ceiling))
        try:
            normal, d, inlier_local, rmse = fit_plane_ransac(pts[candidate_idx], up, seed + index)
        except ValueError as exc:
            results[name] = {"valid": False, "reason": str(exc)}
            continue
        inlier_idx = candidate_idx[inlier_local]
        down = unit(gravity - np.dot(gravity, normal) * normal)
        grade_deg = angle_between(normal, up)
        base = center - (np.dot(normal, center) + d) * normal
        rel = pts[inlier_idx] - base
        angles = (np.arctan2(rel @ e2, rel @ e1) + 2 * np.pi) % (2 * np.pi)
        occupied = len(np.unique(np.floor(angles / (2 * np.pi / 12)).astype(int)))
        coverage = occupied / 12.0
        result = {
            "valid": True,
            "radii": [r0, r1],
            "normal": normal.tolist(),
            "plane_d": d,
            "base": base.tolist(),
            "down": down.tolist(),
            "up_slope": (-down).tolist(),
            "grade_deg": grade_deg,
            "grade_percent": math.tan(math.radians(grade_deg)) * 100,
            "candidate_count": int(len(candidate_idx)),
            "inlier_count": int(len(inlier_idx)),
            "inlier_ratio": float(len(inlier_idx) / len(candidate_idx)),
            "rmse_m": rmse,
            "coverage": coverage,
        }
        results[name] = result

    # Adaptive policy: test a local plane first, then enlarge only as needed.
    # A direction is accepted only if it agrees with the next wider annulus.
    adjacent = {}
    ordered = list(VARIANTS)
    for left, right in zip(ordered, ordered[1:]):
        if results.get(left, {}).get("valid") and results.get(right, {}).get("valid"):
            adjacent[f"{left}_{right}"] = angle_between(results[left]["down"], results[right]["down"])
    selected = None
    for name, next_name in (("mid", "wide"), ("wide", "far")):
        item = results.get(name, {})
        stability = adjacent.get(f"{name}_{next_name}")
        if (item.get("valid") and item["grade_percent"] >= 5 and item["coverage"] >= 0.5
                and item["rmse_m"] <= 0.06 and stability is not None and stability <= 15):
            selected = name
            break
    if selected is None:
        selected = "wide" if results.get("wide", {}).get("valid") else "mid"
    adaptive = dict(results[selected])
    adaptive["source_variant"] = selected
    adaptive["adjacent_angles_deg"] = adjacent
    stability = adjacent.get(f"{selected}_{'wide' if selected == 'mid' else 'far'}")
    reasons = []
    if adaptive.get("valid"):
        if adaptive["grade_percent"] < 5:
            reasons.append("độ dốc < 5%: hướng không xác định chắc chắn")
        if adaptive["coverage"] < 0.5:
            reasons.append("mặt đất không phủ đủ quanh gốc")
        if adaptive["rmse_m"] > 0.06:
            reasons.append("sai số mặt phẳng > 6 cm")
        if stability is None or stability > 15:
            reasons.append("hướng chưa ổn định khi mở rộng vùng fit")
    else:
        reasons.append("không fit được mặt đất")
    adaptive["quality_ok"] = not reasons
    adaptive["quality_reasons"] = reasons
    adaptive["aspect_spread_deg"] = stability
    results["adaptive"] = adaptive

    # Display sample biased toward ground and trunk vicinity, deterministic.
    rng = np.random.default_rng(seed + 100)
    near = np.flatnonzero(radial < 3.2)
    if len(near) > 4200:
        near = rng.choice(near, 4200, replace=False)
    display = pts[near]
    disp_delta = display - center
    display_rows = np.column_stack((disp_delta @ e1, disp_delta @ e2, disp_delta @ up))
    if adaptive.get("valid"):
        base_h = float(np.dot(np.asarray(adaptive["base"]) - center, up))
    else:
        base_h = float(np.quantile(ph, 0.1))
    display_rows[:, 2] -= base_h
    for result in results.values():
        if not result.get("valid"):
            continue
        normal = np.asarray(result["normal"])
        residual = np.abs(pts[near] @ normal + result["plane_d"])
        rr = radial[near]
        ground = (rr >= result["radii"][0]) & (rr <= result["radii"][1]) & (residual < 0.06)
        result["ground_display"] = np.flatnonzero(ground).tolist()
    cam_delta = cameras[:: max(1, len(cameras) // 450)] - center
    cam_rows = np.column_stack((cam_delta @ e1, cam_delta @ e2, cam_delta @ up - base_h))
    return {
        "name": scan_name,
        "point_count": int(len(pts)),
        "pose_count": int(len(cameras)),
        "gravity": gravity.tolist(),
        "trajectory_radius_m": camera_radius,
        "center": center.tolist(),
        "basis": [e1.tolist(), e2.tolist(), up.tolist()],
        "variants": results,
        "display": np.round(display_rows, 4).tolist(),
        "camera": np.round(cam_rows, 4).tolist(),
    }


def analyze_scan(scan: Path, seed: int) -> dict:
    pts = read_ply_xyz(scan / "point_cloud_raw.ply")
    cameras = read_xyz_csv(scan / "odometry.csv")
    gravity = read_gravity(scan / "location.csv")
    return analyze_points(scan.name, pts, cameras, gravity, seed)


def selected_scans(pilot: Path, count: int) -> list[Path]:
    with pilot.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [Path(r["raw_path"]) for r in rows if r["pilot_role"] == "primary_pilot" and r["has_point_cloud"] == "True"]
    return paths[:count]


def build_html(data: list[dict]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prototype — hướng dốc quanh cây quế</title>
<style>
:root{{--bg:#0b1110;--panel:#121c19;--ink:#eef7f1;--muted:#9fb1a8;--line:#294038;--up:#ffb454;--down:#56b4ff;--ground:#77d18b;--bad:#ff7474}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}}header{{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:center;flex-wrap:wrap}}h1{{font-size:20px;margin:0}}.badge{{background:#382c14;color:#ffd98e;padding:4px 9px;border-radius:99px;font-weight:700}}main{{padding:18px 24px;max-width:1500px;margin:auto}}button{{background:#192824;color:var(--ink);border:1px solid #365148;border-radius:7px;padding:8px 11px;cursor:pointer}}button.active{{background:#285e48;border-color:#57b98c}}#scans,#variants{{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:12px}}.grid{{display:grid;grid-template-columns:1fr 1fr 330px;gap:12px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}}canvas{{width:100%;height:430px;background:#0d1513;border-radius:7px}}h2{{font-size:15px;margin:0 0 8px}}.metric{{display:grid;grid-template-columns:1fr auto;gap:7px;border-bottom:1px solid #22342e;padding:7px 0}}.value{{font-variant-numeric:tabular-nums;font-weight:650}}.ok{{color:#7fe3a0}}.bad{{color:var(--bad)}}.legend span{{margin-right:12px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}details{{margin-top:10px}}pre{{white-space:pre-wrap;word-break:break-word;font-size:11px;color:#b9cbc2;max-height:280px;overflow:auto}}.walk{{margin-top:12px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.walk div{{background:#101916;border-left:3px solid #497966;padding:9px}}.note{{color:var(--muted);margin:5px 0 13px}}@media(max-width:1050px){{.grid{{grid-template-columns:1fr}}canvas{{height:350px}}}}
</style></head><body>
<header><span class="badge">PROTOTYPE</span><h1>Nhận diện mặt upslope / downslope từ LiDAR quanh gốc</h1><span class="note">Không phải kết quả production</span></header>
<main><div id="scans"></div><div id="variants"></div>
<p class="note">Màu xanh lá = điểm mặt đất fit; cam = nửa upslope; xanh dương = nửa downslope; đường trắng = quỹ đạo camera. Chế độ adaptive tự mở rộng vùng fit và chỉ nhận hướng khi ổn định.</p>
<div class="grid"><section class="panel"><h2>Nhìn từ trên xuống</h2><canvas id="top"></canvas></section><section class="panel"><h2>Mặt cắt theo hướng dốc</h2><canvas id="side"></canvas></section><aside class="panel"><h2>Chỉ số kiểm soát</h2><div id="metrics"></div><details><summary>Trạng thái đầy đủ</summary><pre id="state"></pre></details></aside></div>
<div class="walk"><div><b>1. Tâm cây</b><br>Fit vòng tròn từ quỹ đạo camera.</div><div><b>2. Mặt đất</b><br>RANSAC có ràng buộc trọng lực trong vành quanh gốc.</div><div><b>3. Hướng dốc</b><br>Chiếu vector trọng lực lên mặt phẳng đất.</div><div><b>4. Chia thân</b><br>Mặt phẳng đứng qua trục cây tạo hai nửa 180°.</div></div></main>
<script>const DATA={payload};let si=0,variant='adaptive';
const scans=document.querySelector('#scans'),variants=document.querySelector('#variants');
function buttons(){{scans.innerHTML=DATA.map((s,i)=>`<button class="${{i===si?'active':''}}" onclick="si=${{i}};render()">${{esc(s.name)}}</button>`).join('');variants.innerHTML=['adaptive','near','mid','wide','far'].map(v=>`<button class="${{v===variant?'active':''}}" onclick="variant='${{v}}';render()">${{v}} ${{({{adaptive:'tự chọn',near:'0.30–1.20 m',mid:'0.35–2.00 m',wide:'0.25–3.00 m',far:'0.25–4.50 m'}})[v]}}</button>`).join('')}}
function esc(x){{return x.replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}})[c])}}
function prep(id){{const c=document.querySelector(id),d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;const x=c.getContext('2d');x.scale(d,d);return [x,r.width,r.height]}}
function scale(points,a,b,w,h,pad=25){{let vals=points.flatMap(p=>[p[a],p[b]]),m=Math.max(1,...vals.map(Math.abs));return p=>[w/2+p[a]/m*(Math.min(w,h)/2-pad),h/2-p[b]/m*(Math.min(w,h)/2-pad)]}}
function draw(){{const s=DATA[si],v=s.variants[variant];for(const spec of [['#top',0,1],['#side',1,2]]){{const [ctx,w,h]=prep(spec[0]);ctx.clearRect(0,0,w,h);const map=scale(s.display,spec[1],spec[2],w,h);ctx.globalAlpha=.28;ctx.fillStyle='#82938b';for(const p of s.display){{const q=map(p);ctx.fillRect(q[0],q[1],1.5,1.5)}}if(v.valid){{const ground=new Set(v.ground_display||[]),b=s.basis,down=v.down;const dx=down[0]*b[0][0]+down[1]*b[0][1]+down[2]*b[0][2],dy=down[0]*b[1][0]+down[1]*b[1][1]+down[2]*b[1][2];ctx.globalAlpha=.9;for(let i=0;i<s.display.length;i++){{const p=s.display[i],r=Math.hypot(p[0],p[1]);let color=null,size=2;if(ground.has(i)){{color='#77d18b';size=3}}else if(r<.48&&p[2]>.05&&p[2]<2.6){{color=(p[0]*dx+p[1]*dy)>=0?'#56b4ff':'#ffb454';size=3}}if(color){{ctx.fillStyle=color;const q=map(p);ctx.fillRect(q[0]-1,q[1]-1,size,size)}}}}if(spec[0]==='#top'){{arrow(ctx,w/2,h/2,dx,dy,'#56b4ff','DOWN');arrow(ctx,w/2,h/2,-dx,-dy,'#ffb454','UP')}}}}ctx.globalAlpha=.7;ctx.strokeStyle='#fff';ctx.beginPath();for(let i=0;i<s.camera.length;i++){{const q=map(s.camera[i]);i?ctx.lineTo(...q):ctx.moveTo(...q)}}ctx.stroke()}}}}
function arrow(ctx,x,y,dx,dy,color,label){{const n=Math.hypot(dx,dy)||1,L=90;dx=dx/n*L;dy=-dy/n*L;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();ctx.fillText(label,x+dx+5,y+dy)}}
function metrics(){{const s=DATA[si],v=s.variants[variant],el=document.querySelector('#metrics');if(!v.valid){{el.innerHTML=`<p class=bad>Không fit được: ${{esc(v.reason)}}</p>`;return}}const ok=variant==='adaptive'?v.quality_ok:null;el.innerHTML=`<div class=metric><span>Trạng thái</span><span class="value ${{ok===false?'bad':'ok'}}">${{ok===false?'CẦN XEM LẠI':'FIT ĐƯỢC'}}</span></div>${{v.source_variant?`<div class=metric><span>Vùng được chọn</span><span class=value>${{v.source_variant}}</span></div>`:''}}<div class=metric><span>Độ dốc</span><span class=value>${{v.grade_deg.toFixed(2)}}° (${{v.grade_percent.toFixed(1)}}%)</span></div><div class=metric><span>RMSE mặt đất</span><span class=value>${{(v.rmse_m*100).toFixed(1)}} cm</span></div><div class=metric><span>Inlier</span><span class=value>${{v.inlier_count}} / ${{v.candidate_count}}</span></div><div class=metric><span>Phủ góc quanh gốc</span><span class=value>${{(v.coverage*100).toFixed(0)}}%</span></div><div class=metric><span>Đổi hướng khi mở rộng</span><span class=value>${{v.aspect_spread_deg==null?'—':v.aspect_spread_deg.toFixed(1)+'°'}}</span></div><div class=metric><span>Bán kính quỹ đạo</span><span class=value>${{s.trajectory_radius_m.toFixed(2)}} m</span></div>${{(v.quality_reasons||[]).map(x=>`<p class=bad>• ${{esc(x)}}</p>`).join('')}}`;document.querySelector('#state').textContent=JSON.stringify({{scan:s.name,gravity:s.gravity,center:s.center,variant:v}},null,2)}}
function render(){{buttons();draw();metrics()}}window.addEventListener('resize',draw);render();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=Path("manifests/pilot.csv"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("prototypes/upslope_downslope/upslope_downslope_prototype.html"))
    args = parser.parse_args()
    scans = selected_scans(args.pilot, args.count)
    data = []
    for i, scan in enumerate(scans):
        print(f"[{i+1}/{len(scans)}] {scan.name}", flush=True)
        data.append(analyze_scan(scan, 4100 + i * 17))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(data), encoding="utf-8")
    summary = args.output.with_suffix(".json")
    compact = [{k: v for k, v in s.items() if k not in ("display", "camera")} for s in data]
    summary.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
