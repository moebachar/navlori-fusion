"""
NavLoRI — Web-based Path Annotation Tool
==========================================
Serves a web page where you click on the building map to define paths.

Usage:
  python annotate_paths.py [--port 5555]

Then open http://localhost:5555 (forward the port if on SSH).

Controls:
  - Left click on map  : add waypoint to current path
  - Right click on map : undo last waypoint
  - [Next Path] button : finish current path, start next
  - [Undo Path] button : delete the last completed path
  - [Save]      button : write paths.json

Author: Mohamed — NavLoRI Project, CESI LINEACT
"""

import argparse
import json
import math
import os

from flask import Flask, jsonify, request, Response

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_FILE = os.path.join(HERE, "paths.json")

# ── Wall geometry (same as viz_paths.py) ───────────────────────────────────
RAW_WALLS = [
    (1.48, -1.47, 2.2327, 0.2, 2.7),
    (1.27, -3.35, -2.5454, 0.2, 2.707),
    (-0.61, -6.52, 2.2301, 0.2, 6.9709),
    (-10.33, -5.40, 2.15, 0.2, 7.5),
    (-2.16, -10.77, -2.6475, 0.2, 5.0),
    (1.10, -11.53, -1.0, 0.2, 5.1671),
    (3.99, -11.34, -2.6168, 0.2, 3.0),
    (-1.06, -16.10, 2.1194, 0.2, 13.8),
    (-8.01, -18.25, 0.6777, 0.2, 1.9),
    (-6.55, -16.13, -1.0151, 0.2, 5.1559),
    (-5.62, -12.74, 0.5114, 0.2, 4.6),
    (-8.43, -11.76, 2.1214, 0.2, 3.8606),
    (-12.77, -8.61, 0.5831, 0.2, 2.9),
    (-10.79, -11.60, 0.5831, 0.2, 2.7),
    (-6.30, -4.47, -2.482, 0.2, 3.1004),
    (-4.05, -4.58, -0.8897, 0.2, 3.6),
    (-3.72, -1.98, 0.5912, 0.2, 3.4447),
    (-2.75, 1.41, -0.8125, 0.2, 6.0),
    (0.84, 1.52, -2.5178, 0.2, 5.3),
]

ROBOT_RADIUS = 0.55
SAFETY_COEFF = 1.1
EXTRA_MARGIN = 0.25
SAFE_CLEARANCE = ROBOT_RADIUS * SAFETY_COEFF + EXTRA_MARGIN


def wall_polygons(inflate=0.0):
    polys = []
    for tx, ty, angle, sx, sy in RAW_WALLS:
        hx = sx / 2 + inflate
        hy = sy / 2 + inflate
        corners_local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        ca, sa = math.cos(angle), math.sin(angle)
        corners = [{"x": round(tx + ca * cx - sa * cy, 4),
                     "y": round(ty + sa * cx + ca * cy, 4)}
                    for cx, cy in corners_local]
        polys.append(corners)
    return polys


app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NavLoRI Path Annotator</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #eee; display: flex; flex-direction: column;
       height: 100vh; }
#toolbar { display: flex; align-items: center; gap: 12px; padding: 10px 16px;
           background: #16213e; border-bottom: 1px solid #0f3460; flex-shrink: 0; }
#toolbar h1 { font-size: 16px; color: #e94560; margin-right: 16px; }
#toolbar button { padding: 8px 16px; border: none; border-radius: 4px;
                  cursor: pointer; font-size: 13px; font-weight: 600; }
#btn-next { background: #00b894; color: #fff; }
#btn-undo-wp { background: #fdcb6e; color: #333; }
#btn-undo-path { background: #e17055; color: #fff; }
#btn-save { background: #0984e3; color: #fff; }
#btn-clear { background: #636e72; color: #fff; }
#toolbar button:hover { opacity: 0.85; }
#status { margin-left: auto; font-size: 13px; color: #dfe6e9; }
#canvas-wrap { flex: 1; overflow: hidden; position: relative; }
canvas { display: block; width: 100%; height: 100%; }
#coords { position: absolute; bottom: 8px; left: 12px; font-size: 12px;
          color: #b2bec3; background: rgba(0,0,0,0.5); padding: 4px 8px;
          border-radius: 4px; pointer-events: none; }
#help { position: absolute; top: 8px; right: 12px; font-size: 11px;
        color: #b2bec3; background: rgba(0,0,0,0.5); padding: 8px 12px;
        border-radius: 4px; pointer-events: none; line-height: 1.6; }
</style>
</head>
<body>
<div id="toolbar">
  <h1>NavLoRI Path Annotator</h1>
  <button id="btn-undo-wp" onclick="undoWaypoint()">Undo Point</button>
  <button id="btn-next" onclick="nextPath()">Next Path</button>
  <button id="btn-undo-path" onclick="undoPath()">Undo Path</button>
  <button id="btn-clear" onclick="clearAll()">Clear All</button>
  <button id="btn-save" onclick="savePaths()">Save</button>
  <span id="status">Path 0 | 0 waypoints | 0 paths done</span>
</div>
<div id="canvas-wrap">
  <canvas id="c"></canvas>
  <div id="coords">x: — y: —</div>
  <div id="help">
    Left click = add waypoint<br>
    Right click = undo last point<br>
    Scroll = zoom &nbsp;|&nbsp; Middle-drag = pan
  </div>
</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const coordsEl = document.getElementById('coords');
const statusEl = document.getElementById('status');

// World data (injected by server)
const WALLS = ##WALLS##;
const SAFETY = ##SAFETY##;

// Map bounds (world coords)
const MAP = { xmin: -14.5, xmax: 5.5, ymin: -20, ymax: 3.5 };

// View state (pan & zoom)
let view = { cx: (MAP.xmin + MAP.xmax) / 2, cy: (MAP.ymin + MAP.ymax) / 2, scale: 1.0 };

// Path data
let completedPaths = [];
let currentPath = [];

// Colours for completed paths
const COLORS = [
  '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6',
  '#bfef45','#fabed4','#469990','#dcbeff','#9A6324','#800000','#aaffc3',
  '#808000','#000075','#a9a9a9','#ffe119','#e6beff','#ffd8b1',
  '#00b894','#e17055','#0984e3','#fdcb6e','#6c5ce7','#00cec9','#ff7675',
  '#74b9ff','#a29bfe','#55efc4'
];

function resize() {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}
window.addEventListener('resize', resize);

// ── Coordinate transforms ──────────────────────────────────────────────
function worldToScreen(wx, wy) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const worldW = (MAP.xmax - MAP.xmin) / view.scale;
  const worldH = (MAP.ymax - MAP.ymin) / view.scale;
  const sx = (wx - view.cx + worldW / 2) / worldW * w;
  const sy = (view.cy - wy + worldH / 2) / worldH * h;  // Y flipped
  return [sx, sy];
}

function screenToWorld(sx, sy) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const worldW = (MAP.xmax - MAP.xmin) / view.scale;
  const worldH = (MAP.ymax - MAP.ymin) / view.scale;
  const wx = (sx / w) * worldW + view.cx - worldW / 2;
  const wy = view.cy + worldH / 2 - (sy / h) * worldH;  // Y flipped
  return [wx, wy];
}

// ── Drawing ────────────────────────────────────────────────────────────
function drawPolygon(pts, fill, stroke, lw) {
  ctx.beginPath();
  const [x0, y0] = worldToScreen(pts[0].x, pts[0].y);
  ctx.moveTo(x0, y0);
  for (let i = 1; i < pts.length; i++) {
    const [x, y] = worldToScreen(pts[i].x, pts[i].y);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
}

function drawCircle(wx, wy, radiusPx, fill, stroke) {
  const [sx, sy] = worldToScreen(wx, wy);
  ctx.beginPath();
  ctx.arc(sx, sy, radiusPx, 0, Math.PI * 2);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
}

function drawPath(waypoints, color, width, showMarkers) {
  if (waypoints.length < 1) return;
  // Line
  if (waypoints.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    const [x0, y0] = worldToScreen(waypoints[0].x, waypoints[0].y);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < waypoints.length; i++) {
      const [x, y] = worldToScreen(waypoints[i].x, waypoints[i].y);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  // Markers
  if (showMarkers) {
    waypoints.forEach((p, i) => {
      const r = (i === 0) ? 7 : 5;
      drawCircle(p.x, p.y, r, color, '#000');
      // Label
      const [sx, sy] = worldToScreen(p.x, p.y);
      ctx.fillStyle = '#fff';
      ctx.font = '10px sans-serif';
      ctx.fillText(i === 0 ? 'S' : (i === waypoints.length - 1 ? 'E' : i), sx + 8, sy - 4);
    });
  }
}

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 0.5;
  for (let gx = Math.floor(MAP.xmin); gx <= MAP.xmax; gx++) {
    const [sx] = worldToScreen(gx, 0);
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
  }
  for (let gy = Math.floor(MAP.ymin); gy <= MAP.ymax; gy++) {
    const [, sy] = worldToScreen(0, gy);
    ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(w, sy); ctx.stroke();
  }

  // Safety zones
  SAFETY.forEach(p => drawPolygon(p, 'rgba(233,69,96,0.1)', null));

  // Walls
  WALLS.forEach(p => drawPolygon(p, 'rgba(136,136,136,0.7)', '#333', 1));

  // Completed paths
  completedPaths.forEach((path, i) => {
    drawPath(path, COLORS[i % COLORS.length], 2, true);
    // Path label
    if (path.length > 0) {
      const [sx, sy] = worldToScreen(path[0].x, path[0].y);
      ctx.fillStyle = COLORS[i % COLORS.length];
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('P' + i, sx - 18, sy - 10);
    }
  });

  // Current path (green, thicker)
  drawPath(currentPath, '#00ff88', 3, true);

  updateStatus();
}

function updateStatus() {
  statusEl.textContent = `Path ${completedPaths.length} | ${currentPath.length} waypoints | ${completedPaths.length} paths done`;
}

// ── Mouse interaction ──────────────────────────────────────────────────
let isPanning = false;
let panStart = null;
let panViewStart = null;

canvas.addEventListener('contextmenu', e => e.preventDefault());

canvas.addEventListener('mousedown', e => {
  if (e.button === 1) {  // Middle button = pan
    isPanning = true;
    panStart = { x: e.offsetX, y: e.offsetY };
    panViewStart = { cx: view.cx, cy: view.cy };
    canvas.style.cursor = 'grabbing';
    e.preventDefault();
  }
});

canvas.addEventListener('mousemove', e => {
  const [wx, wy] = screenToWorld(e.offsetX, e.offsetY);
  coordsEl.textContent = `x: ${wx.toFixed(2)}  y: ${wy.toFixed(2)}`;

  if (isPanning) {
    const dx = e.offsetX - panStart.x;
    const dy = e.offsetY - panStart.y;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const worldW = (MAP.xmax - MAP.xmin) / view.scale;
    const worldH = (MAP.ymax - MAP.ymin) / view.scale;
    view.cx = panViewStart.cx - dx / w * worldW;
    view.cy = panViewStart.cy + dy / h * worldH;
    draw();
  }
});

canvas.addEventListener('mouseup', e => {
  if (e.button === 1) {
    isPanning = false;
    canvas.style.cursor = 'crosshair';
  }
});

canvas.addEventListener('click', e => {
  if (e.button !== 0) return;
  const [wx, wy] = screenToWorld(e.offsetX, e.offsetY);
  currentPath.push({ x: Math.round(wx * 10000) / 10000, y: Math.round(wy * 10000) / 10000 });
  draw();
});

canvas.addEventListener('contextmenu', e => { e.preventDefault(); });
canvas.addEventListener('mousedown', e => {
  if (e.button === 2) {  // Right click = undo
    e.preventDefault();
    undoWaypoint();
  }
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  // Zoom toward mouse position
  const [wx, wy] = screenToWorld(e.offsetX, e.offsetY);
  view.scale *= factor;
  view.scale = Math.max(0.3, Math.min(10, view.scale));
  // Adjust center so mouse world pos stays fixed
  const [wx2, wy2] = screenToWorld(e.offsetX, e.offsetY);
  view.cx -= (wx2 - wx);
  view.cy -= (wy2 - wy);
  draw();
}, { passive: false });

canvas.style.cursor = 'crosshair';

// ── Button handlers ────────────────────────────────────────────────────
function undoWaypoint() {
  if (currentPath.length > 0) {
    currentPath.pop();
    draw();
  }
}

function nextPath() {
  if (currentPath.length < 2) {
    alert('Need at least 2 waypoints!');
    return;
  }
  completedPaths.push([...currentPath]);
  currentPath = [];
  draw();
}

function undoPath() {
  if (completedPaths.length === 0) {
    alert('No completed paths to undo!');
    return;
  }
  completedPaths.pop();
  draw();
}

function clearAll() {
  if (!confirm('Clear ALL paths and current waypoints?')) return;
  completedPaths = [];
  currentPath = [];
  draw();
}

function savePaths() {
  // Auto-finish current if >= 2 points
  if (currentPath.length >= 2) {
    completedPaths.push([...currentPath]);
    currentPath = [];
  }
  if (completedPaths.length === 0) {
    alert('No paths to save!');
    return;
  }

  const data = {
    paths: completedPaths.map((wps, i) => ({
      id: i,
      waypoints: wps.map(p => ({ x: p.x, y: p.y }))
    }))
  };

  fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  })
  .then(r => r.json())
  .then(r => {
    if (r.ok) alert('Saved ' + completedPaths.length + ' paths to paths.json!');
    else alert('Error: ' + r.error);
    draw();
  })
  .catch(err => alert('Error: ' + err));
}

// ── Init ───────────────────────────────────────────────────────────────
resize();
</script>
</body>
</html>"""


@app.route("/")
def index():
    walls = wall_polygons(0.0)
    safety = wall_polygons(SAFE_CLEARANCE)
    html = HTML.replace("##WALLS##", json.dumps(walls))
    html = html.replace("##SAFETY##", json.dumps(safety))
    return Response(html, mimetype="text/html")


@app.route("/save", methods=["POST"])
def save():
    try:
        data = request.get_json()
        paths = data["paths"]

        output = {
            "metadata": {
                "num_paths": len(paths),
                "safe_clearance": SAFE_CLEARANCE,
                "robot_radius": ROBOT_RADIUS,
            },
            "paths": paths,
        }

        with open(PATHS_FILE, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Saved {len(paths)} paths to {PATHS_FILE}")
        return jsonify(ok=True, num_paths=len(paths))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    print("=" * 50)
    print("  NavLoRI Path Annotation Tool")
    print(f"  http://localhost:{args.port}")
    print("=" * 50)
    print("  Left click  = add waypoint")
    print("  Right click  = undo last point")
    print("  Scroll       = zoom")
    print("  Middle-drag  = pan")
    print()

    app.run(host="0.0.0.0", port=args.port, debug=False)
