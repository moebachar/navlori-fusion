"""Re-stage top-20 ILN 2.0 candidates with FULL site IDs in filenames, then
render trajectory overlays with exact site-matching. Fixes the 8-char prefix
collision bug (41 colliding prefixes in this dataset — MongoDB ObjectId
timestamps).
"""
import csv
import json
import math
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CSV = r"X:\navlori-fusion\data\iln20\site_features.csv"
DATA = r"X:\navlori-fusion\data\iln20\data"
INSPECT = r"X:\navlori-fusion\data\iln20\inspection"


def load_rows():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in ["score", "s_data", "s_size", "s_geom", "s_wifi", "med_trace_len",
                  "fr_straight", "fr_smooth", "fr_hard",
                  "med_aps_per_scan", "med_wifi_interval_s", "med_imu_hz"]:
            r[k] = float(r[k]) if r[k] not in ("", "None") else 0.0
        for k in ["n_traces", "loop_traces", "n_bssids"]:
            r[k] = int(r[k]) if r[k] not in ("", "None") else 0
        r["area_m2"] = float(r["area_m2"]) if r["area_m2"] not in ("", "None") else None
        r["has_geojson"] = r["has_geojson"] == "True"
    return rows


def passes(r):
    return (r["has_geojson"] and r["n_traces"] >= 15
            and r["med_imu_hz"] >= 40
            and r["med_aps_per_scan"] >= 20
            and 0 < r["med_wifi_interval_s"] <= 6
            and 30 <= r["med_trace_len"] <= 150)


def build_candidates(rows):
    strict = [r for r in rows if passes(r)]
    out = []
    # top 12 by score then n_traces
    for r in sorted(strict, key=lambda x: (-x["score"], -x["n_traces"]))[:12]:
        out.append(("top", r))
    used = set(((c[1]["site"], c[1]["floor"]) for c in out))

    def pick(label, key):
        for f in sorted(strict, key=key, reverse=True):
            tag = (f["site"], f["floor"])
            if tag not in used:
                used.add(tag)
                out.append((label, f))
                return
    pick("div-straight", lambda f: f["fr_straight"])
    pick("div-loop",     lambda f: f["loop_traces"] / max(1, f["n_traces"]))
    pick("div-hard",     lambda f: f["fr_hard"])

    small = [r for r in strict if r["area_m2"] and r["area_m2"] < 8000]
    for r in sorted(small, key=lambda x: (-x["score"], -x["n_traces"]))[:5]:
        tag = (r["site"], r["floor"])
        if tag not in used:
            used.add(tag)
            out.append(("small", r))
    return out


def fname(label, r):
    area_str = f"{int(r['area_m2'])}" if r["area_m2"] else "NA"
    return (f"sc{r['score']:.2f}_trc{r['n_traces']:03d}_{label:<13}_"
            f"str{int(100*r['fr_straight']):02d}_hd{int(100*r['fr_hard']):02d}_"
            f"wifi{r['med_wifi_interval_s']:.2f}s_area{area_str}m2_"
            f"{r['site']}_{r['floor']}")


def load_waypoints(p: str) -> np.ndarray:
    pts = []
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[1] == "TYPE_WAYPOINT":
                try:
                    pts.append((int(c[0]), float(c[2]), float(c[3])))
                except ValueError:
                    pass
    pts.sort()
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([(x, y) for _, x, y in pts], dtype=np.float32)


def render(site: str, floor: str, base_name: str) -> str:
    fd = os.path.join(DATA, site, floor)
    info_p = os.path.join(fd, "floor_info.json")
    img_p = os.path.join(fd, "floor_image.png")
    pdir = os.path.join(fd, "path_data_files")
    if not (os.path.isfile(info_p) and os.path.isfile(img_p) and os.path.isdir(pdir)):
        return f"missing assets in {fd}"

    info = json.load(open(info_p))
    mi = info.get("map_info", info)
    width_m = float(mi["width"]); height_m = float(mi["height"])

    img = Image.open(img_p)
    polylines = []
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".txt"):
            continue
        wp = load_waypoints(os.path.join(pdir, fn))
        if len(wp) >= 2:
            polylines.append(wp)

    # Official ILN 2.0 convention (per location-competition/io_f.py +
    # visualize_f.py): waypoints are plotted DIRECTLY (no flip, no scale),
    # image placed with origin='upper' so its top row sits at data y=H.
    # Standard y-up axes.
    fig, ax = plt.subplots(figsize=(10, 10 * (img.size[1] / img.size[0])))
    ax.imshow(img, extent=[0, width_m, 0, height_m], origin="upper",
              interpolation="bilinear", aspect="equal")
    for wp in polylines:
        x = wp[:, 0]; y = wp[:, 1]
        ax.plot(x, y, linewidth=0.6, alpha=0.45, color="#ff2a55")
        ax.scatter(x[0], y[0], s=10, c="#00cc44", zorder=4, alpha=0.7)
        ax.scatter(x[-1], y[-1], s=10, c="#e6194b", marker="s", zorder=4, alpha=0.7)
    ax.set_xlim(0, width_m); ax.set_ylim(0, height_m)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{site[:12]}…/{floor}  —  {len(polylines)} traces, "
                 f"{width_m:.0f}×{height_m:.0f} m  ({width_m*height_m:,.0f} m²)",
                 fontsize=10)
    ax.tick_params(labelsize=8)
    out = os.path.join(INSPECT, base_name + "_overlay.png")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"{len(polylines)} traces"


def main():
    # Clean old staged files (8-char-prefix names) — anything matching the
    # old or new pattern but not the new full-site-ID convention.
    for f in os.listdir(INSPECT):
        if f.endswith(".png"):
            os.remove(os.path.join(INSPECT, f))

    rows = load_rows()
    cands = build_candidates(rows)
    print(f"[restage] {len(cands)} candidates", flush=True)

    for i, (label, r) in enumerate(cands, 1):
        base = fname(label, r)
        site, floor = r["site"], r["floor"]
        src_img = os.path.join(DATA, site, floor, "floor_image.png")
        if not os.path.exists(src_img):
            print(f"[{i:>2}/{len(cands)}] MISSING floor_image.png for {site}/{floor}")
            continue
        # plain (no overlay)
        shutil.copy(src_img, os.path.join(INSPECT, base + ".png"))
        # overlay
        msg = render(site, floor, base)
        print(f"[{i:>2}/{len(cands)}] {label:<13} {site} {floor}  -> {msg}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
