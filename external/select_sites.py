#!/usr/bin/env python3
"""
select_sites.py  --  Rank Indoor Location Competition 2.0 floors for Webots replay.

WHAT THIS DOES
--------------
Walks the downloaded ILN 2.0 dataset tree and scores every (site, floor) on the
four things we care about for the hybrid Webots dataset:

  1. DATA SUFFICIENCY   -- enough traces to split train/test
  2. SIZE FIT           -- floor area and median trace length inside a usable band
                           (not too small, not too long)
  3. GEOMETRY RICHNESS  -- the path mix has straight runs, smooth turns,
                           hard turns, AND at least some loops
  4. WIFI QUALITY       -- multi-AP scans (>= ~20 APs) at a usable cadence

It then prints a ranked table plus a per-floor feature dump (CSV) so the choice
is fully auditable, and suggests a DIVERSE top-3 (one straight-dominant, one
loop-rich, one hard-turn-rich) so the synthetic set covers all path types.

IMPORTANT ASSUMPTIONS (read these)
----------------------------------
* Geometry is measured on the TYPE_WAYPOINT polyline (the metric ground truth).
  Waypoints are surveyor-pressed at landmarks, so they cluster at corners --
  good for turn detection, but sparse. For finer turn analysis, swap in the
  step-detected path from the repo's compute_f.py. Waypoints are sufficient for
  RANKING, which is all this script claims to do.
* WiFi "scan" = all TYPE_WIFI rows sharing one system timestamp (col 0).
* All distances are already in METRES (dataset native frame). No unit guessing.
* Stdlib only -- runs anywhere with Python 3.7+. No numpy needed.

USAGE
-----
  python select_sites.py /path/to/iln20/data --csv features.csv

Expected tree (matches the official repo / full-dataset download):
  data/<siteID>/<floor>/path_data_files/<traceID>.txt
  data/<siteID>/<floor>/floor_info.json
  data/<siteID>/<floor>/geojson_map.json
"""

import os
import sys
import json
import math
import csv
import argparse
from collections import defaultdict

# ----------------------------------------------------------------------------
# Tunable thresholds -- change these, the script explains every number it uses.
# ----------------------------------------------------------------------------
AREA_MIN_M2        = 200.0    # floor smaller than this -> "too small"
AREA_MAX_M2        = 3000.0   # floor bigger than this  -> "too big" (per-wing OK)
TRACE_LEN_MIN_M    = 30.0     # median trace shorter than this -> "too short"
TRACE_LEN_MAX_M    = 150.0    # median trace longer than this  -> "too long to replay"
MIN_TRACES         = 15       # fewer traces than this -> not enough to split

# Turn classification (deviation angle at an interior waypoint, degrees)
STRAIGHT_MAX_DEG   = 20.0     # |turn| < 20 deg  -> straight
SMOOTH_MAX_DEG     = 70.0     # 20..70 deg       -> smooth turn
#                              > 70 deg          -> hard turn

# Loop detection
LOOP_CELL_M        = 1.0      # grid cell size for revisit test
LOOP_RESAMPLE_M    = 0.5      # densify polyline to this spacing before gridding
LOOP_MIN_GAP_STEPS = 8        # revisit must be this many samples apart to count

# WiFi targets
WIFI_APS_TARGET    = 20       # median APs/scan we'd like to hit
WIFI_INTERVAL_OK_S = 6.0      # scan interval (s) at/under which we're happy

# Score weights (sum = 1.0)
W_DATA, W_SIZE, W_GEOM, W_WIFI = 0.20, 0.20, 0.40, 0.20


# ----------------------------------------------------------------------------
# Trace parsing
# ----------------------------------------------------------------------------
def parse_trace(path):
    """Return waypoints[(t,x,y)], wifi_scan_aps{t:count}, bssids set, imu_ts list."""
    waypoints = []
    wifi_scan_aps = defaultdict(int)
    bssids = set()
    accel_ts = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 2:
                    continue
                t, typ = c[0], c[1]
                if typ == "TYPE_WAYPOINT" and len(c) >= 4:
                    try:
                        waypoints.append((int(t), float(c[2]), float(c[3])))
                    except ValueError:
                        pass
                elif typ == "TYPE_WIFI" and len(c) >= 4:
                    wifi_scan_aps[t] += 1
                    bssids.add(c[3])           # col 4 = bssid
                elif typ == "TYPE_ACCELEROMETER":
                    try:
                        accel_ts.append(int(t))
                    except ValueError:
                        pass
    except OSError:
        return [], {}, set(), []
    waypoints.sort(key=lambda w: w[0])
    return waypoints, dict(wifi_scan_aps), bssids, accel_ts


# ----------------------------------------------------------------------------
# Geometry helpers (all metric)
# ----------------------------------------------------------------------------
def polyline_length(wps):
    d = 0.0
    for (_, x0, y0), (_, x1, y1) in zip(wps, wps[1:]):
        d += math.hypot(x1 - x0, y1 - y0)
    return d


def turn_counts(wps):
    """Classify deviation angle at each interior vertex."""
    straight = smooth = hard = 0
    for (_, ax, ay), (_, bx, by), (_, cx, cy) in zip(wps, wps[1:], wps[2:]):
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = cx - bx, cy - by
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cross = v1x * v2y - v1y * v2x
        dot = v1x * v2x + v1y * v2y
        ang = abs(math.degrees(math.atan2(cross, dot)))  # deviation from straight
        if ang < STRAIGHT_MAX_DEG:
            straight += 1
        elif ang < SMOOTH_MAX_DEG:
            smooth += 1
        else:
            hard += 1
    return straight, smooth, hard


def has_loop(wps):
    """Densify the polyline, grid it, and flag a revisited cell separated in time."""
    if len(wps) < 3:
        return False
    # densify
    pts = []
    for (_, x0, y0), (_, x1, y1) in zip(wps, wps[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(seg / LOOP_RESAMPLE_M))
        for s in range(steps):
            f = s / steps
            pts.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    pts.append((wps[-1][1], wps[-1][2]))
    last_seen = {}
    for i, (x, y) in enumerate(pts):
        cell = (int(x // LOOP_CELL_M), int(y // LOOP_CELL_M))
        if cell in last_seen and (i - last_seen[cell]) >= LOOP_MIN_GAP_STEPS:
            return True
        last_seen[cell] = i
    return False


def median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ----------------------------------------------------------------------------
# Floor metadata
# ----------------------------------------------------------------------------
def floor_area(floor_dir):
    """Read floor_info.json -> area in m^2. Returns None if unavailable."""
    p = os.path.join(floor_dir, "floor_info.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    mi = d.get("map_info", d)
    w, h = mi.get("width"), mi.get("height")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        return float(w) * float(h)
    return None


def has_geojson(floor_dir):
    return os.path.isfile(os.path.join(floor_dir, "geojson_map.json"))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def clamp01(v):
    return max(0.0, min(1.0, v))


def score_floor(f):
    # data sufficiency
    s_data = clamp01(f["n_traces"] / MIN_TRACES)
    # size fit
    area_ok = (f["area_m2"] is not None and AREA_MIN_M2 <= f["area_m2"] <= AREA_MAX_M2)
    len_ok = TRACE_LEN_MIN_M <= f["med_trace_len"] <= TRACE_LEN_MAX_M
    s_size = 0.5 * (1.0 if area_ok else 0.0) + 0.5 * (1.0 if len_ok else 0.0)
    # geometry richness: reward presence of EACH ingredient
    tot = f["straight"] + f["smooth"] + f["hard"]
    if tot:
        fr_straight = f["straight"] / tot
        fr_smooth = f["smooth"] / tot
        fr_hard = f["hard"] / tot
    else:
        fr_straight = fr_smooth = fr_hard = 0.0
    g_straight = 1.0 if 0.30 <= fr_straight <= 0.85 else 0.4  # some, not all
    g_smooth = clamp01(fr_smooth / 0.12)
    g_hard = clamp01(fr_hard / 0.08)
    g_loop = clamp01(f["loop_traces"] / max(1, 0.15 * f["n_traces"]))
    s_geom = 0.25 * (g_straight + g_smooth + g_hard + g_loop)
    # wifi quality
    s_aps = clamp01(f["med_aps_per_scan"] / WIFI_APS_TARGET)
    s_int = clamp01(WIFI_INTERVAL_OK_S / f["med_wifi_interval_s"]) if f["med_wifi_interval_s"] else 0.0
    s_wifi = 0.5 * (s_aps + s_int)
    total = W_DATA * s_data + W_SIZE * s_size + W_GEOM * s_geom + W_WIFI * s_wifi
    f.update(dict(s_data=s_data, s_size=s_size, s_geom=s_geom, s_wifi=s_wifi,
                  fr_straight=fr_straight, fr_smooth=fr_smooth, fr_hard=fr_hard,
                  score=total))
    return f


# ----------------------------------------------------------------------------
# Walk + aggregate
# ----------------------------------------------------------------------------
def analyze(data_root):
    floors = []
    for site in sorted(os.listdir(data_root)):
        site_dir = os.path.join(data_root, site)
        if not os.path.isdir(site_dir):
            continue
        for floor in sorted(os.listdir(site_dir)):
            fdir = os.path.join(site_dir, floor)
            pdir = os.path.join(fdir, "path_data_files")
            if not os.path.isdir(pdir):
                continue
            trace_lens, all_aps, all_intervals = [], [], []
            straight = smooth = hard = loop_traces = n_traces = 0
            bssids_all = set()
            imu_rates = []
            for fn in os.listdir(pdir):
                if not fn.endswith(".txt"):
                    continue
                wps, wifi, bssids, accel_ts = parse_trace(os.path.join(pdir, fn))
                if len(wps) < 2:
                    continue
                n_traces += 1
                trace_lens.append(polyline_length(wps))
                st, sm, hd = turn_counts(wps)
                straight += st; smooth += sm; hard += hd
                if has_loop(wps):
                    loop_traces += 1
                bssids_all |= bssids
                if wifi:
                    all_aps.extend(wifi.values())
                    ts = sorted(int(t) for t in wifi)
                    gaps = [(b - a) / 1000.0 for a, b in zip(ts, ts[1:]) if b > a]
                    all_intervals.extend(gaps)
                if len(accel_ts) > 10:
                    a = sorted(accel_ts)
                    dts = [b - a_ for a_, b in zip(a, a[1:]) if b > a_]
                    md = median(dts)
                    if md > 0:
                        imu_rates.append(1000.0 / md)
            if n_traces == 0:
                continue
            floors.append(score_floor(dict(
                site=site, floor=floor, n_traces=n_traces,
                area_m2=floor_area(fdir), has_geojson=has_geojson(fdir),
                med_trace_len=round(median(trace_lens), 1),
                straight=straight, smooth=smooth, hard=hard, loop_traces=loop_traces,
                n_bssids=len(bssids_all),
                med_aps_per_scan=round(median(all_aps), 1),
                med_wifi_interval_s=round(median(all_intervals), 2),
                med_imu_hz=round(median(imu_rates), 1),
            )))
    return floors


def diverse_top3(ranked):
    """Pick best straight-dominant, best loop-rich, best hard-turn-rich, all decent."""
    pool = [f for f in ranked if f["score"] >= 0.5] or ranked
    picks, used = [], set()

    def take(key):
        for f in sorted(pool, key=key, reverse=True):
            tag = (f["site"], f["floor"])
            if tag not in used:
                used.add(tag); picks.append(f); return

    take(lambda f: f["fr_straight"])              # straight-dominant
    take(lambda f: f["loop_traces"] / max(1, f["n_traces"]))  # loop-rich
    take(lambda f: f["fr_hard"])                  # hard-turn-rich
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", help="path to ILN 2.0 data/ folder")
    ap.add_argument("--csv", default="site_features.csv")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if not os.path.isdir(args.data_root):
        sys.exit("data_root not found: " + args.data_root)

    floors = analyze(args.data_root)
    if not floors:
        sys.exit("No floors with traces found. Check the folder structure.")
    floors.sort(key=lambda f: f["score"], reverse=True)

    cols = ["site", "floor", "score", "s_data", "s_size", "s_geom", "s_wifi",
            "n_traces", "area_m2", "med_trace_len", "has_geojson",
            "fr_straight", "fr_smooth", "fr_hard", "loop_traces",
            "med_aps_per_scan", "med_wifi_interval_s", "n_bssids", "med_imu_hz"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for f in floors:
            w.writerow({k: f.get(k) for k in cols})

    print(f"\nScored {len(floors)} floors. Full table -> {args.csv}\n")
    print(f"{'site':>26} {'fl':>4} {'score':>6} {'trc':>4} {'area':>6} "
          f"{'len':>5} {'str%':>5} {'sm%':>5} {'hd%':>5} {'loops':>5} "
          f"{'APs':>5} {'int_s':>6} {'gjson':>5}")
    for f in floors[:args.top]:
        print(f"{f['site']:>26} {f['floor']:>4} {f['score']:>6.3f} "
              f"{f['n_traces']:>4} {str(f['area_m2'] or '-'):>6} "
              f"{f['med_trace_len']:>5.0f} {100*f['fr_straight']:>5.0f} "
              f"{100*f['fr_smooth']:>5.0f} {100*f['fr_hard']:>5.0f} "
              f"{f['loop_traces']:>5} {f['med_aps_per_scan']:>5.0f} "
              f"{f['med_wifi_interval_s']:>6.1f} {str(f['has_geojson']):>5}")

    print("\nSUGGESTED DIVERSE TOP-3 (covers straight / loops / hard turns):")
    for f in diverse_top3(floors):
        print(f"  - {f['site']} / {f['floor']}  "
              f"(score {f['score']:.3f}, straight {100*f['fr_straight']:.0f}%, "
              f"hard {100*f['fr_hard']:.0f}%, loops {f['loop_traces']}/{f['n_traces']}, "
              f"geojson={f['has_geojson']})")
    print("\nReject any pick where has_geojson=False (cannot rebuild walls in Webots).")


if __name__ == "__main__":
    main()
