"""PROBE 3 — WiFi transferability + split geometry.

The question that decides whether localization is even possible: when you
take a VAL WiFi scan and find its nearest neighbour in TRAIN RSSI space,
is that train neighbour spatially near the val position? If yes, WiFi
fingerprints transfer and a kNN/encoder can localize. If the spatial
distance is ~random, fingerprints do not transfer and NOTHING downstream
can localize from WiFi.

Compares two regimes:
  * intra-train: train scan -> nearest OTHER train scan (does WiFi even
    encode position within the training distribution?)
  * train->val:  val scan -> nearest train scan (does it transfer?)

Also reports split geometry: spatial overlap of train vs val (bounding box
+ how many val points fall inside the train convex region, via a coarse
grid-occupancy proxy).

RSSI distance uses only co-visible APs (both scans see the AP), which is
how real fingerprinting compares scans; missing entries are not -100-filled
here (we want the honest signal, not the pipeline's -100 artifact).

Pure data. Run:
  .venv/Scripts/python.exe scripts/inspect_03_transfer.py [dataset ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import load_config  # noqa: E402

DATASETS = ["simulation", "ipin2024_floor-2", "ronin_a000"]


def load_scans(root: Path, pids):
    """Return (rssi NxA with NaN for missing, xy Nx2) for all scans in paths."""
    cols = None
    rows_rssi, rows_xy = [], []
    for p in pids:
        pdir = root / f"path_{p:02d}"
        wf = pdir / "wifi.csv"
        gf = pdir / "ground_truth.csv"
        if not (wf.exists() and gf.exists()):
            continue
        w = pd.read_csv(wf)
        g = pd.read_csv(gf)
        if len(w) == 0 or len(g) < 2:
            continue
        c = [c for c in w.columns if c.startswith("wifi_rssi_")]
        if cols is None:
            cols = c
        gt_t = g["sim_time"].values
        gt_xy = g[["gt_x", "gt_y"]].values
        for _, sr in w.iterrows():
            t = sr["sim_time"]
            j = int(np.searchsorted(gt_t, t, side="right") - 1)
            if j < 0:
                continue
            rows_rssi.append(sr[cols].values.astype(np.float64))
            rows_xy.append(gt_xy[j])
    if not rows_rssi:
        return None, None
    return np.array(rows_rssi), np.array(rows_xy)


def covis_dist(a, B):
    """Euclidean RSSI distance between scan a (A,) and each row of B (N,A),
    over APs visible in BOTH (non-NaN). Returns (N,) with inf where no overlap."""
    out = np.full(B.shape[0], np.inf)
    a_seen = np.isfinite(a)
    for i in range(B.shape[0]):
        m = a_seen & np.isfinite(B[i])
        if m.sum() == 0:
            continue
        d = a[m] - B[i][m]
        # normalize by sqrt(#covis) so scans with more overlap aren't penalized
        out[i] = np.sqrt((d * d).sum() / m.sum())
    return out


def nn_spatial_error(query_rssi, query_xy, ref_rssi, ref_xy, exclude_self=False):
    """For each query scan, find nearest ref scan in co-visible RSSI space;
    return spatial distance between query position and that ref's position."""
    errs = []
    for i in range(len(query_rssi)):
        d = covis_dist(query_rssi[i], ref_rssi)
        if exclude_self:
            d[i] = np.inf
        if not np.isfinite(d).any():
            continue
        j = int(np.argmin(d))
        errs.append(np.linalg.norm(query_xy[i] - ref_xy[j]))
    return np.array(errs)


def grid_overlap(train_xy, val_xy, cell=3.0):
    """Fraction of val points whose grid cell is occupied by train points."""
    def cells(xy):
        return set(map(tuple, np.floor(xy / cell).astype(int)))
    tc = cells(train_xy)
    vcells = np.floor(val_xy / cell).astype(int)
    inside = sum((tuple(c) in tc) for c in vcells)
    return inside / len(val_xy)


def analyze(name):
    print(f"\n{'='*80}\n{name}\n{'='*80}")
    cfg = load_config(name)
    d = cfg.dataset
    root = ROOT / str(d.root) / d.collection_dir
    tr_r, tr_xy = load_scans(root, list(d.split.train_paths))
    va_r, va_xy = load_scans(root, list(d.split.val_paths))
    if tr_r is None or va_r is None:
        print("  insufficient scans")
        return
    print(f"  train scans={len(tr_r)}  val scans={len(va_r)}  "
          f"APs={tr_r.shape[1]}")

    # Split geometry
    ov = grid_overlap(tr_xy, va_xy, cell=3.0)
    print(f"  split geometry: {ov*100:.0f}% of val positions fall in a 3m "
          f"cell also visited by train")
    print(f"    train bbox x[{tr_xy[:,0].min():.0f},{tr_xy[:,0].max():.0f}] "
          f"y[{tr_xy[:,1].min():.0f},{tr_xy[:,1].max():.0f}]")
    print(f"    val   bbox x[{va_xy[:,0].min():.0f},{va_xy[:,0].max():.0f}] "
          f"y[{va_xy[:,1].min():.0f},{va_xy[:,1].max():.0f}]")

    # Random-guess baseline: expected spatial dist between two random scans
    rand = []
    rng = np.random.RandomState(0)
    for _ in range(2000):
        i, j = rng.randint(len(va_xy)), rng.randint(len(tr_xy))
        rand.append(np.linalg.norm(va_xy[i] - tr_xy[j]))
    rand = np.array(rand)

    intra = nn_spatial_error(tr_r, tr_xy, tr_r, tr_xy, exclude_self=True)
    trval = nn_spatial_error(va_r, va_xy, tr_r, tr_xy, exclude_self=False)

    print(f"  RSSI-nearest-neighbour spatial error (lower = WiFi encodes position):")
    print(f"    intra-train (train->train): median={np.median(intra):.2f}m  "
          f"mean={intra.mean():.2f}m")
    print(f"    train->val  (val->train):   median={np.median(trval):.2f}m  "
          f"mean={trval.mean():.2f}m")
    print(f"    random guess:               median={np.median(rand):.2f}m  "
          f"mean={rand.mean():.2f}m")
    skill = 1 - trval.mean() / rand.mean()
    print(f"  >>> transfer skill = 1 - (trval/random) = {skill*100:.0f}%  "
          f"(100%=perfect, 0%=no better than random)")


def main():
    for ds in (sys.argv[1:] or DATASETS):
        try:
            analyze(ds)
        except Exception as e:
            print(f"  ERROR {ds}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
