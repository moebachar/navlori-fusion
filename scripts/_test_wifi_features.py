"""Idea #1 probe: session-invariant WiFi features vs raw RSSI, kNN, no retrain.

Between sessions the whole RSSI vector drifts (device/crowd/AP power). Raw values
move; relative structure (ranks, deltas) stays. Test which transform gives the
best WiFi anchor. Same eval as the dead-reckon probe: per-GT-timestep, hold the
last WiFi anchor. 'raw' should reproduce ~8.75 m (sanity), others compare.

Variants (all kNN k=5, euclidean on transformed vectors):
  raw          : RSSI, missing=-105                       (baseline)
  center       : RSSI - mean(heard)   (cancels session offset)
  rel_strong   : RSSI - max(heard)    (delta to strongest AP)
  rank         : normalized rank among heard              (gain-invariant)
  center+vocab : center, keep only APs heard in >=5% of train scans

Run: .venv/Scripts/python.exe scripts/_test_wifi_features.py
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "iln20_5d27099f_F2_replay"
NEURAL = 7.11
RAW_REF = 8.75
FILL = -105.0
ABSENT = -40.0      # sentinel for absent AP AFTER a centering transform
K = 5


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def wifi_raw(rows, cols):
    t = np.array([float(r["sim_time"]) for r in rows])
    X = np.full((len(rows), len(cols)), FILL)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r.get(c, "")
            if v not in ("", "0", "nan", "NaN", None):
                try:
                    X[i, j] = float(v)
                except ValueError:
                    pass
    return t, X


def gt_xy(path):
    g = read_csv(D / path / "ground_truth.csv")
    return (np.array([float(r["sim_time"]) for r in g]),
            np.array([float(r["gt_x"]) for r in g]),
            np.array([float(r["gt_y"]) for r in g]))


def transform(X, mode, keep=None):
    """X: (N, d) raw with missing=FILL. Returns transformed features."""
    heard = X > FILL + 1
    if keep is not None:
        X = X[:, keep]; heard = heard[:, keep]
    if mode == "raw":
        return X.copy()
    out = np.full_like(X, ABSENT)
    for i in range(X.shape[0]):
        h = heard[i]
        if h.sum() == 0:
            continue
        vals = X[i, h]
        if mode == "center":
            out[i, h] = vals - vals.mean()
        elif mode == "rel_strong":
            out[i, h] = vals - vals.max()
        elif mode == "rank":
            # strongest -> 0, weakest -> ~1 among heard; absent -> 1.5
            order = np.argsort(np.argsort(-vals))
            r = order / max(1, h.sum() - 1)
            tmp = np.full(h.sum(), 1.5)
            tmp[:] = r
            out[i, h] = tmp
            out[i, ~h] = 1.5
    return out


def eval_variant(mode, cols, train_ids, test_ids, keep=None):
    DB_X, DB_P = [], []
    for pid in train_ids:
        p = f"path_{pid:02d}"
        if not (D / p / "wifi.csv").is_file():
            continue
        wt, WX = wifi_raw(read_csv(D / p / "wifi.csv"), cols)
        gt, gx, gy = gt_xy(p)
        DB_X.append(WX)
        DB_P.append(np.stack([np.interp(wt, gt, gx), np.interp(wt, gt, gy)], 1))
    DB_X = np.concatenate(DB_X); DB_P = np.concatenate(DB_P)
    DB_T = transform(DB_X, mode, keep)

    errs, per_path = [], {}
    for pid in test_ids:
        p = f"path_{pid:02d}"
        wt, WX = wifi_raw(read_csv(D / p / "wifi.csv"), cols)
        gt, gx, gy = gt_xy(p)
        QT = transform(WX, mode, keep)
        d = np.sqrt(((DB_T[None, :, :] - QT[:, None, :]) ** 2).sum(2))  # (nscan, Ntr)
        idx = np.argpartition(d, K, axis=1)[:, :K]
        anchors = DB_P[idx].mean(1)                                     # (nscan, 2)
        e = []
        for k in range(len(gt)):
            j = max(0, np.searchsorted(wt, gt[k], side="right") - 1)
            e.append(np.hypot(anchors[j, 0] - gx[k], anchors[j, 1] - gy[k]))
        e = np.array(e); errs.append(e); per_path[pid] = float(e.mean())
    A = np.concatenate(errs)
    return float(A.mean()), float(np.median(A)), float(np.mean(list(per_path.values()))), per_path


def main():
    txt = (ROOT / "configs" / "data" / "iln20_5d27099f_F2_replay.yaml").read_text()
    def ids(key):
        m = re.search(key + r"_paths:\s*\[([0-9,\s]*)\]", txt)
        return [int(x) for x in m.group(1).split(",") if x.strip()]
    train_ids, test_ids = ids("train"), ids("test")
    cols = [c for c in read_csv(D / f"path_{train_ids[0]:02d}" / "wifi.csv")[0]
            if c.startswith("wifi_rssi_")]
    print(f"train {len(train_ids)}  test {len(test_ids)}  APs {len(cols)}", flush=True)

    # train coverage for vocab
    cov = np.zeros(len(cols))
    ntot = 0
    for pid in train_ids:
        p = f"path_{pid:02d}"
        if not (D / p / "wifi.csv").is_file():
            continue
        _, WX = wifi_raw(read_csv(D / p / "wifi.csv"), cols)
        cov += (WX > FILL + 1).sum(0); ntot += WX.shape[0]
    cov /= max(1, ntot)
    keep = np.where(cov >= 0.05)[0]
    print(f"APs with >=5% train coverage: {len(keep)}/{len(cols)}", flush=True)

    results = {}
    for mode, kp in [("raw", None), ("center", None), ("rel_strong", None),
                     ("rank", None), ("center", keep)]:
        name = mode + ("+vocab" if kp is not None else "")
        mae, med, macro, pp = eval_variant(mode, cols, train_ids, test_ids, kp)
        results[name] = {"mae": mae, "median": med, "macro": macro, "per_path": pp}
        print(f"  {name:14s} MAE {mae:6.3f} m  median {med:6.3f}  macro {macro:6.3f}", flush=True)

    print("\n  reference: raw kNN ~8.75 m   neural fusion 7.11 m", flush=True)
    best = min(results, key=lambda k: results[k]["mae"])
    print(f"  best: {best} = {results[best]['mae']:.3f} m "
          f"(gain over raw {results['raw']['mae']-results[best]['mae']:+.3f} m)", flush=True)
    (ROOT / "runs" / "wifi_features_probe.json").write_text(json.dumps(results, indent=2))
    print("\ndone -> runs/wifi_features_probe.json", flush=True)


if __name__ == "__main__":
    main()
