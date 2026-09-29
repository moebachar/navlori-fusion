"""Mechanism probe for idea #3 (dead-reckon motion between absolute WiFi fixes).

Neural idea #3 needs a pipeline/eval-anchor rework (the trainer predicts at the
anchor instant, tau=0, so a velocity*time term vanishes at eval, and WiFi
staleness is not exposed). So instead of half-building it, this tests its
UNDERLYING MECHANISM cheaply on real data, no retrain:

  A. WiFi-kNN only  : position(t) = kNN(nearest WiFi scan <= t)     [absolute, sparse]
  B. WiFi + odom DR : position(t) = A + (odom(t) - odom(last scan)) [integrate motion]

If B << A, integrating motion between WiFi fixes has real headroom -> build the
neural version. If B ~ A, the ceiling is WiFi and idea #3 will not help.

HONEST CAVEAT: odometry here is the synthesized (clean) drive odom, so B is an
OPTIMISTIC upper bound on what motion integration can buy.

Run: .venv/Scripts/python.exe scripts/_test_deadreckon.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "iln20_5d27099f_F2_replay"
BASELINE = 7.11
FILL = -105.0   # dBm for an AP not heard
K = 5


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def rssi_cols(rows0):
    return [c for c in rows0[0].keys() if c.startswith("wifi_rssi_")]


def wifi_matrix(rows, cols):
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
    t = np.array([float(r["sim_time"]) for r in g])
    x = np.array([float(r["gt_x"]) for r in g])
    y = np.array([float(r["gt_y"]) for r in g])
    return t, x, y


def odom_xy(path):
    o = read_csv(D / path / "odometry.csv")
    t = np.array([float(r["sim_time"]) for r in o])
    x = np.array([float(r["odom_x"]) for r in o])
    y = np.array([float(r["odom_y"]) for r in o])
    return t, x, y


def main():
    splits = json.loads((D / "replay_manifest.json").read_text())
    cfg = ROOT / "configs" / "data" / "iln20_5d27099f_F2_replay.yaml"
    import re
    txt = cfg.read_text()
    def ids(key):
        m = re.search(key + r"_paths:\s*\[([0-9,\s]*)\]", txt)
        return [int(x) for x in m.group(1).split(",") if x.strip()]
    train_ids, test_ids = ids("train"), ids("test")
    print(f"train paths {len(train_ids)}  test paths {len(test_ids)}", flush=True)

    # column set from first available path
    probe = read_csv(D / f"path_{train_ids[0]:02d}" / "wifi.csv")
    cols = rssi_cols(probe)
    print(f"WiFi APs: {len(cols)}", flush=True)

    # build train fingerprint DB: (rssi_vec, x, y) at each train WiFi scan
    DB_X, DB_P = [], []
    for pid in train_ids:
        p = f"path_{pid:02d}"
        if not (D / p / "wifi.csv").is_file():
            continue
        wt, WX = wifi_matrix(read_csv(D / p / "wifi.csv"), cols)
        gt, gx, gy = gt_xy(p)
        px = np.interp(wt, gt, gx); py = np.interp(wt, gt, gy)
        DB_X.append(WX); DB_P.append(np.stack([px, py], 1))
    DB_X = np.concatenate(DB_X); DB_P = np.concatenate(DB_P)
    print(f"fingerprint DB: {len(DB_X)} scans", flush=True)

    def knn(vec):
        d = np.sqrt(((DB_X - vec) ** 2).sum(1))
        idx = np.argpartition(d, K)[:K]
        return DB_P[idx].mean(0)

    err_A, err_B = [], []           # WiFi-only, WiFi+odom-DR
    per_path = {}
    for pid in test_ids:
        p = f"path_{pid:02d}"
        wt, WX = wifi_matrix(read_csv(D / p / "wifi.csv"), cols)
        gt, gx, gy = gt_xy(p)
        ot, ox, oy = odom_xy(p)
        anchors = np.stack([knn(WX[i]) for i in range(len(WX))])  # (nscan,2)
        # odom at each scan (interp)
        oax = np.interp(wt, ot, ox); oay = np.interp(wt, ot, oy)
        pa, pb = [], []
        for k in range(len(gt)):
            t = gt[k]
            j = np.searchsorted(wt, t, side="right") - 1   # last scan <= t
            if j < 0:
                j = 0
            a = anchors[j]
            pa.append(a)
            # dead-reckon: add odom displacement since that scan
            dx = np.interp(t, ot, ox) - oax[j]
            dy = np.interp(t, ot, oy) - oay[j]
            pb.append(a + np.array([dx, dy]))
        pa = np.array(pa); pb = np.array(pb)
        gtruth = np.stack([gx, gy], 1)
        ea = np.linalg.norm(pa - gtruth, axis=1)
        eb = np.linalg.norm(pb - gtruth, axis=1)
        err_A.append(ea); err_B.append(eb)
        per_path[pid] = (float(ea.mean()), float(eb.mean()))
    A = np.concatenate(err_A); B = np.concatenate(err_B)
    macroA = np.mean([v[0] for v in per_path.values()])
    macroB = np.mean([v[1] for v in per_path.values()])

    print("\n" + "=" * 58, flush=True)
    print(f"  A  WiFi-kNN only        MAE {A.mean():6.3f} m  median {np.median(A):6.3f}  macro {macroA:6.3f}", flush=True)
    print(f"  B  WiFi + odom DR       MAE {B.mean():6.3f} m  median {np.median(B):6.3f}  macro {macroB:6.3f}", flush=True)
    print(f"  neural baseline (query) MAE {BASELINE:6.3f} m", flush=True)
    print(f"  DR gain over WiFi-only  : {A.mean()-B.mean():+.3f} m", flush=True)
    print("  per-path (A -> B):", {k: (round(v[0],2), round(v[1],2)) for k,v in sorted(per_path.items())}, flush=True)

    out = {"knn_k": K, "wifi_only_mae": float(A.mean()), "wifi_only_median": float(np.median(A)),
           "wifi_only_macro": float(macroA), "dr_mae": float(B.mean()),
           "dr_median": float(np.median(B)), "dr_macro": float(macroB),
           "dr_gain_m": float(A.mean() - B.mean()), "neural_baseline": BASELINE,
           "per_path_A_B": {int(k): v for k, v in per_path.items()},
           "caveat": "odom is synthesized clean drive odom -> B is optimistic upper bound"}
    (ROOT / "runs" / "deadreckon_probe.json").parent.mkdir(exist_ok=True)
    (ROOT / "runs" / "deadreckon_probe.json").write_text(json.dumps(out, indent=2))
    print("\ndone -> runs/deadreckon_probe.json", flush=True)


if __name__ == "__main__":
    main()
