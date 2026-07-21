"""H7: kNN noise-floor on MSILN site1/B1 cross-session split.

For each test WiFi scan, find the nearest TRAINING WiFi scan by Euclidean
distance (missing APs filled with -100 dBm = absent). Use that training
scan's GT (x,y) as the prediction. Report MAE per k in {1, 3, 5, 9}.

This is a STRICT lower bound for any model that maps trace-level WiFi to
GT position WITHOUT a floorplan prior or additional inputs. If this floor
is ~10 m, it means the cross-session WiFi-fingerprint signal itself is
insufficient on this split — the gap to published 2-6 m is structural
(more inputs, easier split, or floorplan-aware model), not algorithmic.

Also reports the val kNN floor to sanity-check yesterday's 17.66 m wifi_knn.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path('x:/navlori-fusion/data/msiln_site1_b1')
OUT = Path('x:/navlori-fusion/experiments/diagnostics/h7_knn_floor.json')

# Splits from configs/data/msiln_site1_b1.yaml
TRAIN = list(range(0, 94))
VAL   = list(range(94, 128))
TEST  = [128, 129, 130, 131, 132]

FILL = -100.0   # absent AP convention
KS   = [1, 3, 5, 9]


def load_path(pid: int):
    """Return (wifi_rssi[N, n_aps], gt_xy[N, 2]) aligned by nearest GT in time."""
    pdir = DATA / f'path_{pid:02d}' if pid < 100 else DATA / f'path_{pid}'
    wifi = pd.read_csv(pdir / 'wifi.csv')
    gt   = pd.read_csv(pdir / 'ground_truth.csv')
    # AP columns
    ap_cols = [c for c in wifi.columns if c.startswith('wifi_rssi_')]
    rssi = wifi[ap_cols].to_numpy(dtype=np.float32)
    rssi = np.where(np.isnan(rssi), FILL, rssi)
    t_w = wifi['sim_time'].to_numpy(dtype=np.float64)
    # Nearest-GT lookup
    t_g = gt['sim_time'].to_numpy(dtype=np.float64)
    xy  = gt[['gt_x', 'gt_y']].to_numpy(dtype=np.float32)
    idx = np.clip(np.searchsorted(t_g, t_w), 1, len(t_g) - 1)
    left  = idx - 1
    right = idx
    pick  = np.where(np.abs(t_g[left] - t_w) <= np.abs(t_g[right] - t_w), left, right)
    return rssi, xy[pick], ap_cols


def build_matrix(paths, ref_cols=None):
    """Concatenate all paths' (rssi, xy) into one matrix; align AP columns to ref_cols."""
    all_rssi, all_xy = [], []
    cols_master = ref_cols
    for pid in paths:
        rssi, xy, cols = load_path(pid)
        if cols_master is None:
            cols_master = cols
            all_rssi.append(rssi)
        else:
            # Align: produce full-AP-space matrix with -100 fill for missing cols
            mapping = {c: i for i, c in enumerate(cols)}
            out = np.full((rssi.shape[0], len(cols_master)), FILL, dtype=np.float32)
            for j, c in enumerate(cols_master):
                src = mapping.get(c)
                if src is not None:
                    out[:, j] = rssi[:, src]
            all_rssi.append(out)
        all_xy.append(xy)
    return np.concatenate(all_rssi, axis=0), np.concatenate(all_xy, axis=0), cols_master


def knn_predict(X_train, y_train, X_test, k: int, chunk: int = 256):
    """Brute kNN in RSSI space. Returns predicted xy [N_test, 2] and per-sample
    min-distance to nearest train scan (RSSI L2) for diagnostics."""
    preds = np.zeros((X_test.shape[0], 2), dtype=np.float64)
    nn_dist = np.zeros(X_test.shape[0], dtype=np.float64)
    Xt = X_train.astype(np.float32)
    for start in range(0, X_test.shape[0], chunk):
        Q = X_test[start:start + chunk].astype(np.float32)   # (q, D)
        # squared L2
        d2 = (Q ** 2).sum(1, keepdims=True) + (Xt ** 2).sum(1) - 2.0 * Q @ Xt.T
        d2 = np.maximum(d2, 0.0)
        if k == 1:
            idx = np.argmin(d2, axis=1)
            preds[start:start + chunk] = y_train[idx]
            nn_dist[start:start + chunk] = np.sqrt(d2[np.arange(len(idx)), idx])
        else:
            # k smallest
            part = np.argpartition(d2, kth=k, axis=1)[:, :k]
            # gather and average GT (uniform-weight kNN; matches yesterday's wifi_knn)
            preds[start:start + chunk] = y_train[part].mean(axis=1)
            nn_dist[start:start + chunk] = np.sqrt(d2[np.arange(d2.shape[0]), part[:, 0]])
    return preds, nn_dist


def mae(preds, gt):
    return float(np.linalg.norm(preds - gt, axis=1).mean())


def report_split(name, X_train, y_train, X_query, y_query):
    out = {}
    for k in KS:
        pred, dnn = knn_predict(X_train, y_train, X_query, k)
        err = np.linalg.norm(pred - y_query, axis=1)
        out[f'k={k}'] = {
            'MAE_m': float(err.mean()),
            'median_m': float(np.median(err)),
            'p90_m': float(np.percentile(err, 90)),
            'rmse_m': float(np.sqrt((err ** 2).mean())),
        }
        if k == 1:
            out['k=1_nn_rssi_dist_mean'] = float(dnn.mean())
            out['k=1_nn_rssi_dist_median'] = float(np.median(dnn))
    out['n_query'] = int(len(y_query))
    out['n_train_scans'] = int(len(y_train))
    print(f'\n== {name} ==')
    for k, v in out.items():
        print(f'  {k}: {v}')
    return out


def main():
    print('Loading train…')
    X_tr, y_tr, cols = build_matrix(TRAIN)
    print(f'  train: {X_tr.shape}, APs={len(cols)}')

    print('Loading val…')
    X_va, y_va, _ = build_matrix(VAL, ref_cols=cols)
    print(f'  val:   {X_va.shape}')

    print('Loading test…')
    X_te, y_te, _ = build_matrix(TEST, ref_cols=cols)
    print(f'  test:  {X_te.shape}')

    results = {
        'note': (
            'kNN noise-floor on raw WiFi-RSSI (no PCA, -100 fill). '
            'Strict lower bound for any WiFi-only model without floorplan prior.'
        ),
        'val_floor':  report_split('val (knn train -> val)',  X_tr, y_tr, X_va, y_va),
        'test_floor': report_split('test (knn train -> test)', X_tr, y_tr, X_te, y_te),
    }

    # Oracle (cheat): test scan's own GT — must be 0 by construction.
    # (Sanity check the metric.)
    oracle = mae(y_te, y_te)
    results['oracle_self_lookup_MAE_m'] = oracle
    print(f'\nOracle self-lookup (sanity, must be 0): {oracle:.6f} m')

    OUT.write_text(json.dumps(results, indent=2))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
