from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, load_config

def rssi_clean(R):
    R = np.where(np.isnan(R), -100., R).clip(-100, 0); return (R + 100.) / 100.

def knn_predict(R_te, R_tr, y_tr, k=8):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(R_tr)
    d, idx = nn.kneighbors(R_te); w = 1.0 / (d + 1e-3)
    p = (w[..., None] * y_tr[idx]).sum(1) / w.sum(1, keepdims=True)
    var = ((y_tr[idx] - p[:, None]) ** 2).sum(-1).mean(1)
    return p, np.sqrt(var + 1.0)

def pdr_displacements(imu_win):
    ax, ay = imu_win[..., 0], imu_win[..., 1]
    vx = np.cumsum(ax, axis=-1) * 0.02; vy = np.cumsum(ay, axis=-1) * 0.02
    dx = vx.sum(-1) * 0.02; dy = vy.sum(-1) * 0.02
    return np.stack([dx, dy], axis=-1)

def cost(x_flat, p, sig, dxy, lam_anc, lam_rel, lam_smo):
    x = x_flat.reshape(-1, 2)
    c_anc = ((x - p) ** 2).sum(1) / (sig ** 2)
    diff = x[1:] - x[:-1]
    c_rel = ((diff - dxy[:-1]) ** 2).sum(1)
    smo = x[2:] - 2 * x[1:-1] + x[:-2]
    c_smo = (smo ** 2).sum(1)
    return lam_anc * c_anc.mean() + lam_rel * c_rel.mean() + lam_smo * c_smo.mean()

def solve_path(p, sig, dxy, **lams):
    x0 = p.copy()
    res = minimize(cost, x0.flatten(), args=(p, sig, dxy, lams['a'], lams['r'], lams['s']),
                   method='L-BFGS-B', options=dict(maxiter=300, ftol=1e-7))
    return res.x.reshape(-1, 2)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='msiln_site1_b1')
    ap.add_argument('--seed', type=int, default=42); a = ap.parse_args()
    cfg = load_config(a.dataset); cfg.temporal.n_instants = 1; cfg.data.batch_size = 256
    dm = build_datamodule(cfg)
    def split_arrays(ds):
        W = ds._cache['wifi'].numpy().reshape(len(ds), -1)
        I = ds._cache['imu'].numpy()
        Y = ds._targets.numpy(); pid = np.array([r['path_id'] for r in ds._gt_rows])
        return rssi_clean(W), I, Y, pid
    Rtr, _, Ytr, _ = split_arrays(dm.train_ds)
    for split_name, ds in [('val', dm.val_ds), ('test', dm.test_ds)]:
        Rs, Is, Ys, pids = split_arrays(ds)
        p_all, s_all = knn_predict(Rs, Rtr, Ytr, k=8)
        dxy_all = pdr_displacements(Is)
        preds = np.zeros_like(Ys)
        for pid in np.unique(pids):
            m = pids == pid; preds[m] = solve_path(p_all[m], s_all[m], dxy_all[m], a=1.0, r=0.5, s=0.1)
        mae = np.linalg.norm(preds - Ys, axis=1).mean()
        if split_name == 'val': val_mae = mae
        else: test_mae = mae
    print(f'\nRESULT: dataset={a.dataset} seed={a.seed} val={val_mae:.2f} test={test_mae:.2f}', flush=True)

if __name__ == '__main__': main()