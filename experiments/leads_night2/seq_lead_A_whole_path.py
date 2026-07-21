from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, load_config
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def group_by_path(ds):
    buckets = {}
    for i, r in enumerate(ds._gt_rows):
        buckets.setdefault(r['path_id'], []).append((r['time'], i))
    out = []
    for pid, lst in buckets.items():
        lst.sort()
        out.append((pid, [i for _, i in lst]))
    return out

class PathTransformer(nn.Module):
    def __init__(self, n_aps, imu_T, imu_F, d=192, h=6, L=4, drop=0.15):
        super().__init__()
        self.wifi_enc = nn.Sequential(nn.Linear(n_aps, 256), nn.GELU(), nn.Dropout(drop),
                                       nn.Linear(256, d//2), nn.LayerNorm(d//2))
        self.imu_enc = nn.Sequential(nn.Conv1d(imu_F, 32, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(32, 64, 5, padding=2), nn.GELU(),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                      nn.Linear(64, d - d//2), nn.LayerNorm(d - d//2))
        layer = nn.TransformerEncoderLayer(d, h, 4*d, drop, 'gelu', batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, L)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
    def forward(self, wifi, imu, dt):
        B, T, _ = wifi.shape
        w = self.wifi_enc(wifi)
        # imu shape: (B, T, win, F) -> Conv1d wants (N, C=F, L=win)
        imu_flat = imu.reshape(B*T, imu.size(2), imu.size(3)).transpose(1, 2)
        i = self.imu_enc(imu_flat)
        i = i.view(B, T, -1)
        z = torch.cat([w, i], -1)
        d = z.size(-1)
        freq = torch.exp(torch.arange(0, d, 2, device=z.device) * -(np.log(1e4)/d))
        pe = torch.zeros_like(z)
        pe[..., 0::2] = torch.sin(dt.unsqueeze(-1) * freq)
        pe[..., 1::2] = torch.cos(dt.unsqueeze(-1) * freq)
        z = z + pe
        return self.head(self.trunk(z))

def build_path_batches(ds, groups, max_len=256):
    wifi_all, _ = ds.get_tensors('wifi')
    imu_all, _ = ds.get_tensors('imu')
    wifi_all = wifi_all.view(len(ds), -1)
    imu_all = imu_all.view(len(ds), imu_all.shape[-2], imu_all.shape[-1])
    times = ds._timestamps.float(); tgts = ds._targets
    paths = []
    for pid, idxs in groups:
        idxs = torch.as_tensor(idxs); t0 = times[idxs[0]]
        for start in range(0, len(idxs), max_len):
            sl = idxs[start:start+max_len]
            paths.append(dict(wifi=wifi_all[sl], imu=imu_all[sl], dt=(times[sl]-t0).float(), y=tgts[sl]))
    return paths

def run_split(model, batches, train=False, opt=None):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    for b in batches:
        w = b['wifi'][None].to(DEV); i = b['imu'][None].to(DEV)
        dt = b['dt'][None].to(DEV); y = b['y'].to(DEV)
        if train:
            opt.zero_grad(); p = model(w, i, dt)[0]
            loss = nn.functional.huber_loss(p, y, delta=2.0)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        else:
            with torch.no_grad(): p = model(w, i, dt)[0]
        err = (p - y).norm(dim=-1); total += err.sum().item(); n += err.numel()
    return total / max(n, 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=30)
    a = ap.parse_args(); set_seed(a.seed)
    cfg = load_config('msiln_site1_b1'); cfg.temporal.n_instants = 1
    dm = build_datamodule(cfg)
    train_b = build_path_batches(dm.train_ds, group_by_path(dm.train_ds))
    val_b = build_path_batches(dm.val_ds, group_by_path(dm.val_ds))
    test_b = build_path_batches(dm.test_ds, group_by_path(dm.test_ds))
    n_aps = int(dm.train_ds.feature_dims['wifi'])
    imu_T, imu_F = train_b[0]['imu'].shape[1], train_b[0]['imu'].shape[2]
    model = PathTransformer(n_aps, imu_T, imu_F).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_v, best_t = 1e9, 1e9
    for ep in range(a.epochs):
        random.shuffle(train_b)
        tr = run_split(model, train_b, True, opt)
        v = run_split(model, val_b); t = run_split(model, test_b)
        print(f'[{ep:03d}] train={tr:.3f} val={v:.3f} test={t:.3f}', flush=True)
        if v < best_v: best_v, best_t = v, t
    print(f'\nRESULT: dataset=msiln_site1_b1 seed={a.seed} val={best_v:.3f} test={best_t:.3f}', flush=True)

if __name__ == '__main__': main()