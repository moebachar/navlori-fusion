from __future__ import annotations
import os, sys, math, random, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, load_config
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class MDN(nn.Module):
    def __init__(self, n_aps, imu_dim, K=6, d=128, dropout=0.2):
        super().__init__()
        self.K = K; self.dropout = dropout
        self.wifi = nn.Sequential(nn.Linear(n_aps,256), nn.GELU(), nn.Linear(256,d))
        self.imu = nn.Sequential(nn.Conv1d(imu_dim,64,3,padding=1), nn.GELU(),
                                  nn.Conv1d(64,d,3,padding=1), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.trunk = nn.Linear(2*d, d)
        self.pi = nn.Linear(d, K); self.mu = nn.Linear(d, 2*K); self.L = nn.Linear(d, 3*K)
    def forward(self, w, i):
        h = self.trunk(torch.cat([self.wifi(w), self.imu(i.transpose(1,2))], -1))
        h = F.dropout(F.gelu(h), self.dropout, training=True)
        pi = F.softmax(self.pi(h), -1)
        mu = self.mu(h).reshape(-1, self.K, 2)
        L = self.L(h).reshape(-1, self.K, 3)
        d1 = F.softplus(L[..., 0]) + 1e-2; d2 = F.softplus(L[..., 1]) + 1e-2; o = L[..., 2]
        return pi, mu, d1, d2, o

def gmm_nll(y, pi, mu, d1, d2, o):
    dy = y[:, None] - mu
    z1 = dy[..., 0] / d1
    z2 = (dy[..., 1] - o*z1) / d2
    quad = z1**2 + z2**2
    logdet = torch.log(d1) + torch.log(d2)
    log_n = -0.5*quad - logdet - math.log(2*math.pi)
    log_w = torch.log(pi + 1e-9)
    return -torch.logsumexp(log_w + log_n, -1).mean()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='msiln_site1_b1')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args(); set_seed(args.seed)
    cfg = load_config(args.dataset); cfg.temporal.n_instants = 1
    dm = build_datamodule(cfg)
    n_aps = int(dm.train_ds.feature_dims['wifi']); imu_dim = int(dm.train_ds.feature_dims['imu'])
    Wt = dm.train_ds.get_tensors('wifi')[0].reshape(-1,n_aps).to(DEV)
    It = dm.train_ds.get_tensors('imu')[0].squeeze(1).to(DEV)
    yt = dm.train_ds._targets.to(DEV)
    Wv = dm.val_ds.get_tensors('wifi')[0].reshape(-1,n_aps).to(DEV)
    Iv = dm.val_ds.get_tensors('imu')[0].squeeze(1).to(DEV)
    yv = dm.val_ds._targets.to(DEV)
    Ws = dm.test_ds.get_tensors('wifi')[0].reshape(-1,n_aps).to(DEV)
    Is = dm.test_ds.get_tensors('imu')[0].squeeze(1).to(DEV)
    ys_ = dm.test_ds._targets.to(DEV)
    model = MDN(n_aps, imu_dim).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    H_REG = 0.05; bs = 128
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(Wt))
        for i in range(0, len(Wt), bs):
            idx = perm[i:i+bs]
            pi, mu, d1, d2, o = model(Wt[idx], It[idx])
            ent = -(pi * (pi + 1e-9).log()).sum(-1).mean()
            loss = gmm_nll(yt[idx], pi, mu, d1, d2, o) - H_REG * ent
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            with torch.no_grad():
                preds = []
                for _ in range(10):
                    pi, mu, *_ = model(Wv, Iv)
                    preds.append((pi[..., None] * mu).sum(-2))
                mp = torch.stack(preds).mean(0)
                vmae = (mp - yv).norm(dim=1).mean().item()
            print(f'ep{ep} val_mae={vmae:.3f}', flush=True)
    with torch.no_grad():
        preds = []
        for _ in range(20):
            pi, mu, *_ = model(Ws, Is)
            preds.append((pi[..., None] * mu).sum(-2))
        mp = torch.stack(preds).mean(0)
        tmae = (mp - ys_).norm(dim=1).mean().item()
        preds_v = []
        for _ in range(20):
            pi, mu, *_ = model(Wv, Iv)
            preds_v.append((pi[..., None] * mu).sum(-2))
        mpv = torch.stack(preds_v).mean(0)
        vmae = (mpv - yv).norm(dim=1).mean().item()
    print(f'RESULT: dataset={args.dataset} seed={args.seed} val={vmae:.3f} test={tmae:.3f}', flush=True)

if __name__=='__main__': main()