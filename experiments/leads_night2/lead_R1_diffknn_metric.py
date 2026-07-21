from __future__ import annotations
import os, sys, time, random, argparse
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, load_config
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class RSSIEmbed(nn.Module):
    def __init__(self, n_aps, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_aps, 256), nn.GELU(), nn.LayerNorm(256),
                                  nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
                                  nn.Linear(128, d))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class IMUDelta(nn.Module):
    def __init__(self, imu_dim, T):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(imu_dim, 32, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(32, 64, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(64, 2)
    def forward(self, x):
        h = self.conv(x.transpose(1,2)).squeeze(-1)
        return self.head(h)

def knn_predict(q, mem_emb, mem_xy, k=8, tau=0.07, chunk=512):
    outs = []
    for i in range(0, q.shape[0], chunk):
        qc = q[i:i+chunk]
        sim = qc @ mem_emb.T
        top_v, top_i = sim.topk(k, dim=1)
        w = F.softmax(top_v / tau, dim=1)
        nbrs = mem_xy[top_i]
        outs.append((w.unsqueeze(-1) * nbrs).sum(dim=1))
    return torch.cat(outs, dim=0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=15)
    args = ap.parse_args(); seed_all(args.seed)
    cfg = load_config('msiln_site1_b1'); cfg.temporal.n_instants = 1
    dm = build_datamodule(cfg)
    Xw_tr, Y_tr = dm.train_ds.get_tensors('wifi'); Xi_tr, _ = dm.train_ds.get_tensors('imu')
    Xw_va, Y_va = dm.val_ds.get_tensors('wifi');   Xi_va, _ = dm.val_ds.get_tensors('imu')
    Xw_te, Y_te = dm.test_ds.get_tensors('wifi');  Xi_te, _ = dm.test_ds.get_tensors('imu')
    def flat(x): return x.reshape(x.shape[0], -1)
    Xw_tr, Xw_va, Xw_te = [flat(x).to(DEV) for x in (Xw_tr, Xw_va, Xw_te)]
    Xi_tr, Xi_va, Xi_te = [x.reshape(x.shape[0], x.shape[-2], x.shape[-1]).to(DEV) for x in (Xi_tr, Xi_va, Xi_te)]
    Y_tr, Y_va, Y_te = [y.to(DEV) for y in (Y_tr, Y_va, Y_te)]
    n_aps = Xw_tr.shape[1]; T = Xi_tr.shape[1]; Fdim = Xi_tr.shape[2]
    emb = RSSIEmbed(n_aps).to(DEV); idel = IMUDelta(Fdim, T).to(DEV)
    opt = torch.optim.AdamW(list(emb.parameters()) + list(idel.parameters()), lr=1e-3, weight_decay=1e-4)
    bs = 128; best_val = float('inf'); best_state = None
    for ep in range(args.epochs):
        emb.train(); idel.train()
        perm = torch.randperm(len(Xw_tr), device=DEV)
        with torch.no_grad():
            mem_emb = emb(Xw_tr)
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            q = emb(Xw_tr[idx])
            sim = q @ mem_emb.T
            sim[torch.arange(len(idx), device=DEV), idx] = -1e9
            top_v, top_i = sim.topk(8, dim=1)
            w = F.softmax(top_v / 0.07, dim=1)
            pred_xy = (w.unsqueeze(-1) * Y_tr[top_i]).sum(dim=1)
            pred_xy = pred_xy + idel(Xi_tr[idx])
            loss = F.smooth_l1_loss(pred_xy, Y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        emb.eval(); idel.eval()
        with torch.no_grad():
            mem_emb = emb(Xw_tr)
            pv = knn_predict(emb(Xw_va), mem_emb, Y_tr) + idel(Xi_va)
            mae = (pv - Y_va).norm(dim=1).mean().item()
        if mae < best_val: best_val = mae; best_state = (emb.state_dict(), idel.state_dict())
        print(f'[R1] ep={ep} val_mae={mae:.3f}', flush=True)
    emb.load_state_dict(best_state[0]); idel.load_state_dict(best_state[1])
    with torch.no_grad():
        mem_emb = emb(Xw_tr)
        pt = knn_predict(emb(Xw_te), mem_emb, Y_tr) + idel(Xi_te)
        test_mae = (pt - Y_te).norm(dim=1).mean().item()
    print(f'RESULT: dataset=msiln_site1_b1 seed={args.seed} val={best_val:.3f} test={test_mae:.3f}', flush=True)

if __name__ == '__main__': main()