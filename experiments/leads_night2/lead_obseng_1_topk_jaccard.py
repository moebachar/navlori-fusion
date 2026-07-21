import os, sys, time, argparse, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, load_config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def topk_set(W, K=20, obs_thresh=0.005):
    Wm = W.clone(); Wm[Wm.abs() < obs_thresh] = -1e9
    idx = Wm.topk(K, dim=1).indices
    out = torch.zeros_like(W, dtype=torch.bool)
    out.scatter_(1, idx, True)
    return out

def jaccard_knn(Sq, St, yt, k=8, tau=4.0, chunk=256):
    Stf = St.float()
    St_sum = Stf.sum(1)  # [Nt]
    Nq = Sq.shape[0]
    out = torch.empty((Nq, yt.shape[1]), device=yt.device, dtype=yt.dtype)
    for s in range(0, Nq, chunk):
        e = min(s + chunk, Nq)
        Sqf = Sq[s:e].float()
        inter = Sqf @ Stf.t()
        union = Sqf.sum(1, keepdim=True) + St_sum.unsqueeze(0) - inter
        jacc = inter / union.clamp(min=1.0)
        sims, idx = jacc.topk(k, dim=1)
        w = F.softmax(sims * tau, dim=1)
        out[s:e] = (w.unsqueeze(-1) * yt[idx]).sum(dim=1)
    return out

class IMUDelta(nn.Module):
    def __init__(self, d=9):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(d,64,3,padding=1), nn.GELU(),
                                  nn.Conv1d(64,64,3,padding=1), nn.GELU())
        self.head = nn.Linear(64, 2)
    def forward(self, x): return self.head(self.cnn(x.transpose(1,2)).mean(-1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='msiln_site1_b1')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--K', type=int, default=20); ap.add_argument('--knn', type=int, default=8)
    a = ap.parse_args(); seed_all(a.seed)
    cfg = load_config(a.dataset); cfg.temporal.n_instants = 1
    dm = build_datamodule(cfg)
    t0 = time.time()
    A = int(dm.train_ds.feature_dims['wifi'])
    # Keep WiFi on CPU; only move boolean topk sets to GPU (bool is 1B per element)
    Wtr_cpu = dm.train_ds.get_tensors('wifi')[0].reshape(-1, A)
    Wva_cpu = dm.val_ds.get_tensors('wifi')[0].reshape(-1, A)
    Wte_cpu = dm.test_ds.get_tensors('wifi')[0].reshape(-1, A)
    Str = topk_set(Wtr_cpu, a.K).to(DEVICE)
    Sva = topk_set(Wva_cpu, a.K).to(DEVICE)
    Ste = topk_set(Wte_cpu, a.K).to(DEVICE)
    del Wtr_cpu, Wva_cpu, Wte_cpu
    # IMU on CPU until needed
    Itr = dm.train_ds.get_tensors('imu')[0].squeeze(1)
    Iva = dm.val_ds.get_tensors('imu')[0].squeeze(1)
    Ite = dm.test_ds.get_tensors('imu')[0].squeeze(1)
    ytr = dm.train_ds._targets.to(DEVICE); yva = dm.val_ds._targets.to(DEVICE); yte = dm.test_ds._targets.to(DEVICE)
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    p_va = jaccard_knn(Sva, Str, ytr, k=a.knn, chunk=128)
    print(f'[knn-only] val_mae={(p_va - yva).norm(dim=1).mean().item():.3f}', flush=True)
    base_tr = jaccard_knn(Str, Str, ytr, k=a.knn+1, chunk=128)
    delta_tr = ytr - base_tr
    # Free big Jaccard intermediates before training
    del base_tr
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    Itr_g = Itr.to(DEVICE)
    net = IMUDelta(d=Itr_g.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 128; N = len(Itr_g)
    for ep in range(a.epochs):
        perm = torch.randperm(N, device=DEVICE)
        for s in range(0, N, bs):
            idx = perm[s:s+bs]
            pred = net(Itr_g[idx]); tgt = delta_tr[idx]
            loss = F.smooth_l1_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    del Itr_g
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    base_va = jaccard_knn(Sva, Str, ytr, k=a.knn, chunk=128)
    base_te = jaccard_knn(Ste, Str, ytr, k=a.knn, chunk=128)
    with torch.no_grad():
        p_va = base_va + net(Iva.to(DEVICE))
        p_te = base_te + net(Ite.to(DEVICE))
    val = (p_va - yva).norm(dim=1).mean().item()
    test = (p_te - yte).norm(dim=1).mean().item()
    print(f'\nRESULT: dataset={a.dataset} seed={a.seed} val={val:.3f} test={test:.3f} ({(time.time()-t0)/60:.1f} min)', flush=True)

if __name__ == '__main__': main()