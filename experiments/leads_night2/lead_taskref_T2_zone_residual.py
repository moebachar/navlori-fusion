import os, sys, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, load_config

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class ZoneHead(nn.Module):
    def __init__(self, n_aps, imu_dim, T, K, d=192):
        super().__init__()
        self.K = K
        self.wifi = nn.Sequential(nn.Linear(n_aps, 256), nn.GELU(), nn.Linear(256, d), nn.LayerNorm(d))
        self.imu = nn.Sequential(nn.Conv1d(imu_dim, 64, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(64, d, 3, padding=1), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.trunk = nn.Sequential(nn.Linear(2*d, 256), nn.GELU(), nn.Dropout(0.1))
        self.zone_head = nn.Linear(256, K)
        self.zone_emb = nn.Embedding(K, 64)
        self.res_head = nn.Sequential(nn.Linear(256 + 64, 128), nn.GELU(), nn.Linear(128, 2))
    def forward(self, wifi, imu, true_zone=None):
        h = self.trunk(torch.cat([self.wifi(wifi), self.imu(imu.transpose(1, 2))], -1))
        zlogits = self.zone_head(h)
        z = true_zone if true_zone is not None else zlogits.argmax(-1)
        ze = self.zone_emb(z)
        res = self.res_head(torch.cat([h, ze], -1))
        return zlogits, res

def main():
    set_seed(42)
    cfg = load_config('msiln_site1_b1'); cfg.temporal.n_instants = 1
    dm = build_datamodule(cfg)
    Wt, yt = dm.train_ds.get_tensors('wifi'); It, _ = dm.train_ds.get_tensors('imu')
    Wv, yv = dm.val_ds.get_tensors('wifi'); Iv, _ = dm.val_ds.get_tensors('imu')
    Ws, ys_ = dm.test_ds.get_tensors('wifi'); Is, _ = dm.test_ds.get_tensors('imu')
    flat = lambda w, i: (w.reshape(len(w), -1), i.reshape(len(i), i.shape[-2], i.shape[-1]))
    Wt, It = flat(Wt, It); Wv, Iv = flat(Wv, Iv); Ws, Is = flat(Ws, Is)
    K = 32
    EPOCHS = 15
    km = KMeans(K, n_init=10, random_state=0).fit(yt.numpy())
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    zt = torch.tensor(km.predict(yt.numpy()), dtype=torch.long)
    zv = torch.tensor(km.predict(yv.numpy()), dtype=torch.long)
    res_t = yt - centroids[zt]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ZoneHead(Wt.shape[-1], It.shape[-1], It.shape[-2], K).to(dev)
    cent = centroids.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * (len(Wt) // 64 + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    huber = nn.HuberLoss(delta=0.5)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Wt))
        for i in range(0, len(Wt), 64):
            idx = perm[i:i+64]
            tz = zt[idx].to(dev) if torch.rand(1).item() > 0.3 else None
            zl, res = model(Wt[idx].to(dev), It[idx].to(dev), tz)
            l = F.cross_entropy(zl, zt[idx].to(dev)) + 1.0 * huber(res, res_t[idx].to(dev))
            opt.zero_grad(); l.backward(); opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            zl, res = model(Wv.to(dev), Iv.to(dev))
            pz = zl.argmax(-1); pred = cent[pz] + res
            vmae = (pred - yv.to(dev)).norm(dim=1).mean().item()
            acc = (pz == zv.to(dev)).float().mean().item()
        print(f'ep{ep:02d} val_mae={vmae:.3f} zone_acc={acc:.3f}', flush=True)
    with torch.no_grad():
        zl, res = model(Ws.to(dev), Is.to(dev))
        pred = cent[zl.argmax(-1)] + res
        tmae = (pred - ys_.to(dev)).norm(dim=1).mean().item()
    print(f'RESULT: dataset=msiln_site1_b1 seed=42 val={vmae:.3f} test={tmae:.3f}', flush=True)

if __name__ == '__main__': main()