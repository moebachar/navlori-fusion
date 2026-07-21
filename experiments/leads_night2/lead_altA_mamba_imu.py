from __future__ import annotations
import os, sys, time, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, build_trainer, load_config

class MambaBlockPure(nn.Module):
    def __init__(self, d, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d
        self.in_proj = nn.Linear(d, 2 * d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d)
        self.d_state, self.d_inner = d_state, d_inner
    def forward(self, x):
        xz = self.in_proj(x); xx, z = xz.chunk(2, dim=-1)
        xx = self.conv1d(xx.transpose(1,2))[..., :x.size(1)].transpose(1,2)
        xx = torch.nn.functional.silu(xx)
        dbc = self.x_proj(xx); dt, B, C = dbc.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(x.size(1)):
            dA = torch.exp(dt[:, t, :, None] * A[None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = dA * h + dB * xx[:, t, :, None]
            y = (h * C[:, t, None, :]).sum(-1) + self.D * xx[:, t]
            ys.append(y)
        y = torch.stack(ys, dim=1) * torch.nn.functional.silu(z)
        return self.out_proj(y)

class MambaPlaceIMU(nn.Module):
    readout = 'cls'
    def __init__(self, n_aps, imu_dim=5, T=32, place_dim=48, d=96, depth=2):
        super().__init__()
        self.modalities = ['wifi', 'imu']
        self.place = nn.Sequential(nn.Linear(n_aps,128), nn.GELU(), nn.Linear(128,place_dim), nn.LayerNorm(place_dim))
        self.imu_proj = nn.Linear(imu_dim, d - place_dim)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), MambaBlockPure(d)) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d), nn.GELU(), nn.Linear(d,2))
    def forward(self, inputs, avail=None, dt=None, query_dt=None, **kw):
        w = inputs['wifi'].reshape(inputs['wifi'].shape[0], -1)
        i = inputs['imu'].reshape(inputs['imu'].shape[0], inputs['imu'].shape[-2], -1)
        place = self.place(w).unsqueeze(1).expand(-1, i.size(1), -1)
        z = torch.cat([self.imu_proj(i), place], dim=-1)
        for blk in self.blocks: z = z + blk(z)
        return self.head(z.mean(dim=1))

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='msiln_site1_b1')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=15)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = load_config(a.dataset); cfg.temporal.n_instants = 1
    cfg.train.modality_dropout = 0.0; cfg.train.instant_dropout = 0.0
    cfg.train.modality_balanced_loss = False
    cfg.data.batch_size = 32  # MambaSSM Python-loop scan is memory-heavy; small batch fits 8GB GPU
    dm = build_datamodule(cfg)
    n_aps = int(dm.train_ds.feature_dims['wifi']); imu_dim = int(dm.train_ds.feature_dims['imu'])
    model = MambaPlaceIMU(n_aps=n_aps, imu_dim=imu_dim)
    run_dir = REPO/'runs'/'experiments'/f'altA_mamba_{a.dataset}_s{a.seed}'; run_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg, model, dm, extra_inputs={}, run_dir=str(run_dir))
    t0 = time.time(); trainer.fit(epochs=a.epochs); el = time.time()-t0
    pv, tv = trainer.predict('val'); pt, tt = trainer.predict('test')
    print(f'RESULT: dataset={a.dataset} seed={a.seed} val={(pv-tv).norm(dim=1).mean():.3f} test={(pt-tt).norm(dim=1).mean():.3f} ({el/60:.1f}min)', flush=True)

if __name__ == '__main__': main()