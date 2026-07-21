import os, sys, time, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import build_datamodule, load_config, build_trainer

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class PairwiseRSSIEncoder(nn.Module):
    def __init__(self, n_aps, K=16, d_ap=64, d_model=128, depth=2, heads=4):
        super().__init__()
        self.K = K
        self.ap_emb = nn.Embedding(n_aps + 1, d_ap, padding_idx=0)
        self.pair_mlp = nn.Sequential(nn.Linear(2*d_ap + 3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4*d_model, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=depth)
        self.cls = nn.Parameter(torch.randn(1,1,d_model)*0.02)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, rssi_dbm):
        B, M = rssi_dbm.shape; K = self.K
        rssi, idx = rssi_dbm.topk(K, dim=1)
        conf = torch.sigmoid(rssi + 70.0)
        valid_ap = (rssi > -99.0).float()
        ap_e = self.ap_emb(idx + 1)
        ii, jj = torch.triu_indices(K, K, offset=1, device=rssi.device)
        ei = ap_e[:, ii]; ej = ap_e[:, jj]
        diff = (rssi[:, ii] - rssi[:, jj]).unsqueeze(-1)
        ci = conf[:, ii].unsqueeze(-1); cj = conf[:, jj].unsqueeze(-1)
        tok = self.pair_mlp(torch.cat([ei, ej, diff, ci, cj], -1))
        mask = (valid_ap[:, ii] * valid_ap[:, jj]).bool()
        cls = self.cls.expand(B,-1,-1)
        z = torch.cat([cls, tok], 1)
        kpm = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=z.device), ~mask], 1)
        z = self.trunk(z, src_key_padding_mask=kpm)
        return self.norm(z[:,0])

class PairwiseFusion(nn.Module):
    readout = 'cls'
    def __init__(self, n_aps, imu_dim=5, d=128):
        super().__init__()
        self.modalities=['wifi','imu']
        self.wifi = PairwiseRSSIEncoder(n_aps)
        self.imu = nn.GRU(imu_dim, d, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 2))
    def forward(self, inputs, avail=None, dt=None, query_dt=None, **kw):
        w = inputs['wifi'].reshape(inputs['wifi'].shape[0], -1)
        rssi = w*100.0 - 100.0
        zw = self.wifi(rssi)
        i = inputs['imu']
        i = i.reshape(i.shape[0], i.shape[-2], i.shape[-1])
        _, zi = self.imu(i); zi = zi.squeeze(0)
        return self.head(torch.cat([zw, zi], -1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=12); args=ap.parse_args()
    set_seed(args.seed)
    cfg=load_config('msiln_site1_b1'); cfg.temporal.n_instants=1
    cfg.train.modality_dropout=0.0; cfg.train.instant_dropout=0.0
    cfg.data.batch_size=32
    cfg.train.modality_balanced_loss=False
    dm = build_datamodule(cfg)
    n_aps = int(dm.train_ds.feature_dims['wifi'])
    imu_dim = int(dm.train_ds.feature_dims['imu'])
    model = PairwiseFusion(n_aps, imu_dim=imu_dim)
    run_dir = REPO / 'runs' / 'experiments' / f'phys3_s{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg, model, dm, extra_inputs={}, run_dir=str(run_dir))
    t0=time.time(); trainer.fit(epochs=args.epochs)
    pv, tv = trainer.predict('val');  mae_v = (pv-tv).norm(dim=1).mean().item()
    pt, tt = trainer.predict('test'); mae_t = (pt-tt).norm(dim=1).mean().item()
    print(f'RESULT: dataset=msiln_site1_b1 seed={args.seed} val={mae_v:.2f} test={mae_t:.2f} ({(time.time()-t0)/60:.1f} min)', flush=True)

if __name__=='__main__': main()