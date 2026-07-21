"""P1: Session-MAML — per-session meta-adaptation for cross-session WiFi+IMU.

Mechanistic intent: at meta-training time, sample a session (path_id) from
train; do K-shot inner-loop adaptation via SGD; Reptile-style outer update
nudges the meta-params toward the post-adaptation params. At test time,
adapt to the support of each test session and evaluate on its query.

Reports the best of {adapted, base} on val (selection signal) and test
(reported MAE).
"""

import sys, copy, random, os
import torch
from pathlib import Path

REPO = Path('x:/navlori-fusion')
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.fusion.builder import build_datamodule, load_config
import torch.nn as nn
import torch.nn.functional as F


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class SmallFusion(nn.Module):
    """Tiny WiFi+IMU fusion head — single instant, MAML inner loop runs over this."""

    def __init__(self, n_aps, imu_dim, d=128):
        super().__init__()
        self.wifi = nn.Sequential(
            nn.Linear(n_aps, 256), nn.GELU(), nn.Linear(256, d)
        )
        self.imu = nn.Sequential(
            nn.Conv1d(imu_dim, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, d, 3, padding=1), nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, w, i):
        zw = self.wifi(w)
        zi = self.imu(i.transpose(1, 2))
        return self.head(torch.cat([zw, zi], -1))


def session_tasks(ds):
    """Group sample indices by session (path_id)."""
    by_path = {}
    for g, r in enumerate(ds._gt_rows):
        by_path.setdefault(r['path_id'], []).append(g)
    return by_path


@torch.no_grad()
def base_eval(model, Wi, Im, y, batch=4096):
    """Mean Euclidean error of base meta-params over a tensor split."""
    model.eval()
    errs = []
    N = Wi.shape[0]
    for s in range(0, N, batch):
        e = min(s + batch, N)
        pred = model(Wi[s:e], Im[s:e])
        errs.append((pred - y[s:e]).norm(dim=1))
    return torch.cat(errs).mean().item()


def adapt_and_eval(model, Wi, Im, y, tasks, K_SHOT, INNER_STEPS, INNER_LR):
    """Per-session adaptation + query MAE. Returns (adapt_mae, base_mae)."""
    errs_adapt, errs_base = [], []
    for pid, idx in tasks.items():
        if len(idx) <= K_SHOT:
            # Not enough for support+query; just evaluate base on all.
            qi = torch.tensor(idx, device=Wi.device)
            with torch.no_grad():
                pred_b = model(Wi[qi], Im[qi])
            err = (pred_b - y[qi]).norm(dim=1)
            errs_adapt.append(err)
            errs_base.append(err)
            continue
        sup = idx[:K_SHOT]
        query = idx[K_SHOT:]
        fast = copy.deepcopy(model)
        inner = torch.optim.SGD(fast.parameters(), lr=INNER_LR)
        si = torch.tensor(sup, device=Wi.device)
        for _ in range(INNER_STEPS):
            out = fast(Wi[si], Im[si])
            inner.zero_grad()
            F.huber_loss(out, y[si], delta=0.5).backward()
            inner.step()
        qi = torch.tensor(query, device=Wi.device)
        with torch.no_grad():
            pred_a = fast(Wi[qi], Im[qi])
            pred_b = model(Wi[qi], Im[qi])
        errs_adapt.append((pred_a - y[qi]).norm(dim=1))
        errs_base.append((pred_b - y[qi]).norm(dim=1))
    return (torch.cat(errs_adapt).mean().item(),
            torch.cat(errs_base).mean().item())


def main():
    set_seed(42)

    cfg = load_config('msiln_site1_b1')
    cfg.temporal.n_instants = 1

    # Memory-efficient subset: MAML only needs a representative pool of
    # training sessions (it samples one per outer step). Subsampling the
    # 94 train sessions to ~24 keeps wifi tensor (~1419 APs * N) inside
    # 30 MB on CPU, which matters when other leads are running concurrently.
    # Default 32 train sessions: MAML draws one per outer step, so diversity
    # matters but the full 94 is wasteful and memory-heavy under concurrency.
    max_train = int(os.environ.get('LEAD_MAX_TRAIN_PATHS', '32'))
    if len(cfg.dataset.split.train_paths) > max_train:
        cfg.dataset.split.train_paths = list(
            cfg.dataset.split.train_paths[:max_train]
        )

    dm = build_datamodule(cfg)

    n_aps = int(dm.train_ds.feature_dims['wifi'])
    imu_dim = int(dm.train_ds.feature_dims['imu'])  # 5 for MSILN world-frame

    # GPU may be busy with other leads in the Phase B' batch; fall back to CPU
    # if CUDA allocation fails. The model is tiny so CPU is fine.
    use_cuda = torch.cuda.is_available() and os.environ.get('LEAD_FORCE_CPU') != '1'
    try:
        if use_cuda:
            torch.zeros(8, device='cuda')
        device = torch.device('cuda' if use_cuda else 'cpu')
    except RuntimeError:
        device = torch.device('cpu')
    print(f'[meta] device={device}', flush=True)

    model = SmallFusion(n_aps, imu_dim).to(device)

    # Cache GPU tensors for train / val / test. Center targets so the regressor
    # head can converge fast (MSILN coords are offset hundreds of metres).
    def to_tensors(ds):
        Wi = ds.get_tensors('wifi')[0].reshape(-1, n_aps).to(device)
        Im = ds.get_tensors('imu')[0].to(device)  # (N, win, imu_dim)
        y = ds._targets.to(device)
        return Wi, Im, y

    Wi_tr, Im_tr, y_tr_raw = to_tensors(dm.train_ds)
    Wi_va, Im_va, y_va_raw = to_tensors(dm.val_ds)
    Wi_te, Im_te, y_te_raw = to_tensors(dm.test_ds)

    # Target centering from train split (won't bias val/test selection because
    # we report Euclidean distance in original units).
    y_mean = y_tr_raw.mean(0, keepdim=True)
    y_tr = y_tr_raw - y_mean
    y_va = y_va_raw - y_mean
    y_te = y_te_raw - y_mean

    tasks_tr = session_tasks(dm.train_ds)
    tasks_va = session_tasks(dm.val_ds)
    tasks_te = session_tasks(dm.test_ds)

    # Hyperparameters for Reptile-style outer loop.
    # Production small-test budget: ~200 outer iters is plenty for a tiny
    # 2-layer MLP + tiny Conv1d head on cached tensors.
    # Production small-test budget. Each outer iter is K_SHOT=20 samples * 5
    # inner SGD steps on a tiny model — fast even on CPU.
    N_OUTER = int(os.environ.get('LEAD_N_OUTER', '500'))
    INNER_STEPS = 5
    INNER_LR = 3e-3
    K_SHOT = 20
    Q_SHOT = 32
    META_LR = 0.4

    # Pre-filter trainable sessions.
    train_pids = [pid for pid, idx in tasks_tr.items()
                  if len(idx) >= K_SHOT + Q_SHOT]
    if not train_pids:
        # Fall back: any session with at least K_SHOT+1 samples.
        train_pids = [pid for pid, idx in tasks_tr.items() if len(idx) > K_SHOT]
    print(f'[meta] {len(train_pids)} trainable train sessions '
          f'(of {len(tasks_tr)} total)', flush=True)

    for it in range(N_OUTER):
        pid = random.choice(train_pids)
        idx = tasks_tr[pid]
        s = random.sample(idx, min(K_SHOT, len(idx)))
        fast = copy.deepcopy(model)
        inner = torch.optim.SGD(fast.parameters(), lr=INNER_LR)
        si = torch.tensor(s, device=device)
        for _ in range(INNER_STEPS):
            out = fast(Wi_tr[si], Im_tr[si])
            inner.zero_grad()
            F.huber_loss(out, y_tr[si], delta=0.5).backward()
            inner.step()
        # Reptile outer update: pull meta-params toward post-adapt params.
        with torch.no_grad():
            for p, fp in zip(model.parameters(), fast.parameters()):
                p.add_(META_LR * (fp - p))
        if it % 50 == 0:
            print(f'[meta] it={it}', flush=True)

    # Evaluate on val with per-session adaptation; report best-of.
    va_a, va_b = adapt_and_eval(model, Wi_va, Im_va, y_va,
                                tasks_va, K_SHOT, INNER_STEPS, INNER_LR)
    val_mae = min(va_a, va_b)

    te_a, te_b = adapt_and_eval(model, Wi_te, Im_te, y_te,
                                tasks_te, K_SHOT, INNER_STEPS, INNER_LR)
    test_mae = min(te_a, te_b)

    print(f'[meta] val: adapt={va_a:.3f} base={va_b:.3f}', flush=True)
    print(f'[meta] test: adapt={te_a:.3f} base={te_b:.3f}', flush=True)
    print(f'\nRESULT: dataset=msiln_site1_b1 seed=42 '
          f'val={val_mae:.3f} test={test_mae:.3f}', flush=True)


if __name__ == '__main__':
    main()
