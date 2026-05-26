"""Inline per-encoder training helpers for the publication-grade
walkthrough notebook (PLAN_32).

Each helper trains a single encoder + linear head on the canonical
benchmark + returns the trained components. Used by `notebooks/
run2_walkthrough.ipynb` when ``FAST_MODE=False`` to demonstrate
clone-and-reproduce — the same training recipe as the offline
`scripts/eval_*.py` runners (RESULT_01/04/07/08).

API for each ``train_*`` helper::

    encoder, head, history = train_anchor2vec(Xtr, Ytr, Xva, Yva, ...)

``encoder`` is the trained ``nn.Module`` from ``src.pipeline.encoders``;
``head`` is a ``nn.Linear(embed_dim, 2)``; ``history`` is a dict with
``train_loss``, ``val_mae`` lists.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.encoders import Anchor2Vec


def _ensure_tensor(arr, dtype=torch.float32):
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    return torch.tensor(np.asarray(arr), dtype=dtype)


def anchor2vec_predict(enc: Anchor2Vec, head: nn.Linear,
                        X: np.ndarray | torch.Tensor,
                        batch: int = 1024,
                        device: str | None = None) -> np.ndarray:
    """Predict (x, y) for a batch of UJI scans through Anchor2Vec + head."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    enc.eval(); head.eval()
    X_t = _ensure_tensor(X).to(device)
    if X_t.ndim == 2:
        X_t = X_t.unsqueeze(1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            chunk = X_t[i:i + batch]
            preds.append(head(enc(chunk)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def anchor2vec_val_mae(enc: Anchor2Vec, head: nn.Linear,
                        Xva, Yva, mu) -> float:
    """Compute mean Euclidean error in original (un-centered) target frame."""
    pred = anchor2vec_predict(enc, head, Xva)
    Yva_arr = np.asarray(Yva)
    mu_arr = np.asarray(mu)
    return float(np.linalg.norm((pred + mu_arr) - (Yva_arr + mu_arr), axis=1).mean())


def train_anchor2vec(
    Xtr, Ytr, Xva, Yva,
    n_anchors: int = 64,
    embed_dim: int = 128,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 1.0,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[Anchor2Vec, nn.Linear, dict]:
    """Inline Anchor2Vec training for the UJI per-leg WiFi audit.

    Replicates the RESULT_01 recipe. Returns the best-val checkpoint
    (encoder + head + history). ~3 minutes on Quadro P4000 at the
    canonical 120 epochs + 256 batch.

    Inputs
    ------
    Xtr, Xva : (N, n_aps) RSSI arrays, already preprocessed (NaN/100
        sentinel handled, affine to [0, 1]).
    Ytr, Yva : (N, 2) target arrays (longitude, latitude). Will be
        centered by the train mean for training; final val MAE is
        computed in the centered frame (Euclidean distance is
        centering-invariant).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr_arr = np.asarray(Xtr).astype(np.float32)
    Xva_arr = np.asarray(Xva).astype(np.float32)
    Ytr_arr = np.asarray(Ytr).astype(np.float32)
    Yva_arr = np.asarray(Yva).astype(np.float32)
    mu = Ytr_arr.mean(0)
    Ytr_c = Ytr_arr - mu
    Yva_c = Yva_arr - mu

    Xtr_t = torch.tensor(Xtr_arr, device=device).unsqueeze(1)  # (N, 1, n_aps)
    Ytr_t = torch.tensor(Ytr_c, device=device)
    Xva_t = torch.tensor(Xva_arr, device=device).unsqueeze(1)
    Yva_t = torch.tensor(Yva_c, device=device)

    enc = Anchor2Vec(n_aps=Xtr_arr.shape[1], embed_dim=embed_dim,
                      n_anchors=n_anchors).to(device)
    head = nn.Linear(embed_dim, 2).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                             lr=lr, weight_decay=weight_decay)
    steps = max(1, len(Xtr_t) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=huber_delta)

    history = {"train_loss": [], "val_mae": []}
    best_mae = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); head.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        ep_loss = 0.0
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            loss = crit(head(enc(Xtr_t[idx])), Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ep_loss += loss.item()
        history["train_loss"].append(ep_loss / max(steps, 1))

        enc.eval(); head.eval()
        with torch.no_grad():
            pv = head(enc(Xva_t))
            mae = float(torch.linalg.norm(pv - Yva_t, dim=1).mean())
        history["val_mae"].append(mae)
        if mae < best_mae:
            best_mae = mae
            best_state = (
                {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            )
        if verbose and (ep == 0 or ep % 30 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}/{epochs}  train={history['train_loss'][-1]:.4f}  "
                  f"val_mae={mae:.3f}  (best {best_mae:.3f})", flush=True)

    elapsed = time.time() - t0
    if best_state is not None:
        enc.load_state_dict(best_state[0]); enc.to(device)
        head.load_state_dict(best_state[1]); head.to(device)
    history["best_val_mae"] = best_mae
    history["elapsed_s"] = elapsed
    history["target_mu"] = mu.tolist()
    if verbose:
        print(f"  done in {elapsed:.0f}s; best val mean Euclidean = {best_mae:.3f} m",
              flush=True)
    return enc, head, history


def _umeyama_align(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Sim(3) Umeyama alignment src -> dst with optimal scale."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, d = src.shape
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = (sc ** 2).sum() / n
    H = sc.T @ dc / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0
    R = Vt.T @ D @ U.T
    scale = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - scale * R @ mu_s
    aligned = ((scale * (R @ src.T)).T + t).astype(np.float32)
    return aligned, float(scale)


# ---------------------------------------------------------------------------
# IMUCNN — canonical RoNIN unseen-subjects (replicates RESULT_07 recipe)
# ---------------------------------------------------------------------------

def train_imucnn(
    train_dir,
    test_dir,
    train_seqs=None,
    test_seqs=None,
    *,
    window: int = 200,
    step: int = 10,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 0.5,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
):
    """Inline IMUCNN training for canonical RoNIN unseen-subjects.

    Replicates RESULT_07 / ``scripts/_eval_imucnn_ronin_canonical.py``.
    Returns ``(model_dict, history)`` where ``model_dict`` contains the
    trained encoder, head, per-sequence ATE rows, and aggregate summary
    dict. ~14 minutes on Quadro P4000 at canonical 20 epochs / 128 batch.

    Inputs
    ------
    train_dir, test_dir : ``Path`` to the FRDR ``train/`` and ``unseen/``
        directories.
    train_seqs, test_seqs : optional sequence-name lists. If ``None``,
        loads the canonical RoNIN lists from
        ``external_methods/ronin/lists/`` and filters to seqs present on
        disk.
    """
    from pathlib import Path

    from src.pipeline.baselines import (  # noqa: E402
        GlobSpeedSequence,
        StridedSequenceDataset,
        compute_ate_rte,
        RONIN_LISTS,
    )
    from src.pipeline.encoders import IMUCNN  # noqa: E402

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    if train_seqs is None:
        canonical_train = (RONIN_LISTS / "list_train.txt").read_text().split()
        train_seqs = [s for s in canonical_train if (train_dir / s).is_dir()]
    if test_seqs is None:
        canonical_test = (RONIN_LISTS / "list_test_unseen.txt").read_text().split()
        test_seqs = [s for s in canonical_test if (test_dir / s).is_dir()]

    if verbose:
        print(f"  train seqs: {len(train_seqs)}  test seqs: {len(test_seqs)}", flush=True)

    train_ds = StridedSequenceDataset(
        GlobSpeedSequence, str(train_dir), list(train_seqs), None,
        step, window, random_shift=step // 2, shuffle=False,
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=False,
    )

    enc = IMUCNN(in_features=6, embed_dim=128).to(device)
    head = nn.Linear(128, 2).to(device)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())
    if verbose:
        print(f"  IMUCNN+head params: {n_params/1e6:.3f} M", flush=True)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                             lr=lr, weight_decay=weight_decay)
    steps = max(1, len(train_ds) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=huber_delta)

    history = {"train_loss": []}
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); head.train()
        tot, n = 0.0, 0
        for feat, targ, _, _ in loader:
            x = feat.transpose(1, 2).contiguous().to(device)
            y = targ.to(device)
            pred = head(enc(x))
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        avg = tot / max(n, 1)
        history["train_loss"].append(avg)
        if verbose and (ep <= 1 or ep % 2 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}/{epochs}  vel_huber={avg:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    train_s = time.time() - t0

    enc.eval(); head.eval()
    pred_per_min = 200 * 60
    rows = []
    with torch.no_grad():
        for sname in test_seqs:
            seq = GlobSpeedSequence(
                str(test_dir / sname),
                interval=window, max_ori_error=20.0, grv_only=True,
            )
            feat = seq.features
            ts = seq.ts
            gt = seq.gt_pos[:, :2]
            if len(feat) < window + 5:
                continue
            ends = np.arange(window, len(feat), step)
            vel_chunks = []
            BS = 512
            for i in range(0, len(ends), BS):
                ebatch = ends[i:i + BS]
                wins = np.stack([feat[e - window:e] for e in ebatch]).astype(np.float32)
                xw = torch.tensor(wins, device=device)
                vel_chunks.append(head(enc(xw)).cpu().numpy())
                del xw
            vel = np.concatenate(vel_chunks, axis=0)
            traj = np.zeros((len(ends), 2), np.float32)
            cur = gt[window].copy()
            traj[0] = cur
            for i in range(1, len(ends)):
                dt = ts[ends[i]] - ts[ends[i - 1]]
                cur = cur + vel[i - 1] * dt
                traj[i] = cur
            gtm = gt[ends]
            ate_ronin, rte_ronin = compute_ate_rte(traj, gtm, pred_per_min)
            raw_simple = float(np.sqrt(((traj - gtm) ** 2).sum(1).mean()))
            traj_u, _ = _umeyama_align(traj, gtm)
            umey = float(np.sqrt(((traj_u - gtm) ** 2).sum(1).mean()))
            rows.append({
                "seq": sname,
                "ate_ronin": float(ate_ronin),
                "raw_simple": float(raw_simple),
                "umeyama": float(umey),
                "rte_ronin": float(rte_ronin),
                "n_windows": int(len(ends)),
            })
    if not rows:
        raise RuntimeError("No test sequences yielded an evaluation.")

    summary = {}
    for k in ("ate_ronin", "raw_simple", "umeyama", "rte_ronin"):
        vals = np.array([r[k] for r in rows])
        summary[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "max": float(vals.max()),
            "n": int(len(vals)),
        }
    history["elapsed_s"] = train_s
    history["best_raw_simple_mean"] = summary["raw_simple"]["mean"]
    history["best_umeyama_mean"] = summary["umeyama"]["mean"]
    if verbose:
        print(f"  done in {train_s:.0f}s; per-seq mean raw_simple={summary['raw_simple']['mean']:.3f} m  "
              f"umeyama={summary['umeyama']['mean']:.3f} m", flush=True)

    return {
        "encoder": enc, "head": head,
        "per_seq": rows, "summary": summary,
        "train_seqs": list(train_seqs), "test_seqs": list(test_seqs),
        "n_params": int(n_params),
    }, history


# ---------------------------------------------------------------------------
# OdomCNN — Webots Tiago P-B (Δ-features, winner per RESULT_04)
# ---------------------------------------------------------------------------

def _build_webots_odom_windows(paths, window: int = 16, mode: str = "P-B",
                                 stride: int = 1, data_root=None):
    """Per-path windowed inputs + position targets (matches
    ``scripts/_eval_webots_odom.py::build_windows``)."""
    from pathlib import Path
    import pandas as pd

    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / "data" / "async_collection"
    else:
        data_root = Path(data_root)
    ODOM_COLS = ["odom_x", "odom_y", "odom_theta_deg",
                 "odom_linear_vel", "odom_angular_vel",
                 "wheel_left_vel", "wheel_right_vel"]
    Xs, Ys, pids, times = [], [], [], []
    for pid in paths:
        pdir = data_root / f"path_{pid:02d}"
        odo_path = pdir / "odometry.csv"
        gt_path = pdir / "ground_truth.csv"
        if not (odo_path.is_file() and gt_path.is_file()):
            continue
        odo = pd.read_csv(odo_path)
        gt = pd.read_csv(gt_path)
        if len(odo) < 20 or len(gt) < 5:
            continue
        feat = odo[ODOM_COLS].values.astype(np.float32)
        t = odo["sim_time"].values.astype(np.float32)
        if mode == "P-B":
            df = np.diff(feat[:, :3], axis=0)
            df = np.vstack([np.zeros((1, 3), dtype=np.float32), df])
            feat = np.concatenate([df, feat[:, 3:]], axis=1)
        N = len(feat)
        if N < window + stride:
            continue
        ends = np.arange(window, N, stride)
        for e in ends:
            x = feat[e - window:e]
            t_end = t[e - 1]
            gx = float(np.interp(t_end, gt["sim_time"].values, gt["gt_x"].values))
            gy = float(np.interp(t_end, gt["sim_time"].values, gt["gt_y"].values))
            Xs.append(x)
            Ys.append([gx, gy])
            pids.append(pid)
            times.append(t_end)
    return {
        "X": np.stack(Xs).astype(np.float32) if Xs else np.zeros((0, window, 7), np.float32),
        "Y": np.array(Ys, dtype=np.float32) if Ys else np.zeros((0, 2), np.float32),
        "pid": np.array(pids, dtype=np.int64),
        "t": np.array(times, dtype=np.float32),
    }


def load_webots_odom_pb(train_paths=(1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                         val_paths=(2, 13, 14), test_paths=(15, 16, 17),
                         window: int = 16, data_root=None):
    """Build the OdomCNN P-B (Δ-features) windowed splits for the
    canonical Webots Tiago split (RESULT_04 winner config)."""
    tr = _build_webots_odom_windows(list(train_paths), window=window,
                                     mode="P-B", data_root=data_root)
    va = _build_webots_odom_windows(list(val_paths), window=window,
                                     mode="P-B", data_root=data_root)
    te = _build_webots_odom_windows(list(test_paths), window=window,
                                     mode="P-B", data_root=data_root)
    mu_x = tr["X"].reshape(-1, tr["X"].shape[2]).mean(0)
    sd_x = tr["X"].reshape(-1, tr["X"].shape[2]).std(0) + 1e-6
    mu_y = tr["Y"].mean(0)
    return {
        "Xtr": (tr["X"] - mu_x) / sd_x, "Ytr": tr["Y"] - mu_y, "pid_tr": tr["pid"],
        "Xva": (va["X"] - mu_x) / sd_x, "Yva": va["Y"] - mu_y, "pid_va": va["pid"],
        "Xte": (te["X"] - mu_x) / sd_x, "Yte": te["Y"] - mu_y, "pid_te": te["pid"],
        "Yte_raw": te["Y"], "Yva_raw": va["Y"],
        "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y,
    }


def compute_trivial_integration_floor(dataset: str = "webots",
                                       test_paths=(15, 16, 17),
                                       data_root=None):
    """Trivial-integration floor for OdomCNN: feed odom_x / odom_y
    directly with origin-shift to first GT. RESULT_04 Step 1 baseline.
    Returns dict with ``test_mae`` and per-traj median smoothness ``r``."""
    from pathlib import Path
    import pandas as pd

    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / "data" / "async_collection"
    else:
        data_root = Path(data_root)
    all_pred, all_gt, all_pid = [], [], []
    for pid in test_paths:
        pdir = data_root / f"path_{pid:02d}"
        odo_path = pdir / "odometry.csv"
        gt_path = pdir / "ground_truth.csv"
        if not (odo_path.is_file() and gt_path.is_file()):
            continue
        odo = pd.read_csv(odo_path)
        gt = pd.read_csv(gt_path)
        t0 = float(odo["sim_time"].iloc[0])
        gt_x0 = float(np.interp(t0, gt["sim_time"].values, gt["gt_x"].values))
        gt_y0 = float(np.interp(t0, gt["sim_time"].values, gt["gt_y"].values))
        odom_x0 = float(odo["odom_x"].iloc[0])
        odom_y0 = float(odo["odom_y"].iloc[0])
        pred_x = odo["odom_x"].values - odom_x0 + gt_x0
        pred_y = odo["odom_y"].values - odom_y0 + gt_y0
        pred = np.stack([pred_x, pred_y], axis=1).astype(np.float32)
        t = odo["sim_time"].values.astype(np.float32)
        gt_at_t = np.stack([
            np.interp(t, gt["sim_time"].values, gt["gt_x"].values),
            np.interp(t, gt["sim_time"].values, gt["gt_y"].values),
        ], axis=1).astype(np.float32)
        all_pred.append(pred)
        all_gt.append(gt_at_t)
        all_pid.append(np.full(len(pred), pid, dtype=np.int64))
    if not all_pred:
        return {"test_mae": float("nan"), "smoothness": float("nan")}
    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    pid = np.concatenate(all_pid)
    errs = np.linalg.norm(pred - gt, axis=1)
    rs = []
    for p in np.unique(pid):
        mask = pid == p
        dp = np.linalg.norm(np.diff(pred[mask], axis=0), axis=1)
        dg = np.linalg.norm(np.diff(gt[mask], axis=0), axis=1)
        if dp.std() > 1e-9 and dg.std() > 1e-9:
            rs.append(float(np.corrcoef(dp, dg)[0, 1]))
    return {
        "test_mae": float(errs.mean()),
        "smoothness": float(np.median(rs)) if rs else 0.0,
        "n_samples": int(len(errs)),
    }


def train_odomcnn(data: dict, *,
                   embed_dim: int = 128,
                   epochs: int = 30,
                   batch_size: int = 64,
                   lr: float = 1e-3,
                   weight_decay: float = 1e-4,
                   huber_delta: float = 0.5,
                   seed: int = 42,
                   device: str | None = None,
                   verbose: bool = True):
    """Inline OdomCNN training (Webots Tiago P-B, RESULT_04 winner).

    Accepts the pre-built ``data`` dict from ``load_webots_odom_pb()``.
    Returns ``(model_dict, history)`` with the trained encoder, head, val
    and test MAE, per-path distributions, and smoothness. ~5 minutes on
    Quadro P4000 at 30 epochs.
    """
    from src.pipeline.encoders import OdomCNN  # noqa: E402

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = data["Xtr"]; Ytr = data["Ytr"]
    Xva = data["Xva"]; Yva = data["Yva"]
    Xte = data["Xte"]; Yte = data["Yte"]
    mu_y = data["mu_y"]
    in_features = Xtr.shape[2]

    class _OdomCNNWithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = OdomCNN(in_features=in_features, embed_dim=embed_dim,
                                    channels=(16, 32, 64))
            self.pos = nn.Linear(embed_dim, 2)

        def forward(self, x):
            return self.pos(self.encoder(x))

    model = _OdomCNNWithHead().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  OdomCNN+head params: {n_params/1e6:.3f} M  train={len(Xtr)} val={len(Xva)} test={len(Xte)}",
              flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, len(Xtr) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=huber_delta)
    Xtr_t = torch.tensor(Xtr, device=device)
    Ytr_t = torch.tensor(Ytr, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    Yva_t = torch.tensor(Yva, device=device)

    history = {"train_loss": [], "val_mae": []}
    best = float("inf"); best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        ep_loss = 0.0
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            pred = model(Xtr_t[idx])
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ep_loss += loss.item()
        history["train_loss"].append(ep_loss / max(steps, 1))
        model.eval()
        with torch.no_grad():
            mae = float(torch.linalg.norm(model(Xva_t) - Yva_t, dim=1).mean())
        history["val_mae"].append(mae)
        if mae < best:
            best = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep <= 1 or ep % 5 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}/{epochs}  val_mae={mae:.3f}  (best {best:.3f})", flush=True)
    train_s = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva_t).cpu().numpy() + mu_y
        pred_te = model(torch.tensor(Xte, device=device)).cpu().numpy() + mu_y
    Yva_raw = data["Yva_raw"]
    Yte_raw = data["Yte_raw"]
    val_errs = np.linalg.norm(pred_va - Yva_raw, axis=1)
    test_errs = np.linalg.norm(pred_te - Yte_raw, axis=1)
    val_mae = float(val_errs.mean())
    test_mae = float(test_errs.mean())
    per_path_te = {}
    for p in np.unique(data["pid_te"]):
        mask = data["pid_te"] == p
        per_path_te[int(p)] = float(test_errs[mask].mean())
    rs = []
    for p in np.unique(data["pid_te"]):
        mask = data["pid_te"] == p
        dp = np.linalg.norm(np.diff(pred_te[mask], axis=0), axis=1)
        dg = np.linalg.norm(np.diff(Yte_raw[mask], axis=0), axis=1)
        if dp.std() > 1e-9 and dg.std() > 1e-9:
            rs.append(float(np.corrcoef(dp, dg)[0, 1]))
    smoothness = float(np.median(rs)) if rs else 0.0
    history["elapsed_s"] = train_s
    if verbose:
        print(f"  done in {train_s:.0f}s; val_mae={val_mae:.3f} test_mae={test_mae:.3f} "
              f"smoothness median r={smoothness:.3f}", flush=True)

    return {
        "model": model,
        "encoder": model.encoder, "head": model.pos,
        "val_mae": val_mae, "test_mae": test_mae,
        "per_path_test": per_path_te,
        "smoothness_median_r": smoothness,
        "n_params": int(n_params),
    }, history


# ---------------------------------------------------------------------------
# DPVOMotion head — TartanAir hospital P000 (RESULT_08 Mode 3-prime)
# ---------------------------------------------------------------------------

def train_dpvo_motion_head(
    seq_root,
    *,
    weights_path=None,
    batch: int = 4,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
):
    """Inline DPVOMotion-head training on TartanAir hospital P000.

    Replicates RESULT_08 / ``scripts/_eval_dpvomotion_hospital.py``.
    Frozen DPVO trunk + closed-form linear head trained on the first
    80 % of pairs; eval Umeyama-aligned ATE on the last 20 %.
    Returns ``(model_dict, history)``. ~5 minutes on Quadro P4000
    (extraction dominates; head fit is closed-form).
    """
    from pathlib import Path
    from PIL import Image

    from src.pipeline.encoders.dpvo_motion import DPVOMotionEncoder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    seq_root = Path(seq_root)
    if weights_path is None:
        weights_path = Path(__file__).resolve().parents[3] / "runs" / "_weights" / "dpvo.pth"

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_image(rgb_path: str, target_hw=(480, 640)) -> np.ndarray:
        img = Image.open(rgb_path).convert("RGB").resize((target_hw[1], target_hw[0]))
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = a.transpose(2, 0, 1)
        a = (a - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return a

    enc = DPVOMotionEncoder(weights_path=str(weights_path))
    enc.to(device).eval()
    n_trunk = sum(p.numel() for p in enc.trunk.parameters())

    pose_file = seq_root / "pose_left.txt"
    img_dir = seq_root / "image_left"
    poses = np.loadtxt(pose_file)
    img_files = sorted(img_dir.glob("*.png"))
    n_frames = min(len(poses), len(img_files))
    n_pairs = n_frames - 1
    if verbose:
        print(f"  DPVO trunk params: {n_trunk/1e6:.2f} M (frozen)", flush=True)
        print(f"  TartanAir P000: {n_frames} frames, {n_pairs} pairs", flush=True)

    tokens_list = []
    t_extract0 = time.time()
    with torch.no_grad():
        for i in range(0, n_pairs, batch):
            chunk_idx = list(range(i, min(i + batch, n_pairs)))
            prev_imgs = np.stack([load_image(str(img_files[j])) for j in chunk_idx])
            curr_imgs = np.stack([load_image(str(img_files[j + 1])) for j in chunk_idx])
            prev_t = torch.tensor(prev_imgs, device=device)
            curr_t = torch.tensor(curr_imgs, device=device)
            x = torch.stack([prev_t, curr_t], dim=1)
            tok = enc._frozen_tokens(x)
            tokens_list.append(tok.cpu().numpy())
    tokens = np.concatenate(tokens_list, axis=0)
    extract_s = time.time() - t_extract0
    if verbose:
        print(f"  extracted {tokens.shape} in {extract_s:.0f}s "
              f"({extract_s*1000/max(1, n_pairs):.1f} ms/pair)", flush=True)

    delta_xyz = poses[1:n_frames, :3] - poses[:n_frames - 1, :3]
    n_train = int(0.8 * n_pairs)
    Xtr = tokens[:n_train].mean(axis=1)
    Ytr = delta_xyz[:n_train]
    Xte = tokens[n_train:].mean(axis=1)
    Yte = delta_xyz[n_train:]
    mu_x = Xtr.mean(0); sd_x = Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu_x) / sd_x
    Xte_n = (Xte - mu_x) / sd_x
    mu_y = Ytr.mean(0)
    Ytr_c = Ytr - mu_y

    t_fit0 = time.time()
    A = np.linalg.lstsq(Xtr_n, Ytr_c, rcond=None)[0]
    pred_test = Xte_n @ A + mu_y
    fit_s = time.time() - t_fit0

    per_pair_err = np.linalg.norm(pred_test - Yte, axis=1)
    delta_mae = float(per_pair_err.mean())

    start_xyz = poses[n_train, :3]
    traj = np.zeros((len(pred_test) + 1, 3), dtype=np.float32)
    traj[0] = start_xyz
    for i in range(len(pred_test)):
        traj[i + 1] = traj[i] + pred_test[i]
    gt_traj = poses[n_train: n_train + len(pred_test) + 1, :3]

    aligned, scale = _umeyama_align(traj.astype(np.float64), gt_traj.astype(np.float64))
    errs = np.linalg.norm(aligned - gt_traj, axis=1)
    ate_rmse = float(np.sqrt((errs ** 2).mean()))
    ate_mean = float(errs.mean())

    total_s = time.time() - t_extract0
    history = {
        "extract_s": extract_s, "fit_s": fit_s, "elapsed_s": total_s,
        "delta_mae": delta_mae,
        "ate_rmse": ate_rmse, "ate_mean": ate_mean,
        "scale": scale,
    }
    if verbose:
        print(f"  done in {total_s:.0f}s; per-pair Δ-MAE={delta_mae:.5f} m  "
              f"Umeyama ATE RMSE={ate_rmse:.4f} m  mean={ate_mean:.4f} m  scale={scale:.3f}",
              flush=True)

    return {
        "encoder": enc,
        "head_weights": A,
        "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y,
        "tokens": tokens,
        "predicted_traj": aligned,
        "gt_traj": gt_traj,
        "delta_mae_m": delta_mae,
        "ate_umeyama_rmse_m": ate_rmse,
        "ate_umeyama_mean_m": ate_mean,
        "umeyama_scale": float(scale),
        "n_pairs_train": int(n_train),
        "n_pairs_test": int(len(pred_test)),
        "n_trunk_params": int(n_trunk),
    }, history


# ---------------------------------------------------------------------------
# Fusion arch — Webots Phase B winner reproduction (RESULT_13/17)
# ---------------------------------------------------------------------------

def train_fusion_arch(
    arch: str,
    *,
    dataset: str = "simulation",
    K: int = 4,
    batch_size: int = 128,
    lr: float = 1.3e-3,
    epochs: int = 90,
    seed: int = 42,
    save_dir=None,
    verbose: bool = True,
):
    """Inline FusionTrainer build + fit for one bake-off architecture.

    Replicates the RESULT_13 (incumbent) / RESULT_17 (cnn1d, lstm_attn)
    recipe: 4-modality Webots, K=4 temporal, B=128, lr=1.3e-3, 90 epochs.
    Returns ``(trainer, history, save_path)``. Saves ``model.pt`` under
    ``save_dir`` (default ``runs/overnight/run2_iter_33/<arch>``).

    ~3-4 minutes per arch on Quadro P4000 (RESULT_06/17 actuals).
    """
    from pathlib import Path

    from src.pipeline.fusion import CANDIDATES
    from src.pipeline.fusion.builder import (
        build_datamodule, build_encoders, extract_vision_tokens, load_config,
    )
    from .fusion_trainer import FusionTrainer

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[3] / "runs" / "overnight" / "run2_iter_33" / arch
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = load_config(dataset)
    cfg.temporal.n_instants = int(K)
    cfg.data.batch_size = int(batch_size)
    cfg.train.lr = float(lr)

    dm = build_datamodule(cfg)
    encs, vision = build_encoders(cfg, dm)
    extra = extract_vision_tokens(dm, vision, device="cuda" if torch.cuda.is_available() else "cpu") if vision is not None else {}

    if arch not in CANDIDATES:
        raise KeyError(f"Unknown arch {arch!r}. Available: {list(CANDIDATES)}")
    incumbent_kwargs = dict(
        embed_dim=int(cfg.model.embed_dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.n_heads),
        ff_mult=int(cfg.model.ff_mult),
        dropout=float(cfg.model.dropout),
        use_time=bool(cfg.model.use_time),
        readout=str(cfg.model.readout),
        absolute_modalities=list(cfg.model.get("absolute_modalities", None) or ["wifi"]),
    )
    model = CANDIDATES[arch](incumbent_kwargs, encs)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(model.modalities),
        extra_inputs=extra,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        huber_delta=float(cfg.train.huber_delta),
        grad_clip=float(cfg.train.grad_clip),
        patience=int(cfg.train.patience),
        batch_size=int(cfg.data.batch_size),
        modality_dropout=float(cfg.train.modality_dropout),
        instant_dropout=float(cfg.train.instant_dropout),
        n_instants=int(cfg.temporal.n_instants),
        instant_stride=int(cfg.temporal.instant_stride),
        run_dir=str(save_dir),
    )

    t0 = time.time()
    history = trainer.fit(epochs=epochs, verbose=verbose)
    elapsed = time.time() - t0

    model_pt = save_dir / "model.pt"
    torch.save({"state_dict": trainer.model.state_dict()}, model_pt)
    if verbose:
        print(f"  {arch}: trained in {elapsed/60:.2f} min; saved {model_pt}", flush=True)

    return trainer, history, model_pt


__all__ = [
    "train_anchor2vec", "anchor2vec_predict", "anchor2vec_val_mae",
    "train_imucnn",
    "train_odomcnn", "load_webots_odom_pb", "compute_trivial_integration_floor",
    "train_dpvo_motion_head",
    "train_fusion_arch",
]
