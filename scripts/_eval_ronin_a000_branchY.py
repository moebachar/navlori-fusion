"""IMU encoder audit — Branch Y (a000 intra-session proxy).

PLAN_02 fell back to Branch Y because the full RoNIN dataset
(FRDR_dataset_538...) is not on this machine — only
``data/ronin_a000_intra/`` (215 × 15 s chunks of subject a000,
already converted to the project's async_collection format).

This script trains three IMU encoders on the SAME a000-intra proxy
data with the SAME windowed velocity protocol so the gap between
``IMUCNN``, ``IMUCNN 2× width``, and ``RoNIN ResNet1D`` is measured
apples-to-apples:

    * IMUCNN base (~75 k params, embed_dim=128, channels (32,64,128))
    * IMUCNN 2× (~700 k params, embed_dim=256, channels (64,128,256))
    * RoNIN ResNet1D (~4.6 M params, the vendored open-source SOTA
      architecture from `Sachini/ronin`, imported pure — Demand #3)

Per-chunk evaluation: predicted velocities are integrated into a
trajectory, ATE computed per chunk, then mean ± median ± max are
reported. 6-metric harness (where applicable to motion encoders)
runs on the held-out chunk embeddings.

Caveat written into RESULT_02: this is NOT the canonical RoNIN
unseen-subjects benchmark (single subject, intra-session 15 s
chunks). It answers "is IMUCNN structurally under-powered" but does
NOT discharge C2.

Run: .venv/Scripts/python.exe scripts/_eval_ronin_a000_branchY.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RONIN_SRC = Path(r"C:\Users\FabLab\AppData\Local\Temp\ronin\source")
if str(RONIN_SRC) not in sys.path:
    sys.path.insert(0, str(RONIN_SRC))

# Demand #3 runtime shim — applied here, never edited into vendored source.
np.int = int  # type: ignore[attr-defined]

from model_resnet1d import BasicBlock1D, FCOutputModule, ResNet1D  # noqa: E402  vendored RoNIN

from src.pipeline.encoders import IMUCNN  # noqa: E402
from src.pipeline.evaluation.encoder_eval import (  # noqa: E402
    alignment_uniformity,
    effective_dimensionality,
    knn_probe,
    linear_probe,
    temporal_smoothness,
    trustworthiness,
)


DATA_DIR = ROOT / "data" / "ronin_a000_intra"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_02"


# ---------------------------------------------------------------------------
# Data loading — chunk-level windowed velocity samples
# ---------------------------------------------------------------------------


def load_chunk(path_dir: Path) -> dict | None:
    """Load one path_NN chunk. Returns None if files are missing or short."""
    imu_csv = path_dir / "imu.csv"
    gt_csv = path_dir / "ground_truth.csv"
    if not imu_csv.exists() or not gt_csv.exists():
        return None
    imu = pd.read_csv(imu_csv)
    gt = pd.read_csv(gt_csv)
    # 6 IMU channels in RoNIN order: accel_xyz then gyro_xyz.
    feat = imu[["accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z"]].values.astype(np.float32)
    t_imu = imu["sim_time"].values.astype(np.float32)
    t_gt = gt["sim_time"].values.astype(np.float32)
    xy = gt[["gt_x", "gt_y"]].values.astype(np.float32)
    if len(feat) < 220 or len(xy) < 4:
        return None
    return {"feat": feat, "t_imu": t_imu, "t_gt": t_gt, "xy": xy, "name": path_dir.name}


def build_windows(chunks: list[dict], window: int, stride: int) -> dict:
    """Build (N, window, 6) windowed inputs + (N, 2) velocity targets."""
    Xs, Ys, names_per_window, end_times = [], [], [], []
    for c in chunks:
        feat = c["feat"]
        t_imu = c["t_imu"]
        t_gt = c["t_gt"]
        xy = c["xy"]
        N = len(feat)
        ends = np.arange(window, N, stride)
        for e in ends:
            t0 = t_imu[e - window]
            t1 = t_imu[e - 1]
            # Linear-interp GT position at window endpoints
            xy0 = np.array([np.interp(t0, t_gt, xy[:, 0]),
                            np.interp(t0, t_gt, xy[:, 1])], dtype=np.float32)
            xy1 = np.array([np.interp(t1, t_gt, xy[:, 0]),
                            np.interp(t1, t_gt, xy[:, 1])], dtype=np.float32)
            dt = max(t1 - t0, 1e-3)
            v = (xy1 - xy0) / dt
            Xs.append(feat[e - window:e])
            Ys.append(v)
            names_per_window.append(c["name"])
            end_times.append(t1)
    return {
        "X": np.stack(Xs),  # (N, window, 6)
        "Y": np.stack(Ys),  # (N, 2)
        "name": np.array(names_per_window),
        "t": np.array(end_times),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class IMUCNNWithHead(nn.Module):
    """IMUCNN encoder + linear head; outputs (B, 2) velocity."""

    def __init__(self, in_features=6, embed_dim=128, channels=(32, 64, 128)):
        super().__init__()
        self.encoder = IMUCNN(in_features=in_features, embed_dim=embed_dim,
                              channels=channels)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):  # x: (B, window, 6)
        z = self.encoder(x)
        return self.head(z)


class ResNet1DWithHead(nn.Module):
    """RoNIN ResNet1D (vendored, pure) wrapping (B, window, 6) → (B, 2)."""

    def __init__(self):
        super().__init__()
        # Matches RoNIN's canonical `ronin_resnet.py` `resnet18` config (1:1).
        # GlobAvgOutputModule in the vendored repo has a typo
        # (`self.avg()` missing input) — using FCOutputModule, which is what
        # RoNIN ships as default for their resnet18.
        self.net = ResNet1D(num_inputs=6, num_outputs=2,
                            block_type=BasicBlock1D, group_sizes=[2, 2, 2, 2],
                            base_plane=64, output_block=FCOutputModule,
                            kernel_size=3,
                            fc_dim=512, in_dim=7, dropout=0.5, trans_planes=128)

    def forward(self, x):  # x: (B, window, 6)
        # ResNet1D expects (B, 6, window) — same convention as
        # eval_ronin_ate_fixed.py.
        return self.net(x.transpose(1, 2).contiguous())


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Training + ATE eval helpers
# ---------------------------------------------------------------------------


def memory_budget_check(model_factory, batch=128, window=200) -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = model_factory().cuda()
    x = torch.randn(batch, window, 6, device="cuda", requires_grad=False)
    y = torch.randn(batch, 2, device="cuda")
    pred = m(x)
    loss = nn.functional.huber_loss(pred, y)
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    del m, x, y, pred, loss
    torch.cuda.empty_cache()
    return peak_mb


def train_model(model, Xtr, Ytr, Xva, Yva, epochs, batch, lr, name, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=0.5)
    Xtr_t = torch.tensor(Xtr, device=dev)
    Ytr_t = torch.tensor(Ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)
    Yva_t = torch.tensor(Yva, device=dev)
    best_va = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        tr_loss = 0.0
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            pred = model(Xtr_t[idx])
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tr_loss += float(loss.detach()) * len(idx)
        tr_loss /= max(1, len(Xtr_t))
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t)
            va_loss = float(nn.functional.huber_loss(pv, Yva_t, delta=0.5))
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep <= 1 or ep % 4 == 0 or ep == epochs - 1:
            print(f"  [{name}] ep {ep:3d}  tr_huber={tr_loss:.5f}  va_huber={va_loss:.5f}  "
                  f"(best {best_va:.5f})", flush=True)
    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{name}] done in {elapsed:.1f}s  best val huber {best_va:.5f}", flush=True)
    return best_va, elapsed


def per_chunk_ate(model, chunks_test, window, stride, dev):
    """Integrate predicted velocities → trajectory → ATE per chunk."""
    model.eval()
    ates_raw, ates_aligned, names, lens = [], [], [], []
    with torch.no_grad():
        for c in chunks_test:
            feat = c["feat"]
            t_imu = c["t_imu"]
            t_gt = c["t_gt"]
            xy = c["xy"]
            N = len(feat)
            ends = np.arange(window, N, stride)
            if len(ends) < 2:
                continue
            wins = np.stack([feat[e - window:e] for e in ends]).astype(np.float32)
            x = torch.tensor(wins, device=dev)
            vel = model(x).cpu().numpy()  # (M, 2) m/s
            # Integrate from gt at t_imu[ends[0]].
            t0_traj = t_imu[ends[0]]
            traj = np.zeros((len(ends), 2), np.float32)
            traj[0, 0] = np.interp(t0_traj, t_gt, xy[:, 0])
            traj[0, 1] = np.interp(t0_traj, t_gt, xy[:, 1])
            for i in range(1, len(ends)):
                dt = t_imu[ends[i]] - t_imu[ends[i - 1]]
                traj[i] = traj[i - 1] + vel[i - 1] * dt
            # Compare against GT at ends.
            gt_at_ends = np.stack([
                np.interp(t_imu[ends], t_gt, xy[:, 0]),
                np.interp(t_imu[ends], t_gt, xy[:, 1]),
            ], axis=1).astype(np.float32)
            d = np.sqrt(((traj - gt_at_ends) ** 2).sum(1))
            ate_raw = float(np.sqrt(((traj - gt_at_ends) ** 2).sum(1).mean()))
            # Aligned ATE (Procrustes).
            pc = traj - traj.mean(0)
            gc = gt_at_ends - gt_at_ends.mean(0)
            H = pc.T @ gc
            U, _, Vt = np.linalg.svd(H)
            Rm = Vt.T @ U.T
            if np.linalg.det(Rm) < 0:
                Vt[-1] *= -1
                Rm = Vt.T @ U.T
            aligned = pc @ Rm.T + gt_at_ends.mean(0)
            ate_al = float(np.sqrt(((aligned - gt_at_ends) ** 2).sum(1).mean()))
            ates_raw.append(ate_raw)
            ates_aligned.append(ate_al)
            names.append(c["name"])
            lens.append(int(len(ends)))
    arr_raw = np.array(ates_raw)
    arr_al = np.array(ates_aligned)
    return {
        "raw_mean": float(arr_raw.mean()),
        "raw_median": float(np.median(arr_raw)),
        "raw_p25": float(np.percentile(arr_raw, 25)),
        "raw_p75": float(np.percentile(arr_raw, 75)),
        "raw_p90": float(np.percentile(arr_raw, 90)),
        "raw_max": float(arr_raw.max()),
        "aligned_mean": float(arr_al.mean()),
        "aligned_median": float(np.median(arr_al)),
        "aligned_p90": float(np.percentile(arr_al, 90)),
        "n_chunks": int(len(arr_raw)),
        "n_windows_per_chunk_mean": float(np.mean(lens)),
        "per_chunk_raw": [float(x) for x in arr_raw],
    }


# ---------------------------------------------------------------------------
# Encoder embeddings for 6-metric harness
# ---------------------------------------------------------------------------


@torch.no_grad()
def embed_windows(encoder_fn, X, dev, batch=512):
    """encoder_fn maps (B, window, 6) -> (B, D). Returns numpy (N, D)."""
    out = []
    for i in range(0, len(X), batch):
        x = torch.tensor(X[i:i + batch], device=dev)
        out.append(encoder_fn(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def latency_ms(model, window, dev, batch=1, runs=200):
    model.eval()
    x = torch.zeros(batch, window, 6, device=dev)
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--epochs-pretest", type=int, default=5)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  window={args.window}  stride={args.stride}", flush=True)

    # ---------------- load chunks ----------------
    path_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("path_")])
    print(f"a000-intra: {len(path_dirs)} path dirs", flush=True)
    chunks = []
    for d in path_dirs:
        c = load_chunk(d)
        if c is not None:
            chunks.append(c)
    print(f"loaded chunks: {len(chunks)}", flush=True)
    # Plan-prescribed split: train = 0-184 (185 chunks), test = 185-214 (30 chunks).
    # Use sorted indices on the loaded list (chunks are in name order).
    n_total = len(chunks)
    n_test = 30
    train_chunks = chunks[:max(1, n_total - n_test)]
    test_chunks = chunks[max(1, n_total - n_test):]
    print(f"train chunks: {len(train_chunks)}  test chunks: {len(test_chunks)}", flush=True)

    # ---------------- build windowed datasets ----------------
    train = build_windows(train_chunks, args.window, args.stride)
    test = build_windows(test_chunks, args.window, args.stride)
    print(f"train windows: {len(train['X'])}  test windows: {len(test['X'])}", flush=True)
    # Normalise IMU inputs using train-set per-channel mean/std (Demand #3 stays:
    # vendored RoNIN ResNet1D unchanged; normalisation is in our wrapper).
    mu = train["X"].reshape(-1, 6).mean(0)
    sd = train["X"].reshape(-1, 6).std(0) + 1e-6
    train["X"] = (train["X"] - mu) / sd
    test["X"] = (test["X"] - mu) / sd
    Xtr, Ytr = train["X"].astype(np.float32), train["Y"].astype(np.float32)
    Xva, Yva = test["X"].astype(np.float32), test["Y"].astype(np.float32)

    # ---------------- pre-test gate (per cycle rule) ----------------
    print("\n[pre-test gate] IMUCNN base, 10 % subset, 5 epochs", flush=True)
    n_sub = max(64, int(len(Xtr) * 0.10))
    idx_sub = np.random.RandomState(0).permutation(len(Xtr))[:n_sub]
    pre = IMUCNNWithHead(in_features=6, embed_dim=128).to(dev)
    pre_va, pre_t = train_model(pre, Xtr[idx_sub], Ytr[idx_sub], Xva, Yva,
                                 epochs=args.epochs_pretest, batch=args.batch,
                                 lr=args.lr, name="IMUCNN-pretest", dev=dev)
    pretest_pass = pre_va < 100.0
    print(f"  pre-test val_huber={pre_va:.5f} ({pre_t:.1f}s)  pass={pretest_pass}", flush=True)
    del pre
    torch.cuda.empty_cache()

    # ---------------- memory budget checks ----------------
    print("\n[memory budget] target shape B=128, window=200, 6 ch", flush=True)
    mem_imucnn = memory_budget_check(lambda: IMUCNNWithHead(6, 128), args.batch, args.window)
    mem_imucnn2x = memory_budget_check(lambda: IMUCNNWithHead(6, 256, (64, 128, 256)),
                                       args.batch, args.window)
    mem_resnet = memory_budget_check(ResNet1DWithHead, args.batch, args.window)
    print(f"  IMUCNN base:    peak {mem_imucnn:7.1f} MB", flush=True)
    print(f"  IMUCNN 2×:      peak {mem_imucnn2x:7.1f} MB", flush=True)
    print(f"  ResNet1D:       peak {mem_resnet:7.1f} MB", flush=True)
    for label, val in [("IMUCNN", mem_imucnn), ("IMUCNN 2x", mem_imucnn2x),
                       ("ResNet1D", mem_resnet)]:
        if val > 6000:
            raise RuntimeError(f"{label} peak {val:.0f} MB exceeds 6 GB budget")

    # ---------------- train + eval each model ----------------
    results = {
        "branch": "Y (a000_intra proxy)",
        "split": {"train_chunks": len(train_chunks),
                  "test_chunks": len(test_chunks),
                  "train_windows": int(len(Xtr)), "test_windows": int(len(Xva))},
        "window": args.window, "stride": args.stride, "batch": args.batch,
        "epochs": args.epochs,
        "pretest_val_huber": float(pre_va),
        "memory_budget_mb": {"IMUCNN_base": mem_imucnn,
                              "IMUCNN_2x": mem_imucnn2x,
                              "ResNet1D": mem_resnet},
        "models": {},
    }

    model_specs = [
        ("IMUCNN_base", lambda: IMUCNNWithHead(6, 128)),
        ("IMUCNN_2x", lambda: IMUCNNWithHead(6, 256, (64, 128, 256))),
        ("ResNet1D", ResNet1DWithHead),
    ]

    for name, fac in model_specs:
        print(f"\n[{name}] training ({args.epochs} epochs)", flush=True)
        m = fac().to(dev)
        params = n_params(m)
        best_va, train_s = train_model(m, Xtr, Ytr, Xva, Yva,
                                        epochs=args.epochs, batch=args.batch,
                                        lr=args.lr, name=name, dev=dev)
        ate = per_chunk_ate(m, test_chunks, args.window, args.stride, dev)
        lat = latency_ms(m, args.window, dev)
        # Embeddings — only for IMUCNN variants (ResNet1D doesn't expose an
        # embedding layer; its output IS velocity). Use the trunk output before
        # the head: for IMUCNN, that's m.encoder(x); for ResNet1D we mean-pool
        # the residual_groups output to get a comparable D-vector.
        if isinstance(m, IMUCNNWithHead):
            emb_fn = lambda x, mm=m: mm.encoder(x)
        else:
            def emb_fn(x, mm=m):
                z = mm.net.input_block(x.transpose(1, 2).contiguous())
                z = mm.net.residual_groups(z)
                return z.mean(dim=2)
        z_tr = embed_windows(emb_fn, Xtr[:5000], dev)  # subsample for speed
        z_va = embed_windows(emb_fn, Xva, dev)
        # 6-metric harness (where well-defined for motion encoders).
        # linear_probe and knn_probe predict (vx, vy); they are well-defined
        # for velocity targets. We use train_velocity targets, not positions.
        try:
            lp = linear_probe(z_tr, Ytr[:5000], z_va, Yva, epochs=200, lr=1e-2, device="cpu")
        except Exception as e:
            lp = {"error": str(e)}
        try:
            kp = knn_probe(z_tr, Ytr[:5000], z_va, Yva, k=5)
        except Exception as e:
            kp = {"error": str(e)}
        try:
            au = alignment_uniformity(z_va, Yva, distance_threshold=0.05,
                                       max_samples=1000)
        except Exception as e:
            au = {"error": str(e)}
        try:
            ed = effective_dimensionality(z_va)
        except Exception as e:
            ed = {"error": str(e)}
        try:
            ts = temporal_smoothness(z_va, Yva)
        except Exception as e:
            ts = {"error": str(e)}
        try:
            tw = trustworthiness(Xva.reshape(len(Xva), -1), z_va, k=10)
        except Exception as e:
            tw = {"error": str(e)}
        results["models"][name] = {
            "params": int(params),
            "best_val_huber": float(best_va),
            "train_time_s": float(train_s),
            "latency_ms_per_window_b1": float(lat),
            "ate": ate,
            "linear_probe": lp,
            "knn_probe": kp,
            "alignment_uniformity": au,
            "effective_dimensionality": ed,
            "temporal_smoothness": ts,
            "trustworthiness": tw,
        }
        del m
        torch.cuda.empty_cache()

    out_path = OUT_DIR / "a000_branchY.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # ---------------- comparison summary ----------------
    print(f"\n{'metric':<35} {'IMUCNN_base':>13} {'IMUCNN_2x':>13} {'ResNet1D':>13}")
    a_mm = results["models"]
    print(f"  {'params (M)':<35} {a_mm['IMUCNN_base']['params']/1e6:>12.2f}M "
          f"{a_mm['IMUCNN_2x']['params']/1e6:>12.2f}M {a_mm['ResNet1D']['params']/1e6:>12.2f}M")
    print(f"  {'best val huber':<35} "
          f"{a_mm['IMUCNN_base']['best_val_huber']:>13.5f} "
          f"{a_mm['IMUCNN_2x']['best_val_huber']:>13.5f} "
          f"{a_mm['ResNet1D']['best_val_huber']:>13.5f}")
    print(f"  {'ATE raw (m, mean)':<35} "
          f"{a_mm['IMUCNN_base']['ate']['raw_mean']:>13.3f} "
          f"{a_mm['IMUCNN_2x']['ate']['raw_mean']:>13.3f} "
          f"{a_mm['ResNet1D']['ate']['raw_mean']:>13.3f}")
    print(f"  {'ATE raw (m, median)':<35} "
          f"{a_mm['IMUCNN_base']['ate']['raw_median']:>13.3f} "
          f"{a_mm['IMUCNN_2x']['ate']['raw_median']:>13.3f} "
          f"{a_mm['ResNet1D']['ate']['raw_median']:>13.3f}")
    print(f"  {'ATE aligned (m, mean)':<35} "
          f"{a_mm['IMUCNN_base']['ate']['aligned_mean']:>13.3f} "
          f"{a_mm['IMUCNN_2x']['ate']['aligned_mean']:>13.3f} "
          f"{a_mm['ResNet1D']['ate']['aligned_mean']:>13.3f}")
    print(f"  {'latency b=1 (ms)':<35} "
          f"{a_mm['IMUCNN_base']['latency_ms_per_window_b1']:>13.3f} "
          f"{a_mm['IMUCNN_2x']['latency_ms_per_window_b1']:>13.3f} "
          f"{a_mm['ResNet1D']['latency_ms_per_window_b1']:>13.3f}")
    print(f"  {'kNN-probe vel-MAE':<35} "
          f"{a_mm['IMUCNN_base']['knn_probe'].get('mae', float('nan')):>13.3f} "
          f"{a_mm['IMUCNN_2x']['knn_probe'].get('mae', float('nan')):>13.3f} "
          f"{a_mm['ResNet1D']['knn_probe'].get('mae', float('nan')):>13.3f}")
    print(f"  {'linear-probe vel-MAE':<35} "
          f"{a_mm['IMUCNN_base']['linear_probe'].get('mae', float('nan')):>13.3f} "
          f"{a_mm['IMUCNN_2x']['linear_probe'].get('mae', float('nan')):>13.3f} "
          f"{a_mm['ResNet1D']['linear_probe'].get('mae', float('nan')):>13.3f}")


if __name__ == "__main__":
    main()
