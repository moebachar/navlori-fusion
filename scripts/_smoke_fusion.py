"""Phased smoke / verification harness for the FusionTransformer.

Each phase is independent — run them in order while developing:

    python scripts/_smoke_fusion.py --phase 1   # shape / forward sanity
    python scripts/_smoke_fusion.py --phase 2   # overfit a tiny batch
    python scripts/_smoke_fusion.py --phase 3   # full train + subset eval
    python scripts/_smoke_fusion.py --phase 4   # profile a training step
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data.datamodule import FusionDataModule          # noqa: E402
from src.pipeline.encoders import Anchor2Vec, IMUCNN, OdomCNN      # noqa: E402
from src.pipeline.fusion.transformer import FusionTransformer      # noqa: E402
from src.pipeline.training.fusion_trainer import FusionTrainer     # noqa: E402

MODALITIES = ["imu", "odom", "wifi"]
EMBED_DIM = 128


def _build_dm():
    dm = FusionDataModule.from_paths(modalities=MODALITIES, batch_size=128)
    dm.setup()
    return dm


def _build_encoders(dm, embed_dim=EMBED_DIM):
    n_aps = dm.train_ds.num_wifi_aps
    return {
        "imu": IMUCNN(in_features=9, embed_dim=embed_dim),
        "odom": OdomCNN(in_features=5, embed_dim=embed_dim),
        "wifi": Anchor2Vec(n_aps=n_aps, embed_dim=embed_dim),
    }


# ---------------------------------------------------------------------------
def phase1():
    """Forward pass on random tensors — check shapes and no NaN."""
    print("=== Phase 1: shape / forward sanity ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoders = {
        "imu": IMUCNN(in_features=9, embed_dim=EMBED_DIM),
        "odom": OdomCNN(in_features=7, embed_dim=EMBED_DIM),
        "wifi": Anchor2Vec(n_aps=117, embed_dim=EMBED_DIM),
    }
    for readout in ("cls", "query"):
        model = FusionTransformer(encoders, embed_dim=EMBED_DIM, depth=2,
                                  readout=readout).to(device)
        B = 8
        inputs = {
            "imu": torch.randn(B, 1, 32, 9, device=device),
            "odom": torch.randn(B, 1, 16, 7, device=device),
            "wifi": torch.randn(B, 1, 1, 117, device=device),
        }
        avail = {m: torch.ones(B, 1, dtype=torch.bool, device=device)
                 for m in encoders}
        dt = {m: torch.zeros(B, 1, device=device) for m in encoders}
        out = model(inputs, avail, dt)
        assert out.shape == (B, 2), out.shape
        assert torch.isfinite(out).all(), "non-finite output"

        # Drop two modalities for half the batch — must still be finite.
        avail["wifi"][: B // 2] = False
        avail["odom"][: B // 2] = False
        out2 = model(inputs, avail, dt)
        assert torch.isfinite(out2).all(), "non-finite with dropped modalities"

        # Worst case: a row with *every* modality unavailable. CLS must keep
        # attention well-defined (no all-(-inf) softmax → no NaN).
        allgone = {m: torch.zeros(B, 1, dtype=torch.bool, device=device)
                   for m in encoders}
        out3 = model(inputs, allgone, dt)
        assert torch.isfinite(out3).all(), "non-finite with all modalities gone"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  readout={readout:5s}  out={tuple(out.shape)}  "
              f"params={n_params/1e6:.2f}M  OK")
    print("Phase 1 PASS\n")


def phase2():
    """Overfit a tiny batch — loss must collapse toward zero."""
    print("=== Phase 2: overfit a tiny batch ===")
    dm = _build_dm()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionTransformer(_build_encoders(dm), embed_dim=EMBED_DIM,
                              depth=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    crit = torch.nn.HuberLoss(delta=0.5)

    B = 64
    X = {m: dm.train_ds.get_tensors(m)[0][:B].unsqueeze(1).to(device)
         for m in MODALITIES}
    y = dm.train_ds._targets[:B].to(device)
    avail = {m: torch.ones(B, 1, dtype=torch.bool, device=device)
             for m in MODALITIES}
    dt = {m: torch.zeros(B, 1, device=device) for m in MODALITIES}

    model.train()
    losses = []
    for step in range(300):
        pred = model(X, avail, dt)
        loss = crit(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 299:
            mae = torch.linalg.norm(pred.detach() - y, dim=1).mean().item()
            losses.append(loss.item())
            print(f"  step {step:3d}  loss={loss.item():.5f}  mae={mae:.4f}m")
    assert losses[-1] < losses[0] * 0.2, "loss did not collapse — capacity bug"
    print("Phase 2 PASS — model can fit the task\n")


def phase3(epochs=80, instants=1, stride=1, readout="cls", patience=25,
           instant_dropout=0.0):
    """Full training run + the modality-subset robustness table."""
    print(f"=== Phase 3: full train (instants={instants}, stride={stride}, "
          f"readout={readout}, instant_dropout={instant_dropout}) ===")
    dm = _build_dm()
    print(dm.summary())
    model = FusionTransformer(_build_encoders(dm), embed_dim=EMBED_DIM,
                              depth=4, n_heads=8, dropout=0.1, use_time=True,
                              readout=readout)
    trainer = FusionTrainer(model, dm, MODALITIES, lr=1e-3,
                            modality_dropout=0.3, instant_dropout=instant_dropout,
                            batch_size=128, n_instants=instants,
                            instant_stride=stride, patience=patience)
    hist = trainer.fit(epochs=epochs)
    print("\n  Modality-subset MAE (val):")
    subsets = trainer.evaluate_subsets("val")
    for label, m in subsets.items():
        print(f"    {label:12s}  MAE={m['mae']:.3f}m  RMSE={m['rmse']:.3f}m")
    if instants > 1:
        print("\n  WiFi-staleness MAE (val) — instants of stale WiFi:")
        stale = trainer.evaluate_staleness("wifi", "val")
        for label, m in stale.items():
            print(f"    {label:12s}  MAE={m['mae']:.3f}m  RMSE={m['rmse']:.3f}m")
    print(f"\n  run dir: {trainer.run_path}")
    print("Phase 3 done\n")
    return hist, subsets


def phase4():
    """Profile a single training step."""
    print("=== Phase 4: profile ===")
    dm = _build_dm()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionTransformer(_build_encoders(dm), embed_dim=EMBED_DIM,
                              depth=4).to(device)
    B = 128
    X = {m: dm.train_ds.get_tensors(m)[0][:B].unsqueeze(1).to(device)
         for m in MODALITIES}
    y = dm.train_ds._targets[:B].to(device)
    avail = {m: torch.ones(B, 1, dtype=torch.bool, device=device)
             for m in MODALITIES}
    dt = {m: torch.zeros(B, 1, device=device) for m in MODALITIES}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.HuberLoss(delta=0.5)

    for _ in range(5):  # warmup
        loss = crit(model(X, avail, dt), y)
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n = 50
    for _ in range(n):
        loss = crit(model(X, avail, dt), y)
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.time() - t0) / n * 1e3
    mem = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0
    print(f"  device={device}  step={dt_ms:.2f} ms  "
          f"peak_mem={mem:.0f} MB  (B={B})")
    print("Phase 4 done\n")


def phase5(epochs=80):
    """4-modality fusion including DPVO vision via the patch-token cache.

    The DPVO trunk + correlation tracker is frozen / parameter-free, so its
    per-patch motion tokens (B,64,132) are extracted once and cached; only
    the DPVO ``_MotionHead`` then trains end-to-end inside the fusion model
    — exactly the EncoderTrainer vision-cache pattern.
    """
    from torch.utils.data import DataLoader

    from src.pipeline.encoders import DPVOMotionEncoder

    print("=== Phase 5: 4-modality fusion (imu, odom, wifi, vision/DPVO) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mods = ["imu", "odom", "wifi", "camera"]
    dm = FusionDataModule.from_paths(
        modalities=mods, windows={"camera": 2}, camera_stride=5, batch_size=128,
    )
    dm.setup()
    print(dm.summary())

    dpvo = DPVOMotionEncoder(embed_dim=EMBED_DIM)
    cache = dm.data_dir / ".dpvomotion_fusion_cache"
    # Non-shuffled loaders → extracted tokens align with the tabular caches.
    raw_tr = DataLoader(dm.train_ds, batch_size=128, shuffle=False)
    raw_va = DataLoader(dm.val_ds, batch_size=128, shuffle=False)
    print("  Extracting DPVO patch tokens (one-time, cached to disk)...",
          flush=True)
    feat_tr, _ = dpvo.extract_backbone_features(
        raw_tr, device, cache_path=cache / "train.pt")
    feat_va, _ = dpvo.extract_backbone_features(
        raw_va, device, cache_path=cache / "val.pt")
    print(f"  DPVO tokens: train {tuple(feat_tr.shape)}  "
          f"val {tuple(feat_va.shape)}", flush=True)

    encoders = _build_encoders(dm)
    encoders["camera"] = dpvo.head            # _MotionHead trains end-to-end
    model = FusionTransformer(encoders, embed_dim=EMBED_DIM, depth=4,
                              n_heads=8, dropout=0.1, use_time=True)
    trainer = FusionTrainer(
        model, dm, mods,
        extra_inputs={"camera": {"train": feat_tr, "val": feat_va}},
        lr=1e-3, modality_dropout=0.3, batch_size=128,
    )
    hist = trainer.fit(epochs=epochs)
    print("\n  Modality-subset MAE (val):")
    subsets = trainer.evaluate_subsets("val")
    for label, m in subsets.items():
        print(f"    {label:12s}  MAE={m['mae']:.3f}m  RMSE={m['rmse']:.3f}m")
    print(f"\n  run dir: {trainer.run_path}")
    print("Phase 5 done\n")
    return hist, subsets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--instants", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--readout", type=str, default="cls",
                    choices=["cls", "query"])
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--instant-dropout", type=float, default=0.0)
    args = ap.parse_args()
    {
        1: phase1,
        2: phase2,
        3: lambda: phase3(args.epochs, args.instants, args.stride,
                          args.readout, args.patience, args.instant_dropout),
        4: phase4,
        5: lambda: phase5(args.epochs),
    }[args.phase]()
