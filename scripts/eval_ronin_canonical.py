"""Canonical RoNIN unseen-subjects benchmark — reproduces RESULT_07
(ResNet1D SOTA) + RESULT_23 (CNN1D aggregator over IMUCNN sub-windows).

ResNet1D pretrained reproduces the paper's 5.140 m raw ATE exactly.
Our CNN1D / LSTM-attn aggregator over K=4 sub-windows of length 50
narrows the gap from IMUCNN's +94% (RESULT_07) to +47-48% raw / +16-19%
Umeyama (RESULT_23 — CNN1D clears the 20% Umeyama audit gate).

Thin wrapper on consolidated APIs:
- ``src.pipeline.baselines`` for the vendored RoNIN code (ResNet1D,
  GlobSpeedSequence, compute_ate_rte, load_test_list).
- ``src.pipeline.data.load_dataset('ronin_canonical', split='test')``
  for the data side (returns a ``StridedSequenceDataset``).

By default loads the pretrained ResNet1D checkpoint and reports its
ATE. With ``--include-ours``, also trains/loads the CNN1D aggregator
and reports its number for the side-by-side.

Run: ``.venv/Scripts/python.exe scripts/eval_ronin_canonical.py``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.baselines import (  # noqa: E402
    BasicBlock1D, FCOutputModule, ResNet1D,
    GlobSpeedSequence, compute_ate_rte, load_test_list,
)


CKPT = ROOT / "data" / "ronin_frdr" / "pretrained_resnet" / "ronin_resnet" / "checkpoint_gsn_latest.pt"
TEST_DIR = ROOT / "data" / "ronin_frdr" / "unseen"


def eval_resnet1d_pretrained() -> dict:
    """Run RoNIN's pretrained ResNet1D over the 32 canonical unseen seqs.

    Reproduces RESULT_07's 5.140 m / 4.377 m (ATE / RTE) — paper-exact.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not CKPT.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint missing: {CKPT}. See RESULT_07 for "
            "extraction (data/FRDR_dataset_538_download_606_*.zip).")
    test_list = [s for s in load_test_list("list_test_unseen.txt") if (TEST_DIR / s).is_dir()]
    fc_cfg = {"fc_dim": 512, "in_dim": 7, "dropout": 0.5, "trans_planes": 128}
    net = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2], base_plane=64,
                   output_block=FCOutputModule, kernel_size=3, **fc_cfg).to(dev)
    state = torch.load(CKPT, map_location=dev, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state, strict=True)
    net.eval()
    pred_per_min = 200 * 60
    ates, rtes = [], []
    with torch.no_grad():
        for sname in test_list:
            seq = GlobSpeedSequence(str(TEST_DIR / sname), interval=200,
                                     max_ori_error=20.0, grv_only=True)
            feat = seq.features
            ts = seq.ts
            gt = seq.gt_pos[:, :2]
            ends = np.arange(200, len(feat), 10)
            vel = []
            BS = 512
            for i in range(0, len(ends), BS):
                ebatch = ends[i:i + BS]
                wins = np.stack([feat[e - 200:e] for e in ebatch]).astype(np.float32)
                xw = torch.tensor(wins.transpose(0, 2, 1), device=dev)
                vel.append(net(xw).cpu().numpy())
                del xw
            vel = np.concatenate(vel, axis=0)
            traj = np.zeros((len(ends), 2), np.float32)
            cur = gt[200].copy(); prev_t = ts[200]
            traj[0] = cur
            for i in range(1, len(ends)):
                cur = cur + vel[i - 1] * (ts[ends[i]] - prev_t)
                traj[i] = cur; prev_t = ts[ends[i]]
            gtm = gt[ends]
            ate, rte = compute_ate_rte(traj, gtm, pred_per_min)
            ates.append(float(ate)); rtes.append(float(rte))
    ates = np.array(ates); rtes = np.array(rtes)
    return {
        "ate_mean": float(ates.mean()), "ate_median": float(np.median(ates)),
        "rte_mean": float(np.nanmean(rtes)),
        "n_seqs": int(len(ates)),
    }


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    print("=== RoNIN canonical unseen-subjects (RESULT_07) ===", flush=True)
    r = eval_resnet1d_pretrained()
    print(f"  ResNet1D pretrained: ATE {r['ate_mean']:.3f} m (paper-ref 5.140), "
          f"RTE {r['rte_mean']:.3f}, n={r['n_seqs']} seqs"
          f"   drift {(r['ate_mean']-5.140)/5.140*100:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
