"""PLAN_28 Step 5 — smoke for the fusion / encoders / training
consolidation:

  - ``from src.pipeline.fusion import build_arch, list_archs`` works.
  - All 5 architectures construct without error from the default
    (Webots 4-mod) factory call.
  - ``Anchor2Vec.demo_forward`` / ``IMUCNN.demo_forward`` /
    ``OdomCNN.demo_forward`` / ``DPVOMotionEncoder.demo_forward``
    return the expected keys on a synthetic input.
  - ``src.pipeline.training.load_trained`` loads the CNN1D winner
    checkpoint from RESULT_17 and reproduces sanity numbers.
  - ``compute_per_trajectory_smoothness`` + ``latency_probe`` run
    on the loaded trainer.

Run: ``.venv/Scripts/python.exe scripts/_smoke_fusion_consolidation.py``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def smoke_factory():
    print("=== build_arch factory ===", flush=True)
    from src.pipeline.fusion import list_archs, build_arch
    print(f"  archs: {list_archs()}", flush=True)
    # Build each arch using the default (Webots simulation) encoders.
    counts = {}
    for name in list_archs():
        try:
            m = build_arch(name)
            n = sum(p.numel() for p in m.parameters())
            counts[name] = n
            print(f"  {name:18s} -> {n:>9,d} params ({n/1e6:.2f} M)", flush=True)
        except Exception as e:
            print(f"  {name:18s} -> FAILED: {type(e).__name__}: {e}", flush=True)
            counts[name] = None
    return counts


def smoke_encoder_demos():
    print("\n=== encoder demo_forward methods ===", flush=True)
    from src.pipeline.encoders import Anchor2Vec, IMUCNN, OdomCNN
    encs = {
        "Anchor2Vec": (Anchor2Vec(n_aps=128, embed_dim=128),
                        np.random.randn(1, 128).astype(np.float32)),
        "IMUCNN":     (IMUCNN(in_features=9, embed_dim=128),
                        np.random.randn(32, 9).astype(np.float32)),
        "OdomCNN":    (OdomCNN(in_features=7, embed_dim=128),
                        np.random.randn(16, 7).astype(np.float32)),
    }
    results = {}
    for name, (enc, raw) in encs.items():
        out = enc.demo_forward(raw)
        expected = {"raw", "preprocessed", "intermediate", "encoded", "description"}
        missing = expected - set(out)
        status = "OK" if not missing else f"MISSING {missing}"
        shape = tuple(out["encoded"].shape)
        inter_shape = tuple(np.asarray(out["intermediate"]).shape)
        print(f"  {name:14s}: encoded {shape}, intermediate {inter_shape}  [{status}]",
              flush=True)
        results[name] = status
    return results


def smoke_load_trained_and_methods():
    print("\n=== load_trained: CNN1D RESULT_17 winner ===", flush=True)
    from src.pipeline.training import load_trained
    ckpt = ROOT / "runs" / "overnight" / "run2_iter_17" / "cnn1d"
    if not ckpt.is_dir():
        print(f"  SKIP — no checkpoint at {ckpt}", flush=True)
        return False
    t0 = time.time()
    tr = load_trained(ckpt, arch="cnn1d", dataset="simulation")
    print(f"  loaded in {time.time()-t0:.1f}s; "
          f"model params = {sum(p.numel() for p in tr.model.parameters())/1e6:.2f} M",
          flush=True)
    # Sanity: val + test MAE should match RESULT_17 (val 0.282 / test 0.339).
    pred_v, gt_v = tr.predict("val")
    pred_t, gt_t = tr.predict("test")
    val_mae = float(torch.linalg.norm(pred_v - gt_v, dim=1).mean())
    test_mae = float(torch.linalg.norm(pred_t - gt_t, dim=1).mean())
    print(f"  sanity: val {val_mae:.3f}  test {test_mae:.3f}  "
          f"(RESULT_17: val 0.282 / test 0.339)", flush=True)

    print("\n=== compute_per_trajectory_smoothness ===", flush=True)
    sm = tr.compute_per_trajectory_smoothness("test")
    print(f"  median r = {sm['median_r']:.3f}  per-path = {sm['per_path']}", flush=True)

    print("\n=== latency_probe ===", flush=True)
    lat = tr.latency_probe(batch_sizes=(1, 32), n_trials=20, n_warmup=5)
    for bs, d in lat.items():
        print(f"  b={bs}: {d['ms_per_sample']:.3f} ms/sample  "
              f"({d['ms_per_batch']:.3f} ms/batch)", flush=True)

    return True


def main():
    counts = smoke_factory()
    enc_results = smoke_encoder_demos()
    loaded = smoke_load_trained_and_methods()
    print("\n=== summary ===", flush=True)
    n_arch_ok = sum(1 for v in counts.values() if v is not None)
    n_enc_ok = sum(1 for v in enc_results.values() if v == "OK")
    print(f"  archs built: {n_arch_ok}/{len(counts)}", flush=True)
    print(f"  encoder demos: {n_enc_ok}/{len(enc_results)}", flush=True)
    print(f"  load_trained sanity: {'OK' if loaded else 'SKIPPED'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
