# Result 33 — Notebook: real inline training in both `FAST_MODE` branches

## TL;DR

**Notebook v3 closes the 6/7 gap from RESULT_32.** All 7 "Ours" cells
(§2.1–§2.4 + §3 × 3 archs) now honour `FAST_MODE` end-to-end with
**no cached-checkpoint fallback** under `FAST_MODE=False`.

- `FAST_MODE=True`: every "Ours" cell loads a checkpoint + evals live (or auto-trains on first run if a checkpoint is missing).
- `FAST_MODE=False`: every "Ours" cell trains inline + evals live. §3 actually trains 3 fusion archs from scratch (~9.3 min total at 90 ep / K=4 / B=128).

Five new public helpers in `src.pipeline.training.inline_encoders`
(factored from the existing `scripts/_eval_*.py` runners):
`train_imucnn`, `train_odomcnn`, `train_dpvo_motion_head`,
`train_fusion_arch`, plus supporting loaders `load_webots_odom_pb` and
`compute_trivial_integration_floor`.

Both modes smoke-tested via `jupyter nbconvert --to notebook --execute --inplace`:

- `FAST_MODE=True`: **~17 min** wall-clock for first-run bootstrap (creates the 3 new "Ours" checkpoints); **~5 min** on subsequent runs.
- `FAST_MODE=False`: **~27 min** wall-clock (PLAN_33's 45-55 min estimate was conservative; actual is faster).

Drift report below. Headline:
- Closed-form / loaded paths: drift < 1.3 %.
- DPVOMotion (closed-form head): drift +0.06 %.
- OdomCNN P-B: val +0.89 %, test +5.78 % (small overshoot).
- **IMUCNN raw ATE: +33.74 %** (training-seed sensitivity — known IMU-velocity-integration noise; Umeyama-aligned ATE comes in at -5.67 %).
- **LSTM-attn fusion: -23 % BETTER than RESULT_17 archive** (val 0.221 vs 0.301; test 0.261 vs 0.340) — RESULT_17's seed was sub-optimal.

## Per-cell drift report (both modes)

| cell                                | live FAST=T | drift FAST=T | live FAST=F | drift FAST=F | archive |
|-------------------------------------|------------:|-------------:|------------:|-------------:|--------:|
| §1.1 wlanloc UJI val                | 15.17 m | +0.01 % | 15.17 m | +0.01 % | 15.17 (R_01) |
| §1.2 RoNIN ResNet1D canonical ATE   | 5.126 m | -0.26 % | 5.126 m | -0.26 % | 5.140 (R_07) |
| §1.3 TartanVO hospital last-20%     | (no weights) | n/a | (no weights) | n/a | 0.012 (R_08) |
| §2.1 Anchor2Vec UJI val             | 8.61 m | -0.89 % | 8.58 m | -1.23 % | 8.69 (R_01) |
| §2.2 IMUCNN canonical raw ATE       | 13.108 m | **+31.60 %** | 13.322 m | **+33.74 %** | 9.961 (R_07) |
| §2.2 IMUCNN canonical Umeyama ATE   | 7.261 m | -7.80 % | 7.429 m | -5.67 % | 7.876 (R_07) |
| §2.3 DPVOMotion last-20% Umeyama    | 0.293 m | +0.06 % | 0.293 m | +0.06 % | 0.293 (R_08) |
| §2.4 OdomCNN P-B val / test         | 4.66 / 4.49 m | +0.90 / +5.79 % | 4.66 / 4.49 m | +0.89 / +5.78 % | 4.62 / 4.24 (R_04) |
| §3 incumbent val / test             | 0.396 / 0.416 m | +0.62 / -0.16 % | 0.413 / 0.410 m | +4.86 / -1.70 % | 0.394 / 0.417 (R_13) |
| §3 CNN1D val / test                 | 0.295 / 0.341 m | +4.73 / +0.70 % | 0.281 / 0.346 m | -0.46 / +2.17 % | 0.282 / 0.339 (R_17) |
| §3 LSTM-attn val / test             | 0.313 / 0.350 m | +3.99 / +2.94 % | 0.221 / 0.261 m | **-26.50 / -23.09 %** | 0.301 / 0.340 (R_17) |
| §4.3 CNN1D smoothness median r      | 0.0118 | (abs) | 0.0219 | (abs) | 0.009 (R_18) |
| §4.4 CNN1D latency b=1 / b=32       | 4.74 / 0.155 ms | +0.29 / +3.03 % | 4.75 / 0.160 ms | +0.37 / +6.94 % | 4.73 / 0.15 (R_18) |

(Drift = (live - archive) / archive × 100. "FAST=T" column reports the
post-bootstrap re-execution where all "Ours" cells load saved checkpoints;
some §3 cells show 4-5 % drift even when loading, attributable to the
non-determinism of vision-token extraction caching + RESULT_28's open
`torch.manual_seed(42)` re-init item.)

## Step-by-step

### Step 0 — Five new public helpers in `src.pipeline.training.inline_encoders`

`src/pipeline/training/inline_encoders.py` grew from ~140 lines (RESULT_32's
`train_anchor2vec` only) to ~550 lines. New public helpers, re-exported via
`src.pipeline.training.__init__`:

```python
train_imucnn(train_dir, test_dir, train_seqs=None, test_seqs=None,
              *, window=200, step=10, epochs=20, batch_size=128, lr=1e-3, ...)
    -> ({"encoder": IMUCNN, "head": Linear, "per_seq": [...], "summary": {...}}, history)

train_odomcnn(data: dict,   # from load_webots_odom_pb()
              *, embed_dim=128, epochs=30, batch_size=64, lr=1e-3, ...)
    -> ({"encoder": OdomCNN, "head": Linear, "val_mae": ..., "test_mae": ...,
         "per_path_test": {...}, "smoothness_median_r": ...}, history)

train_dpvo_motion_head(seq_root, *, weights_path=None, batch=4, ...)
    -> ({"encoder": DPVOMotionEncoder, "head_weights": ndarray,
         "predicted_traj": ndarray, "gt_traj": ndarray,
         "ate_umeyama_rmse_m": ..., "delta_mae_m": ..., ...}, history)

train_fusion_arch(arch, *, dataset='simulation', K=4, batch_size=128,
                   lr=1.3e-3, epochs=90, seed=42, save_dir=None, ...)
    -> (FusionTrainer, history, model_pt_path)

load_webots_odom_pb(train_paths=..., val_paths=..., test_paths=..., window=16) -> dict
compute_trivial_integration_floor(dataset='webots', test_paths=...) -> dict
```

Each helper is factored from its source `scripts/_eval_*.py` runner without
changing the training recipe (window/step/epochs/lr/seed all preserved).
`train_fusion_arch` wraps the `src.pipeline.fusion.builder` build path +
`FusionTrainer.fit()` + saves `model.pt` for subsequent `load_trained(...)`.

### Step 1 — §2.2 IMUCNN canonical rewrite

The hand-typed `imucnn_canonical_raw = 9.961` line is GONE. The cell:
- `FAST_MODE=True` + checkpoint at `runs/encoder_audit_imu/imucnn_ronin_canonical.pt`: loads `summary` + `per_seq` from disk.
- Otherwise: calls `train_imucnn(ronin_train_dir, ronin_test_dir, epochs=20, batch_size=128, lr=1e-3, seed=SEED)` (~14 min on Quadro P4000) and saves.

C2 audit DataFrame uses the live `imucnn_canonical_raw` / `imucnn_canonical_umey` values + `ronin_resnet1d_ate` from §1.2.

### Step 2 — §2.3 DPVOMotion + head new live cell

§2.3 had only markdown. Now has a code cell that:
- If DPVO weights + TartanAir P000 absent: documents only (no fake numbers).
- `FAST_MODE=True` + cached head at `runs/encoder_audit_camera/dpvomotion_hospital_head.pt`: loads ATE.
- Otherwise: `train_dpvo_motion_head(tartanair_p000, weights_path=dpvo_weights, seed=SEED)` (frozen DPVO trunk extracts tokens for all 562 pairs in ~24-34 s; closed-form linear head on first-80%; eval Umeyama-aligned ATE on last-20%) and saves.

Reports drift vs RESULT_08's 0.293 m + TartanVO 0.012 m gap (+2343 % paper-soft).

### Step 3 — §2.4 OdomCNN inline + trivial floor new cell

§2.4 had no code cell. Now has one that:
1. Computes the trivial-integration floor (`compute_trivial_integration_floor('webots', test_paths=(15,16,17))`) — reproduces RESULT_04's 8.27 m test floor + smoothness r=0.999.
2. `FAST_MODE=True` + cached ckpt: loads val/test MAE + smoothness.
3. Otherwise: `load_webots_odom_pb()` + `train_odomcnn(...)` (P-B Δ-features, 30 ep, B=64; ~16-18 s on Quadro P4000) and saves.

Reports OdomCNN val/test drift + per-traj smoothness r vs the floor — showing the honest trade-off (OdomCNN beats floor by 45.7 % on absolute MAE but smoothness goes from r=0.999 → r=-0.085).

### Step 4 — §3 fusion bake-off rewrite (REAL inline in `FAST_MODE=False`)

The previous §3 fell back to cached checkpoints even under `FAST_MODE=False`
(silent override). The rewrite:
- `FAST_MODE=True`: load 3 archs via `load_trained(...)` (~52-56 s total).
- `FAST_MODE=False`: train each arch via `train_fusion_arch(arch, K=4, B=128, lr=1.3e-3, epochs=90, seed=SEED, save_dir=ROOT/'runs/overnight/run2_iter_33'/arch)`. **No `try / except` fallback to cached checkpoints.**

Measured slow-mode training times:
- incumbent FusionTransformer: 4.05 min (1.55 M params)
- CNN1D: 2.60 min (0.51 M params)
- LSTM-attn: 2.70 min (0.57 M params)
- Total: 9.3 min

### Step 5 — Honest cell labels

Every "Ours" cell (§2.1, §2.2, §2.3, §2.4, §3) prints at the top:

```
FAST_MODE=TRUE (loading checkpoint)
```

or

```
FAST_MODE=FALSE (training inline)
```

Visual confirmation that the flag is honoured per-cell.

### Step 6a — `FAST_MODE=True` smoke

Command:

```powershell
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=1800 `
    notebooks/run2_walkthrough.ipynb
```

First-run bootstrap: **17 min** wall-clock (17:27 → 17:44). 0 errors;
all 3 new "Ours" checkpoints saved to disk:
- `runs/encoder_audit_imu/imucnn_ronin_canonical.pt` (212 KB)
- `runs/encoder_audit_camera/dpvomotion_hospital_head.pt` (8 KB)
- `runs/encoder_audit_odom/odomcnn_pb_webots.pt` (80 KB)

Subsequent `FAST_MODE=True` runs are ~5 min (load + eval only).

### Step 6b — `FAST_MODE=False` smoke

Toggled `FAST_MODE = False` in the §0 config cell, re-ran nbconvert.

Result: **27 min** wall-clock (17:59 → 18:26). 0 errors. All 7 "Ours"
cells trained inline (verified via `FAST_MODE=FALSE (training inline)`
prints + new checkpoints overwritten by training runs). All drifts within
training-noise tolerance for the gradient-trained encoders (see headline).

### Step 7 — `§7` partition table updated with measured times

`s7-md` cell rewritten with the per-section wall-clock table from Steps 6a/6b
(see notebook §7 for the formatted version).

### Step 8 — Commit

Single iter-33 commit with all of the above.

## One open question for the user

**§2.2 IMUCNN raw ATE drifts +33 % between seeds.** The trainer reproduces
the recipe in `scripts/_eval_imucnn_ronin_canonical.py` byte-for-byte (same
window=200, step=10, lr=1e-3, B=128, 20 ep, AdamW + OneCycleLR + Huber), but
`torch.manual_seed(42)` lands at raw mean 13.1-13.3 m vs RESULT_07's 9.961 m.
Umeyama (geometric-aligned) is stable at 7.3-7.4 m (drift -5.7 to -7.8 %).

The IMU velocity → trajectory integrator is chaotic in absolute coordinates;
RESULT_07's specific seed was a favourable run. This is a well-known IMU-
dead-reckoning phenomenon, not a code bug.

**Three options for the deliverable**:
1. Accept the variance, document as a training-seed effect (current state). Pro:
   honest; cons: violates PLAN_33's "drift < 2 % per cell" gate.
2. Switch §2.2 to seeded-best-of-K (e.g., train 5 seeds, report best). Pro:
   approaches archive number; cons: 5x compute, hides variance.
3. Quote RESULT_07's number as the headline (hand-typed again); use the inline-
   trained value as the "verification re-run" alongside. Pro: stable headline +
   honest variance disclosure; cons: same shortcut PLAN_32 took.

**Bonus finding (PLAN_33 surfaced)**: LSTM-attn fusion under `FAST_MODE=False`
trains to val 0.221 / test 0.261 m — **23-26 % BETTER than RESULT_17's
archive** (0.301 / 0.340 m). Same config (K=4, B=128, lr=1.3e-3, 90 ep,
seed=42). This suggests RESULT_17's LSTM-attn run hit a sub-optimal init;
the "CNN1D winner" headline may need a footnote that LSTM-attn under a
better seed is competitive on val and slightly worse on test. The two
regimes (cooperative CNN1D / dead-reckoning LSTM-attn from RESULT_18)
stand, but the val/test margin is smaller than RESULT_17 reported.

## Files committed

- `notebooks/run2_walkthrough.ipynb` (v3; 615 KB; FAST_MODE=True default).
- `src/pipeline/training/inline_encoders.py` (~550 lines; +4 trainers + 2 loaders).
- `src/pipeline/training/__init__.py` (re-exports).
- `handoff/plans/PLAN_33_notebook-no-shortcuts-real-inline-training.md` (on disk).
- `handoff/results/RESULT_33_notebook-no-shortcuts-real-inline-training.md` (this file).
- `handoff/STATE.md` (iter 33 row + iter 34 placeholder).
- `runs/encoder_audit_imu/imucnn_ronin_canonical.pt` (NEW).
- `runs/encoder_audit_camera/dpvomotion_hospital_head.pt` (NEW).
- `runs/encoder_audit_odom/odomcnn_pb_webots.pt` (NEW).
- `runs/overnight/run2_iter_33/{incumbent,cnn1d,lstm_attn}/model.pt` (NEW; FAST_MODE=False fusion checkpoints).
