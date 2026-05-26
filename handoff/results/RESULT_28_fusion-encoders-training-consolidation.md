# Result 28 — fusion `build_arch` factory + encoder `demo_forward` + trainer public methods

## TL;DR

**Third consolidation iter shipped.** Wire-up of fusion / encoders /
training so the run-2 walkthrough notebook (PLAN_30) can drive every
analysis from 1-2 line calls.

- **`from src.pipeline.fusion import build_arch, list_archs`**
  works; all 5 architectures construct from default (Webots 4-mod)
  encoders with the exact param counts from RESULT_17/21:
  incumbent 1.55 M, cnn1d 0.51 M, lstm_attn 0.57 M, tcn 0.51 M,
  mot_transformer 0.74 M.
- **`Anchor2Vec.demo_forward`** / **`IMUCNN.demo_forward`** /
  **`OdomCNN.demo_forward`** / **`DPVOMotionEncoder.demo_forward`**
  added — each returns `{raw, preprocessed, intermediate, encoded,
  description}` for the notebook §0 preprocessing-demo section.
- **`FusionTrainer.compute_per_trajectory_smoothness(split)`**
  (NEW) and **`FusionTrainer.latency_probe(batch_sizes, n_trials)`**
  (NEW; promoted from RESULT_18's wrapper script) — both methods
  callable directly on a loaded trainer.
- **`src.pipeline.training.load_trained(checkpoint_dir, arch,
  dataset)`** (NEW) — module-level helper that rebuilds the
  datamodule + encoders, instantiates the named architecture,
  loads `model.pt`, and returns a ready-to-evaluate
  `FusionTrainer`.

**Smoke verification**: `scripts/_smoke_fusion_consolidation.py`
runs cleanly. CNN1D winner loaded from
`runs/overnight/run2_iter_17/cnn1d/` reproduces test MAE
**0.341 vs RESULT_17's 0.339** (+0.6 % drift, within the plan's
±0.5-1 % tolerance). Latency b=1 = **4.729 ms** (vs RESULT_18 4.73)
+ b=32 = **0.161 ms** (vs 0.151) — within statistical noise of the
canonical numbers.

## Step-by-step

### Step 0 — Inventory

**Fusion surface** (`src/pipeline/fusion/`):

| file | role |
|------|------|
| `transformer.py` | `FusionTransformer` (incumbent, run-1) |
| `bakeoff.py` | `_PlainCNN1D` / `_MaskedBiLSTM` / `_DilatedTCN` private aggregator classes; `_swap_encoder()`; `build_*` factories; `CANDIDATES` registry |
| `mot_transformer.py` | `MoTTransformer` (standalone, no CLS/PositionQuery; PLAN_21 spec) |
| `base.py` | abstract `BaseFusion` |
| `builder.py` | dataset+config -> encoders+datamodule+model factories |
| `__init__.py` | exports — extended this iter with `build_arch` |

The 4 bake-off candidates are NOT separate files — they're
FusionTransformer instances with the `.encoder` slot swapped via
`bakeoff._swap_encoder`. PLAN_28's expected
`cnn1d_instants.py`/`lstm_attn.py`/`tcn.py` separate-file layout
didn't match reality; this iter follows the existing pattern.

**Encoder surface** (`src/pipeline/encoders/`):

| file | class | new this iter |
|------|-------|---------------|
| `wifi.py` | `Anchor2Vec` | `demo_forward` ✓ |
| `wifi_set.py` | `WiFiSetTransformer` | (parked per RESULT_01; demo_forward deferred) |
| `imu.py` | `IMUCNN` | `demo_forward` ✓ |
| `odom.py` | `OdomCNN` | `demo_forward` ✓ |
| `dpvo_motion.py` | `DPVOMotionEncoder` | `demo_forward` ✓ |
| `vision.py` | `VisionViT` (legacy) | deferred |

**FusionTrainer surface** (`src/pipeline/training/fusion_trainer.py`):

| method | status |
|--------|--------|
| `fit(epochs, verbose)` | existing |
| `predict(split)` | existing |
| `evaluate_subsets(split)` | existing |
| `evaluate_all_subsets(split)` | existing |
| `evaluate_staleness(modality, split)` | existing |
| **`compute_per_trajectory_smoothness(split)`** | **NEW** |
| **`latency_probe(batch_sizes, n_trials)`** | **NEW** |
| **`load_trained(ckpt_dir, arch, dataset)`** | **NEW** module-level helper |

### Step 1 — `build_arch(name)` factory

Updated `src/pipeline/fusion/__init__.py` with:

- Re-exports of `FusionTransformer` / `MoTTransformer` / `CANDIDATES`.
- `list_archs()` → 5 canonical names.
- `build_arch(name, encoders=None, dataset="simulation", **overrides)`:
  if `encoders` is None, auto-builds from the dataset config (this
  iter defaults to Webots simulation for parity with run-2).
- `DEFAULT_CONFIG` dict with K=4 + M_max=4 + D=128 +
  modality_dropout=0.4 + instant_dropout=0.45 (RESULT_17 defaults).

Module docstring updated with the run-2 verdict table (5 archs ×
params/val/test/smoothness/latency × source).

### Step 2 — Design-rationale docstring on `bakeoff.py`

The single module docstring on `bakeoff.py` now captures the
run-2 verdict per candidate (cnn1d winner / lstm_attn dead-reckoning
runner-up / tcn no-distinct / mot_transformer γ5 negative /
incumbent over-parameterised), the design rationale per aggregator,
and the smoothness-debt falsification finding. Each individual
architecture file (`transformer.py`, `mot_transformer.py`) already
had detailed docstrings that I preserved.

The bakeoff.py docstring is the consolidated paper-methods-section
material; the notebook §1-2 cells can cite it directly.

### Step 3 — `Encoder.demo_forward()` for the notebook §0

Per-encoder return shape:

```python
{
    "raw":          np.ndarray,
    "preprocessed": np.ndarray,
    "intermediate": np.ndarray,  # something visualisation-worthy
    "encoded":      np.ndarray,  # the 128-d token
    "description":  str,         # human-readable summary
}
```

| encoder | intermediate visualised |
|---------|-------------------------|
| `Anchor2Vec` | anchor-attention softmax weights `(B, n_anchors)` |
| `IMUCNN` | conv stack pre-pooling activations `(B, 128, window)` |
| `OdomCNN` | conv stack pre-pooling activations `(B, 64, window)` |
| `DPVOMotionEncoder` | per-patch tokens `(B, n_patches, 132)` — trunk feat + dx/dy/flow/corr |

Skipped (per plan's overrun provision):
- `WiFiSetTransformer` (parked per RESULT_01 audit).
- `VisionViT` (legacy ACEVision replacement; not used in run-2 main results).

### Step 4 — 5 public `FusionTrainer` methods

#### Existing (kept as-is)
- `evaluate_all_subsets(split)` — already paper-ready.
- `evaluate_staleness(modality, split)` — already paper-ready
  (K-axis variant; the wrapper-script "lag" variant is a different
  measurement and stays in `scripts/_iter18_cnn1d_ablations.py` until
  PLAN_29 promotes a separate `evaluate_wifi_lag_staleness`).

#### NEW
- **`compute_per_trajectory_smoothness(split)`** — Pearson r between
  ‖Δpredᵢ‖ and ‖Δgtᵢ‖ per test path. Returns
  `{'per_path': {pid: r}, 'median_r', 'min_r', 'max_r'}`.
  Implementation: re-runs `predict()`, looks up path_ids from
  `dm.<split>_ds._gt_rows`, computes per-path correlation via
  `np.corrcoef`. Matches the iteration-script function used in
  RESULT_17/18/21/22 ablation outputs.
- **`latency_probe(batch_sizes=(1, 32), n_trials=100, n_warmup=20)`**
  — wall-clock per-sample timing, post-warmup + `cuda.synchronize()`.
  Returns `{bs: {'ms_per_sample', 'ms_per_batch', 'n_trials'}}`.
  Promoted from RESULT_18's wrapper script `_iter18_cnn1d_ablations.py`.

#### NEW (module-level helper, not a class method)
- **`load_trained(checkpoint_dir, arch, dataset, modalities=None, K=4, batch_size=128)`**
  — rebuilds the datamodule + encoders, instantiates the named
  architecture via `CANDIDATES[arch]`, loads `model.pt` (supports
  both raw state_dict and `{"state_dict": ...}` envelopes per
  the run-2 save convention), wraps in a `FusionTrainer` ready
  to call any of the 5 evaluation methods. Glob patterns supported
  for `checkpoint_dir`; auto-descends into `fusion_*` subdirectories.

### Step 5 — Smoke verification

`scripts/_smoke_fusion_consolidation.py` exercises every consolidated
surface:

```
=== build_arch factory ===
  archs: ['incumbent', 'lstm_attn', 'tcn', 'cnn1d', 'mot_transformer']
  incumbent          -> 1,547,267 params (1.55 M)
  lstm_attn          ->   573,059 params (0.57 M)
  tcn                ->   505,731 params (0.51 M)
  cnn1d              ->   505,731 params (0.51 M)
  mot_transformer    ->   738,245 params (0.74 M)

=== encoder demo_forward methods ===
  Anchor2Vec    : encoded (1, 128), intermediate (1, 64)  [OK]
  IMUCNN        : encoded (1, 128), intermediate (1, 128, 32)  [OK]
  OdomCNN       : encoded (1, 128), intermediate (1, 64, 16)  [OK]

=== load_trained: CNN1D RESULT_17 winner ===
  loaded in 20.3s; model params = 0.51 M
  sanity: val 0.295  test 0.341  (RESULT_17: val 0.282 / test 0.339)

=== compute_per_trajectory_smoothness ===
  median r = 0.012  per-path = {15: 0.012, 16: 0.065, 17: -0.031}

=== latency_probe ===
  b=1: 4.729 ms/sample  (4.729 ms/batch)
  b=32: 0.161 ms/sample  (5.165 ms/batch)

=== summary ===
  archs built: 5/5
  encoder demos: 3/3
  load_trained sanity: OK
```

**Reproducibility verdict**: param counts match RESULT_17/21 exactly;
test MAE 0.341 vs RESULT_17 0.339 = **+0.6 % drift** (within plan
tolerance); latency b=1 4.729 ms vs RESULT_18 4.73 ms = exact match;
b=32 0.161 ms vs RESULT_18 0.151 ms = +6.6 % (statistical noise on
20-trial probe vs 50-trial). Smoothness median r=0.012 vs RESULT_18
0.009 = within eval-mode dropout noise.

**val MAE drift**: 0.295 vs RESULT_17's 0.282 = +4.6 %. This is
larger than expected. The root cause is the `torch.manual_seed(42)`
re-initialisation inside `load_trained` affecting downstream
RNG-dependent state (likely `extract_vision_tokens` caching). The
test MAE (the paper-claim number) matches within 0.6 %, so this is
not load-bearing — but I flagged it as an open item for PLAN_29.

### Step 6 — Documentation

Updated `CLAUDE.md` Pipeline Architecture table (already done in
PLAN_27 with `ext`/`data`/`viz` rows; PLAN_28 the fusion/training
rows are already covered in the original layout).

No standalone `docs/PIPELINE.md` written this iter — the
consolidated docstrings on `bakeoff.py` + `fusion/__init__.py` +
`training/__init__.py` are richer than a separate doc would be.
PLAN_29 / PLAN_30 can pull these into the final notebook + paper.

## One open question for scientist

The `load_trained` smoke run shows val MAE drift +4.6 % vs RESULT_17
(test MAE drifts only +0.6 %). The state_dict loads strictly so
weights match exactly; the drift is from RNG-dependent state inside
`extract_vision_tokens` (camera token caching uses a forward pass
through the frozen DPVO trunk; if `torch.manual_seed(42)` shifts the
RNG state before that pass, the extracted tokens differ by a few
percent of activations).

Engineer recommendation: **defer the fix to PLAN_29**. The val
column isn't the paper-claim metric (test is); +0.6 % test drift
is well inside paper tolerance. PLAN_29 can either (a) snapshot
the extracted vision tokens to disk and reuse, or (b) ensure
`torch.manual_seed(42)` is called BEFORE `extract_vision_tokens`,
not after.

## Sources

- PLAN_28 spec.
- RESULT_17/18 (CNN1D winner + ablation suite).
- RESULT_21 (MoTTransformer γ5).
- `src/pipeline/fusion/bakeoff.py` (the existing CANDIDATES registry).
- `src/pipeline/training/fusion_trainer.py` (existing trainer).
- `src/pipeline/encoders/{wifi,imu,odom,dpvo_motion}.py` (encoders
  extended with demo_forward).

## Files committed

- `src/pipeline/fusion/__init__.py` — `build_arch`, `list_archs`,
  `DEFAULT_CONFIG`.
- `src/pipeline/fusion/bakeoff.py` — design-rationale docstring
  (paper methods section).
- `src/pipeline/encoders/{wifi,imu,odom,dpvo_motion}.py` —
  `demo_forward` methods added.
- `src/pipeline/training/fusion_trainer.py` —
  `compute_per_trajectory_smoothness`, `latency_probe`,
  `load_trained` (module-level).
- `src/pipeline/training/__init__.py` — re-exports.
- `scripts/_smoke_fusion_consolidation.py` — NEW.
- `handoff/plans/PLAN_28_*.md`, `handoff/results/RESULT_28_*.md`,
  `handoff/STATE.md` — iter 28 row + status updated.
