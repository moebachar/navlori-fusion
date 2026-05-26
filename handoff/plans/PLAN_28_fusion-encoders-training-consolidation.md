# Plan 28 — `src/pipeline/fusion/` build_arch factory + `encoders/.demo_forward()` + `training/` public methods

> Third consolidation iter. After PLAN_26 (external_methods +
> baselines) and PLAN_27 (data + visualization), this iteration
> wires up the architecture + encoder + trainer surfaces the
> notebook + canonical eval scripts will call. Goal: every
> training / evaluation / encoder-inspection call from the
> notebook is a 1-2 line invocation into the pipeline.

## Hypothesis

Today the fusion architectures live under
`src/pipeline/fusion/{transformer,cnn1d_instants,lstm_attn,tcn,
mot_transformer}.py` with a dispatch registry in `bakeoff.py`.
The encoders are clean files but have no notebook-friendly
introspection (showing raw → encoded for a single sample).
The `FusionTrainer` has private-feeling methods sprawled across
the file.

After this iter:
- **`from src.pipeline.fusion import build_arch, list_archs`**
  works; `build_arch("cnn1d")` returns a ready-to-train module
  with the documented config (K=4 + 4-mod + B=128 defaults).
- Each architecture file has a paper-citation-ready
  **design-rationale docstring** capturing what was learned in
  run-2 (CNN1D wins, LSTM-attn dead-reckoning, MoTTransformer
  honest negative, incumbent over-parameterised, TCN no
  distinct finding).
- **Each encoder exposes `.demo_forward(raw_input)`** returning
  `{raw, intermediate, encoded}` for the notebook §0
  preprocessing-demo section.
- **`FusionTrainer`** exposes 5 public methods that the notebook
  + canonical eval scripts call directly:
  `evaluate_all_subsets`, `evaluate_staleness`,
  `compute_per_trajectory_smoothness`, `latency_probe`,
  `load_trained(checkpoint_dir)`.
- Smoke test: all 4 architectures build from
  `build_arch(name)`; all 4 evaluation methods run on the
  loaded CNN1D winner checkpoint.

## Steps

### Step 0 — Inventory current fusion/encoder/training surface (10 min)

For each fusion arch file, identify:
- The current build entry point (a class? a function? a factory call?)
- The current init signature (Hydra config vs explicit kwargs)
- Whether any custom ALiBi / temporal bias code needs special handling

For each encoder, identify:
- Forward signature (does it accept raw modality input or already-windowed?)
- Whether it has intermediate states worth surfacing in the demo

For `FusionTrainer`:
- Catalogue every public-ish method already present
- Identify private helpers that should be promoted

**Acceptance**: 3 inventory tables in the engineer's RESULT_28 TL;DR.

### Step 1 — `src/pipeline/fusion/__init__.py` + `build_arch(name)` (15 min)

```python
# src/pipeline/fusion/__init__.py
"""4 fusion architectures + 1 incumbent benchmarked in run-2.

Run-2 verdicts (see handoff/SUMMARY.md):
- cnn1d: Phase B winner — val 0.282 / test 0.339 on Webots
  (RESULT_17/18). 0.51 M params, latency b=1 4.73 ms.
- lstm_attn: runner-up with dead-reckoning fusion regime
  confirmed on 3 datasets (RESULT_18/19/22). 0.57 M params.
- mot_transformer: scientist-designed transformer-from-scratch
  with ALiBi temporal bias. Outcome γ5 — WORST of 4 (RESULT_21).
  0.74 M params. Honest negative result; kept for methods-section
  "we benchmarked 4 architectures" claim.
- tcn: bake-off candidate; no distinct finding. ~0.51 M params.
- incumbent (transformer.py): run-1's design. Over-parameterised
  at 1.55 M. Kept as historical baseline.
"""
from .transformer import FusionTransformer
from .cnn1d_instants import CNN1DFusion
from .lstm_attn import LSTMAttnFusion
from .tcn import TCNFusion
from .mot_transformer import MoTTransformer

_REGISTRY = {
    "incumbent":       FusionTransformer,
    "cnn1d":           CNN1DFusion,
    "lstm_attn":       LSTMAttnFusion,
    "tcn":             TCNFusion,
    "mot_transformer": MoTTransformer,
}

DEFAULT_CONFIG = {
    "K": 4,
    "M_max": 4,            # max modalities (wifi+imu+camera+odom)
    "D": 128,
    "modality_dropout": 0.4,
    "instant_dropout": 0.45,
}

def list_archs() -> list[str]:
    return list(_REGISTRY)

def build_arch(name: str, **kwargs):
    """Construct the named fusion module. Defaults from
    DEFAULT_CONFIG; kwargs override per-arch."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown arch {name!r}. Available: {list_archs()}")
    cfg = {**DEFAULT_CONFIG, **kwargs}
    return _REGISTRY[name](**_filter_kwargs(_REGISTRY[name], cfg))

def _filter_kwargs(cls, cfg):
    """Pass only the kwargs the class accepts (each arch has
    slightly different signature)."""
    import inspect
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in cfg.items() if k in sig.parameters}
```

If `bakeoff.py`'s `CANDIDATES` registry already exists from
RESULT_16, this consolidates / re-exports it (engineer's call —
keep bakeoff.py for backwards compat or remove). The unified
entry point is `build_arch`.

**Acceptance**: `from src.pipeline.fusion import build_arch,
list_archs; print(list_archs()); m = build_arch("cnn1d")` works
without error; param count matches RESULT_17.

### Step 2 — Design-rationale docstrings on each arch file (10 min)

For each of the 5 architecture modules, write/update the module
docstring with run-2 evidence-backed design notes. Template:

```python
"""<Arch name> — <one-liner>

Run-2 verdict: <keep|runner-up|negative-result|baseline>

Design rationale:
  - <choice 1 from arch>: <why, traced to RESULT_NN evidence>
  - <choice 2>: ...

Run-2 measurements (Webots 4-mod canonical split):
  | metric           | value      | source     |
  |------------------|------------|------------|
  | params           | …          | RESULT_NN  |
  | val MAE          | …          | RESULT_NN  |
  | test MAE         | …          | RESULT_NN  |
  | latency b=1 (ms) | …          | RESULT_NN  |
  | smoothness r     | …          | RESULT_NN  |

Citation: <if any prior art, e.g. ALiBi for MoTTransformer>

Honest limitations:
  - <e.g. CNN1D smoothness r=0.009 (architecture-invariant debt)>
"""
```

Each arch gets the same template filled with its actual
RESULT_NN numbers.

**Acceptance**: each fusion module's docstring is paper-citation-
ready; no missing numbers; the engineer can rg "Run-2 verdict"
and find one per file.

### Step 3 — `Encoder.demo_forward()` for the notebook §0 (15 min)

Each encoder file under `src/pipeline/encoders/` gets a
`demo_forward(self, raw_input)` method. Returns:

```python
{
    "raw":           <the input as received>,
    "preprocessed":  <after the encoder's internal preprocessing>,
    "intermediate":  <one or more intermediate activations worth
                      visualising — e.g. anchor-attention for
                      Anchor2Vec, per-channel rotation for IMUCNN,
                      patch tokens for DPVOMotionEncoder>,
    "encoded":       <the final 128-d embedding>,
    "description":   "...",
}
```

This is what the notebook §0 `preprocessing_demo(name, modality)`
chain calls into. The visualization layer (`plot_preprocessing_demo`)
already exists from RESULT_27.

Per encoder:
- `Anchor2Vec.demo_forward(rssi_vector)`: raw RSSI → normalised
  RSSI → anchor attention weights → encoded vector.
- `WiFiSetTransformer.demo_forward(rssi_vector)`: raw → sparse
  attention map → encoded.
- `IMUCNN.demo_forward(imu_window)`: raw 6-channel device →
  conv block activations → encoded.
- `OdomCNN.demo_forward(odom_window)`: raw 7-column → conv
  activations → encoded.
- `DPVOMotionEncoder.demo_forward(image_pair)`: raw 2 RGB frames
  → DPVO patch features → correlation map → encoded.

The library code, not the notebook, owns the modality-specific
visualization data.

**Acceptance**: each encoder's `demo_forward` runs on a single
synthetic sample without error; returns a dict with all 5 keys.

### Step 4 — `FusionTrainer` public methods (20 min)

Promote / consolidate these methods to clean public-API form:

#### `evaluate_all_subsets(self, split="val") -> dict[str, float]`
- 2^M - 1 subset combinations + full → MAE per combination
- Already exists in some form per RESULT_18; clean signature.

#### `evaluate_staleness(self, lags: list[int], modality="wifi", split="test") -> dict`
- For each lag, replace the per-instant modality token with the
  one `lag` instants earlier; report test MAE per lag.
- Returns `{lag: mae}` dict; optionally also a `slope` (linear fit
  m/s).
- Already exists per RESULT_11/14; clean signature.

#### `compute_per_trajectory_smoothness(self, split="test") -> dict[int, float]`
- For each test path, compute median Pearson r between
  ‖Δpredᵢ‖ and ‖Δgtᵢ‖.
- Returns `{path_id: r}`; aggregate median exposed as
  `r.values() → np.median`.

#### `latency_probe(self, batch_sizes=[1, 32], n_trials=100) -> dict`
- Per-batch-size wall-clock per sample; median of n_trials.
- Returns `{batch_size: latency_ms_per_sample}`.

#### `load_trained(checkpoint_dir: Path) -> "FusionTrainer"`
- Class method (or module-level). Loads a checkpoint produced
  by any run-2 iter; restores model + config; ready for any
  of the above methods.
- This is THE notebook's primary entry point: a 1-liner that
  reads `runs/overnight/run2_iter_17/cnn1d/` (the winner) and
  returns a trainer object ready to evaluate.

**Acceptance**: all 5 methods callable on the loaded CNN1D
winner checkpoint; numbers match RESULT_18 (the canonical
ablation suite) within ±0.5 % (eval-time dropout noise).

### Step 5 — Smoke verification (10 min)

Engineer writes `scripts/_smoke_fusion_consolidation.py`:

```python
from src.pipeline.fusion import build_arch, list_archs
from src.pipeline.training import FusionTrainer

# Build each arch from scratch — sanity that the factory works
for arch_name in list_archs():
    model = build_arch(arch_name)
    print(f"{arch_name}: {sum(p.numel() for p in model.parameters())} params")

# Load the CNN1D winner from RESULT_17
trainer = FusionTrainer.load_trained("runs/overnight/run2_iter_17/cnn1d")

# Run each public method; check shapes/types
subsets = trainer.evaluate_all_subsets("test")
staleness = trainer.evaluate_staleness([0, 5, 10, 30])
smoothness = trainer.compute_per_trajectory_smoothness("test")
latency = trainer.latency_probe()

print("All 5 public methods work on the winner checkpoint")
```

**Acceptance**: smoke runs in ≤ 60 s; all 5 architectures build;
all 4 evaluation methods produce non-empty results.

If any RESULT_18 number diverges by > 1 %, debug — but a couple
of tenths-of-percent is acceptable (eval-mode dropout, RNG state).

### Step 6 — Documentation (5 min)

Update `docs/PIPELINE.md` (or similar) with:
- New `build_arch(name)` API + DEFAULT_CONFIG table.
- 5 public `FusionTrainer` methods + signatures.
- Each encoder's `demo_forward` contract.
- Cross-link to `handoff/SUMMARY.md` for run-2 evidence.

## Sources

- `handoff/SUMMARY.md` (run-2 final findings).
- `handoff/results/RESULT_17_*.md` (Phase B winner declaration).
- `handoff/results/RESULT_18_*.md` (canonical ablation suite —
  the verification target for Step 5 numbers).
- `handoff/results/RESULT_21_*.md` (MoTTransformer γ5 honest negative).
- `src/pipeline/fusion/*.py` (existing arch implementations).
- `src/pipeline/encoders/*.py` (existing encoder implementations).
- `src/pipeline/training/fusion_trainer.py` (existing trainer).
- RESULT_26 `src/pipeline/baselines/` (the loader-package pattern;
  apply the same shape: thin uniform API + dispatcher + smoke).
- RESULT_27 `src/pipeline/data/` (the per-dataset module pattern).

## What to report back

In `handoff/results/RESULT_28_fusion-encoders-training-consolidation.md`:

1. **Step 0 inventory** — 3 tables (fusion arch surface, encoder
   surface, FusionTrainer surface).
2. **Step 1** — `build_arch(name)` smoke; param counts table.
3. **Step 2** — 5 arch files' updated docstrings (link to
   each).
4. **Step 3** — encoder `demo_forward` smoke; sample
   intermediate-activation shapes.
5. **Step 4** — 5 public method signatures finalized;
   verification numbers vs RESULT_18.
6. **Step 5** — smoke script output.
7. **Step 6** — docs updated.
8. **One open question** for scientist.

## Reversibility

- Step 1 (`__init__.py` + factory): permanent.
- Step 2 (docstrings): permanent; pure documentation.
- Step 3 (`demo_forward`): permanent; new method per encoder.
- Step 4 (trainer public methods): permanent; engineer either
  promotes existing private helpers or adds new public wrappers
  that call them.
- Step 5–6: throwaway smoke + documentation.

Files committed: updated `src/pipeline/fusion/{__init__.py + 5
arch files}`, `src/pipeline/encoders/{6 encoder files
+ __init__.py}`, `src/pipeline/training/fusion_trainer.py`,
docs.

**Compute budget**: ≤ 80 min.
- Step 0: 10 min.
- Step 1: 15 min.
- Step 2: 10 min.
- Step 3: 15 min (5-6 encoder files × ~3 min each).
- Step 4: 20 min (5 methods; some are existing-private being
  promoted to public).
- Step 5: 10 min (smoke + verification).
- Step 6: 5 min (docs).

If overrun: cut Step 3's `demo_forward` to the 4 modalities used
in the main results table (Anchor2Vec, IMUCNN, OdomCNN,
DPVOMotionEncoder); WiFiSetTransformer's demo can be deferred
since it's "parked" per RESULT_01 anyway.

If `FusionTrainer.load_trained` surfaces a checkpoint-format
mismatch (e.g. CNN1D was saved with a different state_dict
schema than what current code expects), engineer documents the
adapter that bridges the two formats — that's a small inline
fix, not a scope expansion.

## Iteration scope after this plan

- **29**: `src/pipeline/evaluation/MainResultsTable` (reads
  RESULT JSONs and renders the paper-table DataFrame) +
  `scripts/eval_*.py` triage (promote ~5 canonical thin
  wrappers using all 3 consolidated APIs: baselines + data +
  fusion) + configs/docs sweep.
- **30**: `notebooks/run2_walkthrough.ipynb` scaffold using
  every consolidated API. After this, user iterates with
  engineer directly.
