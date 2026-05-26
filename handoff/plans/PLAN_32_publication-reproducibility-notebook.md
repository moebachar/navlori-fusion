# Plan 32 — Publication-grade reproducibility notebook (rewrite from scratch)

> **User directive 2026-05-26 ~16:00 local.** PLAN_30's notebook
> is a summary document with cached numbers; PLAN_31 was a
> half-measure that swapped JSON reads for `load_trained` but
> kept the "load pretrained, eval live" model. **Neither is
> what the user asked for.** The user's actual model:
>
> 1. **SOTA methods are IMPORTED from `external_methods/` via
>    `src.pipeline.baselines`** and run live on canonical data
>    in the notebook.
> 2. **"Ours" (our encoders + fusion architectures) are
>    IMPORTED from `src/pipeline/`** and **TRAINED + EVALUATED
>    INLINE in the notebook** so the reader sees the actual
>    code path from data → trained model → reported number.
> 3. The notebook is **the canonical reproducibility document**
>    for GitHub publication. Clone the repo, init submodules,
>    download data, open the notebook → readers can verify
>    every paper claim themselves.
>
> Take the time to get this right. PLAN_30/31 are superseded;
> the existing `notebooks/run2_walkthrough.ipynb` gets rewritten
> from scratch (engineer keeps a backup at
> `notebooks/_archive/run2_walkthrough_v1_summary.ipynb` for
> the record).

## Hypothesis

After this iter:
- `notebooks/run2_walkthrough.ipynb` is the publication-grade
  reproducibility notebook.
- Top-of-notebook `FAST_MODE = True` flag controls the
  training vs eval-only branch:
  - **`FAST_MODE = True` (default, ~30 min)**: SOTAs run live;
    "Ours" load pretrained checkpoints from
    `runs/overnight/run2_iter_*/` + eval live. For casual
    readers verifying the structure.
  - **`FAST_MODE = False` (~2-3 hours)**: SOTAs run live; "Ours"
    train from scratch + eval live. For strict reproducibility
    reviewers.
  - Both modes produce the same paper numbers within
    training-noise tolerance.
- Every number in the paper's main results table has a cell in
  the notebook that **computes it live** in at least one of the
  two modes. No `json.load(...)` for primary numbers. No
  hand-typed result DataFrames except for explicit "ours vs
  archive" comparison cells.
- Reader's clone-and-run UX: `git clone` → `git submodule
  update --init --recursive` → download datasets per
  `docs/EXTERNAL_DEPENDENCIES.md` → `jupyter nbconvert --execute
  notebooks/run2_walkthrough.ipynb` → all paper claims verified.

## Reproducibility contract

The notebook is the canonical reproducibility document. It
must satisfy:

1. **SOTAs imported from `external_methods/`**:
   - `wlan_localization` (UJI val mean Euclid 15.17 m) — runs
     via `src.pipeline.baselines.load_position_regressor()` +
     `load_preprocessor()`.
   - RoNIN `ResNet1D` (canonical unseen 5.14 m) — pretrained
     checkpoint at `data/ronin_frdr/pretrained_resnet/`
     loaded via `src.pipeline.baselines.ronin`.
   - TartanVO (hospital ATE) — runs via
     `src.pipeline.baselines.tartanvo` with the documented
     compat shims applied.
   - DPVO trunk (used in our Camera encoder) — loaded via
     `src.pipeline.baselines.dpvo_trunk.load_basic_encoder4`.

2. **"Ours" imported from `src/pipeline/`**:
   - Encoders: `Anchor2Vec`, `IMUCNN`, `OdomCNN`,
     `DPVOMotionEncoder` from `src.pipeline.encoders`.
   - Fusion archs: `build_arch("incumbent" | "cnn1d" |
     "lstm_attn")` from `src.pipeline.fusion`.
   - Training: `FusionTrainer` from
     `src.pipeline.training` for fusion;
     `EncoderTrainer` (or inline `train_*_on_*` helpers in
     `src.pipeline.training`) for per-leg encoders.

3. **Every paper number computed live** in at least one
   `FAST_MODE` branch:
   - SOTAs: always live (small / non-parametric / pretrained).
   - "Ours": `FAST_MODE=False` trains + evals live;
     `FAST_MODE=True` loads checkpoint + evals live. Both run
     the same eval code path.

4. **The "ours vs archive" comparison** at the end of each §2
   subsection shows the LIVE number alongside the RESULT_NN
   archive number with drift %. Reader sees the table aligns.

5. **Smoke gate**: `jupyter nbconvert --to notebook --execute`
   succeeds in `FAST_MODE=True` in ≤ 30 min on Quadro P4000.
   The `FAST_MODE=False` long-run is documented + smoke-tested
   on the per-leg (small) trainings; the full fusion training
   triplet (incumbent + CNN1D + LSTM-attn) is documented as a
   ~75-min off-notebook task and the notebook checkpoints exist
   from prior iters (so `FAST_MODE=True` always works).

## Steps

### Step 0 — Archive + scaffold (10 min)

```powershell
# Preserve the v1 summary notebook for the record.
New-Item -ItemType Directory -Force -Path notebooks/_archive
Move-Item notebooks/run2_walkthrough.ipynb notebooks/_archive/run2_walkthrough_v1_summary.ipynb
```

Create the new `notebooks/run2_walkthrough.ipynb` from scratch
with empty cells per the section structure below.

**Section structure** (the engineer renders this as the
top-of-notebook table of contents):

```
§0  Setup + FAST_MODE configuration + dataset overview (6 datasets)
§1  Per-leg SOTA reproductions (4 modalities)
    §1.1 wlan_localization on UJI val
    §1.2 RoNIN ResNet1D pretrained on canonical unseen
    §1.3 TartanVO on TartanAir hospital P000 last-20% slice
    §1.4 (No SOTA for Odom; covered by §2.4's trivial-integration baseline)
§2  "Ours" per-leg (4 modalities) — IMPORT + TRAIN + EVAL
    §2.1 Anchor2Vec on UJI (train inline if !FAST_MODE) → vs §1.1
    §2.2 IMUCNN on canonical RoNIN (train inline if !FAST_MODE) → vs §1.2
    §2.3 DPVOMotionEncoder + head on TartanAir (head training inline if !FAST_MODE) → vs §1.3
    §2.4 OdomCNN on Webots vs trivial-integration floor (train inline if !FAST_MODE)
§3  Phase B fusion bake-off (3 archs on Webots)
    §3.1 incumbent (run-1 FusionTransformer) — train if !FAST_MODE; eval val + test
    §3.2 CNN1D (winner) — train if !FAST_MODE; eval val + test + comparison
    §3.3 LSTM-attn (runner-up) — train if !FAST_MODE; eval val + test + dead-reckoning regime
§4  Ablations on CNN1D winner
    §4.1 16-row subset eval (live)
    §4.2 8-lag staleness sweep (live)
    §4.3 Per-trajectory smoothness (live)
    §4.4 Latency at b=1 and b=32 (live)
§5  Cross-dataset main results table (live aggregation)
    §5.1 Live results assembled into table (this run's numbers)
    §5.2 Comparison vs archive numbers (drift report)
§6  Honest gaps (each derived from §1-5 live results, not hand-typed)
    §6.1 Smoothness debt across 3 archs (from §4.3)
    §6.2 C2 raw-ATE gap (from §1.2 vs §2.2)
    §6.3 IMUWiFine test-no-IMU (cross-session dataset shift)
    §6.4 TartanAir paper-soft (from §1.3 vs §2.3)
§7  Reproducibility model
    §7.1 FAST_MODE explanation + per-mode wall-clock partition
    §7.2 Setup commands (clone, submodules, data download)
    §7.3 Archive cross-references (RESULT_NN files, runs/ paths)
```

### Step 1 — §0 setup + FAST_MODE flag (15 min)

```python
# notebooks/run2_walkthrough.ipynb — top cell

# Reproducibility mode (set BEFORE running any other cell)
FAST_MODE = True   # True = load pretrained "Ours" + eval live (~30 min)
                   # False = train "Ours" inline + eval live (~2-3 hours)
SEED = 42          # deterministic across modes where possible

# --- imports + sys.path setup ---
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
while ROOT.name != 'navlori-fusion' and ROOT.parent != ROOT:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.pipeline.baselines import (
    # SOTAs
    load_position_regressor, load_preprocessor,    # wlanloc
    BasicBlock1D, ResNet1D, FCOutputModule,        # RoNIN
    GlobSpeedSequence, compute_ate_rte,
    apply_tartanvo_shims, load_vo_module,          # TartanVO
    load_basic_encoder4,                           # DPVO trunk
)
from src.pipeline.encoders import (
    Anchor2Vec, IMUCNN, OdomCNN, DPVOMotionEncoder,
)
from src.pipeline.fusion import build_arch, list_archs
from src.pipeline.training import (
    FusionTrainer, load_trained,
    # encoder-level inline trainers (NEW — engineer adds in src/pipeline/training/)
    train_anchor2vec, train_imucnn, train_odomcnn, train_dpvo_motion_head,
)
from src.pipeline.data import (
    load_dataset, dataset_stats, preprocessing_demo, list_datasets,
)
from src.pipeline.visualization import (
    plot_dataset_overview, plot_preprocessing_demo,
    plot_subset_eval_bar, plot_staleness_curve,
    plot_per_trajectory, plot_main_results_heatmap,
    set_paper_style,
)
from src.pipeline.evaluation import MainResultsTable

set_paper_style()
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"ROOT = {ROOT}")
print(f"FAST_MODE = {FAST_MODE}  ({'eval-only on pretrained' if FAST_MODE else 'train + eval from scratch'})")
print(f"available archs = {list_archs()}")
print(f"available datasets = {list_datasets()}")
```

The NEW `src.pipeline.training` per-encoder inline trainers
(`train_anchor2vec`, `train_imucnn`, etc.) are small helper
functions (~30-50 lines each) that engineer adds. They wrap
the same training recipe used by `scripts/eval_*.py` (so the
notebook training reproduces the script numbers).

If `src.pipeline.training` doesn't already have these
helpers, engineer creates them as part of this iter — they're
load-bearing for the notebook's "train inline" cells. The
helpers each accept `(dataset, encoder, epochs, lr, ...)` and
return `(trained_encoder, head, history)`.

§0 dataset overview cells: per-dataset stats + overview figure
+ preprocessing demo (each one ~3 cells; 6 datasets).

**Acceptance**: §0 cell executes without error in both modes;
prints `FAST_MODE` + lists.

### Step 2 — §1 SOTA reproductions (20 min)

Each SOTA gets a "load + run + report" cell pattern. Example
for §1.1:

```python
# §1.1 — wlan_localization on UJIIndoorLoc (SOTA WiFi)
# Imported from external_methods/wlan_localization via
# src.pipeline.baselines (PLAN_26).

# Load UJI val + train from the data factory
uji_train = load_dataset("uji_indoorloc", split="train")
uji_val   = load_dataset("uji_indoorloc", split="validation")
Xtr, Ytr = uji_train["X"], uji_train["Y"]
Xva, Yva = uji_val["X"],   uji_val["Y"]

# Import + load the SOTA's PositionRegressor + DataPreprocessor
PositionRegressor = load_position_regressor()
DataPreprocessor  = load_preprocessor()

# Fit the SOTA on UJI train + predict on UJI val
pre = DataPreprocessor()
Xtr_p = pre.fit_transform(Xtr)
Xva_p = pre.transform(Xva)
reg = PositionRegressor(k=3, metric="manhattan", weights="distance")
reg.fit_global(Xtr_p, Ytr[:, :2])         # global mode (no cascade)
pred = reg.predict_global(Xva_p)

# Live SOTA number
mae_sota = float(np.linalg.norm(pred - Yva[:, :2], axis=1).mean())
print(f"wlan_localization UJI val mean Euclidean: {mae_sota:.2f} m")
print(f"  Archive (RESULT_01): 15.17 m   |   drift: {abs(mae_sota - 15.17) / 15.17 * 100:.1f}%")
```

Same pattern for §1.2 RoNIN ResNet1D pretrained:

```python
# §1.2 — RoNIN ResNet1D (SOTA IMU) on canonical unseen
# Pretrained checkpoint at data/ronin_frdr/pretrained_resnet/

from src.pipeline.baselines import ResNet1D, FCOutputModule, load_test_list, compute_ate_rte

# Build ResNet1D matching RoNIN's published config
model = ResNet1D(num_inputs=6, num_outputs=2, block_type=BasicBlock1D,
                 group_sizes=[2, 2, 2, 2], inter_dim=128,
                 output_block=FCOutputModule, kernel_size=3,
                 fc_dim=512, in_dim=7, dropout=0.5)
ckpt = ROOT / "data" / "ronin_frdr" / "pretrained_resnet" / "ronin_resnet" / "checkpoint_gsn_latest.pt"
sd = torch.load(ckpt, map_location="cpu")
model.load_state_dict(sd["model_state_dict"])
model = model.cuda().eval()

# Eval on the 32 unseen sequences
ronin_canonical_root = ROOT / "data" / "ronin_frdr" / "unseen"
test_seqs = load_test_list("list_test_unseen.txt")
ates = []
for seq_name in test_seqs:
    pred_traj, gt_traj = predict_ronin_seq(model, ronin_canonical_root / seq_name)
    ate, _ = compute_ate_rte(pred_traj, gt_traj, pred_per_min=12000)
    ates.append(ate)
ronin_ate = float(np.mean(ates))
print(f"RoNIN ResNet1D pretrained — canonical unseen mean ATE: {ronin_ate:.3f} m  ({len(ates)} seqs)")
print(f"  Archive (RESULT_07): 5.140 m (paper-exact)   |   drift: {abs(ronin_ate - 5.140) / 5.140 * 100:.2f}%")
```

`predict_ronin_seq(model, seq_dir)` is a 20-line helper in
`src.pipeline.training` (or `src.pipeline.evaluation`) that
runs the canonical-window forward + integrates predicted
velocity into a trajectory. Already implemented in
`scripts/eval_ronin_canonical.py` (RESULT_29); engineer
promotes it to a library function.

§1.3 TartanVO + §1.4 (Odom has no SOTA) follow same shape.

**Acceptance**: §1.1-1.3 run live and produce numbers within
±0.5 % of RESULT_01/07/08 archive values.

### Step 3 — §2 "Ours" per-leg (25 min)

Each "Ours" subsection's pattern:

```python
# §2.1 — Anchor2Vec on UJI (Ours, WiFi)
# Imported from src.pipeline.encoders; trained inline via
# src.pipeline.training.train_anchor2vec (if !FAST_MODE) OR
# loaded from cached checkpoint (if FAST_MODE).

from src.pipeline.encoders import Anchor2Vec
from src.pipeline.training import train_anchor2vec

# Same UJI train/val as §1.1 (already loaded).
ckpt = ROOT / "runs" / "encoder_audit_wifi" / "anchor2vec_uji.pt"

if FAST_MODE and ckpt.is_file():
    enc = Anchor2Vec(n_aps=Xtr.shape[1], embed_dim=128, n_anchors=64)
    sd = torch.load(ckpt, map_location="cpu")
    enc.load_state_dict(sd["encoder_state_dict"])
    head = torch.nn.Linear(128, 2)
    head.load_state_dict(sd["head_state_dict"])
    print(f"Loaded Anchor2Vec checkpoint from {ckpt}")
else:
    print("Training Anchor2Vec on UJI train (120 epochs, ~3 min)...")
    enc, head, history = train_anchor2vec(
        Xtr, Ytr[:, :2], Xva, Yva[:, :2],
        n_aps=Xtr.shape[1], n_anchors=64, embed_dim=128,
        epochs=120, lr=1e-3, batch_size=256, seed=SEED,
    )
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state_dict": enc.state_dict(),
                "head_state_dict": head.state_dict(),
                "history": history}, ckpt)
    print(f"Saved trained checkpoint to {ckpt}")

# Live eval
mae_ours = anchor2vec_eval(enc, head, Xva, Yva[:, :2])
print(f"Anchor2Vec UJI val mean Euclidean: {mae_ours:.2f} m")
print(f"  Archive (RESULT_01): 8.69 m   |   drift: {abs(mae_ours - 8.69) / 8.69 * 100:.1f}%")
print(f"  SOTA (§1.1): {mae_sota:.2f} m   |   Anchor2Vec beats SOTA by {(mae_sota - mae_ours) / mae_sota * 100:.1f}%")
```

Same pattern for §2.2 IMUCNN on canonical RoNIN, §2.3
DPVOMotionEncoder + head training on TartanAir, §2.4 OdomCNN
on Webots + trivial-integration floor.

Helper functions needed in `src.pipeline.training`:
- `train_anchor2vec(Xtr, Ytr, Xva, Yva, **cfg)` → from
  `scripts/eval_uji.py` + RESULT_01 recipe.
- `train_imucnn(...)` → from `scripts/_eval_ronin_imuwifine.py`
  pattern.
- `train_dpvo_motion_head(...)` → from RESULT_08.
- `train_odomcnn(...)` → from RESULT_04.

These are all small (~30-60 lines each). Engineer either
writes them fresh OR factors out from the existing `scripts/`.

**Acceptance**: §2 subsections produce numbers within
training-noise of RESULT_01/07/08/04 archive values.

### Step 4 — §3 Phase B fusion bake-off (15 min)

For each of the 3 paper-facing archs:

```python
# §3.1 — incumbent (run-1 FusionTransformer)
ckpt_dir = ROOT / "runs" / "overnight" / "run2_iter_13" / "incumbent"  # ckpt path

if FAST_MODE and ckpt_dir.is_dir():
    trainer = load_trained(str(ckpt_dir), arch="incumbent", dataset="simulation")
    print(f"Loaded incumbent checkpoint from {ckpt_dir}")
else:
    # Train inline — ~25 min on Quadro P4000
    print("Training incumbent on Webots (90 epochs, K=4, 4-mod, B=128)...")
    trainer = FusionTrainer.from_config("simulation", arch="incumbent",
                                          K=4, batch_size=128, lr=1.3e-3,
                                          epochs=90, seed=SEED)
    trainer.fit()
    trainer.save_checkpoint(ckpt_dir)

val_mae  = trainer.evaluate("val")["mae"]
test_mae = trainer.evaluate("test")["mae"]
print(f"Incumbent: val MAE {val_mae:.3f} / test MAE {test_mae:.3f}")
print(f"  Archive (RESULT_13/14): 0.394 / 0.417   |   drift: ...")
```

Same for §3.2 CNN1D winner, §3.3 LSTM-attn runner-up. The
table at end of §3 is BUILT FROM `val_mae`/`test_mae`
variables computed in the cells, not hand-typed.

`FusionTrainer.from_config(...)` is a class method engineer
either confirms exists OR adds — it's the bridge between the
notebook and the existing builder pattern.
`trainer.save_checkpoint(...)` likewise.

**Acceptance**: §3 produces a 3-row table built from live
variables; numbers within ±1 % of RESULT_13/17 archive.

### Step 5 — §4 ablations on CNN1D winner (10 min)

```python
# §4.1 — 16-row subset eval (LIVE)
subsets = trainer_cnn1d.evaluate_all_subsets("test")
plot_subset_eval_bar(subsets, title="CNN1D Webots test — 16-row subset eval")
plt.show()

# §4.2 — 8-lag staleness sweep (LIVE)
lags = [0, 1, 3, 5, 10, 15, 20, 30]
curve = trainer_cnn1d.evaluate_staleness(lags, modality="wifi", split="test")
plot_staleness_curve(lags, list(curve.values()),
                     title="CNN1D WiFi staleness — LIVE")
plt.show()

# §4.3 — per-trajectory smoothness (LIVE)
r_per_path = trainer_cnn1d.compute_per_trajectory_smoothness("test")
median_r = float(np.median(list(r_per_path.values())))
print(f"CNN1D Webots test per-trajectory smoothness median r = {median_r:.3f}")

# §4.4 — latency probe (LIVE)
lat = trainer_cnn1d.latency_probe(batch_sizes=[1, 32], n_trials=100)
print(f"CNN1D latency b=1 = {lat[1]:.3f} ms/sample, b=32 = {lat[32]:.3f} ms/sample")
```

All cells use the public `FusionTrainer` methods promoted in
PLAN_28 / RESULT_28.

**Acceptance**: §4 produces 4 visualisations / numbers
matching RESULT_18 archive within ±1 %.

### Step 6 — §5 main results table: live aggregation + drift report (10 min)

```python
# §5.1 — Live results assembled into a table built from the
# variables computed in §1-4.
live_table = pd.DataFrame([
    {"dataset": "UJI",       "modality": "WiFi only",     "SOTA": f"{mae_sota_uji:.2f}",
     "Ours_Anchor2Vec": f"{mae_ours_uji:.2f}"},
    {"dataset": "RoNIN can.", "modality": "IMU only",     "SOTA": f"{ronin_ate:.3f}",
     "Ours_IMUCNN":     f"{imucnn_canonical:.3f}"},
    {"dataset": "TartanAir",  "modality": "Camera only",  "SOTA": f"{tartanvo_ate:.3f}",
     "Ours_DPVOMotion": f"{dpvo_motion_ate:.3f}"},
    {"dataset": "Webots",     "modality": "WiFi+IMU+Cam+Odom",
     "Ours_CNN1D_test": f"{cnn1d_test:.3f}",
     "Ours_LSTM_test":  f"{lstm_test:.3f}"},
])
live_table

# §5.2 — Drift report: compare live vs archive
archive_table = MainResultsTable.from_archive().to_dataframe()
# ... build a side-by-side "live vs archive" comparison; print drift %
```

**Acceptance**: §5 table built entirely from live cell
variables; drift per cell < 2 % (training-noise tolerance).

### Step 7 — §6 honest gaps + §7 reproducibility (10 min)

§6 builds tables from §1-4 live data. The smoothness table
in §6.1 is computed from §4.3 across the 3 archs. The C2 raw
gap in §6.2 is computed from §1.2 vs §2.2. §6.3-4 are
markdown + `dataset_stats(...)["known_caveats"]` reference
cells.

§7 reproducibility section:

```markdown
### FAST_MODE controls

| mode | what happens | wall-clock | use case |
|---|---|---|---|
| `FAST_MODE = True` (default) | SOTAs run live; "Ours" load pretrained checkpoints + eval live | ~30 min | casual readers verifying structure |
| `FAST_MODE = False` | SOTAs run live; "Ours" train from scratch + eval live | ~2-3 hours | strict-repro reviewers |

Both modes produce the same paper numbers within training-noise.

### Setup commands

```bash
git clone https://github.com/moebachar/navlori-fusion.git
cd navlori-fusion
git submodule update --init --recursive       # external_methods/{wlan_localization, ronin, tartanvo, dpvo}
python -m venv .venv && .venv/Scripts/activate
pip install -e .
# Datasets — see docs/EXTERNAL_DEPENDENCIES.md for per-dataset URLs
```

### Archive cross-references

- `handoff/results/RESULT_NN_*.md` — per-iteration findings.
- `handoff/SUMMARY.md` — run-2 one-pager.
- `runs/overnight/run2_iter_*/` — saved checkpoints + JSONs.
```

### Step 8 — End-to-end smoke + commit (20 min)

Engineer runs the notebook in BOTH modes:

1. `FAST_MODE = True` → `jupyter nbconvert --to notebook
   --execute notebooks/run2_walkthrough.ipynb` → must succeed
   in ≤ 30 min on Quadro P4000.
2. `FAST_MODE = False` → smoke-test the small encoder trainings
   inline (Anchor2Vec ~3 min, IMUCNN ~14 min, OdomCNN ~5 min,
   DPVO head ~5 min); the 3-arch fusion training (~75 min)
   documented as long-run but engineer either fires it OR
   verifies the cached checkpoints load cleanly under
   `FAST_MODE = True`.

Drift on each "Ours vs archive" comparison cell must be
< 2 %. If anything fails, fix in-place before committing.

**Acceptance**: BOTH modes produce a clean notebook with no
cell errors; drift < 2 % per comparison cell; engineer
commits the final notebook.

## Sources

- User directive 2026-05-26 ~16:00 local.
- RESULT_30 (the v1 summary notebook — archived to
  `notebooks/_archive/`).
- All RESULT_NN files (canonical numbers).
- `src.pipeline.{baselines, encoders, fusion, training,
  evaluation, data, visualization}` (consolidated APIs from
  PLAN_26-29).

## What to report back

In `handoff/results/RESULT_32_publication-reproducibility-notebook.md`:

1. **Step 0** — archive done; new scaffold created.
2. **Step 1** — §0 setup cell; FAST_MODE flag working.
3. **Step 2** — §1 SOTA numbers vs archive: 3-row drift table.
4. **Step 3** — §2 "Ours" numbers vs archive: 4-row drift
   table; encoder helpers added to `src.pipeline.training`.
5. **Step 4** — §3 fusion 3-arch numbers vs archive: 3-row
   drift table.
6. **Step 5** — §4 ablations vs archive: 4-row drift table.
7. **Step 6** — §5 live aggregation table + drift report.
8. **Step 7** — §6 + §7 written.
9. **Step 8** — end-to-end smoke in BOTH modes; final
   wall-clocks recorded.
10. **One open question** for the user.

## Reversibility

- Step 0 (archive v1): permanent move to
  `notebooks/_archive/run2_walkthrough_v1_summary.ipynb`.
- Steps 1-7: new notebook; engineer commits.
- NEW per-encoder inline trainers in `src.pipeline.training/`:
  permanent (small helpers, ~30-60 lines each).

Files committed: `notebooks/run2_walkthrough.ipynb`
(rewritten); `notebooks/_archive/run2_walkthrough_v1_summary.ipynb`
(historical); NEW helpers under `src.pipeline.training/` for
the per-encoder inline trainings.

**Compute budget**: ≤ 4 hours (including the FAST_MODE=False
training smoke).
- Step 0: 10 min.
- Step 1: 15 min.
- Step 2: 20 min.
- Step 3: 25 min (writing) + small smoke-test trainings (~25 min wall in FAST_MODE=False if engineer runs).
- Step 4: 15 min.
- Step 5: 10 min.
- Step 6: 10 min.
- Step 7: 10 min.
- Step 8: 20 min nbconvert + verification.
- Buffer: 60 min for unexpected issues (checkpoint format mismatches, missing helpers, etc.).

If overrun on the FAST_MODE=False smoke: ship `FAST_MODE=True`
clean + document the slow-mode wall-clocks as expectations
based on prior RESULT_NN runs. The reader's main UX is
`FAST_MODE=True`; slow mode is an honest fallback.

If a checkpoint format mismatch is surfaced (e.g. one of the
saved trainers doesn't restore cleanly under the consolidated
API), engineer either: (a) re-saves with the consolidated
format, (b) documents the format adapter in-cell, (c) flags
for the user.

The notebook ships when BOTH modes produce a clean top-to-
bottom execution with every paper number computed live and
drift < 2 % per comparison cell.

## Quality bar

This is a publication-grade deliverable. **Take the time to
get it right.** No shortcuts:

- Don't paste hand-typed DataFrames except for explicit
  comparison vs archive.
- Don't `json.load(...)` cached results for primary numbers.
- Don't skip the per-leg training in `FAST_MODE=False` mode
  even if checkpoints exist — verify the training reproduces
  the archive numbers.
- Each cell that claims a paper number must have a `print(...)`
  showing the live computation + drift vs archive.

Reader's clone-and-reproduce experience is what gets
published. The notebook IS the reproducibility documentation.
