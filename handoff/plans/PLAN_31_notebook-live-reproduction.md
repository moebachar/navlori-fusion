# Plan 31 — Notebook revision: live reproduction (clone-and-run, paper-supplementary-grade)

> **User directive 2026-05-26 ~15:40 local.** The PLAN_30
> notebook reads cached JSONs / pastes hand-typed DataFrames
> for most numbers. The user's deliverable model: the notebook
> is a **clone-and-reproduce** artifact for the scientific
> community — readers should be able to clone the repo,
> install deps, and re-derive the paper claims by running the
> notebook top-to-bottom.
>
> This iter revises `notebooks/run2_walkthrough.ipynb` so every
> claimed number is **either computed live in the notebook
> from a loaded checkpoint OR derived from a `scripts/eval_*.py`
> invocation** — no cached-JSON reads, no hand-typed
> DataFrames.

## Hypothesis

Today the notebook is a "summary document with cached numbers
+ some live demos." After this iter:

- **§0 datasets**: stays as-is (already live — `dataset_stats`,
  `plot_dataset_overview`, `preprocessing_demo` all run library
  code).
- **§1 encoder audit**: per-subsection, the encoder is loaded +
  run on real data + the paper number is **computed in the
  cell**, not pasted.
- **§2 fusion bake-off**: each architecture loaded via
  `load_trained(...)`; subset eval / staleness sweep /
  smoothness all computed by calling `evaluate_all_subsets`,
  `evaluate_staleness`, `compute_per_trajectory_smoothness`
  **live on the loaded checkpoint**. No JSON reads.
- **§3 cross-dataset main table**: `MainResultsTable.from_archive()`
  stays as the source-of-truth narrative, BUT the notebook
  also includes a "Verify one cell live" demonstrator that
  recomputes ONE cell from scratch (e.g. UJI Anchor2Vec 8.69 m
  via `eval_uji.py`-equivalent inline) so the reader sees
  reproducibility at the cell level.
- **§4 honest gaps**: smoothness table computed live from
  `compute_per_trajectory_smoothness` calls on the 3 archs
  Webots checkpoint; C2 raw-ATE table computed live from
  RoNIN canonical eval; IMUWiFine + TartanAir caveats stay as
  text + `dataset_stats` references.
- **§5 reproducibility**: expanded with a "what runs live in
  this notebook" vs "what requires `scripts/_train_*.py` (hours
  of compute)" partition table.

The notebook still executes top-to-bottom in **≤ 5-10 minutes
on the Quadro P4000** (because we eval cached checkpoints, not
retrain). Training-from-scratch is documented but NOT cell-
level work.

## Reproducibility model (load-and-eval, not retrain)

Standard for paper-supplementary notebooks:

| layer | how it's reproduced |
|---|---|
| Pretrained external SOTAs (RoNIN ResNet1D, DPVO trunk, TartanVO) | downloaded as setup (`docs/EXTERNAL_DEPENDENCIES.md`); loaded via `src.pipeline.baselines.*`; eval is live |
| Our trained models (CNN1D winner, LSTM-attn runner-up, incumbent) | checkpoints under `runs/overnight/run2_iter_*/`; loaded via `FusionTrainer.load_trained(...)`; eval is live |
| Our SOTA-comparison numbers (wlanloc on UJI/MSILN/IMUWiFine) | non-parametric (kNN); live re-runs are fast |
| Paper main-table cells | each cell is the output of one of the above; `MainResultsTable` aggregates for paper-facing display |
| Training from scratch | `scripts/_train_*.py` commands in §5; NOT executed by the notebook (would take hours) |

The notebook IS the reproducibility documentation.

## Steps

### Step 0 — Audit the current notebook (5 min)

Engineer greps the existing notebook for:
- `json.load(...)` reads → these are the JSON-cache reads that
  need to be replaced with live calls.
- Hand-typed `pd.DataFrame([{...}])` → these are pasted numbers
  that need to be derived from artifacts or live computation.

Catalogue what stays and what flips. The cell count after
revision should be similar (~25 cells).

**Acceptance**: cell-by-cell audit table in RESULT_31 TL;DR.

### Step 1 — §1 encoder audit: live evaluation per subsection (20 min)

For each of the 4 subsections, replace the demo-with-synthetic
pattern with a live load + eval pattern:

#### §1.1 Anchor2Vec on UJI

```python
# Load UJI val from the data factory (live).
from src.pipeline.data import load_dataset
Xva, Yva = load_dataset('uji_indoorloc', split='validation')

# Load Anchor2Vec from saved encoder weights OR retrain at the
# canonical config from RESULT_01.
# Prefer eval-on-saved-encoder if a checkpoint exists; else
# train inline (~3-5 min on UJI's 19937 train scans).
from src.pipeline.encoders import Anchor2Vec
weights_p = ROOT / "runs" / "encoder_audit_wifi" / "anchor2vec_uji.pt"
if weights_p.is_file():
    enc = Anchor2Vec(n_aps=Xva.shape[1], embed_dim=128, n_anchors=64)
    enc.load_state_dict(torch.load(weights_p, map_location="cpu"))
else:
    # Inline training (the RESULT_01 recipe; ~3 min)
    enc, head = train_anchor2vec_on_uji(Xtr, Ytr, epochs=120)
    torch.save(enc.state_dict(), weights_p)

# Live eval - compute the mean Euclidean error.
pred = predict_xy(enc, head, Xva)
mae = mean_euclidean(pred, Yva)
print(f"Anchor2Vec UJI val mean Euclidean: {mae:.2f} m  (RESULT_01 = 8.69 m)")

# Also run the SOTA via the baselines package (eval-only,
# non-parametric, fast).
from src.pipeline.baselines import load_position_regressor, load_preprocessor
sota_mae = run_wlanloc_uji_global(Xtr, Ytr, Xva, Yva)
print(f"wlanloc UJI val mean Euclidean: {sota_mae:.2f} m  (RESULT_01 = 15.17 m)")

print(f"Anchor2Vec beats wlanloc by {100*(sota_mae - mae)/sota_mae:.1f}%")
```

The helper functions (`train_anchor2vec_on_uji`,
`predict_xy`, `mean_euclidean`, `run_wlanloc_uji_global`) live
in `src/pipeline/eval_helpers.py` (NEW). Engineer's call where
to put them — they're library code, not notebook code.

#### §1.2 IMUCNN on RoNIN canonical

Same pattern: load RoNIN canonical via
`load_dataset("ronin_canonical", split="test")`, load the
pretrained ResNet1D from `external_methods/ronin`, compute the
5.140 m number live via `compute_ate_rte` from the baselines
package. Then either load an IMUCNN checkpoint OR train inline
on canonical RoNIN train (~14 min per RESULT_07 — flag this
in §5 as the "longer cell" if so).

If training takes > 5 min, the notebook prefers loading a saved
checkpoint; the train-from-scratch path is the documented
script. Reader's call which to run.

#### §1.3 DPVOMotion on TartanAir hospital

Live: load TartanAir hospital sample via factory; run TartanVO
via `src.pipeline.baselines.run_vo_module()`; run DPVOMotion
via the encoder; compute ATEs via `evo`-based helper.

#### §1.4 OdomCNN on Webots

Live: load Webots via factory; run trivial integration vs
OdomCNN-P-B; compute test MAE comparison.

**Acceptance**: §1 runs live; all 4 numbers computed and
printed in the cell output; matches RESULT_01/07/08/04 within
training-noise (±0.5 %).

### Step 2 — §2 fusion bake-off: drop JSON reads (15 min)

Replace every `json.load(...)` with the live equivalent:

```python
# BEFORE
with open(ckpt_dir / 'all_subsets_test.json') as f:
    subs = json.load(f)
subset_dict = {k: v['mae'] for k, v in subs.items()}

# AFTER
from src.pipeline.training import load_trained
cnn1d = load_trained("runs/overnight/run2_iter_17/cnn1d", arch="cnn1d", dataset="simulation")
subset_dict = cnn1d.evaluate_all_subsets("test")
# This computes 16 subsets live in ~30 seconds.
```

Same pattern for:
- LSTM-attn subset eval → `lstm_attn.evaluate_all_subsets("test")` live.
- CNN1D staleness sweep → `cnn1d.evaluate_staleness(lags=[0,1,3,5,10,15,20,30], modality="wifi")` live (~80 seconds for 8 lags).
- Phase B 3-arch comparison DataFrame → derive each row's
  numbers from a loaded trainer (`evaluate_test`, `latency_probe`,
  `compute_per_trajectory_smoothness`) instead of hand-typing.

§2 should take ~3 minutes to execute now (was ~5 seconds with
the JSON shortcut; the live calls are the cost of
reproducibility).

**Acceptance**: §2 has zero `json.load` calls; all numbers
derived from `load_trained(...)` + live evaluation methods.

### Step 3 — §3 main results table: live cell verification demo (10 min)

`MainResultsTable.from_archive()` stays — it's the paper-facing
table assembler and the user wants a single source of truth.
But ADD a cell after the table that recomputes ONE specific
cell live as a demonstrator:

```python
# Reproducibility demonstrator: verify the UJI Anchor2Vec cell.
from src.pipeline.evaluation import MainResultsTable
table = MainResultsTable.from_archive()
df = table.to_dataframe()
display(df)

print("--- live verification of one cell ---")
uji_anchor2vec_cell = table.cell("uji_indoorloc", "Anchor2Vec")
print(f"Table reports: {uji_anchor2vec_cell.val} m val")
print(f"Live computation (§1.1 above): {mae:.2f} m val")
print(f"Drift: {abs(uji_anchor2vec_cell.val - mae) / uji_anchor2vec_cell.val * 100:.2f}%")
```

The reader's takeaway: "the table aggregates 30+ numbers; here
is one of them recomputed live; you can recompute any other
via `scripts/eval_*.py`."

**Acceptance**: §3 prints the live-vs-table drift for one cell;
drift within ±2 % (training noise + dropout-eval ordering).

### Step 4 — §4 honest gaps: live smoothness + C2 tables (10 min)

#### Smoothness table — live

```python
# Compute smoothness median r live for the 3 paper-facing archs
# on Webots test paths. Each load_trained + compute call is
# ~5 seconds; total ~15s.
smoothness_rows = []
for arch in ["incumbent", "cnn1d", "lstm_attn"]:
    ckpt_dir = next((Path("runs/overnight").glob(f"run2_iter_*/{arch}")))
    trainer = load_trained(str(ckpt_dir), arch=arch, dataset="simulation")
    r_per_path = trainer.compute_per_trajectory_smoothness("test")
    smoothness_rows.append({
        "arch": arch,
        "median_r": float(np.median(list(r_per_path.values()))),
        "per_path": r_per_path,
    })
pd.DataFrame(smoothness_rows)[["arch", "median_r"]]
```

(Engineer adapts paths/ckpt-naming to actual layout.)

#### C2 raw-ATE table — derived from §1.2

§1.2 already computed IMUCNN and ResNet1D canonical numbers
live. This section just reuses them.

#### IMUWiFine + TartanAir — text-only with `dataset_stats` reference

`dataset_stats("imuwifine_floor4")` already surfaces the
test-no-IMU caveat in `known_caveats`. §4.3 is a markdown
block + a small cell that prints the relevant caveat string.

**Acceptance**: §4.1 smoothness table is computed live (3 archs,
no hand-typed pasted DataFrame); §4.2 C2 reuses §1.2 numbers;
§4.3-4 are markdown + caveat-print cells.

### Step 5 — §5 reproducibility: partition table (5 min)

Expand the existing §5 with an explicit "what's live in this
notebook" vs "what requires offline `scripts/_train_*.py`":

```markdown
### What this notebook does live (≤ 10 min on Quadro P4000)

| section | computation | time |
|---|---|---|
| §0 | dataset stats + overview + preprocessing demos | ~30 s |
| §1.1 | Anchor2Vec UJI eval + wlanloc UJI eval | ~30 s |
| §1.2 | IMUCNN canonical RoNIN eval (loaded checkpoint) | ~15 s |
| §1.3 | TartanVO + DPVOMotion last-20% slice eval | ~20 s |
| §1.4 | OdomCNN + trivial integration Webots eval | ~10 s |
| §2 | 3 archs subset eval + staleness sweep on cached checkpoints | ~3 min |
| §3 | MainResultsTable rendering + 1 cell live verification | ~30 s |
| §4 | 3 archs smoothness compute (Webots test paths) | ~15 s |
| total | | ~5-6 min |

### What requires offline computation (NOT done by this notebook)

| task | command | wall-clock |
|---|---|---|
| Re-train CNN1D winner from scratch | `python scripts/_train_webots_4mod_arch.py --arch cnn1d` | ~25 min |
| Re-train all 3 archs | `python scripts/_train_webots_4mod_arch.py --arch <name>` × 3 | ~75 min |
| Re-run the bake-off (4 archs on 10 % subset) | `python scripts/_bake_off_webots.py` | ~30 min |
| Re-acquire RoNIN FRDR data (15 GB) | manual download per `docs/EXTERNAL_DEPENDENCIES.md` | 30-60 min |
```

(Engineer adjusts wall-clocks to actual machine numbers.)

### Step 6 — End-to-end smoke + commit (10 min)

Run "Restart Kernel and Run All". Expected total wall-clock:
**5-10 minutes**. If any cell errors, fix in-place. Confirm:

- §0-§4 all produce non-empty outputs.
- §3 live-verification drift is < 2 % for the demonstrator cell.
- No `json.load` calls remain in the notebook for primary
  numbers (only acceptable use: reading the `_CANONICAL`
  mapping inside `MainResultsTable`, which is library code).

Commit the revised notebook + any new helpers (NEW
`src/pipeline/eval_helpers.py` if engineer chose to extract
the §1 helpers; alternatively engineer can inline the helpers
in `src/pipeline/training/__init__.py`).

**Acceptance**: clean top-to-bottom execution; total wall ≤
10 min; engineer commits.

## Sources

- User directive 2026-05-26 ~15:40 local (this iter).
- RESULT_30 (the current notebook scaffold).
- All previous RESULT_NN files for the source-of-truth numbers.
- `src.pipeline.training.load_trained` (RESULT_28) — the live
  checkpoint loader that's central to this iter.
- `src.pipeline.training.FusionTrainer.{evaluate_all_subsets,
  evaluate_staleness, compute_per_trajectory_smoothness,
  latency_probe}` (RESULT_28) — the live evaluation methods.

## What to report back

In `handoff/results/RESULT_31_notebook-live-reproduction.md`:

1. **Step 0** — cell-by-cell audit: which cells flipped from
   JSON-read to live-eval, which were re-derived from
   hand-typed pasted DataFrames.
2. **Step 1** — 4 encoder audit subsections live; numbers vs
   RESULT_01/04/07/08 references.
3. **Step 2** — §2 bake-off live; numbers vs RESULT_13/14/17/18.
4. **Step 3** — §3 live-cell verification demonstrator; drift.
5. **Step 4** — §4 smoothness + C2 tables computed live.
6. **Step 5** — §5 partition table.
7. **Step 6** — end-to-end execution time + any cells that
   surfaced unexpected issues.
8. **One open question** for the user.

## Reversibility

- Step 1-6: notebook revision (single file). Engineer commits.
- NEW `src/pipeline/eval_helpers.py` (if extracted): permanent.

Files committed: `notebooks/run2_walkthrough.ipynb` (revised);
optionally `src/pipeline/eval_helpers.py`.

**Compute budget**: ≤ 90 min.
- Step 0: 5 min (audit).
- Step 1: 20 min (4 subsections — most logic exists in
  `scripts/eval_*.py` from RESULT_29; engineer can borrow).
- Step 2: 15 min (JSON-read → live-call swap; mechanical).
- Step 3: 10 min.
- Step 4: 10 min.
- Step 5: 5 min.
- Step 6: 10 min (run + fix).
- Buffer: 15 min for unexpected issues with `load_trained`
  on specific checkpoints, ckpt naming conventions, etc.

If overrun: leave §1.3 (DPVOMotion on TartanAir) as documented
text only (since TartanVO setup has known Windows compat
shims; engineer's RESULT_08 covered them but doing it inline
in the notebook is the riskiest cell). The other 3 §1
subsections are the load-bearing reproducibility demos.

If `load_trained` surfaces a checkpoint-format issue (e.g. one
of the 3 winners' state_dict doesn't restore cleanly), engineer
either: (a) re-saves with the consolidated-API-compatible
format, (b) documents the format adapter in the notebook cell,
(c) flags for the user to look at.

The notebook ships when "Restart Kernel and Run All" produces a
clean top-to-bottom execution in ≤ 10 minutes with every
paper-claim number computed live (or derived from a live
computation in the same section).
