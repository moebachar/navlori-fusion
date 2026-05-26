# Plan 30 — `notebooks/run2_walkthrough.ipynb` scaffold (final consolidation iter)

> **Final iteration of the consolidation roadmap.** PLAN_26-29
> built the library; this plan builds the notebook that drives
> it. After this iter, the user iterates directly with the
> engineer on notebook polish.
>
> **Exclusions still in effect** per
> `handoff/SCIENTIST_NOTE_notebook-exclusions.md`: 6 datasets
> (Webots / IMUWiFine / MSILN / RoNIN canonical / TartanAir
> hospital / UJI; no IPIN) × 3 architectures
> (incumbent + cnn1d + lstm_attn; no MoTTransformer).

## Hypothesis

After this iter:
- `notebooks/run2_walkthrough.ipynb` exists as a 6-section
  notebook (§0 datasets pre-section + §1-§5 phase/findings).
- Every code cell is ≤ 5 lines and imports from
  `src.pipeline.{baselines, data, fusion, training, evaluation,
  visualization}`. No inline boilerplate.
- The notebook executes top-to-bottom in ≤ 5 minutes on the
  Quadro P4000 from a fresh kernel (no training; eval-only on
  cached checkpoints).
- Every figure + table that ends up in the paper has a
  reproducer cell in the notebook.
- The user can hand this to a reviewer alongside the codebase
  and the paper-facing story tells itself.

## Section structure

### §0 — Datasets pre-section

For each of the 6 paper-facing datasets, a 3-cell block:

1. **Stats cell**:
   ```python
   from src.pipeline.data import dataset_stats
   import pandas as pd
   stats = dataset_stats("webots")
   pd.DataFrame(stats["per_path_summary"])  # auto-rendered table
   ```
2. **Overview figure cell**:
   ```python
   from src.pipeline.visualization import plot_dataset_overview
   fig = plot_dataset_overview("webots")
   ```
3. **Preprocessing demo cell** (one per modality the dataset has):
   ```python
   from src.pipeline.data import preprocessing_demo
   from src.pipeline.visualization import plot_preprocessing_demo
   demo = preprocessing_demo("webots", "wifi")
   plot_preprocessing_demo(demo, modality="wifi")
   ```

6 datasets × ~4 modality demos each = ~24 demo cells max.
For modalities a dataset doesn't have (e.g. UJI has no IMU),
the cell is skipped — `dataset_stats` indicates availability.

Engineer organises by collapsing dataset blocks with markdown
headers; each dataset's known-caveats string (already in
`stats["known_caveats"]` per RESULT_27) renders inline.

### §1 — Phase A: encoder audit

Per-modality encoder audit walkthrough:

1. **Anchor2Vec on UJI** — load encoder, call `.demo_forward(rssi)`,
   show the embedding heatmap; compare to wlanloc SOTA (15.17 m)
   vs Anchor2Vec (8.69 m).
2. **IMUCNN on RoNIN canonical** — load encoder, show
   `.demo_forward(imu_window)` with the world-frame rotation;
   compare against ResNet1D pretrained (5.14 m raw) vs IMUCNN
   (9.96 m raw / 7.88 Umeyama).
3. **DPVOMotionEncoder on TartanAir hospital** — load encoder,
   show patch tokens via `.demo_forward(image_pair)`; compare
   against TartanVO (0.518 m full / 0.012 m last-20 % slice) vs
   DPVOMotion (0.293 m last-20 % slice).
4. **OdomCNN on Webots** — internal audit (no public SOTA);
   show `.demo_forward(odom_window)`, the trivial-integration
   floor (test 8.27 m) vs OdomCNN-P-B (test 4.24 m).

Each subsection ends with a verdict label per the run-2 audit
rubric (`keep`, `keep with smoothness debt`, `keep (in-domain
only)`).

### §2 — Phase B: fusion bake-off

3-architecture comparison (incumbent + CNN1D + LSTM-attn).
NOT 4: MoTTransformer dropped per the directive.

1. **Load each checkpoint** (`FusionTrainer.load_trained`):
   ```python
   from src.pipeline.training import load_trained
   incumbent = load_trained("runs/.../incumbent")
   cnn1d     = load_trained("runs/overnight/run2_iter_17/cnn1d")
   lstm_attn = load_trained("runs/overnight/run2_iter_17/lstm_attn")
   ```
2. **Param + latency comparison** (RESULT_17/18 numbers):
   ```python
   table = pd.DataFrame([
       {"arch": "incumbent",  "params_M": 1.55, "test_MAE": 0.417, "latency_b1_ms": 6.41},
       {"arch": "cnn1d",      "params_M": 0.51, "test_MAE": 0.339, "latency_b1_ms": 4.73},
       {"arch": "lstm_attn",  "params_M": 0.57, "test_MAE": 0.340, "latency_b1_ms": 4.73},
   ])
   ```
3. **Subset eval on CNN1D winner**:
   ```python
   subsets = cnn1d.evaluate_all_subsets("test")
   fig = plot_subset_eval_bar(subsets)
   ```
4. **Staleness sweep** (the RESULT_14 paper-figure):
   ```python
   lags = [0, 1, 3, 5, 10, 15, 20, 30]
   curve = cnn1d.evaluate_staleness(lags, modality="wifi")
   fig = plot_staleness_curve(lags, list(curve.values()))
   ```
5. **LSTM-attn dead-reckoning regime** — 3-dataset structural
   finding side-by-side bar chart:
   ```python
   subsets_lstm = lstm_attn.evaluate_all_subsets("test")
   fig = plot_subset_eval_bar(subsets_lstm,
                              title="LSTM-attn: only:X ≈ full")
   ```

(Footnote at end of §2: "`src/pipeline/fusion/mot_transformer.py`
is a documented honest-negative experiment kept in the repo for
reproducibility — see `handoff/results/RESULT_21_*` if curious.")

### §3 — Phase C: cross-dataset main results

The paper main table + the cross-cutting findings:

1. **The 6-row table**:
   ```python
   from src.pipeline.evaluation import MainResultsTable
   table = MainResultsTable.from_archive()
   table.to_dataframe()
   ```
2. **Heatmap visualisation** (optional, RESULT_27 bonus
   plotter):
   ```python
   from src.pipeline.visualization import plot_main_results_heatmap
   fig = plot_main_results_heatmap(table.to_dataframe())
   ```
3. **Per-leg validation status**:
   - WiFi UJI: Anchor2Vec 8.69 m vs wlanloc 15.17 m → ✓ beats by 43 %
   - IMU RoNIN: IMUCNN 7.59-9.96 m vs ResNet1D 5.14 m → partial
     (Umeyama within gate; raw outside)
   - Camera TartanAir: DPVOMotion 0.293 m vs TartanVO 0.012 m
     (last-20 %) → paper-soft
   - Odom Webots: internal, no SOTA
4. **C3 fusion on Webots** — CNN1D 0.339 m clears criterion
   (b) ≤ 0.5 m by 32 %.
5. **C4 cross-session on MSILN** — partial (clean SOTA beat,
   partial kNN gate).

### §4 — Honest gaps

Documented limitations, each as a short markdown block + the
cell that surfaces it:

1. **Smoothness debt** (architecture-invariant across 3 archs ×
   6 datasets):
   ```python
   smoothness_table = pd.DataFrame([
       # arch × dataset matrix of per-trajectory r medians
   ])
   ```
2. **C2 raw-ATE gap** on canonical RoNIN unseen-subjects
   (IMUCNN +47 % raw outside the 20 % gate; +15.7 % Umeyama
   inside).
3. **IMUWiFine test-no-IMU**: documented dataset property; fusion
   test column = WiFi-only on this dataset.
4. **TartanAir paper-soft per-leg**: our DPVOMotion uses
   Webots-trained head out-of-domain; not a fair fresh-data
   comparison to TartanVO.

### §5 — Reproducibility

Closing section: how to reproduce every number in §3:
- Setup: `git submodule update --init --recursive`; pip install
  requirements; download pretrained checkpoints + RoNIN FRDR
  archive (URLs in `docs/EXTERNAL_DEPENDENCIES.md`).
- Per-number reproduction commands:
  ```
  python scripts/eval_uji.py                  # RESULT_01 numbers
  python scripts/eval_ronin_canonical.py      # RESULT_07 numbers
  python scripts/eval_tartanair_hospital.py   # RESULT_08 numbers
  python scripts/eval_msiln.py                # RESULT_15 numbers (if shipped)
  python scripts/eval_imuwifine.py            # RESULT_19 numbers (if shipped)
  python scripts/train_phase_b_winner.py      # RESULT_17 CNN1D 0.339 m
  ```
- Where the RESULT_NN archives live: `handoff/results/` + `runs/
  overnight/run2_iter_*/`.

## Steps

### Step 0 — Notebook scaffold (10 min)

Create `notebooks/run2_walkthrough.ipynb` with the 6 section
headers + markdown intro per section. Each code cell starts as
a placeholder comment that names the consolidated-API call to
make.

```python
# §0.1 Webots stats
# TODO: call dataset_stats("webots") and render the
# per_path_summary as DataFrame
```

The plan's section structure above maps 1:1 to the notebook;
engineer's job is to fill in the placeholders.

**Acceptance**: notebook opens in Jupyter without errors;
section headers + placeholder cells in place.

### Step 1 — Fill §0 datasets pre-section (20 min)

For each of the 6 datasets, populate the 3-cell block. Cells
are mostly `from src.pipeline.{data, visualization} import ...`
+ 1-2 line invocation.

For modalities not in a dataset (e.g. UJI has no IMU), the
notebook **gracefully handles** by catching
`NotImplementedError` from `preprocessing_demo` and rendering
a "modality not applicable" markdown note.

**Acceptance**: §0 runs top-to-bottom in ≤ 90 s on a fresh
kernel; all 6 dataset overview figures rendered inline; ~24
preprocessing demos rendered.

### Step 2 — Fill §1 encoder audit (15 min)

4 subsections (Anchor2Vec / IMUCNN / DPVOMotion / OdomCNN).
Each loads the encoder, calls `.demo_forward` on a real sample
from the relevant dataset, renders the intermediate
visualisation. Verdict labels rendered as markdown.

**Acceptance**: §1 runs top-to-bottom in ≤ 60 s; 4 encoder
introspection visualisations + 4 verdict markdown blocks.

### Step 3 — Fill §2 fusion bake-off (15 min)

Load incumbent + CNN1D + LSTM-attn from their checkpoints via
`load_trained`; render the param/latency comparison table; run
the subset eval + staleness sweep on CNN1D; show LSTM-attn
dead-reckoning subset bar.

**Acceptance**: §2 runs top-to-bottom in ≤ 2 min (the
evaluate_staleness sweep is the long pole at 8 lags × ~10 s);
all paper-grade figures rendered.

### Step 4 — Fill §3 cross-dataset table (10 min)

`MainResultsTable.from_archive().to_dataframe()` →
`to_markdown()` → also render the heatmap. Per-leg validation
status rendered as a markdown checklist with the actual
numbers inline.

**Acceptance**: §3 runs in ≤ 30 s; the 6-row table renders;
heatmap visible.

### Step 5 — Fill §4 honest gaps + §5 reproducibility (10 min)

§4: smoothness-debt table, C2 gap explanation, IMUWiFine
caveat, TartanAir paper-soft caveat. §5: setup commands +
per-script reproduction commands + archive paths.

**Acceptance**: §4-§5 are mostly markdown + 1-2 cells; total
adds ≤ 20 s to notebook runtime.

### Step 6 — End-to-end execution smoke (10 min)

Engineer runs the notebook from "Restart Kernel and Run All".
Expected: all cells execute without error; total wall-clock
≤ 5 minutes.

If any cell errors or any figure is missing, engineer fixes
in-place. The notebook is the deliverable; it ships only when
it executes cleanly.

**Acceptance**: clean top-to-bottom execution; notebook
committed.

## Sources

- `handoff/SUMMARY.md` (the source-of-truth narrative).
- `handoff/SCIENTIST_NOTE_notebook-exclusions.md` (IPIN +
  MoTTransformer exclusions).
- RESULT_26 (`src/pipeline/baselines/` API).
- RESULT_27 (`src/pipeline/data/` + `src/pipeline/visualization/`
  API).
- RESULT_28 (`src/pipeline/fusion/build_arch` + encoder
  `demo_forward` + `FusionTrainer` public methods +
  `load_trained`).
- RESULT_29 (`src/pipeline/evaluation/MainResultsTable`).
- All `handoff/results/RESULT_NN_*.md` for the cell-level
  narrative content.

## What to report back

In `handoff/results/RESULT_30_notebook-walkthrough-scaffold.md`:

1. **Step 0** — notebook scaffold + section headers in place.
2. **Step 1** — §0 dataset pre-section: 6 datasets populated;
   wall-clock per section.
3. **Step 2** — §1 encoder audit: 4 subsections; visualisations
   rendered.
4. **Step 3** — §2 fusion bake-off: 3-arch comparison + subset
   eval + staleness; figures rendered.
5. **Step 4** — §3 cross-dataset table: 6-row main table +
   heatmap.
6. **Step 5** — §4 + §5 honest gaps + reproducibility.
7. **Step 6** — end-to-end execution: total wall-clock; any
   warnings or skipped cells documented.
8. **One open question** for the user (likely about cosmetic
   polish or narrative voice — the user iterates with engineer
   directly from here).

## Reversibility

- Steps 0-5: NEW `notebooks/run2_walkthrough.ipynb`. Engineer
  commits.
- Step 6: smoke run; no permanent changes.

Files committed: `notebooks/run2_walkthrough.ipynb` +
any small helper / sample fixture under `notebooks/_helpers/`
if needed.

**Compute budget**: ≤ 80 min.
- Step 0: 10 min.
- Step 1: 20 min (6 datasets × ~3 min cells; mostly mechanical).
- Step 2: 15 min.
- Step 3: 15 min.
- Step 4: 10 min.
- Step 5: 10 min.
- Step 6: 10 min (run-and-fix loop).

If overrun: cut §0's preprocessing demos to one modality per
dataset (the most representative one — WiFi for WiFi-having
datasets, IMU for RoNIN, Camera for TartanAir, Odom for
Webots). Drop §3's heatmap (it's a bonus visualisation; the
table itself is the load-bearing artifact).

If the smoke (Step 6) surfaces an unexpected cell error,
engineer either fixes the offending cell OR if it's a structural
issue (e.g. a consolidated API behaves differently than
expected), opens a new SCIENTIST_NOTE for the user to look at
next. The notebook ships when it executes cleanly.

## After PLAN_30

**Consolidation roadmap complete.** User picks up direct
iteration with engineer on:
- Notebook narrative polish (markdown voice, figure captions).
- Paper-framing decisions surfaced in §3 (which numbers to
  feature in headline; how to frame each honest gap).
- Optional PLAN_25b / PLAN_26-equivalent extensions (B-1/B-2
  loss-function lever for smoothness; MSILN re-run with new
  CNN1D; canonical RoNIN re-eval with the better aggregator).

`handoff/STATE.md` `STOP_REASON` gets a "Consolidation phase
complete; engineer-user iteration begins on notebook polish"
line after RESULT_30 commits.
