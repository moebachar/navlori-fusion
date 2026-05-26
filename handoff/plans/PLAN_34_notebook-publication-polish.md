# Plan 34 — Notebook publication polish (5 user takes from 2026-05-26 ~18:50)

> Engineer-user iteration on the v3 notebook (RESULT_33). 5
> takes from the user, each translates to concrete cell-level
> changes. The notebook is the **canonical published artifact**;
> the previous "archive drift" prints, "RESULT_NN sources",
> and "we improved from worst version" framing are dev-process
> metrics that don't belong in a publication. This pass strips
> them, adds the things that DO belong (GT trajectories,
> modality samples, preprocessing-influence figures, training
> curves, two paper-grade summary tables), and humanizes the
> tone.

## User's 5 takes

| # | take | concrete action |
|---|---|---|
| 1 | "for data i think plotting the destributions is bad. but i would love to plot some ground truth full path. a sample of data from each modality available in a dataset and some few stats. then gave some figures that show the preprocessing influence on a dataset" | **Reshape §0**: drop distributions; add GT trajectory plot per dataset, raw-modality sample windows, preprocessing-influence figures |
| 2 | "we need some text formating to reduce the markdowns writing only the essentials in a humanized way + in the results remove the hard coded prints of expected results (and minimize sentences through prints)" | **Strip verbose markdown + remove drift-vs-archive prints in cells** |
| 3 | "keep the minimal of results importing for publishing (train val test errors...) and not metrics of our iterations dev process" | **Drop RESULT_NN archive references**: the notebook is the canonical source; reader's live results ARE the publication numbers |
| 4 | "til now i cant see training curves in the notebook i still need that" | **Add training-curve plots** to every inline-train cell (Anchor2Vec, IMUCNN, OdomCNN, DPVOMotion head, 3 fusion archs) |
| 5 | "in the end i need a cell that shows only meters error of ours agains sota methofds pulling values dynamicaly from above cells and have the finidng results bold dynamicaly + a table coparing my own 3 archi on all datasets" | **Add 2 paper-grade summary tables at end**: (a) Ours vs SOTA, (b) Our 3 archs × datasets; dynamic values + dynamic bold-winner |

## Hypothesis

After this iter, `notebooks/run2_walkthrough.ipynb` is a
publication artifact rather than a dev-archive reproduction
trace:

- **§0 (datasets)** has the right shape for a paper-supplement:
  per dataset = 4-line stats + GT trajectory + modality samples
  + preprocessing-influence figure. No distributions.
- **§1-§3** show the live numbers cleanly without
  archive-comparison clutter. Each cell prints **only its own
  result** (val MAE, test MAE, training curve).
- **Every `train_*` call renders a training-curve plot** in the
  cell that called it.
- **§5 final cell** = two paper-grade tables: (a) Ours vs SOTA
  per dataset/modality (bolds the winner); (b) Our 3 archs
  across the cross-dataset rows (bolds the best per row). Both
  tables pull values DYNAMICALLY from variables earlier in the
  notebook — no hard-coded numbers anywhere.
- Markdown is essentialised: only the section/subsection
  headers + 1-2 sentence intro per section. No bullet
  summaries of run-2 iteration history.

## Steps

### Step 0 — Plan the §0 reshape per dataset (10 min)

For each of the 6 paper-facing datasets, decide which content
goes in. Template (engineer adapts per dataset):

```
### §0.1 — Webots (custom 4-modality)

  ~3 lines of essential stats: 18 paths, modalities, splits,
  sensor rates, one-line caveat.

  Figure A: ground-truth (x,y) trajectory of one sample path
  (e.g. path_5; longest of the train set).

  Figure B: 4-panel "what the raw data looks like" — 1 s of
  WiFi RSSI bar chart (per-AP), 1 s of IMU 6-channel time
  series, 1 frame of camera (RGB image), 1 s of odom 7-column
  time series. The modality availability per dataset gates
  which panels appear (UJI has only WiFi → just 1 panel;
  TartanAir has only Camera → 1 panel; etc).

  Figure C: preprocessing-influence (1 figure per dataset's
  primary modality). Examples:
    - Webots WiFi: raw RSSI [-100, 0] sentinel-100 → normalized
      [0, 1] Anchor2Vec-input.
    - IMUWiFine WiFi+IMU: raw IMU device-frame → world-frame
      rotation (RoNIN-style).
    - MSILN WiFi: raw RSSI sparse scan → dense imputed.
    - RoNIN IMU: raw 200 Hz device-frame → 6-channel world-
      frame (the disaster fix from RESULT_02).
    - TartanAir Camera: ImageNet-norm → DPVO-norm (2x − 0.5).
    - UJI WiFi: raw RSSI vector → 520-AP normalized vector
      (no temporal context).
```

Engineer's call on which preprocessing to surface per dataset.
The figures are small (1-row layouts, ~2 inches tall) — keep
them tight; don't blow up the notebook.

**Acceptance**: 6 dataset-blocks documented in plan with
their stats / trajectory / modality-samples / preprocessing
selections.

### Step 1 — Implement the §0 reshape (35 min)

Build the per-dataset blocks using a small helper pattern in
the notebook (not a full library function — these are
display-only):

```python
def show_dataset_block(name):
    stats = dataset_stats(name)
    # 1) Compact stats line
    print_compact_stats(stats)            # ~3-5 lines max
    # 2) GT trajectory of one sample path
    plot_gt_trajectory(name, path_id=stats["sample_path_for_demo"])
    # 3) Modality samples
    plot_modality_samples(name)
    # 4) Preprocessing influence
    plot_preprocessing_influence(name, modality=stats["primary_modality"])
```

The 3 plot helpers either live in
`src.pipeline.visualization` (if the engineer judges they're
reusable beyond this notebook) OR are notebook-local
functions (if they're dataset-specific). For per-dataset
preprocessing-influence figures, engineer's call: extend
`plot_preprocessing_demo` (already exists from RESULT_27) OR
write a new `plot_preprocessing_influence` that emphasizes
"this is what preprocessing changes" rather than the demo
side-by-side. Either works; new code goes in
`src/pipeline/visualization/` if reusable.

`plot_dataset_overview` (the multi-panel distributions
plotter from RESULT_27) **is dropped from the notebook**
(per take #1). The function stays in
`src/pipeline/visualization/` for any future use, but the
notebook doesn't call it.

**Acceptance**: §0 has 6 dataset blocks; each block fits in
roughly 3-4 cells; no distribution histograms; GT
trajectories visible; preprocessing-influence figures clear.

### Step 2 — Strip archive-comparison prints + verbose markdown (15 min)

Sweep §1, §2, §3, §4 for these patterns and remove:

```python
# REMOVE: archive references in cell prints
print(f"  Archive (RESULT_07): 5.140 m   |   drift: {abs(...)/5.140*100:.2f}%")

# REMOVE: hard-coded archive variable comparisons
print(f"  Archive (RESULT_01): 8.69 m   |   drift: {abs(...)/8.69*100:.1f}%")

# REMOVE: long markdown blocks recounting RESULT_NN history
"### §1.1 — wlan_localization SOTA on UJI
- wlan_localization (SOTA, global mode): 15.17 m val mean
  Euclidean.
- Anchor2Vec (ours, 0.075 M params): 8.69 m val (−43% vs SOTA).
Audit verdict: keep. Anchor2Vec beats the SOTA by 43% at one-
quarter the param budget."
```

Replace with:
- **Cells**: print ONLY the live number(s) for the current
  cell. One line per result.
- **Markdown**: section header + 1-2 sentence intro saying
  what this section does. Skip the bullet-list reiteration of
  RESULT_NN findings.

Tone: humanized, direct. "We compare X against Y on dataset Z"
instead of "Per the locked Phase B winner (RESULT_17), our
canonical CNN1D architecture (0.51 M params, K=4, 4-mod
B=128) achieves...".

**Acceptance**: rg `Archive (RESULT_` and rg `drift` find no
matches in cell prints (drift may survive only in the §5 final
tables IF the user wants reproduction-from-archive validation
there — see Step 5).

### Step 3 — Add training-curve plots to every inline-train cell (20 min)

Every `train_*` helper returns a `history` dict
(`{"train_loss": [...], "val_loss": [...], "val_mae": [...]}`).
After every inline training call, the notebook plots:

```python
def plot_training_curves(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"],   label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].set_title(f"{title} — loss")
    axes[1].plot(history["val_mae"], label="val MAE")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MAE (m)")
    axes[1].legend(); axes[1].set_title(f"{title} — val MAE")
    fig.tight_layout(); return fig
```

(Goes into `src.pipeline.visualization` since reusable.)

Cells to update:
- §2.1 Anchor2Vec — plot after `train_anchor2vec(...)`.
- §2.2 IMUCNN — plot after `train_imucnn(...)`.
- §2.3 DPVOMotion head — plot after `train_dpvo_motion_head(...)`.
- §2.4 OdomCNN — plot after `train_odomcnn(...)`.
- §3 × 3 fusion archs — plot after `train_fusion_arch(arch, ...)`.

In `FAST_MODE=True` (load checkpoint), the cell still shows
the training curve IF `history` is loaded from the checkpoint
(engineer ensures `train_*` saves history to the checkpoint
dict; load function reads it back). If the history isn't in
the checkpoint, the cell shows a small note "training curves
saved at <path>, not embedded in this checkpoint".

**Acceptance**: 7 training-curve figures visible in the
notebook (one per `train_*` call); both `FAST_MODE` branches
render the curves (live in slow mode, from checkpoint in
fast mode).

### Step 4 — Restructure §5 into 2 paper-grade summary tables (25 min)

Currently §5 has `MainResultsTable.from_archive()` (cached
numbers) + drift-vs-archive. **Both go**, per takes #3 + #5.

Replace with two final tables computed from variables
populated in §1-§3:

#### Table A: "Ours vs SOTA per dataset / modality"

For each dataset where a published SOTA exists, one row:
- Dataset
- Modality(ies)
- SOTA method name
- SOTA value (live from §1)
- Ours method name (Anchor2Vec / IMUCNN / DPVOMotion / etc.)
- Ours value (live from §2)
- Winner: **bold** the smaller number

```python
# §5.A — Ours vs SOTA
rows_a = [
    {"dataset": "UJI",       "modality": "WiFi",       "SOTA": "wlan_localization", "SOTA_m": wlanloc_uji_mae,    "Ours": "Anchor2Vec",   "Ours_m": anchor2vec_uji_mae},
    {"dataset": "RoNIN can.","modality": "IMU",        "SOTA": "ResNet1D",          "SOTA_m": ronin_resnet_ate,   "Ours": "IMUCNN",       "Ours_m": imucnn_canonical_raw},
    {"dataset": "TartanAir", "modality": "Camera",     "SOTA": "TartanVO",          "SOTA_m": tartanvo_ate,       "Ours": "DPVOMotion",   "Ours_m": dpvo_motion_ate},
]
tbl_a = pd.DataFrame(rows_a)

# Dynamic bold: lower is better, bold the winner per row
def style_winner(row, sota_col="SOTA_m", ours_col="Ours_m"):
    if row[sota_col] < row[ours_col]:
        return ["", "", "", "font-weight: bold", "", ""]
    else:
        return ["", "", "", "", "", "font-weight: bold"]
tbl_a.style.apply(style_winner, axis=1).format({"SOTA_m": "{:.2f}", "Ours_m": "{:.2f}"})
```

(Engineer uses pandas Styler for HTML-rendered bolding in
Jupyter; alternative is to build a markdown table with
`**X.XX**` substituted dynamically — engineer picks whichever
renders cleanly in nbconvert output.)

#### Table B: "Our 3 archs across cross-dataset rows"

For each main-table dataset where all 3 archs were evaluated
(Webots in this run; others optional), one row per dataset
with 3 numeric columns:

```python
rows_b = [
    {"dataset": "Webots sim", "incumbent_test": incumbent_test_mae,
                                "cnn1d_test":     cnn1d_test_mae,
                                "lstm_attn_test": lstm_attn_test_mae},
    # ... future datasets if multi-arch evaluations exist
]
tbl_b = pd.DataFrame(rows_b)

# Dynamic bold: lower test_mae wins; bold the winning column per row.
def style_min_per_row(row, cols=("incumbent_test", "cnn1d_test", "lstm_attn_test")):
    vals = [row[c] for c in cols]
    win = cols[vals.index(min(vals))]
    return ["font-weight: bold" if c == win else "" for c in cols]
tbl_b.style.apply(style_min_per_row, subset=["incumbent_test","cnn1d_test","lstm_attn_test"], axis=1)
```

(Engineer adapts; if other datasets have multi-arch
evaluations, they go as additional rows. In run-2 only Webots
has all 3 archs; the table can still show 1 row meaningfully.)

**Acceptance**: §5 has 2 tables, both with dynamic bolding;
all values pulled from variables defined earlier; no
hard-coded numbers.

### Step 5 — Final markdown sweep + smoke (20 min)

Pass through §1-§7 once more:
- Compress markdown to essentials.
- Remove "RESULT_NN" attributions from in-cell prints (they
  may survive in §7's archive cross-references section as
  appropriate; engineer's call).
- Humanize tone: short, declarative; no "Per the run-2
  archive..." preamble.

Run final `jupyter nbconvert --execute` in `FAST_MODE=True`
(no need to re-smoke `FAST_MODE=False` — the polish is
cosmetic, not behavioural). Confirm:
- All cells produce output.
- Training-curve figures visible in §2-§3.
- Both summary tables in §5 render with dynamic bolding.
- No `Archive (RESULT_NN):` strings in cell output.

**Acceptance**: clean `FAST_MODE=True` execution; visual
inspection of the rendered notebook matches the 5 takes.

### Step 6 — Commit (5 min)

Single commit: notebook + any new visualization helpers
(`plot_training_curves`, optional
`plot_preprocessing_influence`, optional `plot_modality_samples`)
+ updated `__init__.py` re-exports.

## What to report back

In `handoff/results/RESULT_34_notebook-publication-polish.md`:

1. **Step 0** — per-dataset reshape plan; 6 blocks specified.
2. **Step 1** — §0 reshape done; sample figure list.
3. **Step 2** — archive-print sweep; before/after line counts.
4. **Step 3** — 7 training-curve figures filed.
5. **Step 4** — 2 summary tables; styler / markdown approach
   chosen; sample render output.
6. **Step 5** — final smoke; visual inspection notes.
7. **One open question** for the user.

## Reversibility

- Steps 0-5: notebook revisions; engineer commits.
- NEW helpers in `src.pipeline.visualization`: permanent.

Files committed: `notebooks/run2_walkthrough.ipynb` (revised);
NEW visualization helpers if any.

**Compute budget**: ≤ 2 hours.
- Step 0: 10 min.
- Step 1: 35 min (§0 reshape — biggest piece).
- Step 2: 15 min.
- Step 3: 20 min.
- Step 4: 25 min.
- Step 5: 20 min.
- Step 6: 5 min.

If overrun: cut Step 3's training curves to ONLY the 4
encoder cells (skip the 3 fusion cells' curves; document why).
Don't cut Step 4 — the two summary tables are the headline
deliverable per take #5.

If `pandas.Styler` doesn't render cleanly under nbconvert
(known issue with HTML output in some Jupyter versions),
engineer falls back to markdown tables with dynamically-built
`**X.XX**` substitution — same visual result, more robust
to rendering.

## Quality bar

This is the **publication artifact** pass. Match the user's
takes literally:
- No distributions in §0.
- GT trajectories + raw modality samples + preprocessing
  influence per dataset.
- Tone: humanized, essentials only.
- Cell prints: live numbers only, no archive comparisons.
- Training curves: every inline-train shows them.
- §5: 2 dynamic-bold tables.

The notebook ships when these conditions hold and a final
visual inspection (engineer + user) agrees.
