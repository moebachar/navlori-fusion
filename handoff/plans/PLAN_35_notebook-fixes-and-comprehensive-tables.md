# Plan 35 — Notebook fixes + comprehensive summary tables + trajectory comparisons (7 user takes from 2026-05-26 ~19:30)

> Engineer-user iteration on v4 (RESULT_34). 7 takes from the
> user: 3 bugs (camera-source-mix-up, IMUWiFine aspect ratio,
> UJI saturation), 1 rename (incumbent → transformer), 3 polish
> upgrades (comprehensive tables, plot quality, predicted-vs-GT-
> vs-SOTA trajectory plots).

## User's 7 takes

| # | take | concrete action |
|---|---|---|
| 1 | "in webots raw modality samples i noticed in camera an extract from tartanair and not from webots env" | **BUG**: Webots `camera/` only has `depth_*.png` (no RGB); CSV references `rgb_*.png` that don't exist on disk. Notebook fell back to TartanAir frames. Fix: show what we actually have (depth + cached DPVO patch features from `dpvo_features.pt`) + honest "RGB not persisted at collection" note |
| 2 | "imuwifine x range is so huge compared to Y and when plotting a path i can only see a horizontal line" | **BUG**: GT trajectory plot lacks `set_aspect('equal')` for thin-strip datasets. Fix: equal aspect + auto-zoom bounds from path data |
| 3 | "ujiindoorloc wifi samples chow a full saturated samples (i think that is weird and worth investigating)" | **DATA INVESTIGATION**: UJI RSSI sentinel = 100 (non-detected); per scan only ~20-50 of 520 APs are detected. The "saturation" is most likely all the non-detected APs sitting at one normalized value (0 after `(rssi+100)/100`). Fix: show detection count per scan + only-detected-APs bar chart |
| 4 | "incumbent should be renamed to transformer (if that is it) in all the notebook" | **RENAME**: yes, `incumbent` = run-1's `FusionTransformer` (set-transformer). Display label → `transformer` everywhere in the notebook |
| 5 | "summury table dont containt all (all dataset, all incoders, all fusion archis) dynamicaly from notebook" | **EXPAND TABLES**: 2 → 3 paper-grade tables: (a) per-leg encoders × datasets vs SOTAs; (b) fusion-archs × datasets; (c) cross-cutting summary. All dynamic, all bolded winners |
| 6 | "plots quality is poor" | **PLOT STYLING**: DPI ≥ 150, font sizes 11-12, consistent palette, larger axis labels, tight_layout, proper figsize per-plot. Apply across all 7 inline-train curves + GT trajectories + modality samples + preprocessing-influence figures + final tables |
| 7 | "there isnt plots of predicted path by fusion methods vs ground truth vs sota prediction" | **NEW SECTION §4.X**: for each evaluated test-path dataset (Webots paths 15/16/17 + RoNIN canonical + MSILN test), overlay GT + our fusion predictions + SOTA prediction. Single 2D plot per path showing all trajectories color-coded |

## Hypothesis

After this iter, the notebook is publication-clean:

- **3 data-honesty bugs fixed**: Webots camera shows what we
  actually have (depth + cached DPVO features); IMUWiFine
  trajectory readable; UJI saturation explained honestly.
- **Naming is consistent**: `transformer` (the set-transformer
  from run-1) appears throughout instead of `incumbent`.
- **Summary tables comprehensive**: every dataset × every
  method we trained appears, with dynamic-bold winners.
- **Plots look paper-grade**: high DPI, consistent style,
  readable.
- **Trajectory comparisons** prove fusion qualitatively, not
  just numerically.

## Steps

### Step 0 — Inspect current notebook + RESULT_34 (10 min)

Quick audit of what RESULT_34 actually shipped:
- Read the `plot_modality_samples` function to find where the
  Webots camera path goes wrong.
- Note current GT trajectory implementation for IMUWiFine.
- Inspect UJI WiFi sample cell.
- Find every cell that says "incumbent".
- Inspect the v4 summary tables.
- Note plot styling state (DPI, figsize defaults).

**Acceptance**: 7-row "before" inventory in RESULT_35's TL;DR
matching the 7 takes.

### Step 1 — Take 1: fix Webots camera modality sample (15 min)

The Webots camera directory contains only `depth_*.png`. RGB
was never persisted (DPVO features pre-extracted to
`dpvo_features.pt` at collection time per the
`extract_dpvo_features.py` script in `scripts/`).

Fix in `plot_modality_samples("webots")`:

```python
def plot_webots_camera_sample(path_id):
    """Show what Webots ACTUALLY has: one depth frame +
    cached DPVO patch features. Honest about RGB not being
    persisted."""
    pdir = ROOT / "data/async_collection" / f"path_{path_id:02d}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Panel A: one depth frame
    depth_files = sorted((pdir / "camera").glob("depth_*.png"))
    if depth_files:
        depth = plt.imread(depth_files[len(depth_files)//2])
        axes[0].imshow(depth, cmap="viridis")
        axes[0].set_title(f"depth frame ({depth_files[len(depth_files)//2].name})")
    else:
        axes[0].text(0.5, 0.5, "no depth frames", ha="center")

    # Panel B: DPVO patch-feature visualization from
    # dpvo_features.pt
    feat_path = pdir / "dpvo_features.pt"
    if feat_path.exists():
        feats = torch.load(feat_path, map_location="cpu",
                            weights_only=False)
        # Visualize as a heatmap of patch tokens at one
        # mid-sequence instant
        ...
        axes[1].set_title("cached DPVO patch features")

    axes[0].axis("off"); axes[1].axis("off")
    fig.suptitle("Webots camera: RGB not persisted on disk; "
                  "the fusion pipeline consumes cached DPVO features.")
    fig.tight_layout()
    return fig
```

The TartanAir-image fallback in the modality-samples helper
is **removed**. The plot now honestly shows that Webots
collected depth + DPVO features (not raw RGB).

**Acceptance**: §0 Webots block's camera panel shows depth +
DPVO patch feature visualization with the honest caption.

### Step 2 — Take 2: fix IMUWiFine GT trajectory aspect (5 min)

In `plot_gt_trajectory(name, path_id)`:

```python
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.plot(gt[:, 0], gt[:, 1], lw=1.5)
ax.scatter(gt[0, 0], gt[0, 1], c="green", s=40, label="start", zorder=3)
ax.scatter(gt[-1, 0], gt[-1, 1], c="red",   s=40, label="end",   zorder=3)
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"{name} path {path_id} — GT trajectory ({len(gt)} samples)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
fig.tight_layout()
return fig
```

The `set_aspect("equal", adjustable="datalim")` ensures
thin-strip datasets like IMUWiFine still show the path shape
correctly; the figure stretches in the x-direction to fit.

If a path is so thin that "equal aspect" makes the y-axis
invisible (extreme case), engineer adds an auto-detection:
"if (x_range / y_range > 20): use `aspect='auto'` + axis tick
labels show absolute scales".

**Acceptance**: IMUWiFine GT plot shows a visible path
(curves, turns, etc.), not a horizontal line.

### Step 3 — Take 3: investigate + fix UJI WiFi saturation (15 min)

UJI RSSI value `100` is the non-detection sentinel (per
`scripts/eval_uji.py` and RESULT_01). Real RSSI values are in
[-100, 0] dBm. Normalization `(rssi + 100) / 100` maps real
detections to (0, 1] and the sentinel to **2.0**, which when
plotted alongside [0, 1] makes most bars look at one extreme.

Engineer audits the current `plot_modality_samples("uji_indoorloc")`
cell — the saturation is almost certainly because the
plot shows ALL 520 APs (most of which are non-detected per
scan; only ~20-50 are detected).

Fix:

```python
def plot_uji_wifi_sample(scan_idx=0):
    """Show what one UJI scan looks like: 520 APs, only ~20-50
    are detected (real RSSI in [-100, 0] dBm); the rest are at
    the sentinel value 100. Two panels: detected-only bar
    chart + sparsity stat."""
    Xva = load_dataset("uji_indoorloc", split="validation")["X"]
    raw = Xva[scan_idx]
    detected = raw[raw != 100]
    n_detected = len(detected)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # Panel A: detected APs only, bar chart
    axes[0].bar(np.arange(n_detected), detected, color="steelblue")
    axes[0].set_xlabel("detected AP index (sorted)")
    axes[0].set_ylabel("RSSI (dBm)")
    axes[0].set_title(f"scan {scan_idx}: {n_detected}/520 APs detected")
    axes[0].set_ylim(-110, 0)

    # Panel B: distribution of n_detected across the validation set
    n_det_per_scan = (Xva != 100).sum(axis=1)
    axes[1].hist(n_det_per_scan, bins=30, color="steelblue", edgecolor="black")
    axes[1].axvline(n_det_per_scan.mean(), color="red", ls="--",
                     label=f"mean = {n_det_per_scan.mean():.1f}")
    axes[1].set_xlabel("detected APs per scan")
    axes[1].set_ylabel("# scans")
    axes[1].set_title(f"UJI val scan sparsity (n={len(Xva)} scans)")
    axes[1].legend()

    fig.tight_layout()
    return fig
```

This is HONEST about UJI's sparsity (typically 20-50 APs
detected per scan out of 520) — and the "saturated" plot the
user saw was actually 470+ non-detected APs sitting at one
sentinel value.

The `dataset_stats("uji_indoorloc")` should also surface this
as a `known_caveats` entry: "RSSI sparsity ~20-50 detected APs
per scan out of 520; non-detected sentinel value 100".

**Acceptance**: UJI block shows the detected-only RSSI bar +
the sparsity histogram. The "saturated" appearance disappears.

### Step 4 — Take 4: rename incumbent → transformer (10 min)

The `incumbent` is the run-1 `FusionTransformer` (set-
transformer with self-attention cross-modal + cross-time +
cross-attention `PositionQuery` readout). Display name in the
notebook = `transformer`.

Find and replace in the notebook:
- `incumbent` (lowercase) → `transformer` in markdown + cell
  prints
- `Incumbent` → `Transformer` in titles
- `"incumbent"` keys in `trainers` dict / config tables → keep
  as `"transformer"` (engineer's call: keep registry key
  `incumbent` for back-compat AND alias to `transformer`, OR
  rename registry too — both work; the cleaner approach is to
  alias the display name only)

The arch registry in `src.pipeline.fusion.__init__` likely
still uses `"incumbent"` as the key. Engineer's call:
- (a) Add `"transformer"` as an alias → `build_arch("transformer")`
  works AND `build_arch("incumbent")` still works.
- (b) Hard-rename the registry → `build_arch("transformer")`
  only.

Recommend (a) so existing checkpoints + RESULT_NN references
still load. The notebook uses the display name `transformer`
throughout.

**Acceptance**: `rg -i incumbent notebooks/run2_walkthrough.ipynb`
finds zero matches (or only the alias-definition cell);
`transformer` appears consistently.

### Step 5 — Take 5: comprehensive dynamic summary tables (30 min)

The v4 notebook's tables are partial. Expand into 3
paper-grade tables, each with **dynamic values + dynamic
bold-winner**:

#### Table A: Per-leg encoders vs SOTAs (across datasets)

Rows = (dataset × modality); columns = SOTA value, SOTA name,
Our encoder value, Our encoder name. Bold the winner per row.

```
| dataset             | modality  | SOTA               | SOTA m  | Ours              | Ours m  |
|---------------------|-----------|--------------------|--------:|-------------------|--------:|
| UJIIndoorLoc        | WiFi      | wlan_localization  | 15.17   | Anchor2Vec        |  **8.61** |
| RoNIN canonical     | IMU       | ResNet1D           | **5.13**| IMUCNN            |   13.32   |
| TartanAir hospital  | Camera    | TartanVO           | **0.012**| DPVOMotion+head  |   0.29    |
| Webots              | Odom      | trivial integration|   8.27  | OdomCNN-P-B       |  **4.49** |
```

(Engineer pulls values from variables computed in §1-§2;
TartanVO and trivial-integration are SOTA-equivalents per the
run-2 convention.)

#### Table B: Fusion architectures × datasets

Rows = dataset; columns = transformer / CNN1D / LSTM-attn.
Bold the winner per row.

```
| dataset           | metric     | transformer | CNN1D     | LSTM-attn  |
|-------------------|------------|-------------|-----------|------------|
| Webots sim val    | MAE (m)    | 0.413       | **0.281** | 0.221?     |
| Webots sim test   | MAE (m)    | 0.410       | 0.346     | **0.261?** |
```

(LSTM-attn -23% finding from RESULT_33 may show LSTM-attn as
winner under FAST_MODE=False. Engineer reports actual
live-run numbers and bolds per row.)

If we don't have MSILN / IMUWiFine / other dataset numbers
across the 3 archs, the table has only Webots rows. The user's
take #5 ("all dataset, all fusion archis") implies the user
WANTS this filled in for all datasets. Practically:
- For Webots: all 3 archs were trained in §3 → 2 rows (val + test).
- For MSILN / IMUWiFine / IPIN / etc.: only CNN1D and/or
  LSTM-attn were trained in run-2. Add rows showing what we
  HAVE (n/a for missing cells), with a footnote.

Honest table beats inflated table.

#### Table C: Cross-cutting summary — best per dataset

One row per dataset; "Best method" column (whichever beat all
others on test) + "Best test MAE" + "Best method type"
(SOTA / Ours encoder / Ours fusion).

```
| dataset               | best method        | best test MAE | category    |
|-----------------------|--------------------|--------------:|-------------|
| Webots sim            | CNN1D / LSTM-attn  | 0.34 m        | Ours fusion |
| UJIIndoorLoc          | Anchor2Vec         | 8.61 m        | Ours enc.   |
| RoNIN canonical       | ResNet1D           | 5.13 m        | SOTA        |
| TartanAir hospital    | TartanVO           | 0.012 m       | SOTA        |
| MSILN                 | CNN1D / LSTM-attn  | …             | Ours fusion |
| IMUWiFine             | …                  | …             | …           |
```

(Engineer fills dynamically. This is the "headline" table — the
one the paper abstract / intro will cite from.)

All 3 tables use either pandas Styler (`.apply(lambda r:
['font-weight: bold' if ... else '' for c in cols], axis=1)`)
or markdown with computed `**X.XX**` substitution — engineer's
call based on nbconvert HTML render fidelity.

**Acceptance**: 3 tables appear in §5; each has dynamic
values (no hard-coding); each has dynamic bolded winners.

### Step 6 — Take 6: plot quality upgrade (20 min)

Sweep all `plt.subplots(...)` calls in the notebook AND in
`src.pipeline.visualization.*` plotters:

- `dpi=150` (minimum) or `dpi=200` for publication.
- `figsize` matched to content (no tiny figures, no oversized
  empty figures).
- `set_paper_style()` actually applied (font.size=12,
  axes.labelsize=12, axes.titlesize=13, xtick/ytick.labelsize=11,
  legend.fontsize=10, lines.linewidth=1.5, grid.alpha=0.3).
- `fig.tight_layout()` everywhere.
- Consistent color palette: ours blue, SOTAs orange/red,
  GT/floor green, predictions various shades — engineer's
  call but consistent across plots.

Each plot type's checklist:
- Training curves: 2-panel (loss + val MAE), figsize (10, 3),
  legend, grid.
- GT trajectories: figsize (8, 6), equal aspect (with
  override), start/end markers, grid.
- Modality samples: per-modality custom layouts; consistent
  font sizing.
- Preprocessing-influence: side-by-side before/after,
  clear titles.
- Trajectory comparison plots (NEW per take 7): figsize (8, 6),
  GT + our preds + SOTA pred overlaid with distinct line styles.

**Acceptance**: visual sweep through all rendered figures shows
consistent high-DPI, readable labels, paper-grade quality.

### Step 7 — Take 7: predicted-path-vs-GT-vs-SOTA overlay plots (30 min)

NEW §4.X or §6.X section (engineer's call where it fits best;
could be a new §6 "qualitative trajectory comparison" before
the summary tables):

For each test path we can evaluate:

- **Webots sim test paths 15, 16, 17**: 3 plots, each showing
  GT + transformer-fusion pred + CNN1D pred + LSTM-attn pred
  overlaid. ~ The "SOTA" here is the WiFi-only `only:wifi`
  subset eval baseline — engineer pulls from
  `cnn1d.evaluate_all_subsets("test")["only:wifi"]` already
  computed in §4. So the comparison is "WiFi anchor alone vs
  fusion".

- **RoNIN canonical**: pick 1-3 representative test sequences;
  plot GT + IMUCNN pred (Ours) + ResNet1D pred (SOTA). Both
  trajectories are integrated from velocity, anchored to
  GT[0].

- **MSILN cross-session**: pick 1-2 test paths; plot GT + our
  best fusion (CNN1D or whichever) + WiFi-kNN baseline (the
  RESULT_15 reference baseline).

- **UJI**: per-scan dataset, no trajectory. Skip — document.

- **IMUWiFine + TartanAir**: engineer's call whether to
  include based on data availability + time budget.

Plotting helper (NEW in `src.pipeline.visualization`):

```python
def plot_trajectory_comparison(predictions: dict[str, np.ndarray],
                                 gt: np.ndarray,
                                 path_id: int,
                                 title: str = "",
                                 *, equal_aspect: bool = True):
    """Overlay multiple predicted trajectories with GT.
    predictions = {"method_name": (N, 2) array, ...}
    Returns matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2.5, label="ground truth", alpha=0.9, zorder=5)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(pred[:, 0], pred[:, 1], lw=1.5,
                 color=colors[i % len(colors)], label=name, alpha=0.8)
    ax.scatter(gt[0, 0], gt[0, 1], c="green", s=60, marker="o", zorder=6, label="start")
    ax.scatter(gt[-1, 0], gt[-1, 1], c="red",  s=60, marker="s", zorder=6, label="end")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
```

For each dataset, the notebook cell:
1. Loads the test split + ground truth.
2. Runs each method's predictor (loaded trainer or stored
   prediction).
3. Calls `plot_trajectory_comparison(...)`.

For the fusion trainers, predictions come from
`trainer.predict("test")`. Engineer ensures this method exists
on `FusionTrainer` (or adds it if missing — should return a
dict of `{path_id: pred_xy}` or similar).

**Acceptance**: 3 Webots test-path overlay plots + 1-3 RoNIN
overlay plots + 1-2 MSILN overlay plots visible in the
notebook. Each shows GT in black + 2-3 method predictions
color-coded + start/end markers.

### Step 8 — Re-smoke FAST_MODE=True (15 min)

After all 7 fixes/upgrades, run
`jupyter nbconvert --to notebook --execute --inplace
notebooks/run2_walkthrough.ipynb` in `FAST_MODE=True`.

Confirm:
- Webots camera panel shows depth + DPVO patch features (no
  TartanAir image).
- IMUWiFine GT plot shows a visible path (not horizontal line).
- UJI WiFi sample shows detected-only RSSI + sparsity hist.
- "transformer" appears consistently; no "incumbent" in
  display.
- 3 summary tables rendered with bolded winners.
- Plots look paper-grade.
- Trajectory comparison plots visible.
- 0 cell errors.

**Acceptance**: clean nbconvert run; visual inspection confirms
all 7 takes addressed.

### Step 9 — Commit (5 min)

Single commit: notebook + any new visualization helpers
(`plot_trajectory_comparison`, possibly updated
`plot_modality_samples`, updated `set_paper_style`) + arch-
registry alias if added.

## What to report back

In `handoff/results/RESULT_35_notebook-fixes-and-comprehensive-tables.md`:

1. **Step 0** — 7-row before/after inventory.
2. **Steps 1-3** — bug fixes (camera / aspect / UJI).
3. **Step 4** — incumbent → transformer rename audit.
4. **Step 5** — 3 summary tables with sample renders.
5. **Step 6** — plot-style sweep summary.
6. **Step 7** — trajectory comparison plots filed.
7. **Step 8** — nbconvert smoke output.
8. **One open question** for the user.

## Reversibility

- All steps: notebook revisions + library helpers. Engineer
  commits in one go.

Files committed: `notebooks/run2_walkthrough.ipynb` (revised);
NEW `plot_trajectory_comparison` + possibly updated
`plot_modality_samples` / `plot_gt_trajectory` /
`set_paper_style` in `src.pipeline.visualization/`; optional
arch-registry alias `"transformer" = "incumbent"` in
`src.pipeline.fusion.__init__`.

**Compute budget**: ≤ 2.5 hours.
- Step 0: 10 min.
- Step 1: 15 min.
- Step 2: 5 min.
- Step 3: 15 min.
- Step 4: 10 min.
- Step 5: 30 min.
- Step 6: 20 min.
- Step 7: 30 min.
- Step 8: 15 min.
- Step 9: 5 min.

If overrun: cut Step 7's MSILN trajectory comparisons (Webots
+ RoNIN cover the key claims). Don't cut Step 5 (the tables
are the headline deliverable per take #5).

If the `predict()` method on `FusionTrainer` doesn't exist (or
doesn't return predictions in a per-path-compatible format),
engineer adds a `predict_test_paths(split) -> dict[int, ndarray]`
helper as part of Step 7. ~30 lines.

## Quality bar

Same as PLAN_34: publication-grade visual quality. The user
explicitly mentioned **"plots quality is poor"** as take #6 —
this iter is the visual polish pass. Take time to get plot
sizes, fonts, DPI, palettes consistent.

Notebook ships when all 7 takes are visibly addressed and a
final visual inspection (engineer + user) agrees.
