# Plan 36 — Remaining trajectory overlays (RoNIN + MSILN) + full-comparison tables (no-best-only)

> User pickup on RESULT_35's two open items (2026-05-26 ~21:45):
> (a) ship the RoNIN + MSILN trajectory overlays that PLAN_35
> cut per the overrun provision (pure ADD — no removal of the
> Webots overlays); (b) the v5 comparison table(s) only show
> the best value per row — user wants ALL values visible so
> the comparison reads.

## Two items

### Item A — Add RoNIN + MSILN trajectory overlays (pure addition)

NEW cells appended after §4.5 Webots overlays (or in a new
§4.6 / §4.7 — engineer's call). Use the existing
`plot_trajectory_comparison(...)` helper from RESULT_35.

#### §4.6 — RoNIN canonical: IMUCNN vs ResNet1D SOTA

For 1-3 representative unseen test sequences (pick from
RESULT_07's per-sequence ATE distribution — engineer chooses
the median + worst + best to span the range):

```python
# Predictions from:
#   - ResNet1D pretrained (SOTA, from §1.2's load + integrate path)
#   - IMUCNN (Ours, from §2.2's trained-or-loaded model)
# Both integrated forward-Euler from GT[0]; overlaid on GT.
for seq_name in selected_seqs:
    preds = {
        "ResNet1D (SOTA)": predict_ronin_seq(resnet1d_model, ronin_root / seq_name),
        "IMUCNN (Ours)":   predict_ronin_seq(imucnn,           ronin_root / seq_name),
    }
    gt = load_ronin_gt(seq_name)
    fig = plot_trajectory_comparison(preds, gt,
                                      title=f"RoNIN canonical — {seq_name}",
                                      equal_aspect=True)
    plt.show()
```

The `predict_ronin_seq(...)` helper already exists in
`src.pipeline.training` (RESULT_28's promotion from
`scripts/eval_ronin_canonical.py`). If absent, engineer adds
it as ~20 lines.

#### §4.7 — MSILN cross-session: fusion vs WiFi-kNN

MSILN's SOTA-equivalent is the WiFi-kNN baseline (RESULT_15's
gate (c)-1 reference, 9.5 m test). Pick 1-2 longest test paths
(per RESULT_15: path 128/129/130; path 130 dominates kNN test
mean — show it for the comparison).

```python
# Predictions:
#   - WiFi-kNN baseline (run inline on MSILN train/test; ~10 lines)
#   - Our fusion (load from runs/overnight/run2_iter_15 OR re-train if FAST_MODE=False)
for path_id in [128, 129, 130]:
    preds = {
        "WiFi-kNN baseline":     msiln_knn_predict(path_id),
        "Fusion (Ours)":         msiln_fusion_predict(path_id),
    }
    gt = msiln_path_gt(path_id)
    fig = plot_trajectory_comparison(preds, gt,
                                      title=f"MSILN site1/B1 cross-session — path {path_id}",
                                      equal_aspect=True)
    plt.show()
```

The `msiln_knn_predict(path_id)` is a thin in-notebook helper
(~15 lines) implementing WiFi-kNN (k=5 manhattan on training
fingerprints, RESULT_15 protocol). `msiln_fusion_predict`
reuses the loaded MSILN fusion trainer.

If MSILN fusion checkpoint isn't on disk and we're in
FAST_MODE=True: skip the MSILN overlay with a one-line note
("MSILN fusion checkpoint not cached; run scripts/eval_msiln.py
to populate"). Engineer's call — but the user said "only add
cells don't remove the previous", so missing-data fallback is
better than no cell at all.

**Acceptance**: 3-6 new figures appended (3 RoNIN seqs + 1-3
MSILN paths). All existing §4.5 Webots overlays untouched.

### Item B — Fix tables to show ALL values, not only the best

The v5 has 3 Styler tables in §5. User's complaint: the table
"didn't show all values so we can compare it; only shows the
best". Two possible causes:

1. **Table C is the culprit by design** — it's "best per
   dataset" with one value column. User wants to see ALL
   method values per dataset row so the comparison reads.
2. **Styler rendering issue** — pandas Styler with `apply`
   somehow truncates / hides non-winning values in
   nbconvert HTML output. Less likely but worth checking.

Engineer audits the current §5 tables before deciding the fix.
Default interpretation = Table C reshape:

#### Table C reshape — from "best only" to "all methods side-by-side, bold winner"

OLD shape (what RESULT_35 likely shipped):

```
| dataset | best method | best test MAE | category   |
|---------|-------------|---------------|------------|
| Webots  | CNN1D       | 0.34          | Ours fusion|
| UJI     | Anchor2Vec  | 8.69          | Ours enc.  |
| RoNIN   | ResNet1D    | 5.14          | SOTA       |
```

NEW shape:

```
| dataset           | SOTA      | Anchor2Vec | IMUCNN | DPVOMotion | OdomCNN | transformer | CNN1D | LSTM-attn |
|-------------------|----------|-----------:|-------:|-----------:|--------:|------------:|------:|----------:|
| Webots sim test   | n/a      | n/a        | n/a    | n/a        | n/a     | 0.41        | 0.35  | **0.30**  |
| UJI val           | 15.17    | **8.61**   | n/a    | n/a        | n/a     | n/a         | n/a   | n/a       |
| RoNIN canonical   | **5.13** | n/a        | 13.32  | n/a        | n/a     | n/a         | n/a   | n/a       |
| TartanAir last-20%| **0.012**| n/a        | n/a    | 0.29       | n/a     | n/a         | n/a   | n/a       |
| Webots odom test  | 8.27 floor| n/a       | n/a    | n/a        | **4.49**| n/a         | n/a   | n/a       |
| MSILN test        | …        | …          | …      | n/a        | n/a     | …           | …     | …         |
```

Every cell with a value shows the value; absent comparisons
show `n/a`; **bold the row winner**. This is the canonical
paper main-results-table shape.

Tables A and B already show ALL values (they're not "best only"
designs — they have explicit SOTA + Ours columns / explicit 3
arch columns). Engineer verifies the Styler renders cleanly in
nbconvert HTML output. If pandas Styler is the issue, fall back
to markdown table with dynamically-built `**X.XX**` bolding —
same visual result, more robust to rendering.

**Acceptance**: §5 has 3 tables, all values visible, winner
per row bolded.

## Steps

### Step 0 — Audit current §5 tables + §4.5 cell context (10 min)

Engineer reads:
- §5 Table A: confirm all values render (SOTA m + Ours m
  columns both visible).
- §5 Table B: confirm all 3 arch columns render.
- §5 Table C: confirm whether it's "best only" or already
  expanded.
- §4.5 Webots overlays: identify the cell layout to mirror
  for §4.6/§4.7 additions.

**Acceptance**: 3-row before-state inventory + which table needs
the reshape.

### Step 1 — Add §4.6 RoNIN trajectory overlay (30 min)

Engineer either:
- (a) Uses an existing `predict_ronin_seq(...)` helper in
  `src.pipeline.training` (was promoted in RESULT_28). Verify
  it returns `(pred_traj, gt_traj)` in the format
  `plot_trajectory_comparison` expects.
- (b) Implements inline if the helper is missing.

Cell flow:
1. Load ResNet1D pretrained + IMUCNN (Ours) — both already
   loaded in §1.2/§2.2 (engineer reuses the model handles).
2. Pick 3 representative seqs from `list_test_unseen.txt`
   (engineer picks: best, median, worst by RESULT_07
   per-sequence ATE — or any 3 reasonable choices).
3. For each, run both predictors + integrate → plot overlay.

Trajectories are 2D `(x, y)` per-window cumulative
integration anchored at `GT[0]` — both methods anchored the
same way for apples-to-apples.

**Acceptance**: 3 RoNIN overlay figures appended to the
notebook.

### Step 2 — Add §4.7 MSILN trajectory overlay (30 min)

Engineer either:
- Loads cached MSILN fusion checkpoint
  (`runs/overnight/run2_iter_15/.../model.pt`) + runs prediction
  on test paths.
- OR if checkpoint absent in FAST_MODE=True (typical), prints
  a one-line "MSILN fusion checkpoint not cached this run; see
  RESULT_15 / scripts/eval_msiln.py" note and skips the cell
  body. Don't error.

WiFi-kNN baseline:
- Build train fingerprint matrix from MSILN train scans (all
  Nov-24 paths' WiFi scans concatenated).
- For each test scan, k=5 nearest manhattan on RSSI → predict
  weighted-mean (x, y) of neighbours.
- ~15 inline lines. The RESULT_15 protocol.

Plot overlay for 1-3 test paths (128, 129, 130 per
RESULT_15's path-130-dominates-kNN-test finding).

**Acceptance**: 1-3 MSILN overlay figures appended OR a
documented skip note if the fusion checkpoint isn't on disk.

### Step 3 — Reshape Table C to show all methods (20 min)

Currently Table C has columns
`[dataset, best_method, best_test_MAE, category]` — only the
winning value per row.

Reshape to columns
`[dataset, SOTA, Anchor2Vec, IMUCNN, DPVOMotion, OdomCNN,
  transformer, CNN1D, LSTM-attn]` with every applicable cell
populated and `n/a` everywhere else. Bold the row minimum
(across columns with values) via pandas Styler.

```python
def style_winner_per_row(row, value_cols):
    """Bold the smallest numeric value in row[value_cols].
    Cells with n/a / NaN are skipped."""
    vals = {c: row[c] for c in value_cols if isinstance(row[c], (int, float)) and not np.isnan(row[c])}
    if not vals:
        return [""] * len(row)
    win = min(vals, key=vals.get)
    return [("font-weight: bold" if c == win else "") for c in row.index]

tbl_c = (
    pd.DataFrame(rows_c)
      .style
      .apply(lambda r: style_winner_per_row(r, value_cols), axis=1)
      .format("{:.2f}", subset=value_cols, na_rep="n/a")
)
```

Optionally also add a "best method" column at the end of the
row showing the winner's NAME (so the reader knows which
column the bold cell belongs to without counting):

```
| dataset | SOTA | Anchor2Vec | IMUCNN | ... | best |
| UJI val | 15.17| **8.61**   | n/a    | ... | Anchor2Vec |
```

Engineer's call whether to add the "best" name column or just
rely on visual bolding. Both work.

If pandas Styler rendering surfaces issues under nbconvert
HTML (truncation, hidden cells, etc.), fall back to markdown
table:

```python
def to_markdown_with_bold(df, value_cols):
    """Render DataFrame as markdown with **bold** on the row min."""
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, row in df.iterrows():
        vals = {c: row[c] for c in value_cols if isinstance(row[c], (int, float)) and not np.isnan(row[c])}
        win = min(vals, key=vals.get) if vals else None
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("n/a")
            elif c in value_cols and isinstance(v, (int, float)):
                cells.append(f"**{v:.2f}**" if c == win else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

from IPython.display import Markdown
display(Markdown(to_markdown_with_bold(tbl_c_df, value_cols)))
```

The markdown path is more robust across nbconvert versions.

**Acceptance**: Table C shows all method columns; winner per
row bolded; reader can scan and compare each dataset's full
method-row at a glance.

### Step 4 — Verify Tables A and B still show all values (10 min)

If RESULT_35's Tables A and B already show all values per
row (which the v5 plan-spec suggested), no action needed.
Engineer just visually confirms.

If either table is also "best only", same reshape pattern as
Step 3.

**Acceptance**: all 3 tables visibly show every value per
row, with winners bolded.

### Step 5 — Re-smoke FAST_MODE=True (10 min)

`jupyter nbconvert --to notebook --execute --inplace
notebooks/run2_walkthrough.ipynb`.

Confirm:
- §4.5 Webots overlays still there (NOT removed).
- §4.6 RoNIN overlays added.
- §4.7 MSILN overlays added or honestly-skipped.
- §5 Tables A, B, C all show every value with winners bolded.
- 0 cell errors.

**Acceptance**: clean notebook execution; visual inspection
addresses both items.

### Step 6 — Commit (5 min)

Single commit: notebook + any new helpers (e.g.
`msiln_knn_predict` in `src.pipeline.training` or notebook-
local), updated tables.

## What to report back

In `handoff/results/RESULT_36_remaining-overlays-and-full-comparison-tables.md`:

1. **Step 0** — pre-state inventory.
2. **Steps 1-2** — RoNIN + MSILN overlay cells; figures
   rendered.
3. **Steps 3-4** — table reshape; sample render output of
   the 3 tables.
4. **Step 5** — nbconvert smoke output.
5. **One open question** for the user.

## Reversibility

- Steps 1-4: notebook additions/edits. Engineer commits.

Files committed: `notebooks/run2_walkthrough.ipynb` (revised);
NEW `msiln_knn_predict` helper if engineer judges it reusable
beyond the notebook.

**Compute budget**: ≤ 1.5 hours.
- Step 0: 10 min.
- Step 1: 30 min (RoNIN, 3 seqs).
- Step 2: 30 min (MSILN, 1-3 paths).
- Step 3: 20 min (Table C reshape).
- Step 4: 10 min (verify A + B).
- Step 5: 10 min (smoke).
- Step 6: 5 min (commit).

If overrun: cut Step 2 to 1 MSILN path. Don't cut Step 3 —
the table reshape is the user's headline complaint.

If the MSILN fusion checkpoint isn't on disk and the kNN-only
overlay would be misleading without a fusion comparison
counterpart, skip §4.7 cleanly with the documented note and
flag for a follow-up that retrains MSILN under FAST_MODE=False.

## Quality bar

Same as PLAN_35: publication-grade visual quality. The user
asked for ADDITIONS (overlays) and a FIX (table comparison
visibility). Match those literally; don't scope-creep.
