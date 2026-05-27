# Plan 37 — Fill all fusion-arch cells across all datasets + add IMUWiFine row to Table C

> User directive 2026-05-26 ~22:15: Table C must show every
> value that CAN be filled. The v6 has `n/a` in fusion-arch
> columns for every dataset except Webots because those archs
> were never trained on those datasets in the notebook
> execution. Plus IMUWiFine is missing as a row entirely.
>
> This iter trains the missing fusion-arch × dataset
> combinations inline (FAST_MODE=False) or loads from
> bootstrap checkpoints (FAST_MODE=True), adds an IMUWiFine
> row to Table C, and lets the cells fill themselves from live
> variables.

## What needs filling

Cataloguing every "n/a but could be a real number" cell in
Table C, by training a new combination:

| dataset | metric | transformer | cnn1d | lstm_attn | what's needed |
|---|---|---|---|---|---|
| Webots sim | test MAE | 0.41 ✓ | 0.35 ✓ | 0.30 ✓ | already done (RESULT_33) |
| UJI val | mean Euclid | TBD | 8.72 ✓ | 8.43 ✓ | train **transformer at K=1, M=1** (degenerate; matches RESULT_24 protocol). ~5 min. |
| RoNIN canonical | raw ATE | TBD | 7.59 ✓ | 7.50 ✓ | train **transformer aggregator at K=4, M=1** over IMUCNN sub-windows (matches RESULT_23 protocol). ~20 min. |
| TartanAir hospital | last-20 % Umeyama ATE | n/a | n/a | n/a | **genuine n/a** — camera-only, no fusion across modalities applies. Keep n/a. |
| Webots odom-only | test MAE | n/a | n/a | n/a | **genuine n/a** — OdomCNN-only baseline, not a fusion target. Keep n/a. |
| **IMUWiFine fl.4 val** (NEW row) | mean Euclid (WiFi+IMU) | TBD | 1.40 ✓ (R_19) | 1.26 ✓ (R_19) | train **transformer on IMUWiFine** at K=4, 2-mod. ~10-15 min. |
| **IMUWiFine fl.4 test** (NEW row) | mean Euclid (WiFi-only at test by dataset design) | TBD | 7.09 ✓ (R_19) | 7.20 ✓ (R_19) | same checkpoint, eval on test (the WiFi-only column per RESULT_20 audit). 0 extra min. |
| **MSILN site1/B1 val** (NEW row) | mean Euclid (WiFi+IMU cross-session) | TBD | TBD | TBD | train **all 3 archs on MSILN** at K=4, 2-mod. ~20 min each = ~60 min. |
| **MSILN site1/B1 test** (NEW row) | mean Euclid (cross-session test) | TBD | TBD | TBD | same checkpoints, eval on test. 0 extra. |

(NB: RESULT_15's MSILN run used `WiFiSetTransformer` per its
dataset config default. Per RESULT_22 + RESULT_24 the notebook
uses Anchor2Vec encoder consistently. New MSILN trainings use
Anchor2Vec + IMUCNN encoders so the row matches the rest of
the table.)

Cells that stay `n/a` (NOT trainable for reasons of modality
availability):
- Anchor2Vec on RoNIN canonical / TartanAir (no WiFi data).
- IMUCNN on UJI / TartanAir (no IMU data).
- DPVOMotion on UJI / RoNIN / MSILN / IMUWiFine (no camera).
- OdomCNN on UJI / RoNIN / MSILN / IMUWiFine / TartanAir (no odom data).
- Per-leg encoders on the Webots fusion row (the fusion row
  uses all 4 encoders combined; per-encoder rows are listed
  separately for the encoder-only audits).

## Hypothesis

After this iter, Table C has every fillable cell populated by
a live value computed from a loaded-or-trained checkpoint.
The IMUWiFine + MSILN rows appear; `n/a` survives only for
genuine modality-availability reasons (with a one-line caveat
under the table).

The 6 new training combinations are:

1. UJI transformer K=1 M=1 (degenerate) — ~5 min
2. RoNIN canonical transformer aggregator K=4 M=1 — ~20 min
3. IMUWiFine transformer K=4 2-mod — ~10-15 min
4. MSILN transformer K=4 2-mod — ~20 min
5. MSILN CNN1D K=4 2-mod — ~20 min
6. MSILN LSTM-attn K=4 2-mod — ~20 min

Total bootstrap: ~95-100 min wall-clock under FAST_MODE=False
on the first run. Subsequent FAST_MODE=True runs load all 6
new checkpoints in seconds.

## Steps

### Step 0 — Audit current Table C + identify checkpoint paths (10 min)

Engineer reads the current Table C + the `live_numbers` /
`rows_c` building code. Identifies:
- Which cells are currently `n/a` because the value wasn't
  computed in the notebook (the targets of this iter).
- Which cells are `n/a` for genuine modality reasons
  (preserved).

Picks canonical checkpoint paths under `runs/encoder_audit_*/`
or `runs/main_table/<dataset>/<arch>.pt` for the 6 new
trainings. Engineer's choice — but consistent paths help the
clone-and-reproduce UX.

**Acceptance**: a 2-column table in RESULT_37 listing the 6
target combinations + their canonical checkpoint paths.

### Step 1 — Add 6 training cells with FAST_MODE branches (60 min training + 30 min plumbing)

Pattern per combination (mirrors RESULT_32's Anchor2Vec template
and RESULT_33's `train_fusion_arch` helper):

```python
# §X.Y — transformer on MSILN (NEW)
ckpt = ROOT / "runs/main_table/msiln_site1_b1/transformer.pt"
if FAST_MODE and ckpt.is_file():
    print(f"[FAST_MODE=True] loading transformer/MSILN from {ckpt}")
    trainer = load_trained(ckpt.parent, arch="transformer", dataset="msiln_site1_b1")
else:
    print(f"[FAST_MODE=False] training transformer on MSILN (~20 min)...")
    trainer, history, _ = train_fusion_arch(
        arch="transformer",
        dataset="msiln_site1_b1",
        K=4, batch_size=128, lr=1.3e-3, epochs=90, seed=SEED,
        save_dir=ckpt.parent,
    )
val_mae  = trainer.evaluate("val")["mae"]
test_mae = trainer.evaluate("test")["mae"]
print(f"transformer MSILN: val {val_mae:.2f} / test {test_mae:.2f}")
live_numbers["msiln_transformer_val"]  = val_mae
live_numbers["msiln_transformer_test"] = test_mae
plot_training_curves(history if not FAST_MODE else load_history(ckpt.parent),
                      title="transformer @ MSILN")
```

Engineer creates 6 such cells. Group them logically:
- §3.X subsection for fusion-arch trainings on additional
  datasets (MSILN + IMUWiFine) — extends the existing §3
  Phase B bake-off structure.
- §2.X for the RoNIN transformer aggregator + UJI transformer
  cells — extends the existing per-leg "Ours" subsections
  with a multi-arch comparison subsection where multiple
  fusion blocks were evaluated on the single-modality data.

(Alternative: a single new §6 "Cross-dataset fusion-arch
sweep" section that holds all 6. Engineer's call on
organisation; the cells need to exist and populate
`live_numbers[...]`.)

The `train_fusion_arch` helper from RESULT_33 already supports
arbitrary `dataset=` per the consolidated builder pattern
(PLAN_28). Engineer verifies it handles MSILN + IMUWiFine
configs cleanly; if not, ~10 lines of dataset-specific
arg-mapping needed.

For UJI transformer K=1 M=1 (degenerate): the existing
`scripts/_train_uji_arch.py` (RESULT_24) has the recipe.
Engineer factors out a `train_uji_arch(arch, ...)` helper if
not already present.

For RoNIN canonical transformer aggregator K=4 M=1: the
existing `scripts/_train_ronin_canonical_arch.py` (RESULT_23)
has the recipe. Same engineer-factoring pattern.

**Acceptance**: 6 new training cells; each runs in both
FAST_MODE branches; checkpoints saved to documented paths;
`live_numbers[...]` populated with every new value.

### Step 2 — Add the IMUWiFine + MSILN rows to Table C (15 min)

Update the `rows_c` builder to include:

```python
rows_c = [
    # ... existing rows from RESULT_36 ...
    # NEW rows:
    {"dataset": "IMUWiFine fl.4 val",
     "metric": "mean Euclid (WiFi+IMU)",
     "SOTA":   wlanloc_imuwifine_val,                     # NEW from §1.x or live cell
     "Anchor2Vec": np.nan,
     "IMUCNN": resnet1d_imuwifine_val,                    # NEW from §1.x (RoNIN ResNet1D on IMUWiFine)
     "DPVOMotion": np.nan,
     "OdomCNN": np.nan,
     "transformer": live_numbers["imuwifine_transformer_val"],
     "cnn1d":       live_numbers["imuwifine_cnn1d_val"],
     "lstm_attn":   live_numbers["imuwifine_lstm_attn_val"]},
    {"dataset": "IMUWiFine fl.4 test (WiFi-only by design)",
     "metric": "mean Euclid",
     "SOTA":   wlanloc_imuwifine_test,
     "Anchor2Vec": np.nan,
     "IMUCNN": np.nan,                                     # ResNet1D test n/a per RESULT_19 (no test IMU)
     "DPVOMotion": np.nan,
     "OdomCNN": np.nan,
     "transformer": live_numbers["imuwifine_transformer_test"],
     "cnn1d":       live_numbers["imuwifine_cnn1d_test"],
     "lstm_attn":   live_numbers["imuwifine_lstm_attn_test"]},
    {"dataset": "MSILN site1/B1 val (cross-session WiFi+IMU)",
     "metric": "mean Euclid",
     "SOTA":   wlanloc_msiln_val,                         # from RESULT_15
     "Anchor2Vec": np.nan,
     "IMUCNN": np.nan,
     "DPVOMotion": np.nan,
     "OdomCNN": np.nan,
     "transformer": live_numbers["msiln_transformer_val"],
     "cnn1d":       live_numbers["msiln_cnn1d_val"],
     "lstm_attn":   live_numbers["msiln_lstm_attn_val"]},
    {"dataset": "MSILN site1/B1 test",
     ... },
]
```

The IMUCNN cell for IMUWiFine val: RESULT_19 ran RoNIN
ResNet1D on IMUWiFine and got 26.84 m val. That's the IMU SOTA
for that row. Engineer verifies the cell value is the
RESULT_19 number (or recomputes live if the cell isn't
already in the notebook).

For UJI + RoNIN canonical existing rows: update to add the
transformer value:

```python
{"dataset": "UJI val", "metric": "mean Euclid", ...
 "transformer": live_numbers["uji_transformer_val"],   # NEW from Step 1
 "cnn1d":       8.72,  # from §X.X RESULT_24 / existing live
 "lstm_attn":   8.43,  # from §X.X RESULT_24 / existing live
 ...},
{"dataset": "RoNIN canonical", "metric": "raw ATE", ...
 "transformer": live_numbers["ronin_transformer_raw"],  # NEW from Step 1
 "cnn1d":       7.59,
 "lstm_attn":   7.50,
 ...},
```

The Webots row stays unchanged (already filled).

**Acceptance**: Table C has 5-7 rows (Webots fusion + UJI +
RoNIN + TartanAir + Webots odom + IMUWiFine ×2 + MSILN ×2);
every cell that CAN be filled IS filled; `n/a` only where
modality-availability genuinely prevents it.

### Step 3 — Re-style Table C with row-winner bolding (5 min)

The `style_winner_per_row(row, value_cols)` from RESULT_36
already handles `n/a` correctly. No code change needed; the
new rows automatically get the row-min bolded once they have
real values.

If Styler renders awkwardly with so many columns (~10), engineer
may add `.set_table_styles([{"selector": "th",
"props": [("text-align", "right")]}])` or fall back to the
markdown rendering path documented in PLAN_36.

**Acceptance**: rendered Table C shows the new rows with the
row winner bolded per row.

### Step 4 — Caveat footnote under Table C (5 min)

One markdown cell below the table explaining the `n/a`s:

> `n/a` indicates the column's method/encoder isn't applicable
> to the dataset's available modalities (e.g. Anchor2Vec on
> IMU-only RoNIN canonical, DPVOMotion on per-scan UJI). All
> fusion-arch columns are filled where the dataset has ≥ 1
> modality the architecture can process; single-modality
> datasets (RoNIN canonical IMU, UJI WiFi) report the fusion
> arch as a degenerate aggregator over the encoder's tokens
> per the run-2 audit protocol.

This is honest documentation; not paper apology.

**Acceptance**: caveat markdown cell present below Table C.

### Step 5 — Re-smoke BOTH FAST_MODE branches (35 min)

The new training cells justify a full FAST_MODE=False smoke
(first-run bootstrap):

```powershell
jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=10800 `
    notebooks/run2_walkthrough.ipynb
```

Total wall-clock under FAST_MODE=False (first run) ~ 2-2.5 h
(previous 27 min + ~100 min new training). Subsequent runs
under FAST_MODE=True ~ 5-10 min (loads all 6 new checkpoints).

Confirm:
- 6 new training cells produce output in both modes.
- New Table C rows show real values (not `n/a`) for the
  trained combinations.
- IMUWiFine and MSILN rows appear.
- Row-winner bolding works across the wider table.
- 0 cell errors.

**Acceptance**: both modes execute cleanly; bootstrap saves 6
new checkpoints; second run loads them.

### Step 6 — Commit (5 min)

Single commit: notebook + any new helpers (`train_uji_arch`,
`train_ronin_canonical_arch` factored helpers if added) + 6
new checkpoints under `runs/main_table/` (gitignored — only
the notebook + helper code commits).

## What to report back

In `handoff/results/RESULT_37_fill-all-fusion-cells-and-add-imuwifine.md`:

1. **Step 0** — checkpoint-paths inventory.
2. **Step 1** — 6 training cells + per-training wall-clock +
   live `val_mae` / `test_mae` recorded.
3. **Step 2** — Table C row updates (IMUWiFine ×2 + MSILN ×2
   rows added; UJI + RoNIN transformer columns filled).
4. **Step 3** — sample Table C render output.
5. **Step 4** — caveat footnote.
6. **Step 5** — both-modes smoke wall-clocks + cell-error
   count.
7. **One open question** for the user.

## Reversibility

- Step 1 (new training cells): permanent in notebook;
  checkpoints saved (gitignored under `runs/`).
- Step 2 (Table C rows): permanent.
- Step 3-4 (style + caveat): permanent.

Files committed: `notebooks/run2_walkthrough.ipynb` (revised);
optional new helpers in `src/pipeline/training/inline_encoders.py`
if engineer factored from `scripts/_train_*_arch.py`.

**Compute budget**: ≤ 3 hours.
- Step 0: 10 min.
- Step 1 plumbing: 30 min (cell writing + helper factoring).
- Step 1 training (FAST_MODE=False bootstrap): ~100 min.
- Step 2: 15 min.
- Step 3: 5 min.
- Step 4: 5 min.
- Step 5 (both-modes smoke): 30 min (FAST_MODE=False
  ~2-2.5 h is the long pole, but engineer can run it in
  background; FAST_MODE=True ~10 min verification).
- Step 6: 5 min.

If overrun: drop the LSTM-attn MSILN training (keep
transformer + CNN1D, both more important — CNN1D is the run-2
winner; transformer is the new fill cell). Document the cut
explicitly.

If a specific new training diverges or NaN's (e.g. MSILN's
cross-session WiFi sparsity might destabilize one of the
archs), engineer writes a partial RESULT documenting the
training failure + reports honest `n/a (training diverged)`
in Table C rather than fabricating a number.

## Quality bar

Same as the prior 5 polish iters. Every cell value in Table C
either reflects a real live computation OR is `n/a` for a
genuine modality-availability reason — with the caveat
footnote making the distinction explicit. The reader gets the
full comparison.
