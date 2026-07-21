# Plan 38 — Table C per-method columns + run-every-method-where-modality-applies

> User directive 2026-05-27 ~07:20 local. Two coupled changes
> to Table C:
>
> 1. **Split the single `SOTA` column** into separate per-method
>    columns (`wlan_localization`, `RoNIN ResNet1D`, `TartanVO`,
>    `trivial integration`). Each SOTA gets its own column so
>    the reader sees the actual method that competed in each
>    row, not a collapsed "SOTA" placeholder.
> 2. **Run every method on every dataset where the modality
>    requirements are met.** Webots has 4 modalities → every
>    method except the Camera-RGB-required TartanVO can be run
>    on Webots. Same principle for MSILN (WiFi+IMU available)
>    and IMUWiFine (WiFi+IMU available). Stop using `n/a (not
>    tested)`; only `n/a (modality unavailable)` survives.

## What changes per dataset

Audit of "modality available but method not yet tested" gaps:

### Webots sim (WiFi + IMU + Camera-as-DPVO-features + Odom)

Currently the Webots row only has fusion-arch numbers + the
Webots-odom-only sub-row. Everything else is `n/a` purely
because we never ran it.

| method | applies? | currently | this iter |
|---|---|---|---|
| wlan_localization | yes (WiFi avail; non-parametric kNN, ~5 min) | n/a | **NEW: run on Webots WiFi** |
| RoNIN ResNet1D pretrained | yes (IMU avail; eval-only, ~10 min) | n/a | **NEW: run on Webots IMU** |
| TartanVO | n/a — Webots RGB never persisted (only `dpvo_features.pt` + depth on disk per RESULT_35) | n/a | stays n/a, **documented honestly** |
| trivial integration | yes (Odom avail; closed-form, ~1 min) | 8.27 ✓ (RESULT_04) | reuse |
| Anchor2Vec encoder-only | yes (WiFi avail; ~5 min inline train) | n/a | **NEW: train on Webots WiFi only** |
| IMUCNN encoder-only | yes (IMU avail; ~5 min) | n/a | **NEW: train on Webots IMU only** |
| DPVOMotion encoder-only | yes (cached patch features avail; ~5 min head train) | n/a | **NEW: train DPVOMotion head on Webots** |
| OdomCNN encoder-only | yes | 4.49 ✓ (RESULT_04) | reuse |
| transformer fusion | yes | 0.42 ✓ (RESULT_33) | reuse |
| cnn1d fusion | yes | 0.33 ✓ (RESULT_33) | reuse |
| lstm_attn fusion | yes | 0.28 ✓ (RESULT_33) | reuse |

→ 5 NEW Webots cells to populate.

### MSILN site1/B1 (WiFi + IMU; no Camera or Odom)

| method | applies? | currently | this iter |
|---|---|---|---|
| wlan_localization | yes | 21.26 / 28.31 ✓ (RESULT_15) | reuse |
| RoNIN ResNet1D pretrained | yes (IMU avail; eval-only, ~10 min) | n/a | **NEW** |
| Anchor2Vec encoder-only | yes (~5 min inline) | n/a | **NEW** |
| IMUCNN encoder-only | yes (~5 min) | n/a | **NEW** |
| transformer fusion | yes | 15.22 / 10.89 ✓ (RESULT_37) | reuse |
| cnn1d / lstm_attn fusion | yes | n/a (PLAN_37 cut by timeout) | **defer to PLAN_39** |

→ 3 NEW MSILN cells.

### IMUWiFine fl.4 (WiFi train+val+test; IMU train+val only)

| method | applies? | currently | this iter |
|---|---|---|---|
| wlan_localization | yes | 4.17 / 8.50 ✓ (RESULT_19) | reuse |
| RoNIN ResNet1D pretrained on IMU | yes — val only (test no IMU per RESULT_20) | 26.84 / n/a ✓ (RESULT_19) | reuse |
| Anchor2Vec encoder-only | yes — val + test (WiFi available on both) | n/a | **NEW** |
| IMUCNN encoder-only | yes — val only (no test IMU) | n/a | **NEW** |
| transformer / cnn1d / lstm_attn fusion | yes (val); test only meaningful when fusion gracefully degrades to WiFi-only by design | n/a (PLAN_37 cut) | **defer to PLAN_39** |

→ 2 NEW IMUWiFine cells.

### UJI / RoNIN canonical / TartanAir hospital

These are single-modality datasets per their design; no new
fills apply beyond what's already in Table C.

## Total new work in PLAN_38

**10 new evals**, all small and fast:

| # | training/eval | wall-clock |
|---|---|---|
| 1 | wlan_localization on Webots WiFi (eval-only, kNN) | ~5 min |
| 2 | RoNIN ResNet1D pretrained on Webots IMU (eval-only) | ~10 min |
| 3 | Anchor2Vec train+eval on Webots WiFi | ~5 min |
| 4 | IMUCNN train+eval on Webots IMU | ~5 min |
| 5 | DPVOMotion head train+eval on Webots cached features | ~5 min |
| 6 | RoNIN ResNet1D pretrained on MSILN IMU (eval-only) | ~10 min |
| 7 | Anchor2Vec train+eval on MSILN WiFi | ~5 min |
| 8 | IMUCNN train+eval on MSILN IMU | ~5 min |
| 9 | Anchor2Vec train+eval on IMUWiFine WiFi | ~5 min |
| 10 | IMUCNN train+eval on IMUWiFine IMU (val only) | ~5 min |

**Total ~60 min wall-clock** for the new evals.

## Plus: Table C column split (cosmetic, no compute)

Today's columns (per RESULT_37): roughly
`[dataset, metric, SOTA, Anchor2Vec, IMUCNN, DPVOMotion, OdomCNN,
  transformer, cnn1d, lstm_attn]` = 1 collapsed SOTA col.

NEW columns:
`[dataset, metric,
  wlan_localization, RoNIN_ResNet1D, TartanVO, trivial_integration,
  Anchor2Vec, IMUCNN, DPVOMotion, OdomCNN,
  transformer, cnn1d, lstm_attn]`

= 11 method columns. Wider table but each method gets its own
column with proper bolded row-min when applicable.

## Hypothesis

After PLAN_38:
- Table C has 11 method columns; SOTA is split per method.
- Every "could be tested" cell is filled — the only surviving
  `n/a`s are **modality-availability** ones (clearly labelled in
  the caveat footnote).
- 10 new live numbers ground the table.
- Optional follow-up: PLAN_39 fills the 5 remaining fusion-arch
  cells that the PLAN_37 timeout cut (MSILN cnn1d/lstm_attn +
  IMUWiFine transformer/cnn1d/lstm_attn) — ~3 h of training,
  separated to avoid the Jupyter 3h timeout.

## Steps

### Step 0 — Audit current Table C structure + identify checkpoint paths (10 min)

Engineer reads the v6/v7 notebook §5 cell that builds Table C +
the column list. Documents the 10 new checkpoint paths
(`runs/encoder_audit_{wifi,imu,camera}/<dataset>_<encoder>.pt`
OR similar — engineer picks consistent paths).

### Step 1 — Add 5 NEW Webots eval cells (25 min)

Cell pattern per the new eval — each cell either calls a
library function from `src.pipeline.baselines` (for SOTAs) or
trains a fresh encoder using existing inline trainers
(`train_anchor2vec`, `train_imucnn` already in
`src.pipeline.training`). Engineer follows the FAST_MODE
pattern from PLAN_37 — load saved checkpoint if available,
else train inline + save.

For each cell, store result in `live_numbers["<dataset>_<method>"]`
for Table C.

Key points:
- **wlan_localization on Webots WiFi**: use the same `_load_pure`
  + `PositionRegressor` + `DataPreprocessor` machinery from
  `scripts/eval_uji.py`. Adapt dataset loading to read Webots
  WiFi CSVs (gathered across paths). Document the "Webots WiFi
  is GPR-synthesised" caveat in the live print so the reader
  knows the number is optimistic per CLAUDE.md.
- **RoNIN ResNet1D on Webots IMU**: load pretrained checkpoint
  from `data/ronin_frdr/pretrained_resnet/`; run integration on
  Webots paths. Document "Webots IMU is out-of-domain for the
  RoNIN-pretrained model" caveat.
- **Anchor2Vec encoder-only on Webots**: `train_anchor2vec` with
  Webots-formatted WiFi data. Existing helper.
- **IMUCNN encoder-only on Webots**: `train_imucnn` adapted —
  current helper targets RoNIN format; engineer either
  generalises or writes `train_imucnn_webots` (~30 lines).
- **DPVOMotion encoder-only on Webots**: load cached features
  from `data/async_collection/path_*/dpvo_features.pt`, train
  the head with the existing `train_dpvo_motion_head` machinery.

**Acceptance**: 5 new cells render in §5 or a new sub-section;
each populates `live_numbers`.

### Step 2 — Add 3 NEW MSILN eval cells (15 min)

Same pattern:
- RoNIN ResNet1D pretrained on MSILN IMU (val + test).
- Anchor2Vec encoder-only on MSILN WiFi.
- IMUCNN encoder-only on MSILN IMU.

Each ~5-10 min. All small.

### Step 3 — Add 2 NEW IMUWiFine eval cells (10 min)

- Anchor2Vec encoder-only on IMUWiFine WiFi (val + test).
- IMUCNN encoder-only on IMUWiFine IMU (val only — test has no IMU
  per RESULT_20).

### Step 4 — Restructure Table C with 11 method columns (15 min)

Update the `rows_c` builder + `value_cols` list + Styler
bolding. New columns:
`[wlan_localization, RoNIN_ResNet1D, TartanVO, trivial_integration,
   Anchor2Vec, IMUCNN, DPVOMotion, OdomCNN,
   transformer, cnn1d, lstm_attn]`

Per-row mapping: each row populates only the columns where the
method is applicable AND we have a live number. `n/a` everywhere
else.

**Acceptance**: Table C has 11 method columns; row-min bolded
ignoring `n/a`; reader sees every method side-by-side.

### Step 5 — Update caveat footnote (5 min)

Existing caveat (RESULT_37) covers "trainable but not yet run"
vs "genuine modality n/a". After PLAN_38, the "trainable but
not yet run" category shrinks to the 5 fusion-arch cells from
the PLAN_37 timeout (MSILN cnn1d/lstm_attn, IMUWiFine fusion
×3). Update wording:

```
n/a indicates the column's method cannot run on the dataset's
available modalities. Specifically:
 - wlan_localization, Anchor2Vec require WiFi → n/a on RoNIN
   canonical (IMU-only), TartanAir hospital (Camera-only).
 - RoNIN ResNet1D, IMUCNN require IMU → n/a on UJI (per-scan
   WiFi-only), TartanAir hospital (Camera-only).
 - TartanVO, DPVOMotion require RGB camera → n/a on UJI, RoNIN
   canonical, MSILN, IMUWiFine, and on Webots (where camera was
   captured as depth + cached DPVO features only — RGB never
   persisted, per RESULT_35).
 - OdomCNN, trivial_integration require wheel odometry → n/a
   on everything except Webots (the only dataset with odom
   modality).
 - Fusion archs (transformer/cnn1d/lstm_attn) require ≥ 1
   compatible modality and an encoder pipeline; they run on
   every multi-modal dataset.

5 cells remain "training pending" (queued as PLAN_39):
MSILN cnn1d/lstm_attn and IMUWiFine transformer/cnn1d/lstm_attn
— these were cut by the Jupyter 3 h per-cell timeout in PLAN_37.
The clone-and-reproduce path runs scripts/<x>.py to fill them
offline.
```

### Step 6 — Re-smoke FAST_MODE=True (10 min)

`jupyter nbconvert --to notebook --execute --inplace
notebooks/run2_walkthrough.ipynb`. Total expected ~10-15 min
(adds 60 min of new evals to the existing ~3 min smoke; but
FAST_MODE=True loads saved checkpoints so 60 min only on first
run).

**Acceptance**: clean nbconvert; Table C renders with 11 columns
+ 10 new cell values; 5 `training pending` cells with the
documented caveat.

### Step 7 — Commit (5 min)

Notebook + any new inline-trainer helpers (`train_imucnn_webots`
if added).

## What to report back

In `handoff/results/RESULT_38_per-method-columns-and-modality-driven-fills.md`:

1. **Step 0** — checkpoint path inventory.
2. **Steps 1-3** — 10 new cell additions + live values.
3. **Step 4** — new Table C shape (11 method columns); sample
   render.
4. **Step 5** — updated caveat footnote.
5. **Step 6** — smoke output.
6. **One open question** for the user (likely whether to queue
   PLAN_39 for the 5 remaining fusion-arch cells).

## Reversibility

- Steps 1-6: notebook additions + Table C reshape. Engineer
  commits.

Files committed: `notebooks/run2_walkthrough.ipynb` (revised);
optional new `train_imucnn_webots` / `train_dpvomotion_webots`
helpers in `src/pipeline/training/inline_encoders.py`.

**Compute budget**: ≤ 2 hours.
- Step 0: 10 min.
- Steps 1-3: 60 min training + 30 min plumbing = 90 min.
- Step 4: 15 min.
- Step 5: 5 min.
- Step 6: 10 min.
- Step 7: 5 min.

If overrun: cut Step 1's DPVOMotion-encoder-only Webots cell
(it's the most involved + the most subject to data-availability
caveat). The other 9 new cells are higher-value.

If a specific eval surprises (e.g. RoNIN ResNet1D on Webots IMU
diverges because Webots IMU is out-of-distribution from RoNIN's
real-phone training), engineer reports the live number honestly
+ a one-line caveat. Don't fabricate; don't hide.

## Queued: PLAN_39 — remaining fusion-arch fills

Out of scope for PLAN_38. The 5 remaining fusion-arch cells
(MSILN cnn1d/lstm_attn + IMUWiFine transformer/cnn1d/lstm_attn)
need separate per-arch cells in the notebook (not a `for arch
in ...` loop that hits the 3 h timeout) + ~3 h total training
on Quadro P4000 at 90 ep each, or ~1.5 h at 45 ep with early-
stop. Engineer's call when to queue.

## Quality bar

Same as the prior polish iters. The table must be HONEST about
which cells are filled vs which are pending; the column split
must show every method side-by-side. No fabrication; no
"n/a (not tested)" cells when the dataset has the required
modalities.
