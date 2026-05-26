# Result 36 — Remaining trajectory overlays + full-comparison Table C

## TL;DR

Two items from RESULT_35 closed:

- **Item A (pure additions)**: §4.6 RoNIN canonical trajectory overlays
  (IMUCNN vs ResNet1D, 3 representative unseen sequences chosen by IMUCNN
  per-seq raw ATE: best / median / worst). §4.7 MSILN cross-session overlays
  (Fusion-Ours vs WiFi-kNN k=5 manhattan baseline, RESULT_15 protocol)
  on test paths 128/129/130. The Webots §4.5 overlays from PLAN_35 are
  untouched.
- **Item B (table reshape)**: Table C goes from "best method + best MAE only"
  (one value column) to **all methods × all datasets, with the row winner
  bold** — the canonical paper main-results-table shape. Tables A and B were
  already showing every value per row.

## Code changes

- `notebooks/run2_walkthrough.ipynb`:
  - NEW `s46-md` + `s46` — RoNIN per-seq overlay cell. Loads ResNet1D
    pretrained + the cached IMUCNN encoder/head from
    `runs/encoder_audit_imu/imucnn_ronin_canonical.pt`; integrates predicted
    velocity forward-Euler from `GT[window]` for both methods, anchored the
    same way. Picks 3 seqs by IMUCNN per-seq raw ATE (best / median / worst)
    so the figures span the dynamic range.
  - NEW `s47-md` + `s47` — MSILN cross-session overlay cell. Builds the train
    fingerprint matrix from Nov-24 paths inline (~15 lines), runs WiFi-kNN
    k=5 Manhattan + distance-weighted neighbour mean to predict (x, y) per
    test scan. Tries to load the MSILN fusion checkpoint from
    `runs/overnight/run2_iter_15/fusion_*/model.pt` with
    `load_trained(..., arch='lstm_attn', dataset='msiln_site1_b1')` and
    overlays its predictions when available; on load failure the cell prints
    a documented one-line note and shows the kNN-only baseline.
  - `s5-summary` (Table C) reshaped:
    - Columns: `dataset`, `metric`, `SOTA`, `Anchor2Vec`, `IMUCNN`,
      `DPVOMotion`, `OdomCNN`, `transformer`, `cnn1d`, `lstm_attn`.
    - One row per dataset/metric pair (Webots fusion, UJI val, RoNIN raw ATE,
      TartanAir last-20% Umeyama ATE, Webots odom-only).
    - Styler `apply` per row finds the minimum across the value columns and
      bolds that cell; non-applicable cells render as `n/a`.
- No library / helper code changes — the existing
  `plot_trajectory_comparison` from RESULT_35 handles both new sections; the
  per-seq RoNIN integration is inlined per the PLAN_36 "engineer's call"
  provision.

## Smoke result

`jupyter nbconvert --to notebook --execute --inplace` in `FAST_MODE=True`:

- 0 cell errors.
- 34 embedded figures (+6 vs v5: 3 RoNIN + 3 MSILN overlays added; Webots §4.5
  overlays preserved).
- §4.5 (Webots): 3 figures (transformer + cnn1d + lstm_attn per test path 15/16/17).
- §4.6 (RoNIN): 3 figures — ResNet1D + IMUCNN overlaid on GT for best/median/worst
  seqs by IMUCNN per-seq raw ATE.
- §4.7 (MSILN): 3 figures — Fusion (Ours, LSTM-attn) + WiFi-kNN baseline overlaid
  on GT for paths 128/129/130. **MSILN fusion checkpoint loaded successfully**
  from `runs/overnight/run2_iter_15/fusion_20260526_025619/model.pt` — no
  honest-skip needed.
- §5 Tables A, B, C all render with bold-winner styling.
- Output 4.0 MB; 44 cells; ~4 min wall-clock.

## One open question for the user

The MSILN fusion checkpoint loaded cleanly this run, so §4.7 produced the full
GT + Fusion + kNN comparison for all 3 paths — no honest-skip path was needed.
That's the best outcome.

One paper-framing question: **§4.5 Webots overlays compare the three Ours fusion
archs (transformer, cnn1d, lstm_attn) against each other** (no SOTA on Webots),
whereas §4.6 RoNIN and §4.7 MSILN compare **Ours vs published SOTA** (ResNet1D
on RoNIN, WiFi-kNN on MSILN). That's correct given run-2's data, but the user
might want a clarifying one-line note in §4.5 markdown so readers don't expect
a SOTA trace there. Currently the markdown says "three fusion architectures
trained on the same Webots data" — already implicit, but easy to make explicit
if you prefer.

## Files committed

- `notebooks/run2_walkthrough.ipynb` (v6 — RoNIN + MSILN overlays + Table C reshape).
- `handoff/plans/PLAN_36_remaining-overlays-and-full-comparison-tables.md`.
- `handoff/results/RESULT_36_remaining-overlays-and-full-comparison-tables.md` (this file).
- `handoff/STATE.md`.
