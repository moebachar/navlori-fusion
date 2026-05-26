# Result 31 — Notebook revised for live reproduction (clone-and-reproduce)

## TL;DR

**Notebook revision shipped per user directive.** Every paper-claim
number is now computed live in-cell from loaded checkpoints / data
— no cached JSON reads, no hand-typed numbers (except the
`_CANONICAL` source-of-truth mapping inside `MainResultsTable`,
which IS library code).

- `notebooks/run2_walkthrough.ipynb` re-executed cleanly via
  `jupyter nbconvert --execute` → 639 KB output, no cell errors.
- **§1.1 UJI**: wlanloc **15.17 m** (drift +0.0%, paper-exact);
  Anchor2Vec 60ep **8.60 m** (drift -1.0% vs canonical 120ep
  8.69 m).
- **§1.2 RoNIN canonical**: ResNet1D pretrained **5.126 m raw
  ATE** (drift -0.26% vs paper-exact 5.140 m).
- **§2 fusion bake-off**: 3 archs loaded via `load_trained` in
  ~17-21s each; subset eval ~1.5s; K-axis staleness ~0.5s. Param
  counts exact match RESULT_17/21 (incumbent 1.55M / CNN1D
  0.51M / LSTM-attn 0.57M).
- **§3 verification demo**: UJI Anchor2Vec live-vs-table drift
  **1.04 %**; Webots CNN1D **0.59 %**. Both inside the 2% paper-
  tolerance.
- **§4 smoothness**: live computed via `compute_per_trajectory_smoothness`
  across all 3 archs on Webots test paths.
- **§5 partition table**: ~5-min live wall-clock for the
  notebook; offline `_train_*.py` commands documented for the
  hours-long reproductions.

## Step-by-step

### Step 0 — Audit

Catalogued the cells that needed flipping:

| cell# (old) | issue | resolution |
|------------|-------|------------|
| 28         | hand-typed Phase B DataFrame | derive from `trainers[arch].predict()` + `compute_per_trajectory_smoothness()` |
| 30         | `json.load(all_subsets_test.json)` for CNN1D | live `trainers['cnn1d'].evaluate_all_subsets('test')` |
| 32         | `json.load(cnn1d_ablations.json)` for staleness | live `trainers['cnn1d'].evaluate_staleness('wifi', 'test')` |
| 34         | `json.load(all_subsets_test.json)` for LSTM-attn | live `trainers['lstm_attn'].evaluate_all_subsets('test')` |
| 40         | hand-typed smoothness DataFrame | live `compute_per_trajectory_smoothness` across 3 archs |

### Step 1 — §1.1 Anchor2Vec UJI live + §1.2 ResNet1D pretrained live

§1.1 wraps `scripts.eval_uji.run_wlanloc()` + `run_anchor2vec(epochs=60)`
which are already built on the consolidated APIs from PLAN_26.

§1.2 wraps `scripts.eval_ronin_canonical.eval_resnet1d_pretrained()`
which loads the FRDR pretrained checkpoint + iterates the 32 unseen
sequences + computes ATE/RTE via `compute_ate_rte` from the
baselines package.

Both live in ~90s + ~21s wall-clock; reproduce the paper-exact
numbers within 1% drift.

### Step 2 — §2 bake-off via `load_trained`

Replaced all `json.load(...)` calls with live `trainers[arch].evaluate_*`
methods. The 3-arch load takes ~56s; the subset evals + staleness
sweep add ~5s combined (the trainers stage tensors on GPU once,
then iterate fast).

Param counts exact match RESULT_17/21:
- incumbent: 1.55 M
- cnn1d: 0.51 M
- lstm_attn: 0.57 M

K-axis staleness on CNN1D produces 5 lag levels {0,1,2,3,4}
(consistent with K=4 design). The full 8-lag RESULT_14 figure is
the offline runner `scripts/_iter18_cnn1d_ablations.py --lags full`
documented in §5.

### Step 3 — §3 main results + live-cell verification

`MainResultsTable.from_archive()` unchanged (source-of-truth
aggregator). Added the verification demonstrator that cross-
checks 2 cells against the live §1.1 / §2 numbers:

```
UJI Anchor2Vec - table source-of-truth: 8.690 m val   [RESULT_01]
  §1.1 live (60 ep inline):              8.599 m val
  drift: 1.04%  (60 ep < canonical 120 ep, so larger drift expected)

Webots CNN1D - table source-of-truth: 0.339 m test   [RESULT_17]
  §2 live (load_trained + predict):     0.341 m test
  drift: 0.59%
```

Both drifts well inside the 2 % paper-tolerance.

### Step 4 — §4 smoothness live + C2 narrative

Smoothness table computed live via
`compute_per_trajectory_smoothness` on the 3 archs' loaded test
predictions (reuses §2 trainers; no reload cost). Confirms the
"architectural-lever-for-smoothness hypothesis falsified" claim
inline — all 3 archs land r ≤ 0.10 < 0.20 gate.

C2 raw-ATE narrative cross-references §1.2's live ResNet1D number
+ RESULT_07/23 IMUCNN + CNN1D-aggregator numbers (text-only;
re-running IMUCNN canonical inline would take ~14 min of training
per RESULT_07 — that's a script command in §5).

### Step 5 — §5 partition table

Live partition DataFrame split into "live (≤ 10 min)" vs "offline
script" rows:

- 8 LIVE rows: §0 (~30s); §1.1 (~90s); §1.2 (~20s); §2 (~3-5 min);
  §3 (~5s); §4 (~10s).
- 5 OFFLINE rows: 25-min CNN1D retrain, 75-min full bake-off,
  10-min CNN1D ablation suite, 5-min TartanAir 3-shim, 3-h MSILN.

Reader can trivially see what's in the notebook vs what to invoke
via the `scripts/*` commands.

### Step 6 — End-to-end smoke

```
[NbConvertApp] Converting notebook notebooks/run2_walkthrough.ipynb to notebook
[NbConvertApp] Writing 639244 bytes to notebooks\run2_walkthrough.ipynb
```

- Wall-clock: ~4 min (under the 10-min target).
- 0 cell errors.
- 1 RuntimeWarning in §1.2 (empty-slice mean for RoNIN's `a057_3`
  RTE NaN — known issue from RESULT_07 with sequences shorter than
  the RTE sliding window; doesn't affect ATE numbers).
- All paper-grade plots rendered inline (CNN1D subset bar, K-axis
  staleness curve, LSTM-attn dead-reckoning bar, all 4 dataset
  overview figures, WiFi preprocessing demo).

## One open question for the user

The K-axis staleness in §2 shows `{0, 1, 2, 3, 4}` — 5 lags
because the live `FusionTrainer.evaluate_staleness` operates on
the K=4 instant axis (each level zeros the K-most-recent
instants). The full 8-lag RESULT_14 paper figure uses an
*artificial WiFi shift* probe (a different measurement, with lags
out to 30 instants ≈ 27 s) that lives in
`scripts/_iter18_cnn1d_ablations.py` — kept as an offline runner.

Two ways to surface this in the notebook:
- (a) Keep the live K-axis sweep as-is and document the 8-lag
  paper figure as an offline runner (current state).
- (b) Promote the 8-lag artificial-WiFi-shift probe into
  `FusionTrainer.evaluate_wifi_lag_staleness` (or similar) so it's
  a method on the trainer + call it live in §2. Cost: ~80s extra
  wall-clock + a new public method.

Engineer recommendation: **(a)** for the scaffold; (b) is a small
polish item if you want the paper figure inline.

## Files committed

- `notebooks/run2_walkthrough.ipynb` — revised (live cells; 639 KB output).
- `handoff/plans/PLAN_31_*.md`, `handoff/results/RESULT_31_*.md`,
  `handoff/STATE.md` (iter 31 row + status).
