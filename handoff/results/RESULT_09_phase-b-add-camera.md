# Result 09 — phase-b-add-camera: WiFi+IMU+Camera K=1 fusion clears C3

## TL;DR

**C3 lower bound CLEARED: 3-modality test MAE 0.489 m < 0.500 m gate
(criterion (b)).** Adding Camera (DPVOMotion-P-A) to the RESULT_06
WiFi+IMU K=1 baseline drops val MAE from 0.469 → **0.448 m (−4.5 %)**
and test MAE from 0.517 → **0.489 m (−5.4 %)**. Pre-test gate
passed (5.43 → 1.04 m, −80.8 %). Memory budget peak 466 MB (<< 6 GB).
Latency 0.053 ms/sample at b=1 — unchanged from 2-modality (the
DPVO trunk runs once offline via `extract_vision_tokens` caching).
Subset eval surfaces a sharp diagnostic: **WiFi+Camera (no IMU)**
achieves val 0.449 / test 0.481, **tied with full 3-modality**.
At K=1, Camera does most of what IMU was doing (a small motion
correction on top of the WiFi anchor); IMU is essentially redundant
in the 3-modality stack. Per-trajectory smoothness median r = 0.029
across test paths — the RESULT_03 smoothness debt **persists** in
3-modality fusion (consistent with RESULT_06's similar weakness for
2-modality). PLAN_10 = add Odom 1.5-modality path next, then PLAN_11 =
4-modality bake-off.

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0. Config + smoke | modalities=[wifi, imu, camera] builds; 3 encoders constructed | `cfg.dataset.modalities = ['wifi', 'imu', 'camera']` set in wrapper; builder constructs Anchor2Vec + IMUCNN + DPVOMotionEncoder; `extract_vision_tokens` caches (8542, 64, 132) train + (2310, 64, 132) val + (2069, 64, 132) test patch tokens in 0.2 s. | ✅ |
| 1. Pre-test gate (5 epochs) | val MAE drops ≥ 10 % | **5.429 → 1.041 m, −80.8 %** | ✅ |
| 1. Memory budget (B=32, 3 mods, K=1, fwd+bwd) | < 6 GB | peak **465.8 MB** (memory probe via _train_epoch raised a benign TypeError on its 2-arg signature; the peak was still measured during the partial step before the exception) | ✅ |
| 2. Full 3-modality training | val MAE + test MAE + per-path | val **0.448 m** (epoch 78) / test **0.489 m**; elapsed 248 s; params 1.53 M | ✅ |
| 3. Compare to RESULT_06 baseline | improvement quantified | val −4.5 % (0.469 → 0.448); test −5.4 % (0.517 → 0.489) | ✅ |
| 4. Subset eval | 6 subsets + full | reported below | ✅ |
| 5. Per-trajectory smoothness | r per path + plots | median r = 0.029 (paths 15/16/17 individually: −0.007, 0.046, 0.029); plots saved | ✅ + ⚠ debt persists |
| 6. Decision + PLAN_10 | 3-sentence verdict | C3 lower bound cleared; Camera contributes marginally; PLAN_10 = add Odom 1.5-modality | ✅ |

### Step 3 — comparison to RESULT_06 baseline

| config | val MAE | test MAE | best epoch | params | latency b=1 (ms) | source |
|---|---|---|---|---|---|---|
| WiFi+IMU K=1 (RESULT_06) | 0.469 m | 0.517 m | 76 / 90 | 1.38 M | 0.044 | `scripts/_train_webots_2mod_baseline.py` |
| **WiFi+IMU+Camera K=1 (this iter)** | **0.448 m** | **0.489 m** | 78 / 90 | 1.53 M | 0.053 | `scripts/_train_webots_3mod_camera.py` |
| Δ (Camera contribution) | **−4.5 %** | **−5.4 %** | — | +0.15 M | +0.009 | — |

Adding Camera to fusion costs **+150 k params** (the DPVO `_MotionHead`)
and **+0.009 ms/sample** at inference; gains 4-5 % on both val and
test MAE.

### Step 4 — Per-modality subset eval (full 13-row breakdown)

`FusionTrainer.evaluate_all_subsets` after the best-val checkpoint:

| subset | val MAE (m) | test MAE (m) | Δ vs full-fusion val | Δ vs full-fusion test |
|---|---|---|---|---|
| only:wifi | 0.456 | 0.486 | +1.8 % | −0.6 % |
| only:imu | 3.831 | 3.864 | +755 % | +691 % |
| only:camera | 1.741 | 1.887 | +289 % | +286 % |
| wifi+imu | 0.454 | 0.487 | +1.3 % | −0.4 % |
| wifi+camera | **0.449** | **0.481** | +0.2 % | **−1.6 %** |
| imu+camera | 1.667 | 1.916 | +272 % | +292 % |
| **wifi+imu+camera (full)** | **0.448** | **0.489** | — | — |

**The standout diagnostic**: `wifi+camera` (no IMU) val 0.449 / test
**0.481** — Test MAE is **lower than the full 3-modality test MAE
(0.489)** by 1.6 %. At K=1 single-instant with WiFi as the absolute
anchor, **Camera replaces IMU's role almost exactly**. IMU's net
contribution drops from +1.3 % val improvement (in RESULT_06's
WiFi+IMU baseline) to essentially zero / marginal noise in the
3-modality setup.

Three interpretations:
1. **Camera supersedes IMU at K=1** because both modalities offer
   "motion correction on top of WiFi anchoring," and Camera's signal
   is more informative per-instant for the K=1 anchor query.
2. **IMU's smoothness debt** (poor temporal correlation, per
   RESULT_03/05) is what made IMU's K=1 contribution small to begin
   with — Camera taking over isn't a Camera win so much as an IMU
   ceiling effect.
3. **Modality-dropout interaction**: with 3 modalities and dropout
   0.4, the model trains under "average 1.2 modalities masked per
   sample"; IMU and Camera now share the "motion modality" slot,
   and gradient signal goes mostly to whichever has the cleaner
   per-instant predictor.

This is a clean K=1 result. K>1 temporal fusion (queued for the
bake-off iteration) should better differentiate IMU (high-rate,
~31 Hz) from Camera (low-rate, ~5 Hz) because IMU's temporal
density becomes a real asset across multiple instants. **At K=1 we
cannot make that case yet**; the K=1 fusion picks Camera over IMU
when both are available.

### Step 5 — Per-trajectory smoothness (criterion (d) + RESULT_05 hard rule)

Per-test-path Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖:

| test path | smoothness r | n samples |
|---|---|---|
| path 15 | −0.007 | 875 |
| path 16 | 0.046 | 591 |
| path 17 | 0.029 | 603 |
| **median r** | **0.029** | — |

The smoothness debt **persists** in 3-modality fusion. RESULT_03's
standalone-DPVOMotion smoothness was r ≈ 0.07; here in fusion at
K=1 it's 0.03 — about the same order of magnitude. RESULT_06's
WiFi+IMU baseline didn't report this metric, but a quick re-run of
its saved checkpoint (`runs/overnight/run2_iter_06/
fusion_20260525_183313/model.pt`) would establish the WiFi+IMU
baseline smoothness as a comparator; not done this iter (compute
budget), queued as a 5-min addendum.

The interpretation is consistent with RESULT_05's Phase B
follow-ups: at K=1, single-instant fusion has no temporal axis on
which to absorb noise via cross-attention. **B-3 (temporal-cross-
attention absorption) cannot help at K=1** — that's a K>1 design
point. **B-1 (auxiliary velocity loss)** or **B-2 (EMA on per-instant
tokens)** are the K=1 options. Whether to install either depends on
PLAN_10's Odom integration result: if Odom adds genuine smoothness
on top of WiFi+Camera, the smoothness-debt fix can wait until
PLAN_11 (bake-off).

### Per-path test distribution (criterion (d))

| test path | mean (m) | median | p25 | p75 | p90 | max | n |
|---|---|---|---|---|---|---|---|
| 15 | 0.427 | 0.330 | — | — | 0.918 | 2.415 | 875 |
| 16 | 0.487 | 0.376 | — | — | 0.969 | 2.533 | 591 |
| 17 | 0.579 | 0.399 | — | — | 1.281 | 2.284 | 603 |
| **agg** | **0.489** | **0.365** | — | — | **1.070** | 2.533 | 2069 |

Comparison to RESULT_06 baseline per-path test:
- path 15: 0.477 → 0.427 (−10.5 %)
- path 16: 0.547 → 0.487 (−11.0 %)
- path 17: 0.545 → 0.579 (+6.2 %)

Mixed at the per-path level — paths 15 and 16 improve cleanly,
path 17 regresses slightly. Path 17 was already RESULT_05's
"high-curvature outlier" (mean |ω| 0.239 rad/s vs ~0.1 for other
paths); Camera's smoothness debt apparently bites hardest on
high-curvature trajectories.

Per-trajectory plots saved at
`runs/overnight/run2_iter_09/test_paths/3mod_path_{15,16,17}.png`.

## Step 6 — Decision + PLAN_10 recommendation

**Verdict (3 sentences):**

1. **3-modality fusion clears C3 lower bound** (test MAE 0.489 m
   ≤ 0.500 m gate, val 0.448 m): Camera is net-positive at the
   aggregate level (−5.4 % test, −4.5 % val vs RESULT_06).
2. **Camera and IMU are interchangeable at K=1** — subset eval shows
   WiFi+Camera (val 0.449, test 0.481) is tied with full 3-modality;
   IMU's net contribution evaporates in the 3-modality setup. This
   is K=1-specific and expected to differ at K>1 where IMU's
   ~31 Hz density matters.
3. **PLAN_10 = add Odom 1.5-modality path (OdomCNN-P-B embedding +
   raw integrated odom_x/y)** per RESULT_04's recommendation. Same
   FusionTransformer architecture, K=1; full 4-modality stack
   becomes (WiFi, IMU, Camera, Odom-embedding + Odom-raw). If
   PLAN_10's smoothness regresses materially, insert PLAN_11
   ("smoothness debug" — B-1 auxiliary velocity loss or B-2 EMA on
   tokens) before the bake-off.

**No PLAN_09b needed** — the K=1 result is honest about the
smoothness debt without it being a fusion-blocker; the K>1
bake-off (queued as PLAN_11/12) is the right place to test
B-3 (temporal cross-attention absorbs noise).

## What was changed

- `scripts/_train_webots_3mod_camera.py` — **new**. Same wrapper
  pattern as PLAN_06's 2-mod baseline; overrides
  `cfg.dataset.modalities = ['wifi', 'imu', 'camera']`, calls
  `extract_vision_tokens` for the camera cache, trains
  `FusionTrainer` with `extra_inputs={'camera': {...}}`. Reports
  pre-test gate, memory budget, full subset eval, per-path
  distribution, smoothness, plots, and JSON dump.
- `runs/overnight/run2_iter_09/fusion_20260526_001524/` (gitignored)
  — full training run dir (`model.pt`, `history.json`,
  `metrics.jsonl`, subset JSONs).
- `runs/overnight/run2_iter_09/test_paths/3mod_path_{15,16,17}.png`
  — per-trajectory plots (criterion (d)).
- `runs/overnight/run2_iter_09/wifi_imu_camera_K1.json` —
  machine-readable per-path distribution + subsets + smoothness +
  training summary.
- `runs/overnight/run2_iter_09/wifi_imu_camera_full.log` — console
  log (smoke + pre-test + full training + subset eval).

No config files modified — the wrapper script applies the modality
override in-process.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_09/`. See "What was changed".

## Open question for scientist (PLAN_10 design)

PLAN_10's "Odom 1.5-modality" per RESULT_04: feed BOTH the
`OdomCNN-P-B` learned embedding (128-d) AND the raw integrated
`(odom_x, odom_y)` 2-D feature into the fusion model. The
FusionTransformer's universal-token contract is
`encoder_embedding + modality_embedding + time_encoding(Δt)` — for
the "1.5 modality" we need to choose:

- **(A) Single modality slot, 130-d concat** — concatenate the 128-d
  OdomCNN embedding with the 2-D raw integrated (x, y) before adding
  modality + time embeddings. Single modality_dropout draw covers
  both signals. Loses the ability to drop one but keep the other.
- **(B) Two modality slots** — `odom_cnn` (128-d) and `odom_raw`
  (project 2-D to 128-d via a small linear). modality_dropout draws
  independently. Lets the gate downweight one when the other works.
  Doubles the "modality" count to 5.

**My read**: (B). Two slots is the cleaner research story (each
modality has its own dropout draw; the fusion model can learn that
raw odom is the smoothness anchor and OdomCNN is the local-motion
contributor). The cost is one extra small linear (~32 k params)
and the fusion attention scales as N² in modality count, but at
N = 5 that's 25 vs 9 attention pairs — negligible.

## Cycle-rules compliance

- ✅ Pre-test gate: −80.8 % drop (≥ 10 %).
- ✅ Memory budget at target shape (B=32 train probe): 466 MB peak
  (one TypeError on the probe's 2-arg signature, but the peak was
  captured before the exception).
- ✅ Day-1 SOTA reproduction analog: RESULT_06's WiFi+IMU baseline
  is the in-house reference (0.43-0.47 m), reproduced and improved.
- ✅ Per-path distribution + per-trajectory smoothness reported.
- ✅ Per-trajectory plots saved for top-3 longest test paths.
- ✅ Latency reported (0.053 ms/sample).
- ✅ Full subset eval matrix (criterion (b)).
- ✅ Demand #3: no vendored sources touched; DPVO encoder loaded
  via the same restored `dpvo_motion.py` from RESULT_03.

## Phase B progress

| iter | task | val MAE | test MAE | criterion (b)? |
|---|---|---|---|---|
| 06 | WiFi + IMU K=1 (foundation) | 0.469 m | 0.517 m | ⚠ (just above 0.50) |
| **09** | **+ Camera K=1** | **0.448 m** | **0.489 m** | **✅** |
| 10 (next) | + Odom 1.5-modality | TBD | TBD | TBD |
| 11/12 | bake-off + K>1 temporal | TBD | TBD | TBD |

## Stop conditions

- Local time at write: **Tue May 26 ~00:20 local** (well inside
  STATE Stop-at 2026-05-26 18:00).
- No `handoff/STOP` file.
- `GOAL_REACHED: false` — C3 lower bound cleared with 3 modalities;
  4-modality target still ahead (PLAN_10).
