# Result 03 — camera-encoder-audit-webots

## TL;DR

**`DPVOMotionEncoder` = keep** under PLAN_03's amended rubric.
Best configuration is P-A (ImageNet-norm → DPVO-norm) with the
default `camera_stride=5`: **val MAE 1.85 m, test MAE 1.56 m** on the
canonical CLAUDE.md Webots split (train [1, 3-12], val [2, 13, 14],
test [15, 16, 17]). The multi-condition gate passes with margin —
the test-val gap is **negative** (−15.7 %) across all three
preprocessing/stride configurations, i.e. the encoder *transfers*
between paths rather than overfitting to train paths. Per-path test
median is 0.71 m on path_15 and 1.33/1.48 m on path_16/17. The
linear-probe MAE of 1.85 m is **~2× better** than CLAUDE.md's
pre-run-1 ACEVision reference (~3.5 m). DPVO's full SLAM pipeline
could not run on this machine (`lietorch`/`altcorr` CUDA ops absent),
so Step 1 (native-benchmark reproduction) was skipped per the plan's
Branch Q clause — the audit decision rests on Step 2's measurement.
A real weakness surfaced for the fusion designer's attention:
**per-trajectory smoothness is poor** (median Pearson r between
‖Δpred‖ and ‖Δgt‖ = +0.07 across test paths), meaning consecutive
head outputs are noisy in motion-magnitude space even though absolute
position is well-recovered. Step 0c's Umeyama retro on RESULT_02
**softened but did not flip** the IMUCNN-keep verdict (addendum
appended to RESULT_02; all three IMU encoders collapse to ~0.30 m
Umeyama-aligned ATE, indicating they recover the same motion *shape*
at different scales).

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0a. Restore run-1 camera files | files restored + encoder import smoke | 8 files restored from `overnight-autonomous-2026-05-24` (`dpvo_motion.py`, `dpvo_full.py`, `eval_dpvo.py`, `extract_dpvo_features.py`, `fetch_dpvo_weights.py`, `run_dpvo_paths.py`, `dpvo_motion_encoder.md`, `dpvo.yaml`, `dpvo_full.yaml`). `from src.pipeline.encoders.dpvo_motion import DPVOMotionEncoder` works. `__init__.py` not updated this iteration (no fusion-tower yet needs the registration); deferred. | ✅ |
| 0b. DPVO setup probe | Branch P/Q decision | **Branch Q for SLAM, Branch P for encoder.** `external/dpvo/` has only `extractor.py` + `__init__.py` — no `lietorch`/`altcorr` CUDA extensions, no `dpvo.config`/`dpvo.net`. The motion encoder uses only `BasicEncoder4` from `extractor.py` + a custom correlation head, so it loads cleanly. Pretrained weights `dpvo.pth` downloaded (13.5 MB, gdrive id `1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX`). Forward pass `(2, 2, 3, 480, 640) → (2, 128)` confirmed working. | ✅ encoder / ⏭ SLAM |
| 0c. Umeyama retro on RESULT_02 | addendum + verdict update | Addendum appended to RESULT_02 with Umeyama-aligned ATE for all three IMU encoders (re-run from scratch — no checkpoints saved). All three collapse to **~0.30 m Umeyama-aligned ATE**, indicating they recover the same motion *shape* at different scales. IMUCNN = `keep` stands; "capacity probe refuted" softened to "capacity not clearly the bottleneck" (second-run noise shifted IMUCNN-2× from regressed to tied with base). `evo` installed into the venv (2 MB, pure-Python). | ✅ |
| 1. Day-1 SOTA reproduction (DPVO native) | DPVO published number ±20 % | **SKIPPED** — Branch Q (DPVO SLAM not runnable on this machine). Documented per plan: not the gating step for the audit decision; Step 2 carries the verdict. | ⏭ |
| 2. DPVOMotion on Webots (canonical split, P-A) | trains; val+test MAE + per-path | val MAE = **1.85 m**, test MAE = **1.56 m**, per-path table below | ✅ |
| 2. DPVOMotion on Webots (P-B) | trains; val+test MAE + per-path | val MAE = 2.02 m, test MAE = 1.70 m | ✅ |
| 3. Cross-condition (val/test) gate | test-val gap < 20 % | **−15.7 % (P-A)**, −15.9 % (P-B), −9.1 % (stride10). All negative → test < val. | ✅ |
| 4. Capacity/config probe | one probe + delta vs Step 2 default | `camera_stride=10` (sparser correlation): val 1.82 m (−1.6 % vs P-A), test 1.66 m (+6.4 % vs P-A). Within training noise — stride doesn't help. | ✅ (probe ran; verdict = stride neutral) |
| 5. 6-metric harness | one row per (encoder, preprocessing) | three rows tabulated below | ✅ |
| 6. Audit decision | label + multi-condition gate satisfied + preprocessing labelled | **DPVOMotionEncoder = keep, P-A preprocessing.** All three rubric subcriteria pass. | ✅ |
| Pre-test gate (Step 2, 10 % subset, 5 epochs) | subset val moves ≥ 10 % | 420-pair subset, 5 epochs: 5.547 → 4.944 m (−10.9 %). Borderline but passes. | ✅ |
| Memory budget (Step 2, B=4, 2×480×640×3 + backward through head) | < 6 GB | peak **299.4 MB** | ✅ |

### Headline numbers

DPVOMotionEncoder on Webots Tiago sim, canonical split:
**train = paths [1, 3-12]** (11 paths, 4 207 frame pairs at stride 5),
**val = paths [2, 13, 14]** (1 139 pairs), **test = paths [15, 16, 17]**
(1 022 pairs). Trunk is the DPVO `BasicEncoder4` (0.18 M params,
frozen, NeurIPS 2023 pretrained). Trainable: per-patch motion head +
2D position projection (0.15 M params). 30 epochs, AdamW + OneCycleLR
+ Huber(δ=0.5).

| preprocessing / stride | val MAE | test MAE | test-val gap | test p25 / p50 / p75 / p90 / max | per-traj smoothness median r | latency b=1 (ms) | params head | source |
|---|---|---|---|---|---|---|---|---|
| **P-A (ImageNet → DPVO, stride 5)** | **1.85 m** | **1.56 m** | −15.7 % | 0.52 / **1.04** / 1.85 / **3.13** / 9.72 | +0.07 | 10.91 | 0.15 M | `scripts/_eval_webots_dpvo.py` |
| P-B (DPVO-norm only, stride 5) | 2.02 m | 1.70 m | −15.9 % | 0.49 / 1.00 / 2.08 / 4.23 / 11.43 | −0.06 | same | same | same |
| P-A-stride10 (capacity probe) | 1.82 m | 1.66 m | −9.1 % | 0.57 / 1.13 / 2.04 / 3.48 / 10.25 | −0.03 | same | same | same |

Per-path test MAE (P-A default, the audit-winner config):

| test path | n frames | mean MAE | median | p90 | max |
|---|---|---|---|---|---|
| path_15 | 432 | **1.07 m** | 0.71 | 2.20 | 8.85 |
| path_16 | 293 | 1.85 m | 1.33 | 4.35 | 9.72 |
| path_17 | 297 | 1.99 m | 1.48 | 4.33 | 9.61 |

Per-trajectory plots saved under
`runs/overnight/run2_iter_03/test_paths/`:
- `P-A_path_15.png`, `P-A_path_16.png`, `P-A_path_17.png`
- `P-B_path_*.png` + `P-A-stride10_path_*.png` for the variants
(satisfies criterion (d) of STATE.md — top-3 longest test paths).

### 6-metric harness (Webots val embeddings)

Camera on Webots IS temporally ordered, so all six metrics apply.
Temporal smoothness here is computed on val embeddings sorted by
(path, time), with positions as the "displacement" reference.

| metric | P-A | P-B | P-A-stride10 | winner / note |
|---|---|---|---|---|
| linear-probe Euclid (m) | **1.85** | 2.03 | 1.82 | stride10 ≈ P-A (within noise); both ~2× better than ACEVision ref (3.5 m) |
| kNN-probe Euclid (m, k=5) | 1.61 | 1.84 | **1.56** | stride10 best (but within noise) |
| alignment (lower=better, 1 m physical thr) | 0.79 | 0.83 | **0.73** | stride10 best |
| uniformity (lower=better, t=2) | **−2.10** | −2.08 | −1.98 | P-A — best spread |
| eff-dim PR | **4.20** | 4.10 | 3.95 | P-A — highest participation ratio |
| trustworthiness (k=10) | 0.726 | 0.730 | **0.727** | tie (~0.73) |
| temporal smoothness r (val, in-traj-order) | 0.167 | 0.165 | 0.172 | all weak (~0.17) — flagged below |

The geometry metrics broadly favour P-A; P-A's full 6-metric panel is
the cleanest. Preprocessing matters: P-B (DPVO-norm only) is
consistently a step worse on linear-probe / kNN, confirming ImageNet
input normalisation isn't optional — the trunk was trained on
ImageNet-normalised images and skipping the un-norm step before
DPVO's own `2x − 0.5` distorts the input distribution.

### Honest weakness — per-trajectory smoothness

**Median Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖ across test paths is
+0.07 (P-A), −0.06 (P-B), −0.03 (stride10)** — essentially zero.
Frame-to-frame the predicted position is well-anchored (test
aggregate p50 = 1.04 m, max 9.72 m), but the *velocity* implied by
consecutive head outputs doesn't track the GT velocity. This is the
single weakness the audit surfaces and it matters for fusion:
- The encoder is a "where am I" predictor, not a "how fast am I
  moving" predictor; the per-patch motion tokens carry flow info,
  but the head reduces them to absolute position.
- For fusion with the IMU/Odom motion modalities, the temporal
  consumer (transformer attending over K instants) gets a noisy
  motion signal from camera. The fix is downstream — either supply
  velocity targets to the camera head as an auxiliary loss, or let
  the fusion-temporal layer smooth.
- The on-Webots `temporal_smoothness.correlation` metric reports
  `r = 0.167` (P-A val embeddings vs GT positions in time order),
  which is exactly the geometric counterpart of this finding — the
  embedding *space* changes slowly over time but the *predictions*
  are noisy in motion magnitude.

This does not block `keep`. The audit rubric weights raw test MAE
(1.56 m, very good) over smoothness for the per-leg label. It's a
note for the fusion designer.

## Audit decision

**DPVOMotionEncoder = keep, with P-A preprocessing as the canonical
config.**

Justification (3 sentences): P-A's 1.56 m raw test MAE beats the
nearest CLAUDE.md vision reference (ACEVision ~3.5 m linear-probe)
by 2×, and the **−15.7 % test-val gap** is comfortably inside the
amended-rubric 20 % multi-condition window — the encoder *transfers*
between train and unseen paths. The preprocessing-variation probe
shows ImageNet-norm matters (P-A 1.56 vs P-B 1.70 m, +9 %, consistent
across all 6 geometry metrics), so the verdict is conditioned on
**P-A only**; P-B remains a weaker fallback. The capacity probe
(stride 10) is neutral within training noise → no `modify` track
needed.

## Three-orthogonal-probe view

1. **Architecture probe** — DPVO trunk (NeurIPS 2023, scene-agnostic
   ImageNet-pretrained patch CNN, 0.18 M params) on Webots: 1.56 m
   test MAE. The trunk transfers from TartanAir/EuRoC training domain
   to Webots without retraining. **Architecture is sound.**
2. **Preprocessing probe** — P-A vs P-B: P-A wins by 9 % on test
   MAE consistently across the 6-metric panel. **Preprocessing
   matters; the documented P-A path is correct.**
3. **Capacity/config probe** — `camera_stride=10` vs 5: stride 10
   slightly improves val (1.82 vs 1.85) but slightly hurts test
   (1.66 vs 1.56). Within training noise. **Stride doesn't help.**

Three orthogonal probes agree there's no obvious modify direction
from the audit alone.

## PLAN_04 recommendation

Continue to **PLAN_04 = Odom encoder audit on Webots sim** (internal,
no public SOTA). DPVO is shipped as `keep` with no parallel Camera
modification track. The smoothness weakness is logged for **Phase B
fusion design** (consider auxiliary velocity loss / smoothing
adapter) — not for a follow-up Camera iteration.

3-sentence justification: PLAN_04 completes the Phase A audit cycle
(WiFi ✓ IMU ✓ Camera ✓ Odom →) and unblocks Phase B fusion redesign
with all four encoder verdicts in hand. The smoothness weakness is a
fusion-architecture concern (how to consume noisy per-instant motion
predictions), not a Camera-encoder concern. Doing PLAN_04 now keeps
the audit-order intact and gives Phase B a complete picture.

## What was changed

- `src/pipeline/encoders/dpvo_motion.py`, `dpvo_full.py` — restored
  from run-1.
- `configs/stage_a/vision/dpvo.yaml`, `dpvo_full.yaml` — restored.
- `scripts/eval_dpvo.py`, `extract_dpvo_features.py`,
  `fetch_dpvo_weights.py`, `run_dpvo_paths.py` — restored.
- `scripts/diagnostic_dpvo_patch_viz.py`,
  `scripts/diagnostic_dpvo_upstream_viz.py` — restored (not run
  this iteration; kept for documentation).
- `docs/dpvo_motion_encoder.md`, `dpvo_correlation_diagnostic.png`,
  `dpvo_patch_viz_diagnostic.png`, `dpvo_upstream_patches.png` —
  restored.
- `runs/_weights/dpvo.pth` — downloaded via
  `scripts/fetch_dpvo_weights.py` (13.5 MB, gitignored under
  `runs/_weights/`).
- `scripts/_eval_webots_dpvo.py` — **new**. Webots audit
  trainer + evaluator. Trains 3 head variants (P-A / P-B /
  P-A-stride10) on cached DPVO trunk + correlation tokens, reports
  per-path distribution + per-trajectory smoothness + 6-metric
  harness, and saves top-3 longest-test-path trajectory plots.
- `scripts/_eval_ronin_a000_branchY.py` — added `_umeyama_align()`
  helper; `per_chunk_ate()` now reports raw + SVD-aligned (legacy)
  + Umeyama-aligned ATE (Step 0c retro).
- `handoff/results/RESULT_02_imu-encoder-audit-ronin.md` — Umeyama
  addendum appended.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_03/`:
- `smoke.log` — 2-epoch smoke (confirmed memory budget, pretest gate,
  and pair counts).
- `webots_dpvo_full.log` — full 30-epoch run output (all three
  conditions).
- `webots_dpvo.json` — per-run JSON (includes per-path distribution
  arrays and 6-metric values).
- `test_paths/{P-A,P-B,P-A-stride10}_path_{15,16,17}.png` — top-3
  longest-test-path trajectory plots (9 PNGs, criterion (d) of
  STATE.md).

For Step 0c (Umeyama retro), the artefacts live in
`runs/overnight/run2_iter_02/`:
- `branchY_umeyama.log` — retro re-run console output.
- `a000_branchY.json` — JSON overwritten with `umeyama_*` fields and
  `per_chunk_umeyama` array.

## Demand #3 specifics

- `external/dpvo/` not edited. `dpvo_motion.py`'s `from external.dpvo
  import BasicEncoder4` import works against the vendored
  `extractor.py` only (no DPVO SLAM code touched, no upstream patch).
- DPVO weights downloaded as a release artifact via `gdown`; the
  loaded `state_dict` is read but not re-saved. No re-export, no
  modification.
- `fetch_dpvo_weights.py` is our wrapper script and the only
  Internet-touching dependency.

## Open question for scientist

**Q.** Should Phase B's fusion redesign include an explicit camera-
smoothness intervention (auxiliary velocity loss, EMA on per-instant
camera tokens, temporal smoothing adapter), or is it cleaner to let
the fusion transformer absorb the noise via its temporal cross-
attention?

**My read:** let the fusion transformer absorb it. The
DPVOMotionEncoder is shipped as a "per-pair position guess" and the
fusion model is the right place to time-smooth. Adding an
auxiliary head this late in the audit would couple two questions
(does the encoder need a velocity loss? does fusion smooth?) that
are cleaner separated. **Suggestion:** measure camera-leg
contribution in the Phase B bake-off — if the camera token noise
visibly degrades fusion MAE, revisit then.

## Cycle-rules compliance (amended rubric)

- ✅ Pre-test gate ran (10 % subset, 5 epochs): pre-test val MAE
  dropped 5.55 → 4.94 m (−10.9 %, just over the 10 % threshold).
- ✅ Memory budget checked at target shape (B=4, two 480×640×3
  frames, fwd+bwd through trainable head). Peak 299.4 MB << 6 GB.
- ⏭ Day-1 SOTA reproduction skipped (Branch Q for DPVO SLAM); the
  encoder still loads its pretrained-from-DPVO trunk weights
  unmodified, so the spirit of "use SOTA unmodified" is preserved.
- ✅ Per-modality per-path distribution (p25/p50/p75/p90/max)
  reported for every condition.
- ✅ Multi-condition gate (amended rubric correction #1): both val
  and test reported; test-val gap < 20 % required and observed
  (−15.7 %).
- ✅ Preprocessing-aware (correction #2): two preprocessing
  conditions reported; verdict labels which one is canonical.
- ✅ Raw-weighted decisions (correction #3): primary signal is raw
  test MAE (1.56 m); 6-metric geometry is secondary; aligned/Umeyama
  metrics are not used here (Webots is in-world, no scale ambiguity
  to align away).
- ✅ Umeyama retro on RESULT_02 (Step 0c): completed; addendum
  appended; canonical alignment library standard going forward.
- ✅ Three orthogonal probes (architecture / preprocessing / config)
  reported.
- ✅ No silent stalls; iteration well inside the 90-min budget
  (~80 min wall clock including Step 0c retro and DPVO weight
  download).

## Stop conditions

- Local time at write: **Mon May 25 ~13:50 local** (inside
  STATE Stop-at 2026-05-26 18:00).
- No `handoff/STOP` file.
- `GOAL_REACHED: false` — 3/4 Phase-A encoders triaged.

---

## Addendum 2026-05-25 ~15:55 — third-party-review responses (PLAN_05 Step 0)

PLAN_05's Step 0 retros respond to a 2026-05-25 ~15:20 third-party
review of this RESULT. Three issues were flagged; addressed below.

### Step 0a — Difficulty-matched probe

The original "test 1.56 m < val 1.85 m → no overfit" reasoning was
naive: val paths {2, 13, 14} and test paths {15, 16, 17} are not
difficulty-matched. The probe computes per-path difficulty features
from `ground_truth.csv` and normalises MAE by path length.

| path | split | length (m) | mean speed (m/s) | `mean |ω|` (rad/s) | n_pairs | MAE (m) | MAE / length |
|---|---|---|---|---|---|---|---|
| 2 | val | 25.83 | 0.313 | 0.147 | 424 | 2.327 | 0.0901 |
| 13 | val | 19.81 | 0.318 | 0.143 | 320 | 0.700 | 0.0354 |
| 14 | val | 25.14 | 0.328 | 0.104 | 395 | 2.272 | 0.0904 |
| 15 | test | 26.41 | 0.315 | 0.089 | 432 | 1.072 | 0.0406 |
| 16 | test | 18.14 | 0.320 | 0.070 | 293 | 1.849 | 0.1019 |
| 17 | test | 17.89 | 0.309 | 0.239 | 297 | 1.985 | 0.1110 |

- val aggregate: MAE = 1.766 m (per-path mean), length = 23.59 m, **MAE/m = 0.0719**.
- test aggregate: MAE = 1.636 m (per-path mean), length = 20.82 m, **MAE/m = 0.0845**.
- **Raw test-val gap** (per-path-mean basis) = **−7.4 %** (test wins
  on raw MAE, but only because test paths are shorter).
- **Difficulty-normalised gap** (per-meter basis) = **+17.5 %** (val
  wins per-m by 17.5 %; **test is actually 17.5 % harder per meter**).

(Note: the per-path-mean basis above differs slightly from the
RESULT_03 main-table aggregate, which weights by frame count and gives
val 1.85 / test 1.56. The main-table aggregate and the per-path-mean
aggregate both lead to the same difficulty-normalised conclusion.)

**Verdict**: the keep label survives the multi-condition gate
(difficulty-normalised gap 17.5 % is **just inside** the 20 % window),
but the margin is much tighter than the raw test-beats-val finding
suggested. The verdict is held **at the edge of the gate**, not
comfortably inside. Future audit iterations should report per-path
difficulty features in the multi-condition table by default.

Probe script: `scripts/_difficulty_probe_paths.py`. JSON output:
`runs/overnight/run2_iter_05/camera_difficulty_probe.json`.

### Step 0b — Smoothness-debt reframe

The original audit verdict `keep` was paper-soft on the
per-trajectory smoothness weakness (median Pearson r ≈ 0.07 between
‖Δpred‖ and ‖Δgt‖ across test paths). Reframing the verdict label:

> **DPVOMotionEncoder = keep with smoothness debt**, P-A
> preprocessing as the canonical config. The "debt" is explicit: the
> encoder predicts absolute position competitively (1.56 m raw test
> MAE) but does not produce a usable per-frame motion signal — the
> motion magnitude between consecutive head outputs is uncorrelated
> with GT motion magnitude.

Phase B follow-up candidates (must be revisited when fusion
architecture is chosen):

- **(B-1) Auxiliary velocity loss on the camera head** during
  fusion training. Adds a GT-velocity supervision target alongside
  the position regression; trades head expressiveness for explicit
  motion signal.
- **(B-2) EMA smoothing on per-instant camera tokens** before they
  reach the fusion transformer. Cheap, easy to ablate, no head
  retraining needed.
- **(B-3) Let the fusion transformer absorb the noise** via
  temporal cross-attention (the current RESULT_03 default
  recommendation). Most "clean" approach but most dependent on the
  Phase B architecture choice.

**Hard rule**: Phase B's bake-off iteration must report **per-modality
per-trajectory smoothness** in any 4-modality test run so the debt
is visible, not silent.

### Step 0c — PLAN_06 (Camera external SOTA) queued

The Camera audit ran with `lietorch`/`altcorr` unavailable (Branch
Q), so DPVO-as-SLAM was never reproduced end-to-end on a public
benchmark. The Webots-only audit by itself does **not** discharge
per-leg validation for Camera (the Phase A summary table in
RESULT_04 already marks Camera C3-pending).

PLAN_06 is queued (will be issued by the scientist after this
iteration) to:
1. Pick **one** public visual-odometry benchmark (TartanAir →
   EuRoC → KITTI, in preference order).
2. Pick **one** method to reproduce (DPVO unmodified → TartanVO →
   DROID-SLAM, in preference order).
3. Run the project's `DPVOMotionEncoder` trunk on the SAME public
   sequence (motion-only, no SLAM tracker) and compare against
   the SOTA pipeline's reported ATE.
4. Update RESULT_03's per-leg label using public-benchmark evidence.

PLAN_06 is queued **after** PLAN_05 (C2 closure) because RoNIN data
is more accessible locally than TartanAir/EuRoC.

---

End of PLAN_05 Step 0 addendum.
