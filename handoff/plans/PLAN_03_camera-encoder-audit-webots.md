# Plan 03 — Camera encoder audit (DPVOMotionEncoder on Webots sim)

> **Audit rubric amended (2026-05-25 ~12:50 local).** The third-party
> review of RESULT_02 flagged three issues with how Phase A audits
> were graded; this plan and every Phase-A plan after it (PLAN_04
> onwards) apply the amended rubric below. STATE.md is updated to
> match. Brief summary:
>
> 1. **Multi-condition validation.** `keep` requires the encoder
>    holds on ≥ 2 evaluation conditions — ideally 2 datasets; if a
>    modality is single-dataset (e.g. Camera = Webots only), the 2
>    conditions become {nominal split, perturbed/transfer split} in
>    the same world. Document the limitation when only one dataset
>    exists.
> 2. **Preprocessing is a first-class audited variable**, not a
>    nuisance to absorb in a wrapper. Every result must name the
>    preprocessing applied and report at least one preprocessing-
>    variation probe (or cite a prior one).
> 3. **Aligned metrics use Umeyama / a standard library**
>    (`scipy.spatial.procrustes`, `evo`, or RoNIN's own
>    `metric.compute_ate_rte`), never a hand-rolled SVD fit. Raw
>    error is **weighted at least as heavily as aligned** in audit
>    decisions — fusion's downstream consumer (the WiFi anchor)
>    cares about raw error.

## Hypothesis

`DPVOMotionEncoder` (a frozen DPVO patch trunk + trainable correlation
head + attentive pool, per `handoff/archive/run1/results/RESULT_06`
context and `src/pipeline/encoders/dpvo_motion.py` on the run-1 branch)
should produce useful motion descriptors on Webots Tiago video. CLAUDE.md
records a pre-run-1 number: **~3.5 m linear-probe MAE**, **~2.9 m
standalone (DPVOFull line)** on Webots. The Camera audit asks (i)
whether DPVO still works pipeline-end-to-end against its own paper
numbers (Day-1 SOTA reproduction on its native benchmark), (ii) whether
the motion-encoder variant matches the pre-run-1 Webots numbers, and
(iii) whether it generalises across Webots paths.

Because Camera only exists on Webots (no public real-data parallel),
"multi-condition" here is the canonical CLAUDE.md path split:
- train = paths [1, 3-12], val = [2, 13, 14], test = [15, 16, 17].
The "transfer" condition is val → test (different paths, same world).
This is documented as a limitation, not a clean cross-dataset transfer.

Expected outcomes (under the amended rubric):
- `keep` — DPVOMotion within 20 % of the best published learned-VO
  reference AND val → test gap < 20 % (i.e. doesn't overfit to train
  paths).
- `modify` — within 20–50 % OR test holds but linear-probe / 6-metric
  geometry shows a fixable structural issue (capacity, head depth).
- `replace` — gap > 50 % OR test fails to transfer → name the
  alternative (TartanVO, DROID-SLAM frontend, learned monocular flow).

## Steps

### Step 0 — Recovery, DPVO setup probe, and IMU-aligned retro (10–15 min, all gates)

**Step 0a.** Restore run-1 camera-audit files. They are on
`overnight-autonomous-2026-05-24` only:

```powershell
git checkout overnight-autonomous-2026-05-24 -- `
  src/pipeline/encoders/dpvo_motion.py `
  src/pipeline/encoders/dpvo_full.py `
  configs/stage_a/vision/dpvo.yaml `
  configs/stage_a/vision/dpvo_full.yaml `
  scripts/eval_dpvo.py `
  scripts/extract_dpvo_features.py `
  scripts/fetch_dpvo_weights.py `
  scripts/run_dpvo_paths.py `
  scripts/diagnostic_dpvo_patch_viz.py `
  scripts/diagnostic_dpvo_upstream_viz.py `
  docs/dpvo_motion_encoder.md `
  docs/dpvo_correlation_diagnostic.png `
  docs/dpvo_patch_viz_diagnostic.png `
  docs/dpvo_upstream_patches.png
```

Append `DPVOMotionEncoder, DPVOFullEncoder` to
`src/pipeline/encoders/__init__.py` (PLAN_01 only restored the WiFi
exports). Import smoke:
`python -c "from src.pipeline.encoders import DPVOMotionEncoder, DPVOFullEncoder"`
must succeed. If `DPVOFullEncoder` pulls in dependencies that aren't on
this branch and would slow Step 0, restore only `DPVOMotionEncoder`
and document.

**Step 0b — DPVO setup probe.** DPVO needs custom CUDA ops
(`dpvo.lietorch`, `dpvo.altcorr`). Check, in order:
1. `external/dpvo/` already cloned (confirmed present
   2026-05-25 ~12:50); check for a built extension at
   `external/dpvo/dpvo/altcorr/*.pyd` / `*.so`.
2. `python -c "from dpvo.lietorch import SE3"` from the project
   venv (needs `external/dpvo` on `PYTHONPATH`).
3. Pretrained weights `external/dpvo/dpvo.pth` (downloaded by
   `scripts/fetch_dpvo_weights.py` historically).

Branch on outcome:
- **Branch P — DPVO importable.** Proceed with Steps 1–5 in full.
- **Branch Q — DPVO not importable.** The DPVOMotion *encoder* may
  still work if the trunk weights load through the
  `src/pipeline/encoders/dpvo_motion.py` wrapper alone (it
  shouldn't need lietorch for the motion descriptor path). Skip
  Step 1 (DPVO-as-SLAM on its native benchmark) and document the
  obstacle. Continue Steps 2–5 with `DPVOMotionEncoder` only.

**Step 0c — IMU-aligned ATE retro (5 min, retrofitting RESULT_02
under correction #3).** RESULT_02's `_ate_aligned` was a hand-rolled
SVD Procrustes — under the amended rubric it should be Umeyama.
Engineer reruns the aligned-ATE calc on the saved per-chunk
predictions stored in `runs/overnight/run2_iter_02/a000_branchY.json`
using **scipy.spatial.procrustes** OR **evo's
`evo.core.metrics.APE`** (whichever ships in the venv first).
If the venv has neither, install `evo` (it is a 2-MB pure-Python
package). Report:

| encoder | aligned ATE old (hand-rolled SVD) | aligned ATE Umeyama | delta |
|---|---|---|---|
| RoNIN ResNet1D | 0.97 m | … | … |
| IMUCNN base | 1.04 m | … | … |
| IMUCNN 2× | 1.71 m | … | … |

Write the result as a brief **RESULT_02 addendum** appended to
`handoff/results/RESULT_02_imu-encoder-audit-ronin.md` under a new
"### Addendum 2026-05-25 — Umeyama re-alignment" section. The
audit verdict (`IMUCNN = keep`) may or may not change; the addendum
documents the post-correction numbers and either re-affirms or
revises the label. **Raw ATE is unaffected** (no alignment).

This retro is a hard cycle-rules gate: before any future "aligned
metric" claim in Phase A, the alignment must be Umeyama. Engineer
applies the same library to Steps 2–4 below from the outset.

**Acceptance for Step 0:** all three sub-steps either pass or have
one-paragraph obstacle notes; iteration is not blocked even if
Step 0c finds something surprising.

### Step 1 — Day-1 SOTA reproduction (DPVO on a published sequence)

Only if Branch P (DPVO importable). The point is to confirm the
DPVO pipeline produces its published numbers on a sequence DPVO was
designed for, before we trust its trunk as a feature extractor on
Webots.

Pick the **shortest** sequence in the DPVO `evaluation_scripts/`
test set that the vendored repo ships pretrained weights for. By
preference: **TartanAir abandonedfactory_001** (DPVO's training
domain — should pass cleanly) or **EuRoC MH_01_easy** (~80 s).
Run DPVO unmodified:

```powershell
cd external\dpvo
.venv\Scripts\python.exe evaluation_scripts\test_tartan.py `
  --config config/default.yaml `
  --plot --save_ply `
  --datapath <path-to-tartan-sequence>
```

(Adjust the command to whichever entry-point the repo actually
ships; engineer reads `external/dpvo/README.md` first.)

If the vendored data sequence isn't on the machine and downloading
is > 15 min, document and skip — Step 1 is "best-effort," not the
gating step for the Camera audit. The audit decision rests on
Step 2.

- **Acceptance:** at least one DPVO published number reproduced
  within ±20 % (looser than other modalities because DPVO's full
  pipeline has many hyperparameters that aren't ours to tune); OR
  documented obstacle.

### Step 2 — DPVOMotionEncoder on Webots sim (canonical split)

This is the audit's primary measurement. Run
`scripts/eval_dpvo.py` (restored in Step 0a) on the canonical
Webots split:

- train = paths [1, 3-12] (= 11 paths)
- val   = paths [2, 13, 14]
- test  = paths [15, 16, 17]

The encoder is `DPVOMotionEncoder` — frozen DPVO trunk + trainable
head; CLAUDE.md baseline says ~3.5 m linear-probe MAE on Webots.

**Pre-test gate.** Train on 10 % of the train paths for 5 epochs;
expect val MAE to drop ≥ 10 %.

**Memory budget check.** Forward+backward on a synthetic batch at
target shape (B=8, two 480×640×3 frames per sample, frozen trunk +
trainable head). Peak GPU MB reported; < 6 GB.

**Preprocessing variation probe** (cycle rule #2): run Step 2 in
**two** preprocessing conditions and report both:
- (P-A) **default**: ImageNet-norm → DPVO-norm (`2x−0.5`), the
  encoder's documented input convention.
- (P-B) **DPVO-norm only**: skip ImageNet-norm, feed raw `[0, 1]`
  then DPVO-norm directly.
If the two preprocessings give materially different MAE, that's the
preprocessing-variation finding for this modality. If they don't,
document "preprocessing is robust" and move on.

- **Acceptance:** train completes; val MAE and test MAE both
  reported with per-path distribution (median, p25, p75, p90, max,
  per-trajectory smoothness ratio). **Per-path top-3 longest
  trajectories: plot predicted vs GT (saved under
  `runs/overnight/run2_iter_03/test_paths/*.png`)** — required by
  criterion (d) of STATE.md.

### Step 3 — Cross-condition probe (within-sim transfer, the
  multi-condition analog of cross-dataset)

Same data, same encoder, two evaluation conditions:
- **C-1 (nominal):** val MAE on the canonical val=[2, 13, 14]
  split, with train_loss / val_loss / per-path scatter as the
  in-distribution metric.
- **C-2 (transfer):** test MAE on test=[15, 16, 17], NEVER seen
  by the model, as the proxy "transfer" condition.

Compute the **test-val gap ratio** `(test_MAE − val_MAE) / val_MAE`
and report. Acceptance for `keep` per the amended rubric: test-val
gap < 20 %. If gap > 20 %, the encoder doesn't transfer between
paths in the same world, and `keep` is not defensible.

Document explicitly that this is a **same-world cross-path** probe,
**not** a cross-dataset transfer. The C3 (4-modality fusion on
Webots) claim doesn't need cross-dataset on Camera (Camera is sim-
only by project design), so this limitation is intentional, not a
gap.

### Step 4 — Capacity / config probe (orthogonal probe #3)

Vary ONE DPVO config dimension, by preference in order:

- (a) **n_patches** (default in `configs/stage_a/vision/dpvo.yaml`,
  if present — guess: 96 or 128). Probe at 2× the default. If
  default is 96, run 192; if 192, run 96.
- (b) **camera_stride** (default 5 per `dpvo_motion.py` docstring).
  Probe at stride=3 (denser correlation) and stride=10 (sparser).

Pick ONE of (a) or (b), not both. Re-train head only (trunk stays
frozen) for the same epoch budget as Step 2. Report whether the
variation improves test MAE.

- **Acceptance:** one probe + delta vs Step 2's Step 2 default.
  If the probe closes ≥ 50 % of any gap to the published
  pre-run-1 number (~2.9 m on Webots from CLAUDE.md's
  DPVOFullEncoder line), the audit label moves toward `modify`.

### Step 5 — Six-metric harness (with Umeyama alignment if any
  metric needs it — all six are alignment-free on this dataset)

Run `src/pipeline/evaluation/encoder_eval.py` on Webots val
embeddings from the Step 2 default-preprocessing run. All six
metrics apply (Camera on Webots IS temporally ordered, unlike
WiFi/UJI). One row per (encoder, preprocessing-variant).

If `src/pipeline/evaluation/encoder_eval.py` isn't on this
branch, restore from run-1. If non-trivial, skip Step 5 and ship
a partial RESULT — do not let the harness block the audit.

### Step 6 — Audit decision (amended-rubric)

Label `DPVOMotionEncoder` with `keep` / `modify` / `replace`. The
label must satisfy ALL of:
1. **Multi-condition (Step 3) gate:** test-val gap < 20 % for `keep`.
2. **Raw-weighted (correction #3):** raw test MAE is the primary
   signal; alignment metrics are secondary.
3. **Preprocessing-aware (correction #2):** label which
   preprocessing the verdict is conditioned on (P-A or P-B).

One-line justification per label quoting the numbers. Recommend
PLAN_04 (Odom encoder audit on Webots sim, internal eval — no
public SOTA).

## Sources

- DPVO (Teed, Lipson, Deng — NeurIPS 2023):
  https://arxiv.org/abs/2208.04726.
- DPVO repo (vendored): `external/dpvo/` (this branch has it as
  untracked per `git status`).
- DPVOMotionEncoder spec (run-1 doc, restored in Step 0a):
  `docs/dpvo_motion_encoder.md`.
- Webots Tiago dataset: 18 paths, `data/async_collection/path_*/`
  per CLAUDE.md.
- Amended-rubric drivers: scientist review note 2026-05-25 ~12:50
  local (this plan's preamble).
- Pre-run-1 Camera baseline numbers: CLAUDE.md "Stage A Results"
  table — ACEVision ~3.5 m linear-probe / ~4.0 m kNN; DPVOMotion
  presumed comparable, never directly published in the docs that
  exist on this branch.

## What to report back

In `handoff/results/RESULT_03_camera-encoder-audit-webots.md`:

1. **Step 0a/0b outcomes** — files restored, branch (P / Q),
   `dpvo` import status.
2. **Step 0c addendum** — Umeyama re-alignment of RESULT_02 IMU
   numbers + whether the audit label changes; addendum written
   into RESULT_02 itself.
3. **Step 1** — DPVO native benchmark number + ±% vs published.
4. **Step 2 headline:**

   | preprocessing | val MAE | test MAE | test-val gap % | per-path test p25/p50/p75/p90/max | per-traj smoothness median | latency b=1 ms |
   |---|---|---|---|---|---|---|
   | P-A (ImageNet → DPVO) | … | … | … | … | … | … |
   | P-B (DPVO only) | … | … | … | … | … | … |

5. **Step 3** — test-val gap interpretation + multi-condition gate
   pass/fail.
6. **Step 4** — capacity/config probe delta.
7. **Step 5** — 6-metric harness row(s).
8. **Step 6 audit label** + 3-sentence justification quoting numbers.
9. **PLAN_04 recommendation** — proceed to Odom audit OR insert a
   follow-up Camera iteration if Step 4 produces a clean `modify`
   direction worth committing to.
10. **One open question** for scientist.

## Reversibility

- Step 0a (file recovery): permanent; engineer commits.
- Step 0b (DPVO probe): throwaway.
- Step 0c (Umeyama retro): the addendum gets committed with RESULT_02
  edit; throwaway change to a saved JSON if any.
- Step 1 (DPVO native): throwaway (vendored repo, no edits).
- Step 2 (DPVOMotion on Webots): throwaway training; checkpoints
  saved under `runs/overnight/run2_iter_03/` (gitignored).
- Step 3 (transfer probe): same training, just two splits;
  throwaway.
- Step 4 (config probe): one extra training; throwaway.
- Step 5–6: documentation.

All artefacts under `runs/overnight/run2_iter_03/`, gitignored.
Files committed: restored files from Step 0a, RESULT_02 addendum
edit, RESULT_03, `src/pipeline/encoders/__init__.py` update.

**Demand #3** — `external/dpvo/` is not edited (the trunk is loaded
through `dpvo_motion.py`'s wrapper); any runtime shims live in our
scripts.

**Compute budget:** total iteration ≤ 90 min.
- Step 0: 15 min (recovery + DPVO probe + IMU Umeyama retro).
- Step 1: 10 min if DPVO runs out of the box, 0 if Branch Q.
- Step 2: 30 min (two preprocessing runs of head training; trunk
  frozen → fast).
- Step 3: 0 extra min (uses Step 2 outputs).
- Step 4: 15 min (one extra head training).
- Step 5: 5 min (eval-only).
- Step 6: 5 min writeup.

If overrun: cut Step 1 first (best-effort, not gating). Then
Step 4. Step 0c (Umeyama retro) is non-negotiable — it's the
correction-#3 compliance gate.

If Branch Q (DPVO not importable) and `DPVOMotionEncoder` itself
fails to load: write a partial RESULT after 30 min with the
blockage; next iteration is a "Camera env setup" data-acquisition
plan rather than the audit.
