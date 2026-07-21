# Plan 05 — C2 closure: canonical RoNIN unseen-subjects re-eval + RESULT_03 retros

> **Third-party review of RESULT_03 (2026-05-25 ~15:20 local) flagged
> three issues with the Camera audit verdict.** This plan addresses
> two of them in a tight Step 0 retro (same pattern as PLAN_03 Step 0c
> retrofitting RESULT_02 with Umeyama). The third issue — Camera
> needs an external-SOTA validation on a public benchmark — is queued
> as **PLAN_06** (deferred to its own iteration because it needs a
> data download + a non-trivial SOTA reproduction; bundling it here
> would violate the "one focused experiment per plan" rule).
>
> The focused experiment for PLAN_05 itself is **C2 closure** —
> the canonical RoNIN unseen-subjects benchmark per STATE.md's
> already-locked plan. Phase A is technically already closed
> (RESULT_04); PLAN_05 closes the only outstanding *paper-defensible*
> claim (C2) before Phase B fusion redesign begins.

## Hypothesis

C2 ("Our IMU encoder is competitive with RoNIN's ResNet1D on the
canonical RoNIN unseen-subjects benchmark") is the only Phase-A claim
that remains unverified — RESULT_02 used the a000 intra-session proxy.
Run-1 `docs/SOTA_BASELINES.md` reports **IMUCNN 14.41 m raw / 8.41 m
SVD-aligned ATE** vs **RoNIN ResNet1D 5.93 m raw** (paper 5.14 m) on
the canonical `list_test_unseen.txt` (32 sequences) — a ~2.4× raw gap.
Two outcomes:
- **C2 discharged** — IMUCNN within 20 % of ResNet1D on raw test ATE
  on canonical unseen-subjects. Original audit label `keep` is
  paper-defensible.
- **C2 NOT discharged** — gap > 20 %. Audit label moves to `modify`
  or `replace`, and Phase B must use RoNIN ResNet1D (unmodified per
  Demand #3) as the fusion IMU encoder, OR accept the gap with an
  explicit "in-domain only" claim in the paper.

The RESULT_03 retros (Step 0) are tight documentation/computation
fixes, not experiments — they reframe RESULT_03's verdict but do not
re-run iter-03.

## Steps

### Step 0 — RESULT_03 retros (3 issues, ~30 min total)

#### Step 0a — Difficulty-matched probe for Camera test/val (per review issue #1)

The review correctly flagged that "test 1.56 m < val 1.85 m → no
overfitting" is unsafe — test paths {15, 16, 17} and val paths
{2, 13, 14} are not difficulty-matched. The test-beats-val finding
likely reflects easier test paths, not transfer success.

Compute per-path **difficulty features** on all 6 paths from
`data/async_collection/path_*/ground_truth.csv`:
- **Path length** (cumulative Σ‖Δ(x, y)‖).
- **Mean speed** (path_length / wall_time).
- **Mean curvature proxy** (mean ‖Δθ / Δt‖, or 1 / radius-of-curvature
  estimate).
- **Frame count** (n_pairs at stride=5 for the camera).

Report a small table:

| path | length (m) | mean speed (m/s) | mean curvature (rad/s) | n_pairs | split | test/val MAE (P-A) |
|---|---|---|---|---|---|---|
| 2 | … | … | … | … | val | (from RESULT_03) |
| 13 | … | … | … | … | val | (from RESULT_03) |
| 14 | … | … | … | … | val | (from RESULT_03) |
| 15 | … | … | … | … | test | 1.07 |
| 16 | … | … | … | … | test | 1.85 |
| 17 | … | … | … | … | test | 1.99 |

Then compute the **difficulty-stratified gap**: for each path,
compute `MAE / path_length`, then aggregate per split. If
`test_MAE / test_length ≥ val_MAE / val_length` the test paths are
actually harder; if not, the original `keep` verdict is overstated.

Write this as a **RESULT_03 addendum** ("Addendum 2026-05-25 —
difficulty-matched probe") appended to
`handoff/results/RESULT_03_camera-encoder-audit-webots.md`. Update
the audit label if the difficulty-normalised gap exceeds 20 %.

**Reversibility:** documentation + small addendum; no encoder
re-training.

#### Step 0b — Smoothness debt reframe (per review issue #2)

The RESULT_03 verdict `keep` was paper-soft on the smoothness
weakness (median Pearson r ≈ 0.07 between ‖Δpred‖ and ‖Δgt‖).
Reframe the verdict label to **`keep with smoothness debt`** in the
same RESULT_03 addendum (no re-running needed). Add an explicit
**Phase B follow-up entry** to the addendum naming the candidate
fixes:

- (B-1) Auxiliary velocity loss on the camera head during fusion
  training.
- (B-2) EMA smoothing on per-instant camera tokens before they
  reach the fusion transformer.
- (B-3) Let the fusion transformer absorb noise via temporal
  cross-attention (current RESULT_03 default recommendation;
  engineer's open question Q in RESULT_03).

Phase B's bake-off iteration (PLAN_07 or wherever fusion starts)
**must** report per-modality smoothness in any 4-modality test run
so the debt is visible, not silent.

**Reversibility:** documentation only.

#### Step 0c — Queue PLAN_06 (per review issue #3)

The Camera audit ran with `lietorch`/`altcorr` unavailable (Branch Q),
so DPVO-as-SLAM was never reproduced end-to-end on a public
benchmark — the SOTA reproduction step never actually happened.
Webots-only audit cannot discharge per-leg validation for Camera.

PLAN_06 (this plan officially queues it) will:
1. Pick ONE public visual-odometry benchmark — preference order:
   **TartanAir** (DPVO's native training domain, shortest "should
   just work" path) → **EuRoC MAV** (standard real-data benchmark)
   → **KITTI odometry** (largest, slowest).
2. Pick ONE method to reproduce — preference order: **DPVO
   unmodified** (if `lietorch`/`altcorr` install can be unblocked
   on this Windows machine — likely needs a Linux container or a
   prebuilt wheel) → **TartanVO** (TartanAir's official VO
   baseline, MIT-licensed, pure-PyTorch — should run without CUDA
   ops) → **DROID-SLAM** (heavier, fallback).
3. Run our `DPVOMotionEncoder`'s feature trunk on the SAME public
   sequence (motion-only mode, no SLAM tracker) and report
   per-sample MAE relative to the SOTA pipeline's reported ATE on
   the same sequence.
4. Update RESULT_03's per-leg label using the public-benchmark
   evidence.

PLAN_06 is queued AFTER PLAN_05 (C2 first because RoNIN data is
already partly cached locally; TartanVO/TartanAir is a fresh
acquisition).

**No work for PLAN_05 itself on PLAN_06 contents** — engineer just
notes the queueing decision in RESULT_05.

### Step 1 — Acquire the canonical RoNIN unseen dataset

RESULT_02's Branch Y was used because only `data/ronin_a000_intra/`
exists locally — the full FRDR archive isn't on this machine. Two
acquisition options (try in order):

- **(1a) FRDR archive download**: https://doi.org/10.20383/102.0543.
  The RoNIN paper says "50 % of dataset released publicly." Check
  the FRDR page's download mechanism — direct HTTPS, or
  registration-gated? If direct, fetch with `curl` or `wget` into
  `data/_downloads/ronin_frdr/` (gitignored). If registration is
  required and not scriptable, **stop Step 1, document, and run
  Step 2 in Branch Y-only mode** (i.e., do not re-attempt the
  canonical eval; produce an addendum confirming nothing changed
  and queue C2 closure as a manual user task for Phase C).
- **(1b) Use a000 + any other subject already present**: re-check
  with `find` for any `aXYZ_N` directories or unpacked HDF5s on
  any drive (C:, D: if it exists, user's home, OneDrive paths).
  If a multi-subject pool is found, build a proxy "unseen subjects"
  split (train on all-but-one subjects, test on held-out subject).
  Better than Branch Y but still not canonical.

**Acceptance**: report (a) which acquisition path was taken, (b)
total disk used, (c) sequence count on `list_train.txt` and
`list_test_unseen.txt` *after* filtering for what's actually on
disk.

If neither path yields a usable canonical or near-canonical split:
**ship Step 2 as "Branch Y reaffirmation only" + write a SCIENTIST
NOTE flagging C2 as Phase C work**. Do not silently skip.

### Step 2 — Day-1 SOTA reproduction: RoNIN ResNet1D on canonical unseen

Same as PLAN_02 Step 1X but on canonical data. Use the restored
`scripts/eval_ronin_ate_fixed.py` if applicable; otherwise call
vendored `ronin_resnet.py` directly via:

```powershell
.venv\Scripts\python.exe `
  C:\Users\FabLab\AppData\Local\Temp\ronin\source\ronin_resnet.py `
  --mode test `
  --test_list C:\Users\FabLab\AppData\Local\Temp\ronin\lists\list_test_unseen.txt `
  --root_dir <ronin-root-dir-from-Step-1> `
  --out_dir runs\overnight\run2_iter_05\ronin_resnet_eval `
  --model_path <pretrained-or-trained-checkpoint.pt>
```

Prefer the pretrained checkpoint distributed with the FRDR archive
(if present); else train from scratch on the train list (~30 min on
P4000). Demand #3: vendored code unmodified; the
`np.int = int` shim sits in OUR wrapper, not in their source.

**Pre-test gate** (only if training): 5 epochs on first 20 sequences
of `list_train.txt`; train loss drops monotonically.

**Acceptance**: ResNet1D raw ATE within ±10 % of **5.93 m** (run-1
reference) OR ±20 % of **5.14 m** (paper). Aligned ATE with
**Umeyama** (scipy.spatial.procrustes or evo) — never the hand-rolled
SVD that RESULT_02 originally used. Report:
- raw ATE mean / median / p25 / p75 / p90 / max across the 32
  sequences.
- Umeyama-aligned ATE same.
- per-sequence ATE list saved to JSON.

### Step 3 — Reproduce IMUCNN on canonical unseen-subjects

Same as PLAN_02 Step 2X but on canonical data. Use restored
`scripts/eval_ronin_ate_fixed.py` (it already does
forward-Euler velocity integration + per-sequence ATE).
**Replace the hand-rolled SVD `_ate_aligned` with the Umeyama
implementation written in PLAN_03 Step 0c** (already in the
codebase).

**Pre-test gate**: same as Step 2.

**Memory budget**: trivial (IMUCNN < 1 MB peak).

**Acceptance**: report raw ATE + Umeyama-aligned ATE per-sequence.
Same statistics as Step 2.

### Step 4 — C2 audit decision (the focused experiment)

Compare Step 3 against Step 2 on canonical data with Umeyama
alignment:

- **Raw ATE gap**: `(IMUCNN_raw − ResNet1D_raw) / ResNet1D_raw`.
- **Umeyama-aligned ATE gap**: same with aligned numbers.
- **Raw weighted ≥ aligned**: if raw gap > 20 % and Umeyama gap
  ≤ 20 %, the verdict is **C2 NOT discharged**; raw wins.
- **C2 discharged** iff raw gap ≤ 20 %.

Write the verdict label and add it to the Phase A summary table in
RESULT_04. If C2 NOT discharged:
- Update RESULT_02's audit label from `keep` to `keep (in-domain
  only) / queue Phase B IMU replacement` and add an addendum.
- Recommend Phase B uses RoNIN ResNet1D (vendored, unmodified) for
  the IMU branch.
- Note the implication for PerCom paper claim: replace "competitive
  with RoNIN" with "competitive with RoNIN in-domain (single
  subject); cross-subject gap noted."

If C2 discharged: keep IMUCNN as the fusion encoder; RESULT_02
verdict stands without modification.

### Step 5 — Phase A close-out + Phase B prep brief

Write a 1-page "Phase A close-out" in RESULT_05's TL;DR:
- 4 encoder verdicts (with paper-defensible status per claim C1/C2/C3).
- 3 cross-cutting weaknesses + their queued Phase B follow-ups
  (smoothness debt, cross-session WiFi, C2 status).
- Open question for the scientist on Phase B architecture choice
  (transformer / TCN / LSTM-attn / late+gate per the SCIENTIST_BRIEF
  roadmap).

This gives the scientist a clean handoff to design PLAN_06 (Camera
SOTA validation) and PLAN_07 (Phase B bake-off start).

## Sources

- RoNIN paper: Herath, Yan, Furukawa, ICRA 2020.
  https://arxiv.org/abs/1905.12853.
- RoNIN repo (vendored): `C:\Users\FabLab\AppData\Local\Temp\ronin\`.
- RoNIN dataset (FRDR): https://doi.org/10.20383/102.0543.
- run-1 baselines: `docs/SOTA_BASELINES.md` IMU section.
- Amended audit rubric: STATE.md "Amended audit rubric (locked
  2026-05-25 ~12:55 local)".
- C2 placement decision: STATE.md "C2 placement decision (locked
  2026-05-25 ~12:55 local)".
- TartanVO (reference for queued PLAN_06):
  https://github.com/castacks/tartanvo (MIT, pure PyTorch).
- DROID-SLAM (reference for queued PLAN_06):
  https://github.com/princeton-vl/DROID-SLAM (BSD).

## What to report back

In `handoff/results/RESULT_05_c2-closure-ronin-canonical.md`:

1. **Step 0a addendum** — per-path difficulty table + stratified
   gap + whether RESULT_03's `keep` label survives difficulty
   normalisation. Addendum appended to RESULT_03.
2. **Step 0b addendum** — `keep` → `keep with smoothness debt`
   re-label + Phase B follow-up entries (B-1, B-2, B-3).
3. **Step 0c** — confirmation that PLAN_06 (Camera external SOTA) is
   queued; 3-line description of what PLAN_06 will do.
4. **Step 1** — acquisition path taken; sequence count; disk used.
5. **Step 2** — RoNIN ResNet1D canonical numbers (raw + Umeyama
   aligned) + per-sequence distribution. Match against run-1 ref +
   paper.
6. **Step 3** — IMUCNN canonical numbers (raw + Umeyama aligned) +
   per-sequence distribution.
7. **Step 4** — C2 audit decision: **discharged** or **NOT
   discharged**, raw-weighted under correction #3, with explicit
   gap percentages.
8. **Step 5** — updated Phase A close-out table (extending
   RESULT_04's table with canonical C2 column populated).
9. **Open question for scientist** on Phase B architecture (kick off
   PLAN_07 design).

## Reversibility

- Steps 0a–0c: documentation only; addenda appended to existing
  RESULTs.
- Step 1: data downloaded under `data/_downloads/` (gitignored).
- Step 2: vendored repo run, no edits.
- Step 3: existing restored script run; if Umeyama helper isn't
  already in the script (RESULT_03 said it was added), the
  modification is a 10-line patch to OUR wrapper (Demand #3
  untouched).
- Step 4–5: documentation.

Files committed: RESULT_05, addenda to RESULT_02 (if C2 fails) +
RESULT_03 (Step 0 retros), any small wrapper patches.

**Demand #3** — RoNIN vendored repo unmodified. Compat shims live
in our scripts.

**Compute budget:** target ≤ 90 min.
- Step 0a (difficulty probe): 15 min (numpy + writeup).
- Step 0b (smoothness reframe): 5 min (documentation).
- Step 0c (PLAN_06 queue note): 5 min.
- Step 1 (data acquisition): 0–30 min depending on FRDR
  registration. If > 30 min, ship Branch-Y-reaffirmation and
  queue C2 to Phase C with an explicit STATE.md update.
- Step 2 (ResNet1D canonical): 10 min if pretrained, 30 min if
  retrain.
- Step 3 (IMUCNN canonical): 15 min training + eval.
- Step 4 (audit decision): 5 min.
- Step 5 (Phase A close-out): 10 min writeup.

If overrun: keep Steps 0a, 0b, 0c — they're cheap and high-value.
Then prioritise Step 4 (audit decision) using whatever Step 2/3
data was acquired.

If Step 1 fails entirely (FRDR registration-gated, no other
subjects on disk): write a SCIENTIST NOTE relabeling C2 as a
**deferred Phase C task** and proceed to PLAN_06 (Camera external
SOTA) as PLAN_06. STATE.md gets updated accordingly.
