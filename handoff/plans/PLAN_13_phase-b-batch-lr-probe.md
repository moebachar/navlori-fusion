# Plan 13 — Phase B: isolate batch×lr confound (the γ' diagnostic)

> RESULT_11 (K=8 5-mod) and RESULT_12 (K=4 4-mod) both regressed
> +18–34 % vs RESULT_10's K=1 5-mod test of 0.486 m. RESULT_12 fired
> outcome γ' — **K-scale is not the bottleneck**. The two non-K
> changes that ALSO happened starting at RESULT_11 were:
>
> 1. **Batch size halved** (128 → 64) to make K=8 memory-safe.
> 2. **OneCycleLR lr config left at its K=1 default** (max_lr=1.3e-3
>    per fusion.yaml; should rescale with batch under most lr-finder
>    heuristics).
>
> These two are coupled at K=4 too because the engineer kept B=64
> for apples-to-apples comparison with RESULT_11. RESULT_06's K=1
> baseline (test 0.517 m, 2-mod) used B=128 + lr=1.3e-3 and
> reproduced the run-1 number cleanly. Hypothesis: **the batch/lr
> regime — not K — is what broke fresh-data accuracy in RESULT_11/12.**

## Hypothesis

If the batch×lr confound is the regression driver, then K=4 + 4-mod
**at B=128 (restoring RESULT_06's default)** should recover fresh
accuracy comparable to RESULT_09's 3-mod K=1 0.489 m. The staleness
slope and motion-modality contribution discovered at K>1 should
also persist — those are K-axis findings, not batch-axis findings.

Three outcomes:
- **(α'') B=128 + K=4 + 4-mod beats K=1 on fresh AND keeps slope**:
  the Phase B winner is established. PLAN_14 = full ablations +
  Phase C kickoff.
- **(β'') B=128 + K=4 + 4-mod ties or near-ties K=1 (test ≤ 0.50 m)
  + keeps slope**: C3 cleared at K>1; robustness is the paper
  differentiator. PLAN_14 = full robustness ablations + paper-
  framing.
- **(γ'') B=128 + K=4 + 4-mod still regresses (test > 0.50 m)**: it's
  neither K nor batch — there's a deeper issue (maybe the per-instant
  cross-attention readout at K>1 is structurally weaker). PLAN_14 =
  architecture probe (vary readout / attention head count) OR pivot
  to robustness-only framing and ship the β'-equivalent result with
  appropriate paper caveats.

One focused experiment, one variable (batch+lr). PLAN_12's other
levers (K=4, 4-mod) stay constant.

## Steps

### Step 0 — Config (5 min)

In the engineer's `_train_webots_4mod_K4.py` (or equivalent), set:
- **`batch_size: 128`** (restore RESULT_06 default; was 64 in
  RESULT_11/12).
- **`temporal.K: 4`** (unchanged from PLAN_12).
- **`modalities: [wifi, imu, camera, odom]`** (4-mod, drop
  `odom_raw` — same as PLAN_12).
- **lr: keep at 1.3e-3** (RESULT_06 / fusion.yaml default — the
  point of the probe is to isolate the batch effect first;
  rescaling lr is a SECOND probe if needed).

**Memory budget probe**: a forward+backward at B=128, K=4, 4-mod
must fit in 6 GB. RESULT_11's peak at B=64 K=8 5-mod was 471 MB; at
B=128 K=4 4-mod the token count is `128 × 4 × 4 = 2048` per attention
matrix dim, slightly less than RESULT_11's `64 × 8 × 5 = 2560`. So
peak should stay under 500 MB. Confirm explicitly.

If B=128 OOMs unexpectedly, drop to B=96 and document — but B=128
is highly likely to fit.

**Acceptance**: smoke fwd no NaN; B=128 confirmed memory-safe.

### Step 1 — Pre-test gate (5 epochs, 10 % train)

Same pattern. Val MAE drops ≥ 10 % across 5 epochs.

**If the pre-test gate FAILS at B=128 + K=4 + 4-mod**, that's a
strong γ'' signal already (the regression isn't batch-driven). Write
a partial RESULT after Step 1 and stop — don't waste a 90-epoch run.

### Step 2 — Full training (B=128 + K=4 + 4-mod)

Same protocol: 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5), lr=1.3e-3,
`instant_dropout=0.45`, `modality_dropout=0.4`.

**Acceptance**: training completes; val + test + per-path
distribution reported.

### Step 3 — Comparison table

| config | batch | K | mods | val MAE | test MAE | source |
|---|---|---|---|---|---|---|
| WiFi+IMU K=1 | 128 | 1 | 2 | 0.469 | 0.517 | RESULT_06 |
| WiFi+IMU+Camera K=1 | 128 | 1 | 3 | 0.448 | 0.489 | RESULT_09 |
| 5-mod K=1 | 128 | 1 | 5 | 0.491 | 0.486 | RESULT_10 |
| 5-mod K=8 | **64** | 8 | 5 | 0.667 | 0.651 | RESULT_11 |
| 4-mod K=4 (B=64) | **64** | 4 | 4 | 0.579 | 0.575 | RESULT_12 |
| **4-mod K=4 (B=128, this iter)** | **128** | 4 | 4 | **?** | **?** | this iter |

The diagnostic is the 4-mod K=4 row going from B=64 to B=128. If
B=128 recovers ≥ 70 % of the K=12-to-K=1 gap (i.e. test ≤ 0.51 m,
down from 0.575), the batch confound is confirmed.

**Acceptance**: outcome label (α''/β''/γ'') + verdict on whether
fresh accuracy at K=4 is recoverable.

### Step 4 — Staleness probe (gate from PLAN_11/12)

Same 4-lag sweep (0, 3, 10, 20 instants ≈ 0–18 s). The staleness
slope shape should be PRESERVED at B=128 (a K-axis property, not
a batch-axis property). If the slope ALSO degrades at B=128, that's
informative for diagnosis.

### Step 5 — Subset eval (5 informative rows)

| subset | val MAE | test MAE | comment |
|---|---|---|---|
| only:wifi | … | … | should be ≈ 0.49 if WiFi anchoring intact |
| wifi+imu+camera | … | … | RESULT_11 K=8 reference: 0.594 test |
| only:camera | … | … | does Camera still contribute at K=4? |
| wifi+camera+odom | … | … | the "skip IMU" trick from RESULT_10 |
| **full 4-mod (this iter)** | … | … | — |

### Step 6 — Per-trajectory smoothness (gate per RESULT_05 lock)

Median Pearson r across paths 15/16/17. RESULT_12 r=0.048. K=4 +
B=128 doesn't obviously change smoothness one way or the other —
this is a sanity report.

### Step 7 — Decision + PLAN_14 recommendation

Three-sentence verdict:
- Outcome (α''/β''/γ''); quote test MAE.
- Batch confound confirmed or rejected; quote the diff vs RESULT_12.
- PLAN_14 recommendation:
  - (α''): full ablations + Phase C kickoff (MSILN cross-session
    at PLAN_15).
  - (β''): robustness ablations + paper framing pivot.
  - (γ''): readout / attention architecture probe, OR ship the
    robustness story (slope) as the headline with the fresh-
    accuracy gap caveated.

## Sources

- RESULT_06: K=1 B=128 baseline reproduced (val 0.469 / test 0.517,
  WiFi+IMU 2-mod). **The batch/lr config that worked.**
- RESULT_10: K=1 B=128 5-mod saturated (val 0.491 / test 0.486).
- RESULT_11: K=8 B=64 5-mod outcome γ (test 0.651, +33.9 %).
  **Batch dropped here.**
- RESULT_12: K=4 B=64 4-mod outcome γ' (test 0.575, +18 %).
  **Batch stayed at 64; K halved didn't recover.**
- `configs/stage_c/fusion.yaml`: default lr=1.3e-3, default
  batch_size from fusion.yaml (engineer confirms in Step 0).
- OneCycleLR + AdamW lr-scaling heuristic: the standard practice is
  `lr ∝ √batch` (square-root rule) for SGD-like optimizers, often
  `lr ∝ batch` (linear rule) for very small batches; AdamW is in
  between. RESULT_06's combo (B=128, lr=1.3e-3) was already
  hyperparameter-tuned (CLAUDE.md "Optuna budget != production
  budget" caveat); the K>1 iters reused that lr at half the batch.

## What to report back

In `handoff/results/RESULT_13_phase-b-batch-lr-probe.md`:

1. **Step 0** — config diff (just batch_size); memory probe.
2. **Step 1** — pre-test gate (cliff signal if fails).
3. **Step 2** — val + test MAE, best epoch, params, latency,
   per-path distribution.
4. **Step 3** — comparison table; outcome label.
5. **Step 4** — staleness sweep (does the slope persist at B=128?).
6. **Step 5** — subset eval.
7. **Step 6** — smoothness median r.
8. **Step 7** — verdict + PLAN_14 recommendation.
9. **One open question** for scientist.

## Reversibility

- Step 0 (config): permanent. Reversible.
- Step 2: throwaway checkpoint.
- Steps 3–7: documentation.

Files committed: RESULT_13, config change.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 45 min.
- Step 0: 5 min.
- Step 1: 5 min.
- Step 2: 20 min (K=4 B=128 is ~half RESULT_12's wall time at B=64
  since same gradient steps but 2× the batch throughput).
- Step 3: 5 min.
- Step 4: 5 min.
- Step 5: 3 min.
- Step 6: 2 min.
- Step 7: 5 min.

If overrun: cut Step 4 to 2 lags (0 + 18 s only) — the cliff-vs-slope
shape is the headline.

If outcome (γ'') fires (batch isn't the bottleneck either),
PLAN_14 becomes an architecture probe — DON'T scope-creep PLAN_13
to include a second probe. Engineer flags clearly in RESULT TL;DR.
