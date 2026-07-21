# Plan 12 — Phase B: K=4 + drop `odom_raw` (the K-sweet-spot + 4-mod probe)

> RESULT_11 fired outcome γ at K=8: fresh-accuracy regressed +33.9 %
> vs K=1 (test 0.651 m, fails C3 lower bound), but the staleness
> slope unlocked (0.65 → 1.30 m across 18 s; cliff → slope per
> CLAUDE.md run-1 prediction). Engineer's recommendation: PLAN_12 =
> K=4 (halve K) + drop `odom_raw` (RESULT_10 showed it's unattended;
> RESULT_11's 5-mod combinatorics with `modality_dropout=0.4` likely
> contributes to the regression). This iteration tests whether the
> K-axis has a sweet spot **and** whether the modality count was the
> regression driver. ONE focused experiment, two coupled changes.

## Hypothesis

CLAUDE.md cites run-1's K=8 baseline at ≈ 0.43 m val MAE on a
4-modality stack (WiFi + IMU + Camera + Odom). We reproduced that
4-modality K=1 baseline (RESULT_09) at 0.448 / 0.489 val/test. Then
added `odom_raw` (5-mod, K=1) and saw saturation (RESULT_10). Then
went to K=8 (5-mod) and regressed (RESULT_11).

**Two coupled changes** in PLAN_12:

1. **K=4** (vs K=8): halves the temporal token count; if the K=8
   regression is overshoot, K=4 lands in the sweet spot. K=4 also
   halves the modality_dropout combinatorics in the temporal
   direction.
2. **Drop `odom_raw`** (back to 4-modality WiFi + IMU + Camera +
   Odom): RESULT_10 showed `drop:odom_raw` is indistinguishable
   from full-fusion (raw column unattended). Dropping it removes
   an unproductive modality_dropout slot AND matches CLAUDE.md's
   documented 4-mod K=8 config.

Expected outcomes:
- **(α') K=4 + 4-mod beats K=1**: fresh test ≤ RESULT_10's 0.486 m
  AND staleness slope present. The K-axis has a sweet spot at K=4;
  PLAN_13 = full ablations at this config.
- **(β') K=4 + 4-mod ties K=1 + staleness slope**: temporal axis is
  doing its job (robustness, not fresh accuracy). Headline pivots
  to robustness; PLAN_13 = full staleness/modality-dropout
  ablations.
- **(γ') K=4 + 4-mod still regresses (test > 0.50 m)**: it's not the
  K-scale that's the bottleneck; it's the lr/batch/dropout regime.
  PLAN_13 = isolated hyperparameter probe (lr rescaling for halved
  batch, or instant_dropout sweep).

This is a focused-experiment probe of **two coupled levers (K and
modality count)** — they're coupled because both adjust the
effective combinatorics of modality_dropout × temporal dropout.
Decoupling them into two iterations would burn an extra iteration
to answer a smaller question.

## Steps

### Step 0 — Config (5 min)

`configs/stage_c/fusion.yaml`:
- `temporal.K`: 8 → **4** (and `instant_stride` per RESULT_11's
  `instant_stride=9` — keep as-is unless engineer judges otherwise).
- Re-check `temporal.instant_dropout: 0.45` and `modality_dropout:
  0.4` are unchanged (these were the audit-fix values; this
  iteration tests K, not dropout).

The `modalities` list in the wrapper (engineer's
`_train_webots_5mod_K8.py` or equivalent — restored / created in
PLAN_11): drop `odom_raw` so the list becomes `[wifi, imu, camera,
odom]`. This means the `OdomRawEncoder` slot isn't constructed by
the builder.

If the engineer's K=8 wrapper script needs to be cloned to a new
`_train_webots_4mod_K4.py`, do so (one new wrapper, ~40 lines).

**Acceptance**: smoke shows 4 encoders built (not 5); K=4 tokens
per instant axis in the forward pass.

### Step 1 — Pre-test gate (5 epochs, 10 % train)

Same pattern. Acceptance: val MAE drops ≥ 10 % across 5 epochs OR
clear descent.

**Memory budget** at B=64 (RESULT_11's batch size — keep so the
lr/batch comparison is apples-to-apples): expected peak ~300 MB
(K=4 is 1/2 of K=8's tokens, and 4-mod is 4/5 of 5-mod). Report.

### Step 2 — Full training

Same protocol: 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5),
B=64 (RESULT_11 setting), 4 modalities at K=4.

**Acceptance**: training completes; val + test MAE + per-path
distribution reported.

### Step 3 — Compare to K=1 (RESULT_10) and K=8 (RESULT_11) baselines

| config | val MAE | test MAE | latency (ms) | params | source |
|---|---|---|---|---|---|
| 5-mod K=1 | 0.491 | 0.486 | 0.062 | 1.56 M | RESULT_10 |
| 5-mod K=8 | 0.667 | 0.651 | 0.153 | 1.56 M | RESULT_11 |
| **4-mod K=4 (this iter)** | **?** | **?** | ? | ~1.5 M | this iter |
| WiFi+IMU+Camera K=1 (the 4-mod K=1 ref) | 0.448 | 0.489 | 0.053 | 1.53 M | RESULT_09 |

**Acceptance** (raw-weighted, criterion (b) ≤ 0.50 m on test):
- Outcome (α'): test < 0.486 m → K=4 + 4-mod is the Phase B winner;
  PLAN_13 = full ablations.
- Outcome (β'): 0.486 ≤ test ≤ 0.50 → C3 cleared, robustness via
  staleness slope is the differentiator; PLAN_13 = paper-framing +
  full robustness ablations.
- Outcome (γ'): test > 0.50 m → it's not K-scale; PLAN_13 =
  hyperparameter probe (lr × batch sweep; instant_dropout sweep).

### Step 4 — Staleness probe (the K>1 differentiator, gate from RESULT_11)

Same staleness sweep as RESULT_11 Step 4a: lags 0, 3, 10, 20
instants (≈ 0, 2.7, 9, 18 s). Plot test MAE vs lag.

RESULT_11 K=8 reference: 0.65 → 1.30 m across 18 s (2× degradation,
slope). RESULT_10 K=1 reference: implied cliff (no probe ran).

**Acceptance**: K=4's slope shape characterised. Two key questions:
- Is K=4's fresh-data MAE lower than K=8's 0.651?
- Does K=4 preserve the slope (no cliff)?

### Step 5 — Subset eval at K=4 (only the 6 most informative rows)

Run `evaluate_all_subsets` on the K=4 best-val checkpoint. Report:

| subset | val MAE | test MAE | comment |
|---|---|---|---|
| only:wifi | … | … | compare to RESULT_10's 0.489 |
| only:imu | … | … | does IMU contribute at K=4? |
| only:camera | … | … | does Camera contribute at K=4? |
| only:odom | … | … | does Odom contribute at K=4? |
| wifi+imu+camera (the 4-mod minus odom) | … | … | run-1's CLAUDE.md 0.43 ref config |
| **wifi+imu+camera+odom (full 4-mod K=4)** | … | … | — |

Three diagnostic questions:
1. At K=4 does any motion modality move the needle vs `only:wifi`?
2. At K=4 does the 4-mod stack beat 3-mod (with Odom)?
3. Is K=4's `only:wifi` test MAE materially worse than RESULT_10's
   0.489 (indicating the WiFi anchoring degrades when temporal
   tokens flood the attention)?

### Step 6 — Per-trajectory smoothness (gate per RESULT_05 lock)

Median Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖ across paths 15/16/17.
RESULT_11 r=−0.010; RESULT_10 r=0.015. Hypothesis: K=4's smoothness
may be slightly better than K=8's (the per-instant attention has
less to average over), but a meaningful improvement (r > 0.20) is
unlikely without an explicit smoothness loss.

If r ≤ 0.05, the smoothness debt is now a definite Phase B
follow-up (PLAN_13b candidate: auxiliary velocity loss).

Save per-trajectory plots under `runs/overnight/run2_iter_12/test_paths/`.

### Step 7 — Decision + PLAN_13 recommendation

Three-sentence verdict:
- Outcome (α' / β' / γ'); quote test MAE.
- Did smoothness improve at K=4? Quote median r.
- PLAN_13 recommendation, default = full Phase B winner ablations
  (the originally-named slot). Adjust based on outcome:
  - (α'): PLAN_13 = full ablations at K=4 + 4-mod, + Phase C
    kickoff (MSILN cross-session).
  - (β'): PLAN_13 = robustness ablations (staleness × modality_dropout
    matrix) + paper-framing pivot.
  - (γ'): PLAN_13 = hyperparameter probe (lr × batch sweep).

## Sources

- RESULT_11: K=8 5-mod regression, +33.9 % test vs K=1; staleness
  slope present.
- RESULT_10: 5-mod K=1 saturated; `drop:odom_raw` indistinguishable
  from full (the empirical justification for dropping it here).
- RESULT_09: 4-mod (WiFi+IMU+Camera) K=1 val 0.448 / test 0.489.
- CLAUDE.md run-1 reference: 4-mod K=8 ≈ 0.43 m val.
- `configs/stage_c/fusion.yaml` temporal block (restored in
  RESULT_06).
- `src/pipeline/fusion/transformer.py`,
  `src/pipeline/training/fusion_trainer.py` — K plumbing restored.

## What to report back

In `handoff/results/RESULT_12_phase-b-k4-drop-odom-raw.md`:

1. **Step 0** — config diff (K, modalities list).
2. **Step 1** — pre-test gate + memory peak.
3. **Step 2** — val + test MAE, best epoch, params, latency,
   per-path distribution.
4. **Step 3** — comparison table vs RESULT_10/11/09; outcome
   label (α' / β' / γ').
5. **Step 4** — staleness sweep table; cliff-vs-slope shape.
6. **Step 5** — 6-row subset eval.
7. **Step 6** — per-trajectory smoothness median r + plots.
8. **Step 7** — verdict + PLAN_13 recommendation.
9. **One open question** for scientist.

## Reversibility

- Step 0 (config edit): permanent. Reversible.
- Step 2: throwaway checkpoint.
- Steps 3–7: documentation.

Files committed: RESULT_12, config change, optional new wrapper
`scripts/_train_webots_4mod_K4.py`.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 50 min (K=4 is half of K=8's training time).
- Step 0: 5 min.
- Step 1: 5 min.
- Step 2: 18 min (half of RESULT_11's 717 s for K=8).
- Step 3: 5 min.
- Step 4: 8 min (4 lags × ~2 min).
- Step 5: 5 min.
- Step 6: 5 min.
- Step 7: 5 min writeup.

If overrun: cut Step 4 to 2 lags (0 + 18 s endpoints), enough to
confirm cliff-vs-slope; keep Step 6 (smoothness gate locked).

If outcome (γ') fires (test > 0.50 m), DON'T promote to PLAN_13
ablations — the regression is the priority; PLAN_13 becomes the
hyperparameter probe instead. Engineer flags this in the RESULT
TL;DR explicitly.
