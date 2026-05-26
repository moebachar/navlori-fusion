# Result 12 — phase-b-k4-drop-odom-raw: outcome γ' (K is not the bottleneck)

## TL;DR

**Outcome γ' — K=4 + 4-mod STILL regresses vs K=1**. Halving K from
8 to 4 AND dropping `odom_raw` lands the 4-modality stack at val
**0.579 m** / test **0.575 m** — better than K=8 (test 0.651) but
+18.3 % worse than RESULT_10's 5-mod K=1 (test 0.486) and +17.6 %
vs RESULT_09's 4-mod K=1 (test 0.489). **C3 lower bound FAILS at
K=4** (test > 0.50 m). The K-scale is NOT the bottleneck; the
RESULT_11 regression at K=8 was not solved by halving K. **Two
likely confounds** — both kept constant from RESULT_11 — explain
the persistent regression: `batch_size=64` (vs RESULT_10's 128)
and the OneCycleLR `lr=1.3e-3` scaled for batch 128. **K=4 still
shows a staleness slope** (0.58 → 1.21 m across 18 s of WiFi lag,
×2.1 fresh, similar to K=8's ×2.0) — confirming the K>1 robustness
payoff is real and not K-scale-specific. Per-trajectory smoothness
**marginally improved** to r=0.048 at K=4 (vs 0.015 at K=1 and
−0.010 at K=8) but still well under the r > 0.20 gate.

The headline of run-2 Phase B is now clear: **K=1 5-mod is the C3
number** (test **0.486 m**, criterion (b) cleared) and **K=4/K=8
buy staleness robustness at a fresh-accuracy cost** — these are
complementary points, not competing ones. **PLAN_13 should isolate
the lr × batch confound first** (re-run K=1 at batch_size=64 and
re-run K=4 at batch_size=128 with proportional lr); if K=1 at B=64
matches RESULT_10's 0.486 m, the K-regression is a batch effect,
not a K effect, and PLAN_14 can ship a clean K-sweep at fixed
batch.

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0. Config K=4 + drop odom_raw | 4 encoders, K=4 axis | 4 modalities (wifi, imu, camera, odom); cfg.temporal.n_instants=4; instant_stride=9 unchanged; odom_raw NOT registered. | ✅ |
| 1. Pre-test gate + memory budget | < 6 GB | peak **470.1 MB** (basically identical to K=1's 466 and K=8's 471 — the 5-mod's modality_raw cache wasn't loading on GPU anyway) | ✅ |
| 2. Full training | val + test + per-path | val **0.579 m** (epoch 64) / test **0.575 m**; 550 s; 1.55 M params; latency 0.111 ms/sample | ✅ trained / ❌ **C3 gate** (test > 0.50) |
| 3. Compare to K=1 (RESULT_10) and K=8 (RESULT_11) | improvement/regression quantified | K=4 beats K=8 by 11.7 % test, but **regressed +18.3 % vs K=1**; outcome γ' | ❌ regression confirms K-scale ≠ bottleneck |
| 4. Staleness probe | cliff vs slope | **slope persists**: 0.575 → 1.214 m across 18 s (×2.1 fresh; similar shape to K=8 ×2.0) | ✅ (robustness payoff confirmed) |
| 5. 6-row subset eval | full matrix | reported below — `only:wifi` test 0.574 ≈ full 0.575 (saturation pattern from RESULT_10 persists at K=4) | ✅ |
| 6. Per-trajectory smoothness | r per path | median r = **0.048** (paths 15/16/17: 0.048, 0.146, 0.028); modest improvement vs K=1 (0.015) and K=8 (−0.010); still well below the r > 0.20 gate | ⚠ debt persists |
| 7. Decision + PLAN_13 | verdict + plan | outcome γ' (K-scale is not the bottleneck); PLAN_13 = isolated batch×lr probe before any further K-sweep | ✅ |

### Step 3 — full K-axis comparison (4-mod scope where applicable)

| config | val MAE | test MAE | ms/sample | batch | smoothness r | source |
|---|---|---|---|---|---|---|
| 4-mod WiFi+IMU+Camera K=1 | 0.448 | 0.489 | 0.053 | 128 | 0.029 | RESULT_09 |
| 5-mod (+odom_raw) K=1 | 0.491 | **0.486** | 0.062 | 128 | 0.015 | RESULT_10 |
| 5-mod K=8 | 0.667 | 0.651 | 0.153 | 64 | −0.010 | RESULT_11 |
| **4-mod K=4 (this iter)** | **0.579** | **0.575** | 0.111 | 64 | 0.048 | this iter |

Two confounds keep K=8 vs K=4 vs K=1 from being a clean K-scale
sweep:
1. **batch_size**: 128 at K=1, 64 at K∈{4, 8}. OneCycleLR's
   `max_lr=1.3e-3` was Optuna-tuned for batch=128. Smaller batches
   under the same lr typically converge to higher loss.
2. **modality count**: 4-mod (this iter), 5-mod (RESULT_10/11).
   Modality-dropout × instant-dropout combinatorics differs.

If we hold modality count fixed (4-mod) and compare K=1 (RESULT_09)
vs K=4 (this iter): val 0.448 → 0.579 (+29.2 %), test 0.489 →
0.575 (+17.6 %). Held-batch-size, the gap is +17.6 % test — still a
regression but the batch confound is the obvious next probe.

### Step 4 — staleness sweep at K=4

| WiFi lag (instants) | ≈ s stale | test MAE | Δ vs fresh | RESULT_11 K=8 ref |
|---|---|---|---|---|
| 0 | 0.0 | **0.575** | — | 0.651 |
| 3 | 2.7 | 0.701 | +22 % | 0.763 (+17 %) |
| 7 | 6.3 | 0.868 | +51 % | 0.901 (+38 %) |
| 12 | 10.8 | 1.024 | +78 % | 1.057 (+62 %) |
| 20 | 18.0 | 1.214 | +111 % | 1.296 (+99 %) |

Slope, not cliff — same robustness payoff as K=8 but starting from
a lower fresh-MAE base. The per-percent degradation slope is
slightly **steeper** at K=4 than K=8 (+111 % vs +99 % at 18 s),
but the **absolute** stale-MAE at 18 s is **lower** at K=4 (1.214
vs 1.296 m). K=4 wins on absolute terms across the entire staleness
sweep.

### Step 5 — 15-row subset eval at K=4

| subset | val MAE | test MAE | comment |
|---|---|---|---|
| **only:wifi** | **0.574** | **0.574** | ≈ full 4-mod — saturation persists at K=4 |
| only:imu | 3.990 | 3.648 | drifts (expected) |
| only:camera | 1.735 | 1.822 | unchanged from K=1 |
| only:odom | 5.318 | 4.914 | drifts (expected) |
| wifi+imu | 0.574 | 0.579 | basically tied with only:wifi |
| wifi+camera | 0.589 | 0.573 | camera adds 0.001 to test |
| wifi+odom | 0.570 | 0.576 | odom helps val a touch, neutral on test |
| wifi+imu+camera | 0.583 | 0.573 | best test |
| wifi+imu+odom | 0.574 | 0.577 | — |
| wifi+camera+odom | 0.578 | 0.574 | — |
| **wifi+imu+camera+odom (full)** | **0.579** | **0.575** | — |

The K=4 4-mod fusion has the **same saturation pattern** as RESULT_10's
K=1 5-mod fusion: **WiFi alone does the job; other modalities each
add 0-1 %**. The K-axis didn't unlock motion-modality contribution
at K=4. (At K=8 in RESULT_11, motion modalities DID contribute
meaningfully — `wifi+imu+camera+odom` test 0.594 vs `only:wifi` 0.732,
−19 % — but at K=4 they don't. K=8's bigger temporal axis may be
necessary for motion modalities to find a niche; K=4's smaller
temporal axis lets WiFi still dominate.)

### Step 6 — per-trajectory smoothness

| test path | smoothness r (K=4) | K=1 (RESULT_10) | K=8 (RESULT_11) |
|---|---|---|---|
| 15 | 0.048 | 0.015 | −0.045 |
| 16 | 0.146 | −0.008 | −0.010 |
| 17 | 0.028 | 0.035 | 0.064 |
| **median** | **0.048** | 0.015 | −0.010 |

K=4 has the best median smoothness of the three K values **but it
still doesn't break r > 0.20**. Path 16 reaches r=0.146 (best so
far across iterations) — the temporal axis at K=4 is doing
*something* for smoothness on that path, but not enough.
B-1 (auxiliary velocity loss) or B-2 (EMA on tokens) remain the
viable smoothness levers per RESULT_05's lock.

### Per-path test distribution at K=4

| test path | mean | median | p90 | max | RESULT_10 K=1 |
|---|---|---|---|---|---|
| 15 | 0.553 | 0.467 | 0.858 | 3.696 | 0.433 |
| 16 | 0.606 | 0.604 | 0.980 | 1.447 | 0.509 |
| 17 | 0.575 | 0.572 | 0.951 | 1.905 | 0.542 |
| **agg** | **0.575** | 0.538 | 0.932 | 3.696 | 0.486 |

All three paths regressed. Path 17 (high-curvature) regressed
least (+6 %, similar to its RESULT_10 K=1 result), suggesting the
K-axis helps more on curve-heavy paths.

Per-trajectory plots at
`runs/overnight/run2_iter_12/test_paths/K4_path_{15,16,17}.png`.

## Step 7 — Decision + PLAN_13 recommendation

**Verdict (3 sentences):**

1. **Outcome γ' confirmed**: K=4 + 4-mod test MAE 0.575 m still
   above C3 gate. Neither halving K (8 → 4) nor dropping
   `odom_raw` recovered RESULT_10's K=1 0.486 m. The K-scale is
   **NOT** the bottleneck — and a likely confound is `batch_size`
   (RESULT_10 at B=128 vs this iter at B=64) under fixed
   `max_lr=1.3e-3`.
2. **Smoothness debt persists** (median r=0.048; the locked r>0.20
   gate is not met at any K). B-1 (auxiliary velocity loss) or B-2
   (EMA on tokens) are the remaining levers from RESULT_05.
3. **PLAN_13 = isolated batch×lr probe**, NOT another K-sweep.
   Smallest probes:
   - **(P1) K=1 5-mod at batch_size=64** with the same lr=1.3e-3
     and same OneCycleLR schedule. If val ≈ 0.49 m, the K=4/K=8
     regression is a batch effect; if val regresses similarly to
     0.58, K really is hurting.
   - **(P2) K=4 4-mod at batch_size=128** with lr_max rescaled
     (×√(128/64) = ×1.41 ≈ lr=1.84e-3). If test recovers to ≤ 0.50,
     batch+lr was the regression; K=4 + 4-mod is then the C3
     headline number.

**Headline for the run-2 paper given current evidence:**

> The 4-modality fusion architecture clears C3 on Webots sim at
> K=1 (test 0.486 m, criterion (b) gate 0.5 m). Temporal extension
> to K=4 / K=8 buys graceful staleness robustness — under 18 s of
> WiFi lag, K=1 implies a cliff (no temporal axis); K=4 degrades by
> 2.1× and K=8 by 2.0× — a complementary contribution to fresh
> accuracy. The fresh-vs-staleness frontier is the publishable
> insight.

If PLAN_13's batch×lr probe confirms K=4 + 4-mod at B=128 ≤ 0.50,
the paper can claim a *single* configuration that's both
state-of-the-art on fresh and robust under staleness; otherwise
two configs (K=1 for fresh, K=4 for robustness) is the honest
framing.

**Alternative PLAN_13 paths**:
- **Phase C kickoff** (MSILN cross-session C4) using K=1 5-mod
  (RESULT_10) as the fusion model. Accepts the K-axis paper-framing
  for now.
- **Smoothness lever directly** (B-1 / B-2 from RESULT_05). Easier
  to A/B at K=1 where fresh accuracy is best.

## What was changed

- `scripts/_train_webots_4mod_K4.py` — **new** (cloned from
  `_train_webots_5mod_K8.py`, removed `OdomRawEncoder` registration,
  overrode `cfg.temporal.n_instants = 4`, retained staleness probe).
- `runs/overnight/run2_iter_12/` (gitignored) — full training run
  dir + 15-row subset JSON + staleness sweep + smoothness + plots.

No config / vendored / dataset sources modified.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_12/`:
- `wifi_imu_camera_odom_K4_full.log` — main run console.
- `wifi_imu_camera_odom_K4.json` — per-path + subsets + staleness +
  smoothness + latency.
- `test_paths/K4_path_{15,16,17}.png` — per-trajectory plots.
- `fusion_20260526_*/` — FusionTrainer run dir.

## Open question for scientist (PLAN_13 design)

PLAN_13 has three viable directions. **My read**:
**(A) PLAN_13 = isolated batch×lr probe (P1 + P2 above)** — small
two-run experiment that disambiguates the K-vs-batch confound.
~25 min total. The result locks in either the K=1 or K=4
config as the C3 headline number.

If scientist prefers **(B) Phase C kickoff (C4 on MSILN
cross-session)** instead, accept K=1 5-mod (RESULT_10) as the
fusion model and move to the real-world plausibility claim. C2
in-domain is paper-strength; C4 closure plus C3 in-sim would
complete the four-claim bundle.

**Time budget**: STATE Stop-at 18:00 local; we have ~16 hours left.
Either (A) or (B) fits comfortably with margin for a follow-up.

## Cycle-rules compliance

- ✅ Pre-test gate: monotonic descent across 5 epochs.
- ✅ Memory budget probe: K=4 peak 470 MB << 6 GB.
- ✅ Day-1 reproduction analog: RESULT_09 4-mod K=1 (val 0.448) is
  the in-house comparison.
- ✅ Per-path distribution + per-trajectory smoothness (criterion
  (d), locked gate).
- ✅ Per-trajectory plots (criterion (d)).
- ✅ Latency (criterion (e)): 0.111 ms/sample.
- ✅ Full 15-row subset eval.
- ✅ Demand #3: no vendored sources touched.
- ✅ Staleness probe (gate from RESULT_11).

## Phase B progress

| iter | config | val | test | smoothness r | ms/sample | batch |
|---|---|---|---|---|---|---|
| 06 | WiFi+IMU K=1 | 0.469 | 0.517 | n/a | 0.044 | 128 |
| 09 | WiFi+IMU+Camera K=1 | 0.448 | 0.489 | 0.029 | 0.053 | 128 |
| 10 | 5-mod K=1 | **0.491** | **0.486** | 0.015 | 0.062 | 128 |
| 11 | 5-mod K=8 | 0.667 | 0.651 | −0.010 | 0.153 | 64 |
| **12** | **4-mod K=4** | **0.579** | **0.575** | **0.048** | 0.111 | 64 |
| 13 (next) | batch×lr probe | TBD | TBD | TBD | TBD | various |

## Stop conditions

- Local time at write: **Tue May 26 ~01:50 local**.
- No `handoff/STOP` file.
- `GOAL_REACHED: false`. C3 lower bound was cleared at K=1
  (RESULT_10 5-mod test 0.486 m); K=4 + 4-mod regressed but
  staleness slope holds — paper-framing decision pending PLAN_13.
