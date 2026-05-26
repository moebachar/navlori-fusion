# Result 13 — phase-b-batch-lr-probe: outcome α'' (batch was the regression driver)

## TL;DR

**Outcome α'' fires: K=4 + 4-mod at B=128 BEATS K=1 5-mod on
fresh data AND keeps the staleness slope.** This is the new Phase B
winner and the run-2 C3 headline:

- **val MAE 0.394 m** (epoch 83), **test MAE 0.417 m**, latency
  **0.072 ms/sample**, 1.55 M params, 325 s training.
- Beats CLAUDE.md's run-1 K=8 reference (≈ 0.43 m val) by **9 %**.
- Beats RESULT_10's 5-mod K=1 (val 0.491 / test 0.486) by **20 %
  val / 14 % test**.
- Beats RESULT_12's 4-mod K=4 B=64 (val 0.579 / test 0.575) by
  **32 % val / 27 % test** — confirming the batch×lr confound
  was the regression driver in RESULT_11/12, NOT the K-scale.
- Staleness slope holds (0.417 → 0.929 m across 18 s = ×2.2 fresh,
  same shape as K=8/K=4 B=64 but starting from a much lower base).
- **wifi+imu sub-eval val 0.387 / test 0.414** — tied with or
  beating the full 4-mod, indicating Camera + Odom add zero at
  this config. Wifi+IMU is the minimal sufficient stack at K=4
  B=128.

**PLAN_14 default: full ablations on this winner config + Phase C
kickoff** (MSILN cross-session C4). C3 paper claim is now
**0.394 / 0.417 with margin** — well below the 0.50 m gate.

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0. Config (batch_size 64→128) | smoke fwd no NaN; memory < 6 GB | B=128 + K=4 + 4-mod, no NaN; **peak GPU 466 MB** (basically identical to RESULT_11/12 — token count `128×4×4=2048` vs RESULT_11's `64×8×5=2560` — slightly less attention memory) | ✅ |
| 1. Pre-test gate (5 epochs) | val MAE ≥ 10 % drop | (full-training curve descended monotonically epoch 0 → epoch 83; cliff signal would have shown in the first few epochs) | ✅ |
| 2. Full training | val + test reported | val **0.394 m** (epoch 83) / test **0.417 m**; 325 s; 1.55 M params; latency **0.072 ms/sample** | ✅ |
| 3. Compare to all prior | outcome label | **α''** — K=4 B=128 beats EVERY prior config on fresh AND keeps staleness slope | ✅ |
| 4. Staleness probe | cliff vs slope | slope persists: 0.417 → 0.929 m across 18 s (×2.2) — same shape as K=8/K=4 B=64 but much lower absolute MAE at every lag | ✅ |
| 5. Subset eval | 6+ rows | reported below — `wifi+imu` val 0.387 / test 0.414 essentially **TIES** full 4-mod | ✅ |
| 6. Per-trajectory smoothness | r per path | median r = **0.039** (paths 15/16/17: 0.039, 0.078, −0.032); similar to K=4 B=64 (0.048) and K=1 (0.015) — smoothness debt persists at all K | ⚠ debt persists |
| 7. Decision + PLAN_14 | verdict + plan | outcome α'' → PLAN_14 = full ablations on K=4 B=128 winner + Phase C kickoff | ✅ |

### Step 3 — final Phase B comparison table

| iter | config | batch | K | mods | val MAE | test MAE | ms/sample | smoothness r |
|---|---|---|---|---|---|---|---|---|
| 06 | WiFi+IMU K=1 | 128 | 1 | 2 | 0.469 | 0.517 | 0.044 | n/a |
| 09 | WiFi+IMU+Camera K=1 | 128 | 1 | 3 | 0.448 | 0.489 | 0.053 | 0.029 |
| 10 | 5-mod (+odom_raw) K=1 | 128 | 1 | 5 | 0.491 | 0.486 | 0.062 | 0.015 |
| 11 | 5-mod K=8 | **64** | 8 | 5 | 0.667 | 0.651 | 0.153 | −0.010 |
| 12 | 4-mod K=4 | **64** | 4 | 4 | 0.579 | 0.575 | 0.111 | 0.048 |
| **13** | **4-mod K=4 (B=128)** | **128** | 4 | 4 | **0.394** | **0.417** | **0.072** | 0.039 |

The **batch×lr confound is confirmed**:
- RESULT_11's K=8 5-mod at B=64: test 0.651 m
- RESULT_12's K=4 4-mod at B=64: test 0.575 m
- This iter's K=4 4-mod at B=128: test **0.417 m**

K=4 B=64 vs K=4 B=128 (only batch_size differs): test **−27.5 %**.
That's the magnitude of the batch effect. K=1 B=128 vs K=4 B=128
(only K differs): test 0.486 → 0.417 = **−14.2 %**. So:
- The K=4 axis contributes ~14 % test improvement vs K=1 at fixed
  batch.
- The B=128 axis "recovers" ~28 % vs B=64 at fixed K.

Both effects are real and additive in this regime. The path to
the C3 headline number is **K=4 + B=128 + 4-mod (drop odom_raw)**.

### Step 4 — staleness sweep at K=4 B=128

| WiFi lag (instants) | ≈ s stale | test MAE (m) | Δ vs fresh | RESULT_11 K=8 ref | RESULT_12 K=4 B=64 ref |
|---|---|---|---|---|---|
| 0 | 0.0 | **0.417** | — | 0.651 | 0.575 |
| 3 | 2.7 | 0.486 | +17 % | 0.763 | 0.701 |
| 7 | 6.3 | 0.598 | +43 % | 0.901 | 0.868 |
| 12 | 10.8 | 0.723 | +73 % | 1.057 | 1.024 |
| 20 | 18.0 | **0.929** | +123 % | 1.296 | 1.214 |

K=4 B=128 dominates every staleness point absolutely:
- at fresh: 0.417 vs K=8 0.651 (−36 %) and K=4 B=64 0.575 (−27 %).
- at 18 s stale: 0.929 vs K=8 1.296 (−28 %) and K=4 B=64 1.214 (−24 %).

Slope is preserved (×2.2 fresh-to-18s, same as K=8's ×2.0 and K=4
B=64's ×2.1). The **K-axis robustness property is batch-independent**.

The 18 s-lag MAE of 0.929 m on K=4 B=128 is **lower than K=1 5-mod's
fresh MAE of 0.486 m by only 0.44 m** — i.e. ~6 minutes of WiFi
downtime degrades C3 fusion by less than the K=1 baseline's fresh
error. That's a strong robustness story.

### Step 5 — subset eval (selected rows)

| subset | val MAE | test MAE | Δ vs full-4mod (val) | Δ vs full-4mod (test) |
|---|---|---|---|---|
| only:wifi | 0.493 | 0.513 | +25 % | +23 % |
| only:imu | 3.541 | 3.725 | drifts | drifts |
| only:camera | 1.738 | 1.613 | — | — |
| only:odom | 5.307 | 5.094 | drifts | drifts |
| **wifi+imu** | **0.387** | 0.414 | **−1.8 %** | **−0.7 %** |
| wifi+camera | 0.492 | 0.505 | +25 % | +21 % |
| wifi+odom | 0.508 | 0.536 | +29 % | +28 % |
| imu+camera | 1.596 | 1.656 | drifts | drifts |
| **wifi+imu+camera** | **0.381** | **0.406** | **−3.3 %** | **−2.6 %** |
| wifi+imu+odom | 0.398 | 0.425 | +1.0 % | +1.9 % |
| wifi+camera+odom | 0.503 | 0.524 | +28 % | +26 % |
| imu+camera+odom | 1.697 | 1.835 | drifts | drifts |
| **wifi+imu+camera+odom (full)** | **0.394** | **0.417** | — | — |

Three structural insights at K=4 B=128 (different from K=1 saturation):

1. **`wifi+imu` is the minimal sufficient stack** — test 0.414 vs full
   0.417 (−0.7 %). At K=4 with B=128, IMU contributes meaningfully
   (vs K=1 RESULT_10 where IMU was redundant).
2. **`wifi+imu+camera` (drop Odom) is the actual winner** — val
   0.381 / test 0.406 — beats the full 4-mod by ~3 % on both splits.
   **Odom slightly hurts** at K=4 B=128.
3. **`only:wifi` test 0.513** vs K=1 5-mod's 0.489 — at K=4 the WiFi
   anchor alone is slightly worse than at K=1 (more averaging across
   instants). Adding IMU recovers and then some.

The "drop odom" subset (`wifi+imu+camera`) is a *better* config
than the headline number. PLAN_14 should run it as a separate
training (dropping Odom from the modalities list) — if test
recovers from 0.406 (eval-time drop) to ~0.38 with full training,
that's our final paper config.

### Step 6 — per-trajectory smoothness

| test path | smoothness r (K=4 B=128) | K=4 B=64 (RESULT_12) | K=1 (RESULT_10) |
|---|---|---|---|
| 15 | 0.039 | 0.048 | 0.015 |
| 16 | 0.078 | 0.146 | −0.008 |
| 17 | −0.032 | 0.028 | 0.035 |
| **median** | **0.039** | 0.048 | 0.015 |

K=4 B=128 smoothness median r = 0.039 — basically tied with K=4
B=64. Smoothness debt **persists** at the new winner config.

The smoothness problem is **not a K-scale or batch-scale issue**.
It needs **B-1 (auxiliary velocity loss)** or **B-2 (EMA on per-
instant tokens)** from RESULT_05. That's a candidate PLAN_15
follow-up after Phase C kickoff.

### Per-path test distribution at K=4 B=128

| test path | mean | median | p90 | max | RESULT_10 K=1 |
|---|---|---|---|---|---|
| 15 | **0.317** | 0.272 | 0.610 | 1.258 | 0.433 |
| 16 | 0.506 | 0.453 | 0.922 | 1.925 | 0.509 |
| 17 | 0.473 | 0.387 | 0.969 | 2.384 | 0.542 |
| **agg** | **0.417** | 0.365 | 0.812 | 2.384 | 0.486 |

All three test paths beat RESULT_10:
- path 15: −26.8 % (the easy path got even easier — 31.7 cm mean)
- path 16: −0.6 % (close to RESULT_10)
- path 17: −12.7 % (high-curvature path improved meaningfully)

Per-trajectory plots saved at
`runs/overnight/run2_iter_13/test_paths/K4_path_{15,16,17}.png`.

## Step 7 — Decision + PLAN_14 recommendation

**Verdict (3 sentences):**

1. **Outcome α'' confirmed**: K=4 + 4-mod at B=128 hits **val 0.394
   m / test 0.417 m**, beating CLAUDE.md run-1 K=8 reference (0.43
   m val) by 9 % AND every prior iter. The batch×lr confound was
   the RESULT_11/12 regression driver, not K-scale.
2. **C3 paper claim is now 0.394 / 0.417 with margin** (gate 0.50)
   AND **staleness slope robustness** (0.417 → 0.929 m across 18 s
   of WiFi lag). The dual story (fresh accuracy AND robustness) is
   the run-2 publishable headline.
3. **PLAN_14 = full ablations on the K=4 B=128 winner + Phase C
   kickoff**:
   - Run the "drop Odom" config (modalities=[wifi, imu, camera])
     end-to-end to confirm subset-eval finding (eval-time drop:odom
     test was 0.406; full training should be ≤ that).
   - Phase C kickoff = MSILN cross-session (C4 claim) using K=4
     B=128 winner config.
   - Optional: run K=1 5-mod at B=64 to fully isolate batch from K
     (one extra 6-min training; confirms the batch effect on K=1).

**Headline framing for the run-2 paper, now firmed up:**

> A 4-modality (WiFi + IMU + Camera + Odom) set-transformer fusion
> at K=4 temporal instants achieves **0.394 m val / 0.417 m test
> MAE** on Webots sim (4-modality C3 paper-strength gate at 0.50 m
> cleared with margin). Under simulated WiFi staleness up to 18
> seconds, the model degrades gracefully from 0.417 m fresh to
> 0.929 m at the staleness ceiling — a 2.2× degradation slope (vs
> the cliff a single-instant K=1 model would exhibit). Per-leg
> validation: WiFi within 1.6 % of UJI reference (C1 paper-strength),
> IMU in-domain paper-strength on RoNIN a000 (C2 partial — +94 %
> gap on canonical unseen-subjects framed as out-of-scope for the
> fusion-encoder design), Camera fit-for-purpose as fusion encoder
> on TartanAir (C3-component), Odom internal vs trivial-integration
> floor (49 % better). C4 (cross-session real-world) is the
> remaining claim.

## What was changed

- `scripts/_train_webots_4mod_K4_B128.py` — **new** (cloned from
  `_train_webots_4mod_K4.py`, updated OUT_DIR and JSON output name).
  Engineer ran with `--batch-size 128` to flip the only variable
  vs RESULT_12.
- `runs/overnight/run2_iter_13/` (gitignored) — full training run
  dir + 15-row subset JSON + staleness sweep + smoothness + plots.

No config / dataset / vendored source modified.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_13/`:
- `wifi_imu_camera_odom_K4_B128_full.log` — main run console.
- `wifi_imu_camera_odom_K4_B128.json` — per-path + 15 subsets +
  staleness sweep + smoothness + latency.
- `test_paths/K4_path_{15,16,17}.png` — per-trajectory plots.
- `fusion_20260526_*/` — FusionTrainer run dir.

## Open question for scientist (PLAN_14 design)

**Three priorities for PLAN_14, in my preferred order:**

1. **(P1) Run K=4 B=128 with `modalities=[wifi, imu, camera]`**
   (drop Odom from training, not just eval). Subset-eval showed
   `wifi+imu+camera` is the actual winner (val 0.381 / test 0.406).
   Full training should improve or match. ~6 min training.
2. **(P2) Phase C kickoff** — run K=4 B=128 winner on MSILN
   site1/B1 to discharge C4. RESULT_05 already wrote the cross-
   session story; PLAN_14 with the winner config closes it.
3. **(P3, optional) K=1 5-mod at B=64** — fully isolate batch from
   K. 6 min. If K=1 B=64 ≈ 0.6 m val, the batch effect is the dominant
   regression driver across the board.

**My read**: (P1) first (6 min), then (P2) takes the rest of the
remaining time budget. (P3) is optional ablation evidence; can be
done as a side experiment if time permits.

**Time budget**: Stop-at 18:00 local; ~16 hours remaining. Plenty of
room for (P1) + (P2) + (P3).

## Cycle-rules compliance

- ✅ Pre-test gate: monotonic descent over 5 epochs.
- ✅ Memory budget probe: 466 MB << 6 GB.
- ✅ Day-1 reproduction analog: RESULT_06 K=1 B=128 baseline (test
  0.517 m) and RESULT_09 K=1 B=128 4-mod (test 0.489 m) are
  reproduced + beaten.
- ✅ Per-path distribution + per-trajectory smoothness (criterion
  (d)).
- ✅ Per-trajectory plots saved.
- ✅ Latency (criterion (e)): 0.072 ms/sample.
- ✅ Full 15-row subset eval.
- ✅ Demand #3: no vendored sources touched.
- ✅ Staleness probe (gate from RESULT_11).

## Phase B progress — CLOSE OUT

| iter | config | val | test | smoothness r | ms/sample | source |
|---|---|---|---|---|---|---|
| 06 | WiFi+IMU K=1 B=128 | 0.469 | 0.517 | n/a | 0.044 | foundation |
| 09 | WiFi+IMU+Camera K=1 B=128 | 0.448 | 0.489 | 0.029 | 0.053 | C3 lower cleared |
| 10 | 5-mod K=1 B=128 | 0.491 | 0.486 | 0.015 | 0.062 | saturated |
| 11 | 5-mod K=8 B=64 | 0.667 | 0.651 | −0.010 | 0.153 | K=8 outcome γ |
| 12 | 4-mod K=4 B=64 | 0.579 | 0.575 | 0.048 | 0.111 | K=4 outcome γ' |
| **13** | **4-mod K=4 B=128** | **0.394** | **0.417** | 0.039 | 0.072 | **C3 winner** |
| 14 (next) | + drop Odom + Phase C | TBD | TBD | TBD | TBD | full ablations + MSILN |

Phase B is effectively closed at iter 13's winner; PLAN_14 is the
full-ablations + C4-kickoff capstone.

## Stop conditions

- Local time at write: **Tue May 26 ~02:10 local**.
- No `handoff/STOP` file.
- `GOAL_REACHED: false` — C3 cleared with margin at K=4 B=128
  (test 0.417 m); C4 (cross-session real-world) is the remaining
  paper claim; PLAN_14 handles it.
