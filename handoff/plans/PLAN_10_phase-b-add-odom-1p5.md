# Plan 10 — Phase B: add Odom 1.5-modality (OdomCNN-P-B + raw integrated `(odom_x, odom_y)`) on Webots

## Hypothesis

RESULT_09 cleared C3 lower bound at 3-modality
(WiFi+IMU+Camera test 0.489 m ≤ 0.50 gate) but surfaced a sharp
diagnostic: **WiFi+Camera (no IMU) at K=1 ties full 3-modality on
val and beats it by 1.6 % on test**. IMU's marginal contribution at
K=1 is consistent with RESULT_03's smoothness debt (and RESULT_05's
per-trajectory smoothness r ≈ 0.03 finding which persists in
3-modality fusion). **Odom is the next modality on the C3 critical
path AND the only one with a clean smoothness signal:** RESULT_04's
trivial cumulative integration has Pearson r = 0.999 between
‖Δpred‖ and ‖Δgt‖, while OdomCNN-P-B has the anchoring win
(test MAE 4.24 m vs trivial 8.27 m on standalone, 49 % better).

RESULT_04's recommended fusion path is **(iii)** — the
"1.5-modality" approach: feed BOTH the OdomCNN-P-B embedding (for
absolute MAE / anchoring contribution) AND the raw integrated
`(odom_x, odom_y)` (for the smoothness property). This plan tests
that recommendation directly.

Expected outcomes:
- **4-modality clears C3 with margin** (test MAE meaningfully under
  0.50 m, ideally < 0.45 m) — Odom adds value via smoothness;
  PLAN_11 (bake-off) probes whether other fusion architectures
  exploit the 1.5-modality split better.
- **4-modality regresses or ties 3-modality** — at K=1 with WiFi as
  the anchor, additional modalities saturate; this is informative
  for the bake-off (suggests late+gate might be the winning arch
  because it can route around saturated modalities). Still ship
  4-modality as the headline C3 number with the finding noted.
- **Smoothness improves** (median r > 0.20 from RESULT_09's 0.03)
  — the raw integrated odom column would be doing what RESULT_04
  predicted. If smoothness stays ~0, the raw column isn't being
  attended to (modality_dropout + per-instant noise in fusion
  attention) and PLAN_11 should probe an explicit gating mechanism.

This is the focused experiment that closes the 4-modality story
for C3. One plan.

## Steps

### Step 0 — Config + 1.5-modality wiring (10 min)

`configs/stage_c/fusion.yaml` modality list extends to 4 entries.
RESULT_04's "1.5-modality" framing means we need TWO separate
encoder paths from the same odometry CSV:

- **odom (embedding)**: `OdomCNN` with P-B Δ-features preprocessing
  (the audit-winner config from RESULT_04). Already on this branch
  via `src/pipeline/encoders/odom.py`.
- **odom_raw (raw integrated column)**: a tiny "identity-style"
  feature path — the raw `(odom_x, odom_y)` columns from
  `odometry.csv`, projected to the model's embedding dim via a
  2-layer MLP (input 2 → 128). This gives the fusion attention a
  direct line to the smooth-but-drifty trivial integration signal.

Two clean implementation options — pick (A) for minimum surface
area:

- **(A) Single modality key `odom`, two-feature input.** The Odom
  dataloader emits a per-window dict `{"emb": (B, W=16, 7), "raw":
  (B, 2)}`. The encoder is a small wrapper `OdomDualEncoder` that
  applies OdomCNN to `emb` and the 2 → 128 MLP to `raw`, then
  concatenates or sums the two 128-d outputs. From the fusion
  transformer's perspective, it's still one modality, one
  per-instant token. **Smallest config / dataloader change.**
- **(B) Two modality keys `odom_emb` and `odom_raw`.** Each appears
  as its own per-instant token to the fusion transformer.
  Requires extending the modality registry, dataloader keys, and
  fusion config — bigger surface change for the same downstream
  signal.

**Pick (A)** unless engineer judges (B) substantially cleaner. RESULT
documents the choice. Wire via:

```yaml
modalities: [wifi, imu, camera, odom]   # add odom; dual-feature inside
```

`src/pipeline/encoders/odom.py` already implements P-B. Engineer
either (i) adds an `OdomDualEncoder` subclass that wraps OdomCNN +
the 2 → 128 MLP, or (ii) adds the raw-column projection inline in
the existing `OdomCNN`. Smallest path is (i); engineer commits
either.

If `configs/data/simulation.yaml` doesn't have an `odom_window:`
or related field, restore from run-1 the same way prior plans have
restored. The CSV column header is already known per RESULT_04
Step 0: `sim_time, odom_x, odom_y, odom_theta_deg, odom_linear_vel,
odom_angular_vel, wheel_left_vel, wheel_right_vel`.

**Acceptance**: builder constructs **4 encoders**; smoke forward
produces a per-instant 128-d Odom token without NaN.

### Step 1 — Pre-test gate (5 epochs, 10 % train)

Same pattern as PLAN_09 Step 1. Acceptance: val MAE drops ≥ 10 %
across 5 epochs.

**Memory budget**: forward+backward at B=32, 4 modalities, K=1.
Peak GPU MB; < 6 GB. The added Odom path is ~30 k params (negligible);
overall peak should be ~470 MB like RESULT_09's 3-modality.

If pre-test fails (NaN, divergence, no descent), STOP and write
partial RESULT — do not promote.

### Step 2 — Full 4-modality training

Same protocol as PLAN_09:
- 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5).
- modality_dropout 0.4, instant_dropout 0.45 (config defaults).
- K=1.

**Acceptance**: training completes; val + test MAE reported with
per-path distribution (criterion (d)).

### Step 3 — Compare to RESULT_09 (3-modality) baseline

| config | val MAE | test MAE | Δ vs 3-mod val | Δ vs 3-mod test | params | latency |
|---|---|---|---|---|---|---|
| WiFi+IMU+Camera K=1 (RESULT_09) | 0.448 | 0.489 | — | — | 1.53 M | 0.053 ms |
| **WiFi+IMU+Camera+Odom K=1 (this iter)** | **?** | **?** | ? | ? | ? | ? |

**Acceptance** (raw-weighted):
- Test MAE < 0.45 m → 4-modality is the headline C3 number with
  margin; PLAN_11 bake-off probes alternatives.
- 0.45 m ≤ test MAE ≤ 0.50 m → 4-modality clears C3 lower bound
  but at saturation; PLAN_11 explores whether late+gate exploits
  the 1.5-modality split better.
- Test MAE > 0.50 m → 4-modality regressed; analyse via Step 4
  subset eval before deciding next iteration.

### Step 4 — Per-modality subset eval (10 subsets)

Same `evaluate_all_subsets` machinery. Key rows to surface in the
RESULT:

| subset | val MAE | test MAE | interpretation hook |
|---|---|---|---|
| only:wifi | … | … | (compare to RESULT_09's 0.456 / 0.486) |
| only:imu | … | … | redundancy from RESULT_09 should persist |
| only:camera | … | … | (compare to RESULT_09's 1.741 / 1.887) |
| only:odom | … | … | **new** — is OdomCNN-1.5 informative standalone? |
| wifi+camera | … | … | RESULT_09 winner: 0.449 / 0.481 |
| wifi+odom | … | … | **new** — does Odom replace IMU's role? |
| wifi+imu+camera | … | … | RESULT_09 reproduction (sanity check) |
| wifi+camera+odom | … | … | **new** — is IMU still redundant when Odom enters? |
| **wifi+imu+camera+odom (full)** | … | … | — |

Two diagnostic questions:
1. **Does Odom-1.5 supersede IMU** (same as Camera did in RESULT_09)?
   If `wifi+camera+odom` (no IMU) ≥ full 4-modality, IMU drops from
   "redundant" to "remove-able" — late+gate becomes attractive in
   the bake-off.
2. **Does the raw-integrated path contribute the predicted
   smoothness?** Compare Step 5 per-trajectory r across configs.

### Step 5 — Per-trajectory smoothness ratio (gate per RESULT_05 lock)

Compute median Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖ across
test paths 15/16/17. Compare to RESULT_09's 0.029.

**Hypothesis-success criterion**: median r > 0.20 → the
1.5-modality raw odom column is doing what RESULT_04 predicted.
If r stays near 0, the raw column isn't being attended to;
flag explicitly for PLAN_11 bake-off (late+gate may help).

Save per-trajectory plots under
`runs/overnight/run2_iter_10/test_paths/`.

### Step 6 — Decision + PLAN_11 recommendation

Three-sentence verdict:
- Does 4-modality clear C3 with margin (test ≤ 0.45 m)?
- Does the 1.5-modality split deliver the smoothness improvement
  (r > 0.20)?
- PLAN_11 = architecture bake-off (set-transformer vs TCN vs
  LSTM-attn vs late+gate) on 10 % Webots subset, OR — if
  RESULT_10 surfaces a clean late+gate hypothesis (IMU consistently
  redundant, raw-odom-path under-attended) — go directly to a
  late+gate prototype as PLAN_11 with the other 3 candidates as
  the *fallback bake-off* if late+gate doesn't win cleanly.

## Sources

- RESULT_04: OdomCNN-P-B val 4.62 / test 4.24 m; trivial integration
  floor val 12.17 / test 8.27 m with per-trajectory r = 0.999.
- RESULT_04 open question Q: explicit (iii) recommendation = feed
  both OdomCNN embedding + raw integrated `(odom_x, odom_y)`.
- RESULT_06: WiFi+IMU K=1 val 0.469 / test 0.517 m.
- RESULT_09: WiFi+IMU+Camera K=1 val 0.448 / test 0.489 m; IMU
  marginally redundant; smoothness r = 0.029 persists.
- RESULT_05 locked gate: Phase B bake-off MUST report per-modality
  per-trajectory smoothness in every 4-modality run.
- `src/pipeline/encoders/odom.py` — OdomCNN class, untouched since
  the public restructure (P-B is a preprocessing choice, not an
  encoder change).
- CLAUDE.md "Universal token: encoder_embedding + modality_embedding
  + time_encoding(Δt)" — handles the WiFi 1 Hz / IMU 31 Hz / Camera
  5 Hz / Odom 15 Hz rate mismatch via the per-modality time encoding.

## What to report back

In `handoff/results/RESULT_10_phase-b-add-odom-1p5.md`:

1. **Step 0** — `OdomDualEncoder` (or inline) implementation choice;
   config diff.
2. **Step 1** — pre-test gate + memory budget peak.
3. **Step 2** — val + test MAE, best epoch, params, latency,
   per-path distribution.
4. **Step 3** — table vs RESULT_09; verdict on Odom contribution.
5. **Step 4** — full subset-eval table (10 rows + full-fusion);
   answer the two diagnostic questions.
6. **Step 5** — per-trajectory r median + plots; does the raw-odom
   path deliver smoothness as predicted?
7. **Step 6** — decision + PLAN_11 recommendation (bake-off or
   direct-to-late+gate).
8. **One open question** for scientist.

## Reversibility

- Step 0: permanent (config + small encoder wrapper). Reversible.
- Step 2: throwaway checkpoint under `runs/overnight/run2_iter_10/`.
- Steps 3–6: documentation.

Files committed: RESULT_10; config change to
`configs/stage_c/fusion.yaml`; `OdomDualEncoder` wrapper (small,
~30 lines) or equivalent inline change to `odom.py`.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 60 min.
- Step 0: 10 min.
- Step 1: 5 min.
- Step 2: 25 min (90-epoch training; DPVO trunk frozen, only heads
  + transformer train; close to RESULT_09's 248 s).
- Step 3: 5 min.
- Step 4: 5 min.
- Step 5: 5 min.
- Step 6: 5 min writeup.

If overrun: cut Step 4 to a 4-row subset (`only:wifi`,
`wifi+camera`, `wifi+camera+odom`, full) instead of all 10 — the
"does Odom replace IMU?" question is the most load-bearing.
