# Plan 11 — Phase B: K>1 temporal fusion on the 4-modality stack

> RESULT_10 surfaced a sharp diagnostic: **K=1 fusion saturates at
> the WiFi anchor.** `only:wifi` test 0.489 m ≈ full 5-modality test
> 0.486 m. Adding modalities at K=1 contributes ≤ 1 %. The right
> architectural lever is NOT a late+gate bake-off (engineer's
> RESULT_10 reasoning: "late+gate at K=1 won't help when the fusion
> is already WiFi-dominated and the other modalities have no temporal
> axis on which to be useful"), it is **K>1 temporal fusion** — give
> motion modalities multiple instants to integrate against WiFi
> anchors. Run-1's archived experience (per CLAUDE.md "honest
> findings"): naïve K>1 overfit to 0.69 m; per-instant dropout
> recovered ≈ 0.44 m **and** unlocked graceful degradation under
> stale WiFi (cliff → slope). This iteration replicates that and
> probes whether K>1 + 4 modalities surfaces the contribution the
> motion encoders have been waiting to deliver.

## Hypothesis

At K=8 (the run-1 documented config), with per-instant dropout
preserved at the run-1 audit-fix value (`instant_dropout=0.45`,
already in `configs/stage_c/fusion.yaml`), the 4-modality stack
(WiFi + IMU + Camera + Odom + Odom_raw) does ONE of:

- **(α) Beats K=1 on fresh WiFi data** (test MAE drops below 0.486 m)
  AND demonstrates graceful staleness — Camera/IMU/Odom finally
  contribute meaningfully because they have a temporal axis to
  encode motion against. Headline: temporal fusion is the
  architectural lever; PLAN_12 closes Phase B with full training +
  ablations.
- **(β) Ties K=1 on fresh, but shows graceful staleness** — fresh-
  data accuracy is already at the WiFi ceiling; the value of
  temporal + 4-modality is robustness, not fresh accuracy. Headline:
  the run-2 paper claim shifts from "fresh accuracy" to
  "robustness under modality dropout/staleness." Still
  publishable, but a different framing.
- **(γ) Regresses or fails to recover from naïve K>1**
  (per-instant dropout doesn't help as run-1 reported, or modality
  contributions don't surface). Triggers a focused architecture
  probe (PLAN_11b: cross-attention readout variation, or explicit
  late+gate).

This is the architectural pivot iteration — one focused experiment
on the K dimension.

## Steps

### Step 0 — Config: K=8 with per-instant dropout (5 min)

`configs/stage_c/fusion.yaml` should already have `temporal.K: 1`
as the default; change to `temporal.K: 8` for this iteration.
Verify `temporal.instant_dropout: 0.45` is set (CLAUDE.md cites
this as the run-1 audit-fix value).

Verify the FusionTransformer + FusionTrainer machinery accepts
K=8 — `temporal_index` / `K` parameters per CLAUDE.md should be
plumbed through `builder.load_config` → `build_datamodule` → `build_model`.

**Acceptance**: a `_smoke_fusion.py --phase 1` (shape check) with
K=8 produces a forward pass without NaN; per-instant masks
attached.

### Step 1 — Pre-test gate (5 epochs, 10 % train)

Same pattern as PLAN_09/10. Acceptance: val MAE drops ≥ 10 %
across 5 epochs.

**Memory budget**: K=8 means 8× the per-sample modality tokens
in the transformer's attention. RESULT_10's K=1 peak was 466 MB;
K=8 expected ~3–4 GB depending on transformer config. Peak GPU MB
< 6 GB required.

If pre-test FAILS (NaN, divergence, no descent) OR memory blows
the 6 GB budget, write a partial RESULT documenting the failure
mode and STOP — do not promote to full training. Backup plan: K=4
instead of K=8 to halve memory; if K=4 also fails, K=2 is the
floor.

### Step 2 — Full 4-modality K=8 training

Same protocol as PLAN_09/10 (AdamW + OneCycleLR + Huber(δ=0.5),
90 epochs, patience 25, modality_dropout 0.4, instant_dropout 0.45).

**Acceptance**: training completes; val + test MAE reported with
per-path distribution (criterion (d)).

### Step 3 — Compare to RESULT_09 and RESULT_10 K=1 baselines

| config | val MAE | test MAE | params | latency (ms/sample) | source |
|---|---|---|---|---|---|
| WiFi+IMU K=1 | 0.469 | 0.517 | 1.38 M | 0.044 | RESULT_06 |
| WiFi+IMU+Camera K=1 | 0.448 | 0.489 | 1.53 M | 0.053 | RESULT_09 |
| 5-mod (Odom-1.5) K=1 | 0.491 | 0.486 | 1.56 M | 0.062 | RESULT_10 |
| **5-mod K=8 (this iter)** | **?** | **?** | ? | ? | this iter |

CLAUDE.md run-1 reference: K=8 single-instant ≈ 0.43 m val MAE
(but that was 2-modality WiFi+IMU per run-1). Expectation: K=8
on 4-modality should land at or under 0.45 m val. Test MAE bar:
beat **0.486 m** (RESULT_10's K=1 test). Latency expected ~3–5
ms/sample (CLAUDE.md says run-1 K=8 was 4.2 ms) — still well
under the 100 ms gate.

**Acceptance** (raw-weighted):
- Outcome (α): K=8 beats K=1 test by ≥ 5 % AND staleness probe
  (Step 4) shows graceful degradation.
- Outcome (β): K=8 ties K=1 on fresh test but staleness probe
  shows graceful degradation — paper-framing pivot.
- Outcome (γ): K=8 regresses (test > 0.50 m) or NaN — write
  diagnostic + scientist plans PLAN_11b.

### Step 4 — Staleness probe (the K>1 differentiator)

This is the criterion (d)-style "robustness under modality
dropout/staleness" measurement. Two variants — run BOTH:

**Step 4a — WiFi staleness sweep.** At eval time, simulate WiFi
staleness by replacing the per-instant WiFi token with the
*previous* WiFi token at various lags (0 = fresh, 5 = ~5 s
stale, 15 = ~15 s stale, 30 = ~30 s stale). Plot test MAE vs WiFi
lag. Expected pattern per CLAUDE.md run-1: K=1 shows a cliff (MAE
jumps to several m once WiFi is stale); K>1 shows a slope (gradual
degradation).

For comparison, ALSO run this same staleness sweep on the K=1
checkpoint from RESULT_10 (use the cached weights if still on
disk). If RESULT_10's checkpoint isn't saved, just compare K=8
staleness against the K=8 fresh number.

**Step 4b — Per-modality dropout robustness at eval.** Re-run the
subset eval at K=8 to see if motion modalities finally contribute.
Key rows: `only:wifi` (should match RESULT_10's 0.489 if WiFi
anchor still dominates fresh), `drop:wifi` (should now degrade
gracefully because IMU+Camera+Odom have a temporal axis to
dead-reckon), `wifi+imu+camera+odom+odom_raw` full (the
headline).

### Step 5 — Per-trajectory smoothness (gate from RESULT_05 lock)

Compute median per-trajectory Pearson r between ‖Δpredᵢ‖ and
‖Δgtᵢ‖ across test paths 15/16/17. RESULT_09 r=0.029, RESULT_10
r=0.015. **Expectation**: K=8 with motion modalities finally
having a temporal axis should produce r > 0.20. If r stays near
0, the temporal fusion isn't smoothing predictions — flag for
PLAN_11b architecture probe.

Save per-trajectory plots under `runs/overnight/run2_iter_11/test_paths/`.

### Step 6 — Decision + PLAN_12 recommendation

Three-sentence verdict:
- Which outcome (α/β/γ) landed; quote test MAE + staleness slope.
- Did smoothness recover (r > 0.20)? If yes, K>1 is the architectural
  lever; if no, name the next probe.
- PLAN_12 default = Phase B winner full-training + per-modality
  ablations (the original Phase B PLAN_12 slot). Update accordingly:
  - Outcome (α): PLAN_12 = K=8 full ablations + Phase C kickoff
    (cross-session MSILN at PLAN_13).
  - Outcome (β): PLAN_12 = staleness ablations + paper-framing
    decision.
  - Outcome (γ): PLAN_12 = architecture probe (cross-attention
    readout variation, OR late+gate at K=8).

## Sources

- CLAUDE.md "Honest findings" (run-1):
  - Single-instant fusion ≈ 0.43 m; only-WiFi ≈ 0.46 m.
  - Naïve K>1 regressed to 0.69 m (overfit); per-instant dropout
    recovered ≈ 0.44 m AND unlocked graceful staleness.
- RESULT_06: WiFi+IMU K=1 baseline.
- RESULT_09: WiFi+IMU+Camera K=1, C3 lower bound cleared.
- RESULT_10: 5-mod K=1 saturated at 0.486 m test, smoothness r=0.015.
- `configs/stage_c/fusion.yaml`: temporal config keys (`temporal.K`,
  `temporal.instant_dropout`); restored in RESULT_06.
- `src/pipeline/fusion/transformer.py`,
  `src/pipeline/training/fusion_trainer.py`: K plumbing; restored.

## What to report back

In `handoff/results/RESULT_11_phase-b-k-gt-1-temporal.md`:

1. **Step 0** — K, instant_dropout, modality_dropout values used;
   smoke pass.
2. **Step 1** — pre-test gate; memory budget peak (K=8 is the
   memory test).
3. **Step 2** — val + test MAE, best epoch, params, latency,
   per-path distribution. Per-trajectory plots for test paths
   15/16/17.
4. **Step 3** — comparison table vs RESULT_06/09/10; outcome
   label (α/β/γ).
5. **Step 4a** — WiFi-staleness sweep table + plot (test MAE vs
   lag). The shape (cliff vs slope) is the headline.
6. **Step 4b** — K=8 subset eval (focus rows: `only:wifi`,
   `drop:wifi`, full-fusion).
7. **Step 5** — per-trajectory smoothness median r; comparison
   to RESULT_09/10.
8. **Step 6** — verdict + PLAN_12 recommendation.
9. **One open question** for scientist.

## Reversibility

- Step 0 (config edit): permanent. Engineer commits.
- Step 2 (training): throwaway checkpoint under
  `runs/overnight/run2_iter_11/` (gitignored).
- Steps 3–6: documentation.

Files committed: RESULT_11, config change to fusion.yaml
(`temporal.K` from 1 → 8), small wrapper script if added.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 75 min.
- Step 0: 5 min.
- Step 1: 5 min (pre-test).
- Step 2: 35 min (90 epochs at K=8; ~6× the K=1 time per CLAUDE.md
  "K=8 was 4.2 ms latency vs K=1's 0.044 ms" — training scales
  similarly).
- Step 3: 5 min.
- Step 4a: 10 min (5 lags × ~1 min eval each).
- Step 4b: 5 min.
- Step 5: 5 min.
- Step 6: 5 min writeup.

If overrun: cut Step 4a's lag count from 5 to 3 (0, 5, 15) to
save 2 lag-evals; keep Step 5 (smoothness gate is locked).

If memory blows the 6 GB budget at K=8: drop to K=4 and document
the cap; the experiment still answers the qualitative K>1 question
(temporal axis enables motion contribution / staleness graceful
degradation).
