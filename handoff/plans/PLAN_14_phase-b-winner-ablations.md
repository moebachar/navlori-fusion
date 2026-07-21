# Plan 14 — Phase B winner: full ablation suite on the K=4 B=128 4-mod config

> RESULT_13 fired outcome α'' — **C3 cleared with margin**. K=4 +
> 4-mod (WiFi+IMU+Camera+Odom) at B=128 trains to **val 0.394 /
> test 0.417 m**, beating CLAUDE.md's run-1 K=8 reference (0.43 m
> val) by 9 % AND preserving the staleness slope (0.417 →
> 0.929 m across 18 s). A surprise from RESULT_13's subset eval:
> **wifi+imu+camera (drop Odom)** lands at val 0.381 / test 0.406
> — slightly better than the full 4-mod. The Phase B winner is
> established; this iteration characterises it with the ablation
> set the PerCom paper will need.

## Hypothesis

The Phase B winner is the **K=4 + 4-mod + B=128** config. The
PerCom-paper-strength claim requires:

1. **Reproducibility**: the RESULT_13 number reproduces under a
   second training (paper-clean seed or same seed re-run).
2. **Full per-modality subset eval**: 15 non-empty 4-mod subsets +
   full, to verify every modality contributes (or to honestly
   document which ones don't — RESULT_13's "drop Odom is better"
   finding needs an addendum).
3. **Full staleness sweep**: a paper-figure showing the slope
   shape across multiple lags (criterion (b)/(d)/(c)-style
   robustness evidence).
4. **Per-trajectory smoothness**: top 5 longest test paths with
   plots (criterion (d) — explicitly required).
5. **Latency at multiple batch sizes**: criterion (e) gate at < 100 ms
   per sample on the Quadro P4000.

This is one focused experiment in the cycle-rule sense — the
ablation study of the chosen architecture. The trained model is
shared across Steps 2–6 (single training run, many evaluations).

## Steps

### Step 0 — Recover RESULT_13's checkpoint OR re-train (5–10 min)

Two paths — engineer picks whichever is faster:

- **(0A) Use RESULT_13's saved checkpoint** at
  `runs/overnight/run2_iter_13/<run-dir>/model.pt` if it exists.
  Skip Step 1 (re-training) entirely. Just verify the checkpoint
  loads and emits the RESULT_13 val/test numbers (sanity within
  ±0.005 m due to dropout-eval mode).
- **(0B) Re-train fresh** at K=4 + 4-mod + B=128 with a paper-
  clean fixed seed (`seed=42`). ~20 min wall on this hardware per
  RESULT_13's timing. Confirms reproducibility.

**Acceptance**: a working checkpoint at hand; one-line val MAE
sanity check matches RESULT_13's 0.394 m within ±0.01 m.

### Step 1 — Pre-test gate (only if Step 0B)

Same pattern as previous iterations. Skip if Step 0A used.

### Step 2 — Full subset eval (all 15 non-empty 4-mod subsets + full)

Run `FusionTrainer.evaluate_all_subsets`. The 4-mod stack has 2^4 −
1 = 15 non-empty subsets; report all + full (16 rows).

Headline questions the result must answer:

1. **Is wifi+imu+camera (drop Odom) genuinely better than full
   4-mod?** RESULT_13's drop-odom subset val/test 0.381/0.406 vs
   full 0.394/0.417 — Δ ~3 %. Could be training noise OR a real
   Odom-injects-noise finding. Either way, the paper claim has to
   address this. Best practice: train BOTH configs from scratch and
   report — but that's 2 trainings, scope creep. **Settle for
   subset-eval evidence from the single trained model.**
2. **Is `only:wifi` still close to the full** (the K=1 saturation
   pattern from RESULT_10)? At K=4, expect the temporal axis to
   *separate* full from `only:wifi`. If the gap is still < 10 %, the
   K=4 win is again WiFi-dominated; if > 20 %, motion modalities
   genuinely contribute under the corrected lr/batch regime.
3. **Per-modality contribution ranking**: rank IMU/Camera/Odom by
   how much each adds vs `only:wifi`. CLAUDE.md run-1 said
   "IMU injects noise at high embed_dim" — does Odom now do the
   same at K=4?

**Acceptance**: 16-row table with val + test MAE.

### Step 3 — Full staleness sweep (paper-figure-grade)

Lags: 0, 1, 3, 5, 10, 15, 20, 30 instants (= 0, 0.9, 2.7, 4.5, 9,
13.5, 18, 27 s at the per-instant stride). Same WiFi-staleness
mechanism as RESULT_11/12 Step 4a.

Plot test MAE vs lag. Report the line shape:
- Smooth monotonic slope → graceful degradation (the headline
  claim).
- Plateau then cliff → fragile under heavy staleness.
- Anything weird (oscillation, sudden drops) → diagnostic.

Save plot under
`runs/overnight/run2_iter_14/staleness_curve.png`.

**Acceptance**: 8-row staleness table + plot. The plot is a
paper-figure candidate; engineer optimises x/y labels for
publication.

### Step 4 — Per-trajectory smoothness on the top-5 longest test paths

Criterion (d) of STATE.md explicitly requires this for the paper.
The Webots test split per CLAUDE.md is paths [15, 16, 17] (3 paths
only). The "top-5 longest test paths" criterion implies we may want
to extend the test set. **Default interpretation**: the 3 test
paths constitute "all available," and the per-trajectory plots
under `runs/overnight/run2_iter_14/test_paths/` for paths 15/16/17
satisfy the gate.

Per-trajectory Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖. Report
median + per-path. RESULT_13 reported some smoothness; this
iteration confirms.

If r ≤ 0.05 (the persistent debt from RESULT_03 onwards), flag
as a Phase C follow-up (auxiliary velocity loss). The Phase B
winner is shipped with the smoothness debt documented — same
framing as RESULT_03's `keep with smoothness debt`.

### Step 5 — Latency probe (criterion (e))

Run two measurements:

- **b=1 latency**: per-sample wall-clock for a single fwd pass on
  CUDA. Repeat ≥ 100 times, report median ms.
- **b=32 latency**: per-batch wall-clock, divided by 32. Reflects
  the realistic streaming-eval throughput.

RESULT_11 reported 0.153 ms at b=1 K=8 5-mod. K=4 4-mod should be
~half that. **Acceptance: < 100 ms/sample on the Quadro P4000.**

### Step 6 — Phase B winner declaration + PerCom main-results table

Write the Phase B winner declaration in RESULT_14's TL;DR with:

- The headline number (test MAE, val MAE, latency).
- The criterion-(b) status — C3 cleared by N m.
- The criterion-(d) status — per-path distribution + smoothness +
  plots filed.
- The criterion-(e) status — latency under the cap by Nx.
- The criterion-(a) status — C1 ✓ (RESULT_01 Anchor2Vec UJI), C2
  PARTIAL (RESULT_02+07 in-domain only), Camera paper-soft
  (RESULT_08), Odom internal (RESULT_04).
- The criterion-(c) status — open; queued as Phase C (PLAN_15+).

### Step 7 — Decision + PLAN_15 recommendation

Three-sentence verdict:
- Phase B winner reproduces / doesn't reproduce; quote numbers.
- Best subset across all 16 (likely wifi+imu+camera per RESULT_13);
  the Phase B paper claim sticks with 4-mod (run-2 thesis) OR
  pivots to 3-mod with Odom noted as redundant — engineer recommends
  with justification.
- PLAN_15 = Phase C kickoff. Recommend:
  - **(default) Cross-session real-world on Microsoft ILN 2.0
    site1/B1** (criterion (c) — the next paper-defensible claim).
  - **(alternative) K-axis sweep at B=128** (K=1, 2, 4, 8 with
    fixed B=128) to characterise the K-axis cleanly — gives a
    paper-figure of K-vs-MAE that closes the architectural-choice
    discussion.
  - Engineer picks based on time-to-stop remaining and outcome of
    Step 2 (if subset eval reveals something needing follow-up,
    do that instead of Phase C kickoff).

## Sources

- RESULT_13: K=4 + 4-mod + B=128 val 0.394 / test 0.417 (Phase B
  winner numbers); subset 'drop Odom' val 0.381 / test 0.406;
  staleness 0.417 → 0.929 m across 18 s.
- RESULT_06: K=1 2-mod baseline (val 0.469 / test 0.517) — the
  C3-with-2-modalities reference.
- RESULT_10: K=1 5-mod saturation (val 0.491 / test 0.486).
- CLAUDE.md "Honest findings" + "Stage A + B/C Complete" — run-1's
  K=8 ≈ 0.43 m reference.
- `configs/stage_c/fusion.yaml` (restored RESULT_06): default
  hyperparameters.
- `src/pipeline/training/fusion_trainer.py` — `evaluate_all_subsets`
  + `evaluate_staleness` methods (restored RESULT_06).

## What to report back

In `handoff/results/RESULT_14_phase-b-winner-ablations.md`:

1. **Step 0** — checkpoint reuse vs re-train; reproducibility
   sanity check.
2. **Step 2** — 16-row subset eval table; per-modality contribution
   ranking; verdict on drop-Odom-vs-full.
3. **Step 3** — 8-row staleness table + line plot; degradation
   shape verdict.
4. **Step 4** — per-trajectory smoothness (median r + per-path);
   plots filed.
5. **Step 5** — latency at b=1 + b=32; criterion (e) gate status.
6. **Step 6** — Phase B winner declaration; full criteria status
   panel (a/b/c/d/e).
7. **Step 7** — PLAN_15 recommendation (default = Phase C kickoff
   on MSILN; alternative = K-axis sweep).
8. **One open question** for scientist.

## Reversibility

- Step 0: no permanent changes.
- Step 2–5: throwaway evaluation outputs.
- Steps 6–7: documentation.

Files committed: RESULT_14 + any small evaluation script changes
(under `scripts/_eval_*.py`).

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 40 min if Step 0A (use checkpoint); ≤ 70 min
if Step 0B (re-train + eval).
- Step 0A: 5 min OR Step 0B: 20–30 min (training).
- Step 2: 8 min (16 subsets × ~30 s eval each).
- Step 3: 8 min (8 lags × ~1 min eval each).
- Step 4: 5 min (smoothness + plots).
- Step 5: 5 min (latency).
- Step 6: 5 min writeup.
- Step 7: 5 min PLAN_15 thinking.

If overrun: drop Step 5 first (latency was already confirmed
< 100 ms in earlier iters; this is a paper-grade re-measurement).
Then drop Step 3's lags from 8 to 4 (0, 5, 15, 30) — keep the
shape, lose the resolution.
