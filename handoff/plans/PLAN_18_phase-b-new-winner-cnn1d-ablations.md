# Plan 18 — Phase B new winner (CNN1D) full ablation suite + LSTM-attn dead-reckoning probe

> RESULT_17 fired outcome α''' decisively: **CNN1D is the new Phase
> B winner** at val 0.282 / test 0.339 (−18.7 % test vs incumbent's
> 0.417 at 1/3 params + 145× lower latency). LSTM-attn essentially
> tied on test (0.340) but with a structurally different fusion
> regime — `only:imu` test 0.339 ≈ full 0.340, i.e. each motion
> modality dead-reckons independently. This iteration produces the
> paper-grade ablation suite on CNN1D + characterises LSTM-attn's
> dead-reckoning more carefully (the paper's "discussion" finding).

## Hypothesis

CNN1D is the headline paper claim for criterion (b) / C3. The
ablation suite mirrors RESULT_14's incumbent ablations so the
PerCom main-results table can replace RESULT_14's incumbent
numbers with CNN1D's, in the same shape.

LSTM-attn's dead-reckoning behaviour is a secondary finding worth
one extra step:
- If `only:imu` and `only:camera` and `only:odom` each ≈ full
  on test for LSTM-attn but NOT for CNN1D, the **fusion regime
  difference is real and matters**. Paper claim: "we observed
  two distinct fusion regimes — LSTM-attn dead-reckons
  independently per modality; CNN1D fuses cooperatively."
- If the dead-reckoning finding doesn't hold under staleness or
  on different paths, it's training noise.

This is one focused experiment (CNN1D's full ablation, RESULT_14
shape) plus a tightly-bounded follow-up probe on LSTM-attn's
dead-reckoning structural finding.

## Steps

### Step 0 — Load CNN1D + LSTM-attn checkpoints + sanity (5 min)

`runs/overnight/run2_iter_17/<arch>/model.pt` should contain both
checkpoints from RESULT_17. Load each; emit val + test MAE.
Sanity check matches RESULT_17 within ±0.005 m.

If checkpoints aren't saved (only metrics emitted), re-train from
scratch with the seed=42 / RESULT_17 protocol — but this adds 25+25
min. Prefer checkpoint reuse.

**Acceptance**: both checkpoints loaded; sanity emits matching
numbers.

### Step 1 — CNN1D 16-row subset eval (paper main-results table)

Run `evaluate_all_subsets` on the CNN1D checkpoint. The 4-mod stack
has 15 non-empty subsets + full = 16 rows.

Per RESULT_14's pattern, surface the diagnostic rows:
- `only:wifi` (RESULT_17 reports 0.393 val / 0.402 test) —
  confirms WiFi-anchor saturation pattern.
- `wifi+imu` / `wifi+camera` / `wifi+odom` — pairwise additions to
  WiFi.
- `drop:wifi` — robustness under WiFi outage.
- `drop:imu` / `drop:camera` / `drop:odom` — single-modality
  removal.
- `wifi+imu+camera` (3-mod drop-odom; RESULT_14 had this as the
  best variant on incumbent at 0.406; check for CNN1D).
- **`wifi+imu+camera+odom` (full)** — the headline.

**Acceptance**: 16-row table reported; verdict on whether the
"drop-Odom" RESULT_14 finding holds under CNN1D.

### Step 2 — CNN1D 8-lag staleness sweep (paper robustness figure)

Same lag grid as RESULT_14: 0, 1, 3, 5, 10, 15, 20, 30 instants
(= 0–27 s WiFi staleness). Plot test MAE vs lag. Report:
- Slope (test MAE vs lag, linear fit).
- Comparison vs incumbent's RESULT_14 0.029 m/s slope across 27 s.

If CNN1D's slope is similar (or shallower), the K=4 temporal axis
property holds under the new architecture. If it's much steeper,
CNN1D may rely more on the WiFi token per-instant than the
incumbent did — informative.

**Acceptance**: 8-row staleness table + plot under
`runs/overnight/run2_iter_18/cnn1d_staleness.png`.

### Step 3 — CNN1D per-trajectory smoothness + plots (criterion (d))

Median Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖ across test paths
15/16/17. RESULT_17 reported CNN1D r=0.009 — substantially worse
than incumbent's 0.039. The CNN1D's temporal locality bias
(dilated convs) didn't translate to smoother predictions at full
data — confirming RESULT_17's "smoothness is a loss-function
problem" verdict.

Save per-trajectory plots for test paths 15/16/17 under
`runs/overnight/run2_iter_18/test_paths/`.

### Step 4 — LSTM-attn dead-reckoning probe

Run two diagnostic measurements on the LSTM-attn checkpoint:

**Step 4a — extended subset eval.** Beyond RESULT_17's single
`only:X` row per modality, run the 16-row subset eval on LSTM-attn
too. Check: do ALL `only:X` rows tie full-fusion (CNN1D's
`only:wifi` 0.393 is the closest; the others are 0.7-5+ m)? The
table will show whether the dead-reckoning regime is uniform or
limited to certain modalities.

**Step 4b — staleness on LSTM-attn.** Lag 0, 5, 15, 30 instants on
LSTM-attn (subset of Step 2's grid to save compute). If LSTM-attn
dead-reckons better, its staleness slope should be MUCH shallower
than CNN1D's (the per-modality LSTM bridges across stale-WiFi
instants).

**Acceptance**: 16-row subset table + 4-lag staleness table for
LSTM-attn. Verdict on whether the dead-reckoning finding is
training noise or structural.

### Step 5 — CNN1D latency probe (criterion (e))

Per-sample b=1 (single-sample wall time) and b=32 throughput
divided by batch size. RESULT_17 reported b=1 0.044 ms; this
iteration re-measures with statistics (median of 100 trials) +
b=32.

**Acceptance**: < 100 ms / sample on Quadro P4000 (gate clears
trivially).

### Step 6 — New PerCom main-results panel + Phase status update

Write the updated panel:
- Criterion (a) — per-leg validation: C1 ✓ (Anchor2Vec UJI),
  C2 NOT discharged (canonical RoNIN, kept as "in-domain only"),
  Camera paper-soft (TartanAir hospital), Odom internal (no SOTA).
- Criterion (b) — 4-modality fusion test MAE ≤ 0.5 m: **CNN1D
  clears by 32 %** (test 0.339 m).
- Criterion (c) — MSILN cross-session: clean SOTA beat
  (RESULT_15), partial kNN gate.
- Criterion (d) — per-path + per-trajectory smoothness: CNN1D
  ablation plots filed; smoothness debt documented.
- Criterion (e) — latency < 100 ms: **CNN1D 0.044 ms/sample**
  (2300× under).

### Step 7 — Decision + PLAN_19 recommendation

Three-sentence verdict:
- Does CNN1D's 16-row ablation hold the paper-strength shape?
- Is LSTM-attn's dead-reckoning regime confirmed?
- PLAN_19 recommendation. Default options:
  - **(a) MSILN cross-session re-run with the new CNN1D winner**
    (re-do RESULT_15 with the new architecture; the previous
    K=4 + 2-mod result was on the incumbent). If CNN1D's bigger
    margin holds cross-session, gate (c)-1 might finally close.
  - **(b) Conformal coverage on CNN1D winner** (criterion (d)
    extension; uses `src/pipeline/uncertainty/conformal.py`
    restored in RESULT_06).
  - **(c) SUMMARY draft** — write `handoff/SUMMARY.md` capturing
    Phase A/B/C findings; the run-2 archive is ready for handoff.

Engineer recommends with justification.

## Sources

- RESULT_17: CNN1D + LSTM-attn full-data results.
- RESULT_14: incumbent ablation pattern (16-row + 8-lag mirror).
- RESULT_05: smoothness debt B-1/B-2/B-3 follow-up entries.
- `runs/overnight/run2_iter_17/<arch>/model.pt` checkpoints.
- `src/pipeline/fusion/{cnn1d_instants,lstm_attn}.py` (committed
  RESULT_16).
- `src/pipeline/training/fusion_trainer.py` (`evaluate_all_subsets`,
  `evaluate_staleness`).

## What to report back

In `handoff/results/RESULT_18_phase-b-new-winner-cnn1d-ablations.md`:

1. **Step 0** — checkpoint reuse vs re-train; sanity numbers.
2. **Step 1** — CNN1D 16-row subset eval; verdict on
   drop-Odom-vs-full.
3. **Step 2** — CNN1D 8-lag staleness sweep; slope comparison vs
   incumbent.
4. **Step 3** — CNN1D smoothness + plots.
5. **Step 4a** — LSTM-attn 16-row subset; dead-reckoning verdict
   (uniform or limited to certain modalities).
6. **Step 4b** — LSTM-attn 4-lag staleness; slope vs CNN1D.
7. **Step 5** — CNN1D latency.
8. **Step 6** — PerCom main-results panel update.
9. **Step 7** — PLAN_19 recommendation + open question.

## Reversibility

- All steps: throwaway eval outputs.
- Step 5–7: documentation.

Files committed: RESULT_18, any eval script enhancements.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 50 min (no training; all eval on
checkpoints).
- Step 0: 5 min.
- Step 1: 8 min (16 subsets × ~30 s).
- Step 2: 8 min (8 lags).
- Step 3: 5 min (smoothness + plots).
- Step 4a: 8 min (16 subsets on LSTM-attn).
- Step 4b: 4 min (4 lags).
- Step 5: 3 min.
- Step 6: 5 min.
- Step 7: 5 min.

If overrun: cut Step 4a's LSTM-attn 16-row to a 6-row diagnostic
(only:* for all 4 modalities + full + drop:wifi). The
dead-reckoning verdict is the headline; the full 16 rows are bonus.

If checkpoints aren't saved (Step 0 fails), defer Step 4 entirely
and ship Step 1-3 + 5-7 with CNN1D-only — the new winner's
ablation is the load-bearing finding.
