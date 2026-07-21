# Plan 16 — Phase B architecture bake-off (4 candidates) on 10 % Webots subset

> **Inserted per third-party review 2026-05-26 ~04:10 local.** The
> bake-off was dropped at iter 10 on the premise "K=1 saturation
> means architecture isn't the bottleneck." RESULT_13 refuted that
> premise: K=4 + B=128 + 4-mod surfaces real motion-modality
> contribution (`only:wifi` 0.732 → 4-mod 0.594 at K=8 = −19 %).
> The incumbent (run-1's `FusionTransformer`, restored RESULT_06)
> is now defensible only if we compared it against alternatives.
> This iteration makes the comparison.
>
> Time-context: RESULT_15 just closed Phase C kickoff with outcome
> β (clean wlan_localization beat, partial kNN gate). The bake-off
> runs BEFORE further Phase C work because (a) it's a paper-
> strength fix for the run-2 methods section and (b) it's
> independent of MSILN's test-composition anomaly. Phase C
> continues at PLAN_17+.

## Hypothesis

The Phase B winner from RESULT_14 is **set-transformer (run-1's
`FusionTransformer`) at K=4 + 4-mod + B=128: val 0.394 / test
0.417 m, smoothness r=0.039**. The bake-off asks: does any of
**{LSTM-with-attention, TCN dilated-conv, 1D-CNN over instants,
Transformer-from-scratch}** beat this incumbent on the same
inputs?

The "Transformer-from-scratch" candidate exists separately from
the incumbent to control for "is the run-1 design (per-modality
embedding + time encoding + cross-attention PositionQuery readout)
fundamentally better than a vanilla transformer at the same param
budget?" — answers a fair-comparison reviewer question for the
paper.

Decision rule:
- **New winner**: a candidate beats incumbent 0.417 m on test AND
  lifts per-trajectory smoothness median r above the locked
  > 0.20 gate.
- **Incumbent stands defensible**: no candidate clears both bars;
  the paper methods section is rewritten as "we benchmarked 4
  fusion architectures and kept this one" instead of "we used
  this one without comparison."

This is one focused experiment: an architecture A/B/C/D
comparison under identical input pipeline + readout + train
protocol. Single iteration; 4 candidates × small subset = fast.

## Steps

### Step 0 — Subset + candidate scaffolding (10–15 min)

**Step 0a — Webots 10 % subset.** The 11 train paths split per
CLAUDE.md ([1, 3-12]) contain ~80–500 frames each per modality.
10 % = ~1.1 paths (round to 1 path = path_1, or to 2 paths if
single-path is too noisy for pre-test). Engineer picks the
smallest split that satisfies the pre-test gate's signal-to-noise
requirement (val MAE moves ≥ 10 % across 5 epochs reliably).

Val + test splits stay unchanged ([2, 13, 14] / [15, 16, 17]) so
all 4 candidates report numbers directly comparable to the
incumbent's RESULT_14 / RESULT_13 baselines.

**Step 0b — Candidate scaffolds**. Each candidate is a new file
under `src/pipeline/fusion/` with the same forward-pass interface
as `transformer.py`. Engineer picks ONE implementation pattern
(simplest), e.g. all candidates take `(B, K, M, D)` per-instant
per-modality tokens + a learnable PositionQuery `(B, D)` and emit
a `(B, 2)` xy prediction. Same param budget target (~1.5 M params)
for fairness.

Candidate sketches:

1. **LSTM-with-attention** (`fusion/lstm_attn.py`).
   - Per-modality LSTM over K instants (M independent LSTMs) → per-modality summary `(B, M, D)`.
   - Cross-modal attention: PositionQuery attends over the M summaries → `(B, D)`.
   - MLP head → `(B, 2)`.

2. **TCN (dilated conv)** (`fusion/tcn.py`).
   - Stack of 1D dilated convolutions over the K-instant axis,
     per modality, with shared kernel weights across modalities.
     Dilations [1, 2, 4] (covers K=4 receptive field).
   - Per-modality summary → cross-modal attention pool → MLP.

3. **1D-CNN over instants** (`fusion/cnn1d_instants.py`).
   - Simpler than TCN: 3-layer 1D conv on K axis without dilation,
     ReLU + BN.
   - Per-modality summary → mean-pool across modalities → MLP.
   - This is the minimum-baseline candidate.

4. **Transformer-from-scratch** (`fusion/transformer_scratch.py`).
   - Vanilla 2-layer transformer encoder, batch_first=True, M×K
     tokens flattened with learnable modality + instant positional
     embeddings.
   - CLS token at index 0, output projected by MLP.
   - **NO** custom PositionQuery cross-attention readout (the
     incumbent's design). NO per-modality embedding bank with
     `time_encoding(Δt)`. Just a stock transformer.

**Acceptance for Step 0**: 4 candidate modules import cleanly; a
synthetic-input smoke fwd produces `(B, 2)` output with correct
shape and no NaN; param count within ±20 % of 1.5 M for each.

If Step 0 surfaces an unworkable implementation issue (e.g.
input-shape mismatch with the dataloader), document and STOP —
do NOT attempt 4 architecture rewrites in one iteration.

### Step 1 — Pre-test gate per candidate (5 epochs × 4 candidates = ~5 min)

Each candidate: 5 epochs on the 10 % subset. Acceptance: val MAE
drops ≥ 10 % OR clear descent.

Any candidate that fails pre-test → label `unworkable` and skip
to Step 2 with that candidate's "did not converge" noted.

### Step 2 — Full training on 10 % subset (30 epochs × 4 candidates)

Reduced epoch count (30 not 90) because the subset is small and
overfitting risk is high. Same lr/optimizer schedule as
RESULT_13/14 (AdamW + OneCycleLR + Huber(δ=0.5), B=128, K=4,
4-mod, `instant_dropout=0.45`, `modality_dropout=0.4`).

**Memory budget per candidate**: < 6 GB. Should be trivially met
at K=4 + 4-mod + B=128 + 10 % subset.

Run all 4 candidates back-to-back. Each ~5 min at 30 epochs
(scaling from RESULT_14's 90-epoch / B=128 / full-data 0.394
val); total bake-off training ≈ 20 min.

**Acceptance**: 4 trained models; val + test MAE recorded for each.

### Step 3 — Subset eval (each candidate)

For each candidate, run `evaluate_all_subsets` (3 minimum rows
per candidate: `only:wifi`, full-fusion, and one motion-only baseline
like `wifi+imu+camera` to detect saturation patterns). Reduced
from RESULT_14's 16-row sweep because the headline question is
"does this candidate fuse better than wifi alone?"

### Step 4 — Per-trajectory smoothness (gate per RESULT_05 lock)

Median Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖ across test paths
15/16/17 for each candidate. The locked gate (r > 0.20) is what
separates "candidate has a smoothness inductive bias" from "just
another fusion architecture."

TCN's dilated conv has a smoothness inductive bias by construction
(temporal locality); 1D-CNN has it weakly; LSTM-attn possibly;
transformer-from-scratch typically not. The gate is the
differentiator.

### Step 5 — Bake-off summary table

```
| candidate                     | params | val MAE | test MAE | only:wifi test | smoothness r | both gates pass? |
|-------------------------------|-------:|--------:|---------:|---------------:|-------------:|:----------------:|
| Incumbent (FusionTransformer) | 1.53 M | (RESULT_14: 0.394) | (RESULT_14: 0.417, smoothness r=0.039 on FULL data) | — | reference | (reference, incumbent) |
| LSTM-with-attention           |        |         |          |                |              |                  |
| TCN dilated-conv              |        |         |          |                |              |                  |
| 1D-CNN over instants          |        |         |          |                |              |                  |
| Transformer-from-scratch      |        |         |          |                |              |                  |
```

**NOTE**: incumbent's RESULT_14 numbers are on FULL Webots data,
not the 10 % subset. The candidates are on 10 % subset.
**For a fair comparison, train the incumbent on the same 10 %
subset as a 5th column** so all five numbers are apples-to-apples
on the bake-off's data partition. Engineer adds this row.

### Step 6 — Decision + PLAN_17 recommendation

Three-sentence verdict:
- Did any candidate beat the incumbent on 10 % subset val + test
  AND clear smoothness r > 0.20?
- If yes: name the new winner; PLAN_17 = full-data re-training of
  the new winner at the Phase-B-winner config (analog of
  RESULT_13/14 with new architecture).
- If no: incumbent stands defensible; PLAN_17 = Phase C continuation
  (MSILN follow-up addressing RESULT_15's β outcome: either
  conformal coverage or WiFiSetTransformer re-eval cross-session
  — depending on the "test-composition anomaly" diagnosis the
  scientist resolves before PLAN_17).

## Sources

- Third-party review note 2026-05-26 ~04:10 local: architecture
  bake-off requirement.
- RESULT_13: K=4 + 4-mod + B=128 0.417 m test (the incumbent).
- RESULT_14: ablation suite confirming the incumbent.
- RESULT_05 lock: per-trajectory smoothness > 0.20 gate.
- `src/pipeline/fusion/transformer.py` (incumbent, restored
  RESULT_06).
- `src/pipeline/training/fusion_trainer.py` (training loop,
  restored RESULT_06).
- CLAUDE.md "Phase B candidates" — original SCIENTIST_BRIEF
  listing of LSTM-attn / TCN / late+gate; the four-candidate
  set here substitutes 1D-CNN + transformer-from-scratch for the
  late+gate variant the brief mentioned (late+gate's natural
  habitat is K=1 where modality_dropout drives the gate; at K=4
  the per-instant attention does that role).

## What to report back

In `handoff/results/RESULT_16_phase-b-architecture-bakeoff.md`:

1. **Step 0** — subset choice; 4 candidate scaffold paths;
   smoke + param counts.
2. **Step 1** — pre-test gate results per candidate.
3. **Step 2** — training summary per candidate (loss curves
   pointer; final val/test MAE).
4. **Step 3** — subset eval per candidate (3 rows minimum).
5. **Step 4** — per-trajectory smoothness median r per candidate.
6. **Step 5** — 5-row bake-off summary table (4 candidates +
   incumbent on same 10 % subset).
7. **Step 6** — outcome label + PLAN_17 recommendation.
8. **One open question** for scientist.

## Reversibility

- Step 0a (subset): throwaway split.
- Step 0b (candidate scaffolds): **permanent** under
  `src/pipeline/fusion/{lstm_attn,tcn,cnn1d_instants,transformer_scratch}.py`.
  Engineer commits. Keeps them in tree even if none wins (for
  reviewer-facing "we tried these alternatives" claim).
- Step 2 (training): throwaway checkpoints.
- Steps 3–6: documentation.

Files committed: RESULT_16 + 4 new fusion candidates +
`src/pipeline/fusion/__init__.py` re-exports + any small
factory plumbing in `builder.py` if needed.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 75 min.
- Step 0a: 5 min.
- Step 0b: 30 min (4 candidate implementations + smoke; engineer
  uses smallest viable patterns).
- Step 1: 5 min (4 × 5-epoch pre-tests).
- Step 2: 20 min (4 × 30-epoch + 1 × 30-epoch incumbent re-train
  on same subset).
- Step 3–4: 5 min.
- Step 5–6: 10 min.

If overrun: cut Step 0b candidate count from 4 to 3 (drop the
weakest-prior candidate, likely 1D-CNN; document the cut). Don't
silently skip — the paper claim is "4 candidates benchmarked," so
the cut needs to be explicit.

If any candidate refuses to converge on the 10 % subset (pre-test
fails), don't promote that candidate to Step 2 — write "did not
converge in pre-test on subset; not pursued further" and move on.
The paper-defensible claim is still "4 candidates attempted" if
the failure mode is honestly reported.

If Step 5 reveals a new winner, **do NOT scope-creep PLAN_16 to
full-data re-train** — PLAN_17 picks that up. This iteration's
scope is the bake-off itself.
