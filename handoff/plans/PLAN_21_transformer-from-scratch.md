# Plan 21 — Transformer-from-scratch candidate (`MoTTransformer`) on full Webots

> **Restoring the cut bake-off candidate** per third-party
> directive 2026-05-26 ~09:00 local. PLAN_16 dropped
> Transformer-from-scratch under its "cut to 3 if overrun"
> provision; that left the bake-off comparing only ONE transformer
> family (run-1's incumbent vs three non-transformer candidates).
> For a fusion paper, the methods section needs an *informed*
> transformer baseline — i.e. a transformer designed AFTER we
> learned what RESULTs 17/18 revealed about the K=4 + 4-mod regime.
> Scientist designs (this plan); engineer implements.
>
> IPIN floor 0 (originally PLAN_21) slides to PLAN_22. Main-results
> table chain becomes: PLAN_22 IPIN, PLAN_23 RoNIN single-mod,
> PLAN_24 UJI degenerate, PLAN_25 SUMMARY + table assembly.

## Hypothesis

A transformer designed from first principles for (modality × time)
fusion under K=4 + 4-mod + B=128 should be able to (a) match or
beat CNN1D's test 0.339, AND (b) take a real swing at the
smoothness gate via an explicit inductive bias. The incumbent's
run-1 design (over-parameterised 6-layer 4-head transformer at
1.55 M params with PositionQuery readout) was the wrong design;
this iteration tests whether transformers as a class are wrong
or whether the run-1 design was wrong.

Three outcomes:
- **(α5) MoTTransformer beats CNN1D (test < 0.322 = −5 % vs 0.339)
  AND clears smoothness r > 0.20**: transformer family wins on
  fresh accuracy AND on smoothness — strongest possible bake-off
  outcome.
- **(β5) MoTTransformer competitive with CNN1D/LSTM-attn (test
  within ±5 %)**: 4-architecture bake-off honestly inconclusive
  at fresh accuracy; smoothness verdict separates them. Defensible
  paper claim "we benchmarked CNN1D vs LSTM-attn vs MoTTransformer
  + incumbent."
- **(γ5) MoTTransformer regresses (> +5 % vs CNN1D test)**:
  transformer family confirmed wrong choice for this task at this
  scale. Paper claim: "transformer-family losses 18 % at 1/3
  params on this K=4 + 4-mod regime — recommended against."

This is one focused experiment: a fair-design transformer trained
at the production protocol.

## MoTTransformer architecture — design rationale + spec

### What RESULTs 17/18 revealed (and how each finding maps to a design choice)

1. **CNN1D wins at 1/3 params (0.51 M vs incumbent's 1.55 M)** —
   the incumbent over-parameterised for this data scale. Design
   choice: target ~0.45–0.50 M params for MoTTransformer.
2. **LSTM-attn dead-reckons from any single motion modality**
   (`only:imu` ≈ `only:camera` ≈ full) — temporal ordering inside
   each modality matters more than the incumbent's cross-modal
   self-attention assumed. Design choice: explicit temporal axis
   processing in the attention pattern.
3. **Smoothness debt is architecture-invariant across all 4 archs
   tried so far** (r ≤ 0.085) — no architecture so far has a
   strong inductive bias toward smooth embeddings. Design choice:
   **ALiBi-style relative-position bias on the temporal axis** —
   attention scores decay with temporal distance, producing a
   built-in smoothness prior that's not learned (so doesn't get
   regularised away).
4. **K=4 is small** — full self-attention over K=4 instants is
   trivially cheap; no need for sparse/sliding attention.

### Architecture spec — `MoTTransformer` (Modality-of-Time Transformer)

**Input tokens.** Same FusionBlock interface as the other
bakeoff candidates: per-modality per-instant tokens at shape
`(B, K=4, M=4, D=128)` plus a `(B, K, M)` valid-mask.

**Stage 1 — Token flattening + learnable modality embeddings.**
- Reshape `(B, K, M, D)` → `(B, K*M=16, D)`.
- Add learnable modality embeddings: shape `(M, D)`, broadcast
  to all K instants for each modality.
- **No** learnable temporal positional embedding (ALiBi handles
  it). **No** CLS token.

**Stage 2 — 3-layer transformer encoder block.** Pre-norm
(LayerNorm BEFORE attention + FFN). Each layer:
  - Multi-head self-attention over the 16 tokens, **2 heads**.
  - **ALiBi temporal bias** on the attention scores: for token
    pairs `(t1, m1)` and `(t2, m2)`, add bias `−|t1 − t2| / s_h`
    where `s_h` is a per-head learnable inverse temperature
    (or fixed at `s = {1, 2}` for the two heads per the canonical
    ALiBi formulation). **Bias is only on the temporal axis**;
    modality-modality pairs get no positional bias (the modality
    embedding handles modality identity).
  - FFN: `D → 2D → D` (smaller than the standard 4D to keep
    param count down).
  - Residual connections around attention + FFN.
- 3 layers (matches RESULT_06 incumbent's depth ÷ 2; param-budget
  control).

**Stage 3 — Single-query cross-attention readout.**
- Learnable query token `Q (1, D)`, broadcast to batch as
  `Q (B, 1, D)`.
- 1-head cross-attention: Q queries the 16 tokens from Stage 2.
  No positional encoding on K or V (the encoder already mixed
  positions).
- Output: `(B, D)`.

**Stage 4 — Head.**
- MLP: `D → 64 → 2`. ReLU between layers. Outputs the (x, y)
  position prediction.

**Mask handling.** The `(B, K, M)` valid-mask reshapes to
`(B, K*M=16)`; tokens flagged invalid get attention-mask = −∞
in Stage 2's self-attention AND in Stage 3's cross-attention.
Standard modality_dropout / instant_dropout (set at
`modality_dropout=0.4`, `instant_dropout=0.45` per RESULT_17/18
default) applies at training time as in other candidates.

### Parameter budget (estimate)

| component | params |
|---|---|
| Modality embeddings `(M=4, D=128)` | 512 |
| Layer × 3 each: QKV proj `(3·D·D)` + out proj `(D·D)` + FFN `(2·D·D + D + 2·D·D + D)` + 2 LayerNorms `(4·D)` | ~135 k × 3 = 405 k |
| ALiBi inverse temperatures `(2 heads)` | 2 |
| Stage 3 cross-attn: Q proj + KV proj + out proj | ~65 k |
| Learnable query `(D)` | 128 |
| Head MLP `(D → 64 + 64 → 2)` | 8 322 |
| **TOTAL** | **~478 k** (~0.48 M) |

Matches CNN1D's 0.51 M and LSTM-attn's 0.57 M — fair-comparison
parameter count.

### Why this design beats the incumbent on first principles

- **Permutation-invariant over modalities** (modality embedding,
  not positional encoding) — no fragile modality ordering.
- **ALiBi temporal bias** — explicit smoothness inductive bias
  the incumbent lacked.
- **Shallow (3 layers, 2 heads)** — fights over-parameterisation
  that RESULT_16/17 surfaced.
- **Smaller FFN (2D not 4D)** — fits the param budget.
- **Single learnable query readout** — same family as the
  incumbent's PositionQuery but without per-modality-embedding
  bookkeeping; simpler.

### What this design does NOT borrow from the incumbent

- No 6-layer 4-head over-parameterisation.
- No CLS token / no PositionQuery `(time_encoding(Δt))` machinery.
- No per-modality embedding bank with `time_encoding(Δt)` (the
  ALiBi bias substitutes for the explicit Δt encoding).
- No cross-attention-readout-as-separate-stage with its own MLP
  before the head (single cross-attn → head is cleaner).

The design is recognisably a transformer encoder + cross-attention
readout, BUT every choice was driven by what RESULTs 17/18 said
should be different from the incumbent.

## Steps

### Step 0 — Implement `MoTTransformer` (engineer's task, ~25 min)

Engineer writes `src/pipeline/fusion/mot_transformer.py` matching
the FusionBlock interface used by `bakeoff.py` (CNN1D / LSTM-attn /
TCN). The spec above is the exact architecture; engineer translates
it to PyTorch.

ALiBi reference for engineer: arXiv:2108.12409 (Press et al.,
ICLR 2022). Standard implementation is ~15 lines of bias
computation in the attention forward.

**Acceptance**: import smoke + 1-epoch synthetic forward+backward
at `(B=128, K=4, M=4, D=128)` produces a `(B, 2)` output without
NaN; total param count within ±10 % of 478 k.

### Step 1 — Pre-test gate (5 min)

5-epoch training on 10 % Webots train. Val MAE drops ≥ 10 % OR
clear descent. Memory budget: K=4 + 4-mod + B=128 + 0.48 M
params should peak ~250 MB.

If pre-test fails (NaN, divergence, no descent), STOP — engineer
checks ALiBi implementation OR the QKV projection shapes.

### Step 2 — Full training on Webots

Same protocol as RESULT_17:
- 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5), B=128, K=4,
  4-mod (WiFi + IMU + Camera + Odom), lr=1.3e-3.
- `modality_dropout=0.4`, `instant_dropout=0.45`.
- Same canonical Webots split.

Wall-clock target: ~12-15 min (similar to CNN1D's 196 s in
RESULT_17; MoTTransformer has more compute per token but K=4
is small).

**Acceptance**: training completes; val + test MAE recorded.

### Step 3 — Compare to RESULT_17/18 leaders

| arch | params | val MAE | test MAE | smoothness r | latency b=1 (ms) |
|---|---|---|---|---|---|
| Incumbent (FusionTransformer) | 1.55 M | 0.394 | 0.417 | 0.039 | 4.73 |
| CNN1D (PHASE B WINNER) | 0.51 M | 0.282 | 0.339 | 0.009 | TBD |
| LSTM-attn | 0.57 M | 0.301 | 0.340 | 0.051 | TBD |
| **MoTTransformer (this iter)** | ~0.48 M | ? | ? | ? | ? |

**Acceptance** (outcome label per the hypothesis: α5 / β5 / γ5).

### Step 4 — Subset eval (6 key rows)

`only:wifi`, `only:imu`, `only:camera`, `only:odom`, `wifi+imu+camera`,
full. Surfaces whether MoTTransformer fuses cooperatively (CNN1D
pattern) or per-modality-dead-reckons (LSTM-attn pattern) or
something new.

### Step 5 — Per-trajectory smoothness (the load-bearing secondary
test)

Median Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖ across test paths
15/16/17. **This is the key measurement** — does ALiBi's relative
position bias finally lift smoothness above the 0.20 gate?

- If r > 0.20: paper-strength architectural finding. ALiBi (or
  any explicit smoothness prior) is the lever the 4-arch
  comparison has been missing.
- If r still ≤ 0.20: confirms smoothness is loss-function-bound,
  not architecturally tractable.

Save per-trajectory plots under
`runs/overnight/run2_iter_21/test_paths/`.

### Step 6 — Latency probe + decision

- Latency at b=1 (100-trial median) + b=32 throughput.
- Outcome label.
- **PLAN_22 recommendation**: continue main-results table at IPIN
  2026 floor 0 (the scheduled next row). If α5 fires (new
  winner), update PLAN_22 to use MoTTransformer as the
  paper-claim model going forward. If β5 / γ5, CNN1D remains the
  Phase B winner.

## Sources

- Third-party directive 2026-05-26 ~09:00 local: restore the cut
  transformer-from-scratch candidate.
- RESULT_17: CNN1D + LSTM-attn vs incumbent (full data).
- RESULT_18: ablation suite on CNN1D + LSTM-attn dead-reckoning
  confirmation.
- RESULT_16: bake-off subset table (incumbent vs CNN1D vs LSTM-attn
  vs TCN; transformer_scratch slot empty).
- ALiBi: Press, Smith, Lewis (ICLR 2022) — arXiv:2108.12409.
- `src/pipeline/fusion/{cnn1d_instants,lstm_attn,tcn}.py` (committed
  RESULT_16/17) for the FusionBlock interface pattern.
- `src/pipeline/training/fusion_trainer.py` (training loop, arch
  agnostic).

## What to report back

In `handoff/results/RESULT_21_transformer-from-scratch.md`:

1. **Step 0** — `mot_transformer.py` param count + smoke; any
   implementation deviations from spec (e.g. ALiBi formulation
   choice).
2. **Step 1** — pre-test gate outcome.
3. **Step 2** — training summary (loss curves, val + test,
   wall, peak GPU).
4. **Step 3** — 4-row comparison table; outcome label (α5/β5/γ5).
5. **Step 4** — 6-row subset eval.
6. **Step 5** — per-trajectory smoothness median r + plots; **does
   ALiBi lift r above 0.20?**
7. **Step 6** — latency + PLAN_22 recommendation.
8. **One open question** for scientist.

## Reversibility

- Step 0: permanent — `src/pipeline/fusion/mot_transformer.py`
  committed. Stays in tree even if MoTTransformer loses (the
  paper claim is "4 architectures benchmarked"; the file is the
  evidence).
- Step 2: throwaway checkpoint under
  `runs/overnight/run2_iter_21/` (gitignored).
- Steps 3–6: documentation.

Files committed: RESULT_21, `src/pipeline/fusion/mot_transformer.py`,
factory/registry update in `bakeoff.py` if used.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 50 min.
- Step 0: 25 min (architecture implementation; standard
  transformer encoder + ALiBi + cross-attn-readout; engineer
  experience determines exact time).
- Step 1: 5 min.
- Step 2: 15 min training.
- Step 3: 3 min.
- Step 4: 3 min.
- Step 5: 4 min smoothness + plots.
- Step 6: 5 min decision + writeup.

If Step 0 overruns (ALiBi math gets fiddly), pause and check
with scientist BEFORE writing inefficient code. The architecture
spec is fixed; the engineering risk is implementation depth, not
design choices.

If γ5 fires (MoTTransformer regresses substantially), the
honest paper claim is "transformer family + this design choice
ALiBi underperforms; CNN1D's temporal-locality bias wins." Engineer
NOT to attempt PLAN_21b at different design points — that's
scientist's call after seeing the result.
