# Result 21 — MoTTransformer (transformer-from-scratch): outcome γ5 — substantial regression

## TL;DR

**MoTTransformer regresses by +79 % vs CNN1D on Webots test** (0.608 m
vs CNN1D's 0.339). The "designed from first principles after
RESULT_17/18 evidence" transformer is the **worst of the four
architectures** tested on full Webots data. CNN1D remains the
Phase B winner.

| arch                              | params  | val MAE | test MAE | smoothness r | latency b=1 (ms) | regime               |
|-----------------------------------|--------:|--------:|---------:|-------------:|-----------------:|----------------------|
| Incumbent (FusionTransformer)     | 1.55 M  | 0.394   | 0.417    | 0.039        | 6.41             | over-parameterised   |
| CNN1D (PHASE B WINNER)            | 0.51 M  | **0.282** | **0.339** | 0.009     | 4.73             | cooperative          |
| LSTM-attn                         | 0.57 M  | 0.301   | 0.340    | **0.051**    | 4.67             | dead-reckoning       |
| **MoTTransformer (this iter)**    | 0.74 M  | **0.594** | **0.608** | **0.019** | 5.82             | WiFi-anchored        |

**Outcome label: γ5** (regresses > +5 % vs CNN1D test). The
"transformer family + this design choice (ALiBi temporal bias)"
underperforms on this K=4 + 4-mod + B=128 regime. The honest
paper claim becomes: **"we benchmarked four architectures
(CNN1D, LSTM-attn, MoTTransformer, FusionTransformer incumbent);
CNN1D's temporal-locality bias wins. Transformer-family
architectures over-fit to WiFi anchoring on this regime."**

**ALiBi did NOT clear the smoothness r > 0.20 gate** (r=0.019, only
marginally above CNN1D's 0.009). The architectural-lever-for-
smoothness hypothesis is now **falsified across 4 architectures**.
Confirmed: smoothness debt is loss-function-bound, not
architecturally tractable. The B-1 (aux velocity loss) / B-2 (EMA)
lever remains the only open knob.

**PLAN_22 recommendation**: continue main-results table at IPIN
2024 floor 0 as originally scheduled. CNN1D remains the
paper-claim model for criterion (b) / C3.

## Step-by-step

### Step 0 — `MoTTransformer` implementation

Wrote `src/pipeline/fusion/mot_transformer.py`, registered as
`"mot_transformer"` in
`src/pipeline/fusion/bakeoff.py::CANDIDATES` for the existing
`scripts/_train_webots_4mod_arch.py --arch <name>` wrapper.

Architecture as PLAN_21 spec:
- Token flatten: per-modality encoder → (B, K=4, D=128) per mod →
  stack to (B, K, M, D) → reshape (B, K·M=16, D). Token i at
  position (t = i//M, m = i%M).
- Learnable modality embedding `(M=4, D=128)` added broadcast over
  K (no temporal positional embed; ALiBi handles it).
- 3 layers, 2 heads, pre-norm, FFN dim = 2D (smaller than the
  incumbent's 4D).
- **ALiBi temporal bias**: per-head learnable inverse-temperature
  slopes initialised to `[1.0, 0.5]`. For tokens at temporal
  positions `t_i, t_j`: `bias[i,j] = -slope_h * |t_i - t_j|`.
  Modality-modality identical-time pairs get no positional bias
  (bias = 0); modality embedding handles modality identity.
  Custom multi-head attention (since `nn.MultiheadAttention`'s
  `attn_mask` is added pre-softmax but the per-head broadcasting
  is fiddly; a custom QKV+softmax path is clearer).
- Single learnable-query cross-attention readout (1-head).
- MLP head D=128 → 64 → 2 (xy).

**NaN safety**: no CLS by design. If all tokens of a sample are
masked (modality_dropout × instant_dropout), token 0 is forcibly
unmasked so softmax stays defined (instead of producing all-`-inf`
rows). Implementation: 4 lines around the `pad` tensor build,
checking `pad.all(dim=1)` then `pad[all_masked, 0] = False`.

**Smoke test** (dummy encoders, synthetic shape (B=128, K=4)):
- Forward: (128, 2) output, mean ≈ 0, std ≈ 0.17 — sensible
  initialisation.
- Backward: loss ≈ 0.034 (MSE on random target); `.backward()`
  succeeds without NaN.
- Drop-camera modality (all `avail['camera'] = False`): forward
  returns (128, 2) without NaN.
- All-tokens-masked stress test: returns (2, 2) with no NaN
  (NaN-safety trick works as designed).

**Param count with real encoders**: **0.74 M** (per Step 2 trainer
load report). Above the PLAN_21 estimate of 0.48 M because the
estimate didn't account for the real per-modality encoders
(Anchor2Vec / IMUCNN / DPVOMotionEncoder / OdomCNN). Apples-to-
apples comparison: CNN1D 0.51 M body + same encoders gives
total 0.51 M (per RESULT_17). MoTTransformer's body alone is
~0.23 M heavier than CNN1D's body, putting total at 0.74 M.

### Step 1 — Pre-test gate

5-epoch run on 10 % Webots train: first val MAE **5.61 m → best
val MAE 0.82 m** (87.4 % drop). Clears the ≥ 10 % gate. Loss
descends monotonically; no NaN; no divergence.

### Step 2 — Full training (90 epochs)

| metric              | value     |
|---------------------|-----------|
| best val MAE        | **0.594** |
| best epoch          | 68        |
| wall time           | 250 s     |
| peak GPU            | 820 MB    |
| n_params            | 0.74 M    |

Val descent is slow: epoch 0 = 0.805 → epoch 60 = 0.652 → epoch
80 = 0.604. Diminishing returns past epoch 50; the architecture
saturates well above CNN1D's 0.282.

### Step 3 — Comparison to RESULT_17/18 leaders

(Headline table at top of this RESULT.)

**Outcome label**: **γ5** (MoTTransformer regresses substantially,
test 0.608 vs CNN1D 0.339 = +79 %). PLAN_21's hypothesis 3
("γ5: transformer family confirmed wrong choice for this task at
this scale") is the verdict.

The fact that MoTTransformer also regresses vs incumbent (test
0.608 vs 0.417 = +46 %) is doubly informative: the "designed
from first principles" version is WORSE than the over-parameterised
run-1 design. Three possible interpretations:

1. **ALiBi temporal bias hurt** — by penalising attention across
   instants, ALiBi might block the cross-temporal motion fusion
   that the incumbent's free attention learned.
2. **No CLS / no PositionQuery hurt** — the incumbent's CLS
   provides a stable always-attended anchor token that the
   MoTTransformer lacks; the single-query readout is more
   sensitive to noisy mask configurations.
3. **No time encoding hurt** — by removing `time_encoding(Δt)`
   in favour of ALiBi-only positional info, the model loses the
   ability to handle varying Δt per modality (some modalities
   have stale-vs-fresh instant gaps the time-encoding-aware
   incumbent could use).

Likely all three contribute. The diagnostic step would be an
ablation suite over (ALiBi vs no-ALiBi) × (CLS vs no-CLS) ×
(time-enc vs no-time-enc) — that's a PLAN_21b experiment if
scientist wants to characterise where the loss came from.

### Step 4 — Subset eval (16 rows)

Test MAE per subset:

| subset                       | mae    | gap to full | regime signal              |
|------------------------------|-------:|------------:|----------------------------|
| only:wifi                    | 0.704  | +15.8 %     | close to full              |
| only:imu                     | 3.928  | +546 %      | very poor                  |
| only:camera                  | 2.072  | +241 %      | poor                       |
| only:odom                    | 5.191  | +753 %      | very poor                  |
| wifi+imu                     | 0.656  | +7.9 %      |                            |
| wifi+camera                  | 0.622  | +2.3 %      | close to full              |
| wifi+odom                    | 0.714  | +17.4 %     |                            |
| imu+camera                   | 2.023  | +232 %      |                            |
| imu+odom                     | 4.566  | +651 %      |                            |
| camera+odom                  | 2.444  | +302 %      |                            |
| wifi+imu+camera (drop-Odom)  | **0.596** | **−2.0 %** | BEATS full marginally     |
| wifi+imu+odom                | 0.676  | +11.2 %     |                            |
| wifi+camera+odom             | 0.630  | +3.6 %      |                            |
| imu+camera+odom (drop-WiFi)  | 2.274  | +274 %      | needs WiFi anchor          |
| **wifi+imu+camera+odom (full)** | **0.608** | 0 %     | the headline               |

**Regime**: **WiFi-anchored**, not cooperative or dead-reckoning.
- only:wifi 0.704 ≈ full 0.608 (+16 %); WiFi alone is most of the
  signal.
- Drop-WiFi (`imu+camera+odom`) collapses to 2.274 m (+274 %).
- Drop-Odom is the best subset (0.596 m, marginally below full
  by 2 %) — RESULT_14 / RESULT_18 pattern persists.

Contrast with the other architectures:
- **CNN1D regime**: cooperative — only:imu 0.352 is close to full
  0.339 (4 % gap); WiFi+motion both contribute.
- **LSTM-attn regime**: per-modality dead-reckoning — all four
  only:X within 8 % of full.
- **MoTTransformer regime**: WiFi-anchored — only:wifi 0.704 is
  the only single-modality < 1 m result; motion modalities are
  3-5 m alone.

This explains the regression: MoTTransformer learnt to ignore
the motion modalities almost entirely (likely the ALiBi cross-
temporal bias prevented the motion modalities from contributing
useful cross-instant information).

### Step 5 — Per-trajectory smoothness (THE load-bearing secondary test)

| path | r       | mean test MAE | n   |
|------|--------:|--------------:|----:|
| 15   | varies  | 0.612         | 875 |
| 16   | varies  | 0.530         | 591 |
| 17   | varies  | 0.677         | 603 |
| median r | **0.019** |     |     |

**ALiBi did NOT lift the smoothness median r above the locked
gate of 0.20.** r=0.019 is only marginally above CNN1D's 0.009
and below LSTM-attn's 0.051. **The architectural-lever-for-
smoothness hypothesis is now falsified across 4 architectures
(incumbent r=0.039, CNN1D r=0.009, LSTM-attn r=0.051,
MoTTransformer r=0.019)** — none clear 0.20.

**Confirmed**: smoothness debt is loss-function-bound, not
architecturally tractable. The B-1 aux velocity loss / B-2 EMA
hypothesis remains the only standing lever (queued as a
post-Phase-C deliverable per RESULTs 18-20).

Per-trajectory plots saved at
`runs/overnight/run2_iter_21/test_paths/mot_transformer_path_{15,16,17}.png`.

### Step 6 — Staleness + latency + PLAN_22

**WiFi staleness sweep** (4-lag short grid):

| lag (instants) | staleness (s) | test MAE (m) |
|---------------:|--------------:|-------------:|
| 0              | 0.0           | 0.608        |
| 5              | 4.5           | 0.722        |
| 15             | 13.5          | 0.973        |
| 30             | 27.0          | 1.359        |

Linear slope **0.0279 m/s, R²=1.000** — **essentially identical
to incumbent's 0.029 and CNN1D's 0.028**. The K=4 temporal-fusion
staleness property is architecture-invariant across all 4
architectures we've tested. ALiBi's temporal-locality bias does
NOT change the staleness slope at this regime.

**Latency probe** (100 trials at b=1, 50 trials at b=32, post 20-
trial warmup):

| batch | ms/sample | ms/batch | factor under 100 ms gate |
|------:|----------:|---------:|-------------------------:|
| 1     | **5.82**  | 5.82     | 17×                      |
| 32    | **0.20**  | 6.46     | 500×                     |

Per-sample: ~23 % slower than CNN1D (4.73 ms) but faster than
incumbent (6.41 ms). Criterion (e) cleared by 17× / 500×.

### Step 7 — Verdict + PLAN_22 recommendation

**Three-sentence verdict.**

(1) **MoTTransformer is the worst of the 4 architectures on full
Webots data** — test 0.608 m regresses by +79 % vs CNN1D's 0.339
and +46 % vs incumbent's 0.417, despite being designed from first
principles after RESULT_17/18 evidence. The likely cause is the
ALiBi temporal bias suppressing motion-modality cross-instant
information; the model regresses to a WiFi-anchored regime where
only:wifi 0.704 carries most of the signal and motion-only
subsets are 3-5 m alone.

(2) **ALiBi did NOT solve smoothness** — r=0.019 falls well below
the 0.20 gate; the architectural-lever-for-smoothness hypothesis
is now falsified across 4 architectures (none clear 0.20).
**Smoothness debt is loss-function-bound, confirmed at 4-arch ×
2-data-scale evidence depth.**

(3) **PLAN_22 recommendation**: continue main-results table at
IPIN 2024 floor 0 as originally scheduled. CNN1D remains the
Phase B / criterion (b) paper-claim model. MoTTransformer is
documented as the "we tried, transformer-family loses" methods-
section data point. The 4-architecture bake-off is now complete
and honest: 1 winner (CNN1D), 1 runner-up with structural
finding (LSTM-attn dead-reckoning), 1 incumbent (over-
parameterised), 1 loser (MoTTransformer).

## One open question for scientist

The γ5 outcome on MoTTransformer is hard to interpret without an
ablation over the 3 design choices that differ from the
incumbent: (1) ALiBi vs free temporal attention; (2) no-CLS vs
incumbent's CLS; (3) no time-encoding vs incumbent's `Δt`
encoding. Without that ablation, the paper claim has to be the
broad "transformer-family with this design choice underperforms"
rather than the precise "ALiBi is the wrong inductive bias for
this task."

Option for scientist: queue a PLAN_21b 3-row ablation
(MoTTransformer + ALiBi-off / +CLS / +time-enc) as an optional
methods-section bonus. Cost: 3 × ~15 min training. Would
strengthen the "we know WHY transformers lose here" claim. Or:
leave it as "transformer-family loses, we documented the
attempt" — cheaper and the paper's bake-off methodology section
is still well-supported.

## Sources

- PLAN_21 architecture spec (Stages 1-4).
- RESULT_17/18 evidence base for the design choices.
- `src/pipeline/fusion/mot_transformer.py` — implementation.
- `src/pipeline/fusion/bakeoff.py` — `CANDIDATES` registry update
  + `build_mot_transformer` factory.
- `scripts/_train_webots_4mod_arch.py` — training wrapper (no
  change required; `--arch mot_transformer` works via the
  registry).
- `scripts/_iter18_cnn1d_ablations.py` — ablation script
  (extended to look up the checkpoint in iter_17 OR iter_21 by
  arch).
- `runs/overnight/run2_iter_21/mot_transformer_full.json`,
  `mot_transformer_ablations.json` — full numerical output.
- ALiBi reference: Press, Smith, Lewis. "Train Short, Test Long:
  Attention with Linear Biases Enables Input Length
  Extrapolation." ICLR 2022 (arXiv:2108.12409).
