# The Fusion Pipeline — A Step-by-Step Walkthrough

This document explains the **Stage B+C fusion pipeline** from the ground up:
how raw sensor data becomes an `(x, y)` position, *why* each piece exists, and
*how it was built* (it was developed iteratively — each iteration added one
mechanism and was smoke-tested before the next).

Each step follows the same shape:

> **Intuition** → **How it relates to the rest** → **Formula** → **Worked
> example with simple numbers** → **In the code**.

Read it top to bottom once; after that each step stands alone as a reference.

---

## Table of contents

0. [The problem](#step-0--the-problem)
1. [Data — async sensor windows](#step-1--data--async-sensor-windows)
2. [Encoders — raw window → 128-d token](#step-2--encoders--raw-window--128-d-token)
3. [The universal token](#step-3--the-universal-token)
4. [Attention — the one primitive](#step-4--attention--the-one-primitive)
5. [Self-attention = cross-modal fusion (Iteration 1)](#step-5--self-attention--cross-modal-fusion-iteration-1)
6. [The padding mask — handling missing sensors](#step-6--the-padding-mask)
7. [Temporal self-attention (Iteration 2)](#step-7--temporal-self-attention-iteration-2)
8. [Continuous-time encoding](#step-8--continuous-time-encoding)
9. [Cross-attention readout — the PositionQuery (Iteration 3)](#step-9--cross-attention-readout-iteration-3)
10. [Robustness training — modality & instant dropout (Iteration 4)](#step-10--robustness-training-iteration-4)
11. [Conformal prediction — calibrated uncertainty](#step-11--conformal-prediction)
12. [Config, builder, Optuna — how it is wired and tuned](#step-12--config-builder-optuna)
13. [Honest findings](#step-13--honest-findings)

---

## Step 0 — The problem

**Intuition.** A robot drives around a building. Four sensors report at
different rates: WiFi signal strength (~1 Hz), an IMU (accelerometer/gyro,
~31 Hz), wheel odometry (~15 Hz), and a camera (~5 Hz). We want its position
`(x, y)` in metres. No single sensor is enough: WiFi gives a noisy *absolute*
fix, the others give *motion*. The job is to **fuse** them — and to keep working
when some sensors are missing or late.

**How it relates.** Everything below is machinery for that one sentence. The
encoders turn each sensor into a comparable form; the transformer fuses them;
conformal prediction puts an honest error bar on the answer.

**In the code.** Targets are GT `(x, y)` from `ground_truth.csv`; the data
lives in `data/async_collection/path_XX/`.

---

## Step 1 — Data — async sensor windows

**Intuition.** At the moment we want a position (a "ground-truth timestamp"),
each sensor has a recent *history*. We grab a fixed-size **window** of the most
recent readings from each. Fast sensors get longer windows (more readings per
second), slow sensors get short ones.

**How it relates.** Windows are the raw input to the encoders (Step 2). Fixing
the window size makes every sample the same shape, so we can batch them.

**Formula.** For modality *m* with rate *r* and window length *W*, a window
covers roughly `W / r` seconds. The window for a sample at time *t* is the last
*W* readings with timestamp ≤ *t* (front-padded with zeros if history is short).

**Worked example.** IMU runs at ~31 Hz with `W = 32` → the window is
`32 / 31 ≈ 1.0 s` of motion, shaped `(32, 9)` (32 timesteps × 9 features:
accel xyz, gyro xyz, roll/pitch/yaw). WiFi runs at ~1 Hz with `W = 1` → a
single 117-number scan, shaped `(1, 117)`.

| Modality | Rate | Window `W` | Shape | ≈ seconds |
|---|---|---|---|---|
| IMU | 31 Hz | 32 | (32, 9) | 1.0 |
| Odom | 15 Hz | 16 | (16, 7) | 1.0 |
| WiFi | 1 Hz | 1 | (1, 117) | 1.0 |
| Camera | 5 Hz | 2 (a pair) | (2, 3, 480, 640) | ~1.0 (stride 5) |

**In the code.** `src/pipeline/data/dataset.py` — `FusionDataset` builds one
sample per GT timestamp and caches every window as a tensor for fast access.
`FusionDataModule` (`datamodule.py`) makes train/val/test splits and shares
normalisation statistics from the training set only (no leakage).

---

## Step 2 — Encoders — raw window → 128-d token

**Intuition.** A WiFi scan (117 numbers) and an IMU window (32×9 numbers) look
nothing alike. We can't fuse them while they're in different "languages." Each
**encoder** is a small neural network that translates its modality's window
into the *same* shape: a single 128-number vector called a **token**. Think of
it as a 128-word summary every sensor is forced to write.

**How it relates.** Encoders are Stage A (they already existed before this
work). The fusion transformer never sees raw sensors — only their 128-d tokens.
That uniformity is what lets one transformer handle all modalities.

**Formula.** Each encoder is a function `f_m : window_m → ℝ¹²⁸`.

**Worked example.** `IMUCNN` takes `(32, 9)`, runs three 1-D convolutions
(32→64→128 channels), averages over time, and projects to 128 → output
`(128,)`. `Anchor2Vec` takes the 117-d WiFi scan, compares it to 64 learned
"anchor" fingerprints, and mixes their embeddings → output `(128,)`.

| Modality | Encoder | What it does |
|---|---|---|
| WiFi | `Anchor2Vec` | RSSI → similarity to 64 learned anchors → 128-d |
| IMU | `IMUCNN` | 1-D CNN over the 32-step window → 128-d |
| Odom | `OdomCNN` | 1-D CNN over the 16-step window → 128-d |
| Camera | `DPVOMotionEncoder` | frozen DPVO trunk + patch tracking → 128-d motion token |

**In the code.** `src/pipeline/encoders/`. The camera one is special: its heavy
DPVO trunk is *frozen*, so its `(64, 132)` per-patch tokens are computed once
and cached to disk; only its small `_MotionHead` trains inside fusion.

---

## Step 3 — The universal token

**Intuition.** A 128-d encoder output says *what* a sensor saw, but not *which*
sensor it was or *when*. The transformer needs all three. So we **add** two
more vectors to every token:

- a **modality embedding** — a learned 128-d vector, one per sensor type, that
  stamps "I am WiFi" / "I am IMU" onto the token;
- a **time encoding** — a 128-d vector derived from the token's age Δt, that
  stamps "I was observed 3.2 seconds ago."

The result is the **universal token**: every observation, from any sensor, at
any time, is one 128-d vector of the same kind.

**How it relates.** This is the central design choice of the whole pipeline.
Because every token is the same kind of object, *one* transformer handles any
mix of modalities and any number of time instants — nothing downstream needs to
know how many sensors there are.

**Formula.**

```
token = encoder_embedding(window)        # what was sensed   (Step 2)
      + modality_embedding[m]            # which sensor      (learned, per modality)
      + time_encoding(Δt)                # when             (Step 8)
```

with `Δt = t_observed − t_query` (seconds; 0 = now, negative = the past).

**Worked example.** Suppose (toy 4-d instead of 128-d) an IMU encoder outputs
`[0.5, -0.2, 0.1, 0.9]`, the IMU modality embedding is `[0.0, 0.1, 0.0, -0.1]`,
and the time encoding for Δt = 0 is `[0.2, 0.0, 0.0, 0.0]`. Then:

```
token = [0.5, -0.2, 0.1, 0.9] + [0.0, 0.1, 0.0, -0.1] + [0.2, 0.0, 0.0, 0.0]
      = [0.7, -0.1, 0.1, 0.8]
```

That single vector now encodes content + identity + timing.

**In the code.** `src/pipeline/fusion/transformer.py` — `FusionTransformer.encode_tokens()`.
`modality_emb` is an `nn.Parameter` of shape `(num_modalities, 128)`;
`time_enc` is the `ContinuousTimeEncoding` module (Step 8).

---

## Step 4 — Attention — the one primitive

You need this before Steps 5/7/9, because all three "attentions" are the same
operation used three ways.

**Intuition.** Attention lets one token *look at* a set of other tokens and pull
in a weighted blend of them — paying more attention to the relevant ones. Each
token asks a **query** ("what am I looking for?"); every token offers a **key**
("what do I match?") and a **value** ("what I'll contribute if you pick me").
Match queries against keys → weights → blend the values.

**How it relates.** "Self-attention" = the queries and keys come from the *same*
set (Steps 5, 7). "Cross-attention" = the query is a *separate* token, keys come
from the set (Step 9). Same maths, different wiring.

**Formula.** For query matrix `Q`, keys `K`, values `V` (each row a token, `d` =
dimension):

```
                     Q · Kᵀ
attention(Q,K,V) = softmax( ────── ) · V
                            √d
```

`softmax` turns a row of scores into positive weights that sum to 1.

**Worked example.** One query `q = [1, 0]` and two key/value tokens
`k₁ = [1, 0], v₁ = [10, 0]` and `k₂ = [0, 1], v₂ = [0, 20]`, with `d = 2`:

```
scores = [ q·k₁ , q·k₂ ] / √2 = [1, 0] / 1.414 = [0.71, 0.00]
weights = softmax([0.71, 0.00]) = [0.67, 0.33]
output  = 0.67·v₁ + 0.33·v₂ = 0.67·[10,0] + 0.33·[0,20] = [6.7, 6.6]
```

The query matched `k₁` better, so the output leans toward `v₁` — but still
mixes in some of `v₂`.

**In the code.** PyTorch's `nn.TransformerEncoderLayer` (self-attention) and
`nn.MultiheadAttention` (the cross-attention readout). "Multi-head" just runs
several of these in parallel on slices of the 128 dims and concatenates — it
lets the model attend to several things at once.

---

## Step 5 — Self-attention = cross-modal fusion (Iteration 1)

**Intuition.** Put the modality tokens of *one instant* into a set —
`{wifi, imu, odom, camera}` — and run self-attention. Each token looks at the
others and updates itself. The WiFi token (a noisy absolute fix) and the IMU
token (recent motion) exchange information. **That exchange is the fusion.**

**How it relates.** This was the first thing built (Iteration 1). It already
delivers the "dynamic" property — predict from whatever sensors are present —
once paired with the mask (Step 6) and dropout (Step 10).

**Formula.** With token set `X` (rows = tokens), one encoder layer is

```
X ← X + SelfAttention(X)        # tokens exchange information
X ← X + FeedForward(X)          # each token is refined individually
```

stacked `depth` times. A learned **CLS token** is added to the set; after the
layers, *its* final vector is the fused summary → an MLP maps it to `(x, y)`.

**Worked example.** Tokens (toy 2-d): `wifi = [9, 1]` (says "x≈9"),
`imu = [0, 5]` (says "moving +y"). Self-attention lets the CLS token attend to
both; if it weights them `[0.8, 0.2]` it reads
`0.8·[9,1] + 0.2·[0,5] = [7.2, 1.8]` — mostly the WiFi fix, nudged by motion.
The MLP turns that into a position.

**In the code.** `FusionTransformer.forward()` with `n_instants = 1`. Result:
**≈ 0.43 m** mean error on the simulation test set.

---

## Step 6 — The padding mask

**Intuition.** If a sensor is missing for a sample, its token slot still exists
(fixed shapes for batching) — but we must tell attention to **ignore** it. The
mask is a list of booleans: `True` = "this slot is empty, do not attend to it."

**How it relates.** The mask is *the* mechanism for "works with any subset of
sensors." Drop WiFi? Set its mask entry `True`; attention skips it; everything
else runs unchanged. No architectural branch per sensor combination.

**Formula.** Before `softmax`, masked scores are set to `−∞`, so their weight
becomes `e^{−∞} = 0`:

```
score_j ← −∞   if token j is masked
weights = softmax(scores)        # masked tokens get weight 0
```

**Worked example.** Three tokens, scores `[2.0, 1.0, 0.5]`, but token 3 is
masked:

```
masked scores = [2.0, 1.0, −∞]
weights = softmax([2.0, 1.0, −∞]) = [0.73, 0.27, 0.00]
```

Token 3 contributes nothing.

**Edge case (a real bug that was found and fixed).** If *every* token in a row
is masked, all scores are `−∞` and `softmax` divides by zero → `NaN`. Fix: the
CLS token is **never** masked, so every row always has ≥ 1 live token. The
cross-attention readout (Step 9) attends to `[CLS] + tokens` for the same
reason.

**In the code.** `src_key_padding_mask` in `FusionTransformer.forward()`.

---

## Step 7 — Temporal self-attention (Iteration 2)

**Intuition.** One instant tells you where you *are*; several recent instants
tell you how you *got here*. Feed the transformer the modality tokens of the
last **K** instants at once. The same self-attention layers now also let
instant *k* attend to instant *k−3* — so "temporal attention" is not new code,
it is self-attention the moment the token set spans more than one instant.

**How it relates.** It reuses Steps 3–6 unchanged; only the *number of tokens*
grows from `M` to `K × M`. Each token still carries its time encoding (Step 8),
so the model knows which token is recent and which is old.

**Formula.** Token set size goes from `M` to `K · M` (+1 for CLS). Attention
cost is `O((K·M)²)` — with `K = 8, M = 4` that is `32² ≈ 1024`, trivial.

**Worked example.** `K = 8` instants spaced `stride = 9` GT rows apart. GT is
~10 Hz, so the window spans `8 × 9 / 10 ≈ 7 s` of history — about 8 WiFi scans
and 8 IMU windows, each tagged with its own Δt (0 s, −0.9 s, −1.8 s, …).

**The honest twist (this is "how it was built").** Iteration 2, as first built,
**made things worse** — 0.69 m vs 0.43 m. Diagnosis: WiFi already nearly solves
the fresh-data problem, so the 7 extra instants gave the model no new *absolute*
signal — just more parameters to overfit. This was not hidden; it was measured,
analysed, and fixed in Iteration 4 (Step 10). Temporal fusion's real value
turned out to be **robustness to stale sensors**, not fresh-data accuracy.

**In the code.** `FusionTrainer._build_temporal_index()` finds, for each
sample, the K recent instants within the same path; `_batch()` assembles the
`(B, K, …)` tensors.

---

## Step 8 — Continuous-time encoding

**Intuition.** Sensors are *asynchronous* — a WiFi scan might be 3.2 s old while
an IMU window is 0.1 s old. The model must know each token's age. We can't just
feed the raw number "3.2"; neural nets prefer a smooth, multi-scale code. So we
turn Δt into a vector using sine waves of many frequencies — fast waves resolve
small time differences, slow waves resolve large ones.

**How it relates.** This is the third additive part of the universal token
(Step 3). It is also where the idea behind **mTAN** (continuous-time attention)
lives — folded into a token feature instead of being a separate pipeline stage.

**Formula.** For Δt (seconds) and a bank of `n` periods `Pᵢ`:

```
ωᵢ = 2π / Pᵢ
features(Δt) = [ sin(ω₁Δt), …, sin(ωₙΔt), cos(ω₁Δt), …, cos(ωₙΔt) ]
time_encoding(Δt) = Linear( features(Δt) )      # → 128-d
```

The periods are spaced geometrically from 0.05 s to 120 s (32 of them).

**Worked example.** Take one period `P = 4 s` → `ω = 2π/4 ≈ 1.57 rad/s`.

```
Δt =  0.0 s →  sin(0)    = 0.00 ,  cos(0)    = 1.00
Δt = −1.0 s →  sin(−1.57)= −1.00,  cos(−1.57)= 0.00
Δt = −2.0 s →  sin(−3.14)= 0.00 ,  cos(−3.14)= −1.00
```

Different ages → distinctly different `(sin, cos)` pairs; nearby ages → nearby
pairs. With 32 periods at once, every Δt in the useful range gets a unique,
smooth fingerprint, which a `Linear` layer maps into the 128-d token space.

**In the code.** `ContinuousTimeEncoding` in `transformer.py`.

---

## Step 9 — Cross-attention readout (Iteration 3)

**Intuition.** After the encoder layers, the token set is a contextualised pile
of information. We need to *extract one answer* from it: the position at a
chosen time τ. We use a special **PositionQuery** token — it does not carry
sensor data, it carries the *question* "where am I at time τ?" — and let it
**cross-attend** to the whole token set. The query is not part of the set; that
is what makes this cross-attention rather than self-attention.

**How it relates.** A CLS token (Step 5) can summarise the set, but it cannot be
*parameterised by a continuous time τ*. The PositionQuery can: add
`time_encoding(τ)` to it and you can ask for the position *between* observations
or just *past* the last one — the asynchronous capability.

**Formula.**

```
query = PositionQuery + time_encoding(τ)         # the question, at time τ
readout = CrossAttention( Q = query,  K = V = token_set )
(x, y)  = MLP(readout)
```

**Worked example.** Observations exist at Δt = 0 s and Δt = −1 s, but you want
the position at τ = −0.5 s (between them). Build `query = PositionQuery +
time_encoding(−0.5)`. Its time stamp sits *between* the two observation tokens,
so cross-attention weights them roughly equally and the MLP interpolates a
position. During training τ is randomised over the K instants so the query
genuinely learns to route by time.

**In the code.** `FusionTransformer.forward(..., readout="query")` —
`PositionQuery` is a learned parameter, `cross_attn` is an `nn.MultiheadAttention`,
and it attends to `[CLS] + tokens` (Step 6 keeps it NaN-safe).

---

## Step 10 — Robustness training (Iteration 4)

**Intuition.** A model is only robust to situations it *practised*. If it always
trains with all 4 sensors fresh, it collapses the first time one is missing. So
during training we deliberately **hide** sensors at random, and force the model
to cope. Two kinds of hiding:

- **modality dropout** — hide a whole sensor (all its instants). Teaches
  "predict from any subset."
- **instant dropout** — hide individual (instant, sensor) tokens. Teaches "cope
  with intermittent / late / stale observations" — *and* it regularises the
  large temporal token set, which is what fixed Iteration 2's overfitting.

**How it relates.** This is why the system is **dynamic and asynchronous**. The
robustness is not in the architecture — it is in the training distribution. The
mask (Step 6) is the tool; dropout is how we exercise it.

**Formula.** For each sample, each modality *m* (and each instant *k*):

```
hide modality m   with probability  p_mod        # whole sensor gone
hide token (k,m)  with probability  p_inst        # one observation gone
constraint: at least one token survives per sample (else restore the anchor)
```

**Worked example.** `p_mod = 0.16`, `p_inst = 0.45`, modalities
`[imu, odom, wifi, camera]`. One training sample might roll:

```
modality dropout → hide camera        → camera tokens all masked
instant dropout  → hide wifi at 3 of 8 instants, imu at 2 of 8, …
```

The model must still predict `(x, y)` from imu + odom + (partial) wifi. Over
millions of such randomised samples it learns every degradation mode.

**The staleness payoff.** Evaluate by masking WiFi at the *N most-recent*
instants (it went stale N seconds ago):

| Stale WiFi | Single-instant model | Temporal model |
|---|---|---|
| 0 s (fresh) | 0.43 m | 0.44 m |
| ~2 s | **~4 m** (cliff) | 0.8 m |
| ~4 s | ~4 m | 1.8 m |

A single-instant model only ever sees *now*; the instant WiFi is not fresh it
falls to its no-WiFi error — a cliff. The temporal model dead-reckons from the
last good fix — a graceful slope. **That is what temporal fusion is for.**

**In the code.** `FusionTrainer._apply_dropout()`, `evaluate_staleness()`.

---

## Step 11 — Conformal prediction

**Intuition.** A point `(x, y)` is not enough — we want "the robot is here,
**± r metres**, with 90% confidence." Conformal prediction gets that *r* with no
assumption about the error distribution: it just measures how wrong the model
was on held-out data and takes a high percentile.

**How it relates.** This is Stage E — a thin wrapper around the *already-trained*
model. It does not change the model; it calibrates a number.

**Formula.** On a calibration set, compute each residual
`rᵢ = ‖prediction_i − target_i‖`. For target coverage `1 − α`:

```
radius = the q-th quantile of {r₁, …, rₙ},   q = ⌈(n+1)(1−α)⌉ / n
```

The `(n+1)` correction makes the guarantee exact for finite *n*.

**Worked example.** `α = 0.1` (want 90% coverage), `n = 9` calibration
residuals sorted: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2.0] m`.

```
q = ⌈(9+1)(0.9)⌉ / 9 = ⌈9⌉ / 9 = 9/9 = 1.0   → radius = max = 2.0 m
```

(With more calibration points `q < 1` and the radius is a non-max percentile —
e.g. `n = 2000` gives `q ≈ 0.9005`.) Any new prediction then comes with a
2.0 m disc; ~90% of true positions fall inside.

**The catch (a real correction made in the notebook).** The guarantee holds only
if calibration and test data are **exchangeable** (same distribution).
Calibrating on `val` paths and testing on `test` paths — different physical
trajectories — *under-covered* (79%). Fixing it to random halves of one pool
restored ~90–92%.

**In the code.** `src/pipeline/uncertainty/conformal.py` — `ConformalPosition`.

---

## Step 12 — Config, builder, Optuna

**Intuition.** A research pipeline has dozens of knobs (model depth, dropout
rates, learning rate, window size…). Scattering them across scripts makes
experiments irreproducible. So **every knob lives in one YAML file**, one helper
module **wires** the pipeline from it, and an automated search **tunes** it.

**How it relates.** This is the plumbing that keeps Steps 1–11 reproducible and
lets the notebook, the smoke harness, and the Optuna search all build the
*identical* pipeline.

**The three pieces.**

- `configs/stage_c/fusion.yaml` — every hyperparameter (data, model, training,
  temporal, conformal, Optuna search space).
- `src/pipeline/fusion/builder.py` — `load_config → build_datamodule →
  build_encoders → extract_vision_tokens → build_model → build_trainer`. The
  notebook calls these and writes almost no logic itself.
- `scripts/optuna_fusion.py` — **Optuna** hyperparameter search.

**How Optuna works.** It repeatedly proposes a set of hyperparameters (a
"trial"), trains a short model with them, and reads back the validation MAE. A
**TPE sampler** models which regions of the search space produced low MAE and
samples the next trial there — it learns where good configs live instead of
guessing randomly.

**Formula / procedure.**

```
for trial = 1 … N:
    θ ← sampler.suggest()                  # depth, lr, dropouts, K, …
    model ← train(θ, short budget)
    score ← best validation MAE
    sampler.observe(θ, score)
best θ* = argmin score
```

**Worked example (from the actual run).** 20 trials × 30 epochs. Trial values
ranged 0.41–1.18 m; the search converged on **trial 2 = 0.409 m**:

```
depth 6 · heads 4 · ff_mult 4 · lr 1.3e-3
modality_dropout 0.16 · instant_dropout 0.47 · K = 8 · stride 9
```

Those values are now the defaults in `fusion.yaml`. (One judgement call:
Optuna searched at a 30-epoch budget; for the full 90-epoch run `dropout` was
kept at 0.1 instead of the search's 0.035, because a longer run needs slightly
more regularisation.)

**In the code.** Run it with `python scripts/optuna_fusion.py`; results land in
`runs/optuna_fusion/best.json` and `trials.csv`.

---

## Step 13 — Honest findings

How the pieces actually performed on the simulation data — including what did
*not* work, because that is part of understanding the build:

1. **Self-attention fusion works and is strong.** Single-instant fusion reaches
   ≈ 0.43 m. But it is mostly a *WiFi localiser* — WiFi RSSI is a direct
   absolute observation; IMU/Odom/Vision add only a few centimetres on fresh
   data.

2. **Temporal fusion first regressed (0.69 m), then was fixed.** Extra instants
   are not extra *absolute* information when WiFi is already fresh — just extra
   capacity to overfit. **Per-instant dropout** (Step 10) regularised it back to
   ≈ 0.44 m *and* unlocked its real purpose.

3. **Temporal fusion's real job is staleness robustness.** Under stale WiFi a
   single-instant model jumps to ~4 m (a cliff); the temporal model degrades
   gracefully (0.8 m at 2 s, 1.8 m at 4 s). That is the dynamic/asynchronous
   payoff.

4. **Vision (DPVO) plugs in as a 4th modality with one config line.** Alone it
   reaches ≈ 2.9 m — the best of the non-WiFi sensors — and it confirms the
   set-transformer is genuinely modality-count-agnostic.

5. **`drop:wifi` stays ~4 m and that is correct.** With no absolute reference at
   *any* instant, the position is genuinely unobservable — fusion cannot invent
   an anchor that was never measured. Reporting this honestly matters.

6. **Conformal coverage needs exchangeability.** ~90–92% when calibration and
   inference data share a distribution; it under-covers across different
   building regions. The error bar is only as honest as its calibration set.

---

## How it was built (the iterative method)

The pipeline was developed autonomously in iterations, each one **built →
smoke-tested → profiled → analysed → debugged** before the next:

| Iteration | Added | Outcome |
|---|---|---|
| 1 | self-attention (cross-modal fusion) | 0.43 m — works |
| 2 | temporal self-attention (K instants) | 0.69 m — **regressed**, diagnosed as overfitting |
| 3 | cross-attention readout (PositionQuery) | query-at-any-time; a NaN bug found & fixed |
| 4 | modality + instant dropout, staleness eval | 0.44 m + graceful staleness — fixes Iter 2 |
| — | vision (DPVO) as 4th modality | works; vision alone ≈ 2.9 m |
| — | config + builder + Optuna search | reproducible & tuned (best 0.409 m) |

## File map

| File | Role |
|---|---|
| `src/pipeline/data/dataset.py`, `datamodule.py` | async windows, splits, normalisation |
| `src/pipeline/encoders/` | Stage-A per-modality encoders → 128-d tokens |
| `src/pipeline/fusion/transformer.py` | `FusionTransformer` + `ContinuousTimeEncoding` |
| `src/pipeline/fusion/builder.py` | wiring from the config |
| `src/pipeline/training/fusion_trainer.py` | `FusionTrainer` — dropout, temporal index, eval |
| `src/pipeline/uncertainty/conformal.py` | `ConformalPosition` |
| `configs/stage_c/fusion.yaml` | every hyperparameter |
| `scripts/_smoke_fusion.py` | 5-phase smoke / profile harness |
| `scripts/optuna_fusion.py` | hyperparameter search |
| `notebooks/fusion_workbench.ipynb` | runnable end-to-end demo |
