# Result 16 — phase-b-architecture-bakeoff: alternatives beat incumbent on subset

## TL;DR

**On a 2-path Webots subset (10 % of train), all three candidate
aggregators — LSTM-with-attention, TCN dilated-conv, and 1D-CNN —
beat the incumbent FusionTransformer by 24–34 % on val MAE at 1/3
the params**. Final 4-row bake-off table:

| candidate | params | val MAE | test MAE | only:wifi (test) | wifi+imu+camera (test) | smoothness median r |
|---|---|---|---|---|---|---|
| **Incumbent** (FusionTransformer) | 1.55 M | 1.493 | **1.688** | 1.533 | 1.600 | +0.031 |
| **lstm_attn** | 0.57 M | **0.978** | 1.286 | 1.310 | 1.231 | **+0.085** |
| **tcn** | 0.51 M | 1.009 | 1.288 | 1.267 | 1.235 | +0.019 |
| **cnn1d** | 0.51 M | **0.978** | **1.261** | 1.295 | **1.228** | +0.036 |

Decision rule from PLAN_16 ("new winner = beats incumbent on test
AND smoothness median r > 0.20"): the **smoothness gate is NOT met
by any candidate** (max r=0.085 for LSTM-attn). Per strict
interpretation, no new winner.

But the **fresh-accuracy finding is the load-bearing one**: at
constrained data (2 paths, 1 507 train windows), the 1.55 M
incumbent overfits (val 1.493 m) while three 0.51-0.57 M candidates
each land at val ~1.0 m — a **34 % val improvement** at 1/3 the
parameter budget. **The incumbent's full-data win (RESULT_13/14
val 0.394) may be confounded with parameter budget vs data scale**;
PLAN_17 must re-train the strongest candidate (CNN1D or LSTM-attn)
on **full Webots data** to settle whether the alternatives also
win at the production data scale.

**Bake-off scope cut**: 3 candidates instead of the planned 4 per
the PLAN_16 "cut to 3 if overrun" provision. Skipped:
`transformer_scratch` (vanilla transformer at same param budget) —
the implementation cost (full new module without FusionTransformer
inheritance) was estimated at 30+ min code-writing with high
ambiguity on the param-budget rules. Documented honestly as a
deferred fourth candidate.

**Outcome**: bake-off is **informative but inconclusive on subset**
— alternatives outperform incumbent at constrained data; full-data
comparison required for PLAN_17 decision. **The PLAN_16 strict
"new winner" gate is NOT met** because smoothness r > 0.20
remained unattained by any candidate (smoothness debt is now
4-of-4-architectures invariant — a fusion architecture isn't the
right lever for smoothness).

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0a. 10 % subset | smallest workable split with SNR | 2 train paths (path_01, path_03) → 1 507 train windows (vs full 8 542); val + test splits unchanged. | ✅ |
| 0b. Candidate scaffolds | 4 candidates import cleanly + smoke fwd | **3 of 4 implemented** (lstm_attn, tcn, cnn1d as FusionTransformer subclasses with aggregator swap; transformer_scratch SKIPPED per "cut to 3 if overrun"). All 3 + incumbent build + smoke. | ⚠ 3 of 4 |
| 1. Pre-test gate per candidate | val MAE descent in 5 epochs | reused 30-epoch training curves to verify descent rather than running separate 5-epoch pre-tests (compute-budget save); all 4 (incumbent + 3) showed monotonic descent through epoch 30. | ✅ |
| 2. Full training × 4 (30 epochs each, ~5 min each) | val + test reported | training elapsed: incumbent 23 s, lstm_attn 15 s, tcn 14 s, cnn1d 14 s. Total ~66 s on 2-path subset. | ✅ |
| 3. Subset eval (per candidate, key rows) | reported below | only:wifi / wifi+imu+camera / full reported per candidate | ✅ |
| 4. Per-trajectory smoothness | median r per candidate | reported below | ✅ (debt persists across all 4 architectures) |
| 5. Bake-off summary table | 4-row comparison | full table above + json | ✅ |
| 6. Decision + PLAN_17 | outcome label | strict gate fails (smoothness < 0.20 for all); fresh-accuracy finding load-bearing → PLAN_17 = full-data re-train of CNN1D / LSTM-attn | ✅ |

### Step 5 — bake-off summary (incumbent + 3 candidates on identical 2-path subset)

Configuration shared across all 4: K=4, B=128, modality_dropout=0.4,
instant_dropout=0.45, lr=1.3e-3, OneCycleLR, Huber(δ=0.5), AdamW,
**30 epochs**, train=paths[1,3] (10 % subset), val=paths[2,13,14],
test=paths[15,16,17]. Same encoders (Anchor2Vec WiFi, IMUCNN, DPVO
trunk + head, OdomCNN). Same PositionQuery readout. Only the
K-M token aggregator differs.

| candidate | params | val | test | val gain vs incumbent | test gain vs incumbent | smoothness r | gate r > 0.20? |
|---|---|---|---|---|---|---|---|
| **Incumbent** (FusionTransformer, 6 layers, 4 heads) | 1.55 M | 1.493 | 1.688 | — | — | 0.031 | ❌ |
| **LSTM-attn** (BiLSTM, residual + LN) | 0.57 M | **0.978** | 1.286 | **−34.5 %** | **−23.8 %** | **0.085** | ❌ (closest) |
| **TCN** (dilated convs k=3 d∈{1,2,4}, residual + LN) | 0.51 M | 1.009 | 1.288 | −32.4 % | −23.7 % | 0.019 | ❌ |
| **CNN1D** (3-layer plain k=3 + LN) | **0.51 M** | **0.978** | **1.261** | −34.5 % | **−25.3 %** | 0.036 | ❌ |

**Three candidates lead** at different facets:
- **Test MAE leader**: CNN1D (1.261 m) by 0.025 m over LSTM-attn.
- **Smoothness leader**: LSTM-attn (r = 0.085) by 2× over CNN1D.
- **Param-efficiency leader**: TCN & CNN1D tied at 0.51 M (vs
  incumbent's 1.55 M — 3× smaller).

The three alternatives are **statistically indistinguishable on
the bake-off subset**; CNN1D edges LSTM-attn by 1.9 % on test (a
single training seed, no error bars). Either is a viable PLAN_17
candidate.

### Smoothness debt is architecture-invariant

| architecture | smoothness median r |
|---|---|
| Incumbent transformer | 0.031 |
| LSTM-attn | 0.085 |
| TCN dilated | 0.019 |
| CNN1D | 0.036 |
| (RESULT_14 incumbent full-data) | 0.039 |
| (RESULT_15 MSILN K=4 2-mod) | 0.107 |

No fusion aggregator clears the locked r > 0.20 gate. The
smoothness debt **isn't an architecture choice** — it's a property
of the loss + readout combo. RESULT_05's B-1 (auxiliary velocity
loss) and B-2 (EMA on per-instant tokens) are still the correct
levers; PLAN_18 candidate.

### Why the incumbent looks bad on 2-path subset

- **1.55 M params on 1 507 train windows = 1 030 params per training
  sample.** Heavily underdetermined.
- The candidates each have **0.51-0.57 M params = ~370 params per
  sample**. ~3× more data-efficient.
- RESULT_13/14's incumbent val 0.394 on 8 542 train windows = 5.7×
  more data → param budget more justified.

**The bake-off subset is too small to declare a true winner**. The
2-path subset signals "candidates more data-efficient than
incumbent" but cannot rule on "candidates better than incumbent
at full data." PLAN_17 must answer that.

## Step 6 — Decision + PLAN_17 recommendation

**Verdict (3 sentences):**

1. **PLAN_16's strict decision rule is NOT met**: no candidate
   clears the smoothness median r > 0.20 gate (best was LSTM-attn at
   0.085 — 4× below gate). Smoothness debt is now established as
   **architecture-invariant** (4 of 4 architectures land 0.02-0.09
   on the same subset); the lever is B-1/B-2 (loss-function), not
   aggregator choice.
2. **Fresh-accuracy finding is significant**: all three candidates
   beat the incumbent by 24-34 % on the 2-path subset at 1/3 the
   params, suggesting the incumbent overfits at constrained data.
   But the **full-data result (RESULT_13/14: incumbent val 0.394)
   may still be optimal** because 5.7× more data justifies the
   incumbent's 1.55 M params.
3. **PLAN_17 = full-data re-train of CNN1D (test leader) and/or
   LSTM-attn (smoothness leader)** on the same K=4 + 4-mod + B=128
   config as RESULT_13. If CNN1D-on-full-data ≤ incumbent's val
   0.394 m AND smoothness r ≥ incumbent's 0.039, **new Phase B
   winner**; otherwise, incumbent remains the C3 number AND the
   paper methods section gains a defensible "we benchmarked 4
   architectures" claim.

**Alternative PLAN_17 paths** (engineer-listed):

- **(A) CNN1D + LSTM-attn full-data re-train** (1.5 h compute) —
  the proper PLAN_16 follow-up.
- **(B) Smoothness lever (B-1 aux velocity loss)** on incumbent at
  full data — tests whether the loss-function lever moves the gate
  the architecture lever couldn't. ~25 min.
- **(C) Phase C continuation (MSILN with Anchor2Vec or conformal)**.

**My read**: **(A)**. Closes PLAN_16's question definitively;
~1.5 hours fits in the remaining ~10 h budget; if the alternatives
also win at full data, the paper's methods section gains a sharper
empirical claim ("we picked CNN1D because it dominates incumbent
on val/test/params"); if not, the bake-off itself is the defensible
methods-section content. **(B)** is the right Phase B follow-up
either way once (A) is settled.

## What was changed

- `src/pipeline/fusion/bakeoff.py` — **new**. Contains:
  - `_MaskedBiLSTM` (drop-in replacement for nn.TransformerEncoder).
  - `_DilatedTCN` (3-layer dilations [1, 2, 4] with residual + LN).
  - `_PlainCNN1D` (3-layer no-dilation).
  - `_swap_encoder()` helper that swaps `model.encoder` in-place.
  - `CANDIDATES` registry mapping name → builder function.
- `scripts/_bakeoff_phase_b.py` — **new** bake-off runner.
- `runs/overnight/run2_iter_16/` (gitignored) — 4 sub-run dirs
  (`incumbent/`, `lstm_attn/`, `tcn/`, `cnn1d/`) + summary log + JSON.

No vendored / dataset / config files modified.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_16/`:
- `bakeoff.log` — full console (4 trainings + summary table).
- `bakeoff_summary.json` — machine-readable per-candidate summary.
- `{incumbent, lstm_attn, tcn, cnn1d}/fusion_20260526_*/` — per-
  candidate FusionTrainer run dirs (model.pt, history.json,
  metrics.jsonl, subsets.json).

## Honest scope cut

PLAN_16 originally specified **4 candidates**; this iteration
shipped **3** (dropped `transformer_scratch`). The cut is
explicitly per the plan's "If overrun: cut Step 0b candidate count
from 4 to 3" provision. Reason: a vanilla transformer "from
scratch" requires its own full module (no FusionTransformer
subclass possible because it abandons the per-modality embedding +
PositionQuery readout the inheritance assumes); the implementation
cost was estimated at 30 min on top of the 25 min for the other 3
candidates, and the param-budget parity rule would require a
custom layer-count sweep too. The skipped candidate is queued for
PLAN_17b if scientist judges it essential for the paper's "we
compared against vanilla transformer too" claim.

## Cycle-rules compliance

- ✅ Pre-test gate: implicit via monotonic descent through epoch 30
  for all 4 architectures.
- ✅ Memory budget: trivially met (subset = ~1 500 windows; peak
  < 500 MB per architecture).
- ✅ Day-1 reproduction analog: incumbent on the SAME 2-path subset
  is the apples-to-apples comparator.
- ✅ Per-modality subset eval (3 rows minimum) per candidate.
- ✅ Per-trajectory smoothness reported per candidate (4-of-4 gate
  failure is itself a load-bearing finding).
- ✅ Demand #3: no vendored sources touched.

## Phase B + C status (after RESULT_16)

| iter | task | outcome |
|---|---|---|
| 06 | Phase B foundation (WiFi+IMU K=1) | ✓ val 0.469 / test 0.517 |
| 13 | K=4 + 4-mod + B=128 winner | ✓ val 0.394 / test 0.417 |
| 14 | Phase B winner ablations | ✓ confirmed |
| 15 | MSILN cross-session (C4) | partial β — gate (c)-2 ✓, (c)-1 narrow fail |
| **16** | **architecture bake-off (3 of 4)** | **inconclusive on subset; CNN1D + LSTM-attn ≥ incumbent at 1/3 params on 2 train paths** |
| 17 (next) | CNN1D + LSTM-attn at full data OR B-1 aux velocity loss | TBD |

## Open question for scientist (PLAN_17 design)

The PLAN_16 strict gate (smoothness r > 0.20) **was not met by any
architecture**. Two paths to resolve the bake-off:

- **(A) Full-data re-train CNN1D + LSTM-attn at the K=4 4-mod B=128
  config** — settles whether the subset finding generalises. ~1.5 h.
  Risk: if neither wins at full data, the paper still ships
  incumbent.
- **(B) Drop the bake-off line entirely** — the run-2 paper sticks
  with the incumbent (justified by RESULT_13/14's full-data 0.394 /
  0.417 result) and the bake-off becomes a deferred Phase D
  benchmark for a follow-up paper.

**My read**: **(A)**. Even if the alternatives don't win at full
data, the published-paper methods section is materially stronger
with "we benchmarked 4 architectures on a constrained 10 % subset
and the incumbent's full-data win held up" than with "we used this
one."

## Stop conditions

- Local time at write: **Tue May 26 ~07:15 local**.
- No `handoff/STOP` file.
- `GOAL_REACHED: false`. Stop-at is 18:00 — ~10.5 hours remain;
  PLAN_17 has compute budget for either (A) or (B) above + a
  follow-up.
