# Plan 17 — Phase B full-data retrain: CNN1D + LSTM-attn vs incumbent (settle the bake-off)

> RESULT_16 ran a 4-architecture bake-off on a 2-path 10 % Webots
> subset. Three candidates (LSTM-attn, TCN, CNN1D) all beat the
> incumbent FusionTransformer by 24–34 % at 1/3 the params, but
> none cleared the smoothness gate (r > 0.20). Engineer's
> recommendation, accepted: **full-data retrain of the two strongest
> candidates** (CNN1D test-leader 1.261, LSTM-attn val-tied 0.978)
> at the Phase-B-winner config (K=4 + 4-mod + B=128). Decision:
> which one — if either — beats the incumbent's full-data RESULT_13
> number (val 0.394 / test 0.417)?

## Hypothesis

The incumbent's full-data RESULT_13 win (0.394 / 0.417) may be a
**data-scale × parameter-budget** confound rather than an
architectural advantage:

- On the 2-path subset, the incumbent (1.55 M params) likely overfits
  (val 1.493 m at 30 epochs); the smaller candidates (~0.51–0.57 M)
  regularise implicitly via their capacity ceiling and land at val
  ~1.0 m.
- On full data (8 542 train windows vs 1 507 in the subset), the
  incumbent has the capacity to fit; the smaller candidates may
  either also scale and stay competitive, OR saturate at higher
  MAE because they can't model the richer signal.

Three outcomes:
- **(α''') CNN1D and/or LSTM-attn beat incumbent's RESULT_13 0.417
  on full Webots data**: the bake-off finding holds at scale. New
  Phase B winner. RESULT_17's winner becomes the Phase B
  paper-claim model. PLAN_18 = full ablations on the new winner
  (analog of RESULT_14 but on the bake-off winner).
- **(β''') Candidates tie or come within 5 % of incumbent**:
  candidates are competitive and **paper-defensibly cheaper**
  (1/3 params, similar / slightly worse fresh accuracy). Paper
  reports both with the comp-cost trade-off explicit. PLAN_18
  proceeds with the incumbent (which is the safer choice for
  remaining Phase C work) but the methods section cites the
  bake-off honestly.
- **(γ''') Candidates regress at scale (test > 0.50 m)**: the
  subset advantage was a regularisation artifact; incumbent
  decisively wins on full data. Paper claim sticks with
  incumbent; the bake-off documents "we tried 4 architectures
  and the data-rich regime decisively favours the transformer."

This is one focused experiment: validate the bake-off subset
finding at production data scale. Two candidates × one config × full
data = a clean apples-to-apples comparison against RESULT_13.

## Steps

### Step 0 — Confirm candidate scaffolds + config (5 min)

The 3 candidates were committed in RESULT_16 under
`src/pipeline/fusion/{lstm_attn,tcn,cnn1d_instants}.py`. Confirm
they import cleanly + the builder's factory route picks them up
when the config names them. If RESULT_16 used a separate runner
script (`scripts/_eval_webots_bakeoff.py` or similar), this iter
can write `scripts/_train_webots_4mod_full_<arch>.py` clones from
RESULT_13's training wrapper, or — preferred — pass an architecture
name to the existing RESULT_13 wrapper via a `--arch` flag.

Pick the minimum-surface implementation. **Acceptance**: each
candidate (`cnn1d`, `lstm_attn`) trains on a 1-epoch synthetic
fwd+bwd at the full-data tensor shape (B=128, K=4, 4-mod) without
NaN.

### Step 1 — Pre-test gate per candidate (5 min × 2 = ~10 min)

Each candidate: 5 epochs on 10 % full train, val MAE drops ≥ 10 %
OR clear descent. RESULT_16 already demonstrated descent for both
candidates on the smaller subset — this is the formal pre-test
gate restated at full-data tensor sizes (window count, batch).

If a pre-test FAILS at full-data scale (e.g. lr is wrong for the
much larger batch×K combination), STOP that candidate and document.

### Step 2 — Full training × 2 (~25–35 min each, sequential)

Same protocol as RESULT_13/14:
- 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5), B=128, K=4,
  4-mod, `instant_dropout=0.45`, `modality_dropout=0.4`.
- Modalities: WiFi (Anchor2Vec) + IMU (IMUCNN) + Camera
  (DPVOMotion-P-A) + Odom (OdomCNN-P-B).
- lr=1.3e-3 (the RESULT_14 default).

Train **CNN1D** first (test-leader on subset), then **LSTM-attn**
(val-tied with CNN1D on subset; potentially better at the
sequential-motion inductive bias).

**Memory budget**: candidates are ~1/3 of incumbent's params;
RESULT_14 peak was 466 MB at K=4 4-mod B=128. Candidates should
land at ~150-200 MB. Report.

**Acceptance**: both trainings complete; val + test MAE recorded.

### Step 3 — Compare against RESULT_13's incumbent

| config | params | val MAE | test MAE | smoothness r | latency b=1 (ms) | source |
|---|---|---|---|---|---|---|
| Incumbent (FusionTransformer) | 1.55 M | 0.394 | 0.417 | 0.039 | 6.41 | RESULT_13/14 |
| **CNN1D** (this iter) | ~0.51 M | ? | ? | ? | ? | this iter |
| **LSTM-attn** (this iter) | ~0.57 M | ? | ? | ? | ? | this iter |

**Acceptance** (raw-weighted per amended-rubric correction #3):
- α''' (new winner): the better of {CNN1D, LSTM-attn} beats
  incumbent test 0.417 by ≥ 5 % (i.e. test ≤ 0.396).
- β''' (competitive): within 5 % of 0.417 (test ≤ 0.438) →
  documented trade-off.
- γ''' (regresses): test > 0.500 → bake-off subset advantage
  was an artifact.

### Step 4 — Per-modality subset eval (key rows) — only on the winner

If α''' or β''' fires, run subset eval on the leading candidate
ONLY (no need to run 16 subsets on both). Key rows:

| subset | val | test | comment |
|---|---|---|---|
| only:wifi | ? | ? | (RESULT_14: 0.492/0.489) |
| wifi+camera | ? | ? | (RESULT_14: 0.492/0.482 — drop-Odom on full was BEST) |
| **full 4-mod** | ? | ? | the headline |

The "drop-Odom is best" pattern from RESULT_14 — does it persist
under the new architecture? If yes, the architectural choice
doesn't change the modality-set conclusion.

### Step 5 — Per-trajectory smoothness on test paths 15/16/17

Median Pearson r between ‖Δpredᵢ‖ and ‖Δgtᵢ‖. RESULT_16's
subset smoothness was max 0.085 (LSTM-attn). At full data, with
more temporal coverage, the CNN1D's temporal locality bias might
finally cross the r > 0.20 gate. **If it does, that's the
load-bearing finding for the run-2 paper's smoothness claim** —
the architecture lever (not just the loss) finally moves the
smoothness debt.

Save per-trajectory plots under
`runs/overnight/run2_iter_17/test_paths/`.

### Step 6 — Latency probe (criterion (e))

For the winning candidate (if α''' or β'''): per-sample b=1
latency + per-sample b=32 throughput. The candidates are smaller
than the incumbent; latency should be ≤ RESULT_14's 6.41 ms b=1 /
0.20 ms b=32 (gate < 100 ms cleared).

### Step 7 — Decision + PLAN_18 recommendation

Three-sentence verdict:
- Outcome (α''' / β''' / γ'''); quote both candidate's test MAE
  vs incumbent.
- Does the smoothness gate (r > 0.20) clear for the winner? If yes
  — paper-strength finding.
- PLAN_18 recommendation:
  - **α'''**: PLAN_18 = full ablations on the new winner (subset
    eval + staleness sweep, mirroring RESULT_14). Phase C
    follow-ups (conformal, MSILN re-run with audit-winner
    Anchor2Vec) slide to PLAN_19+.
  - **β'''**: PLAN_18 = Phase C continuation (the bake-off has
    yielded its comparison; document and move on). PLAN_18 =
    MSILN re-run with Anchor2Vec (the divergence the engineer
    flagged in RESULT_15).
  - **γ'''**: PLAN_18 = Phase C continuation (incumbent stands).

## Sources

- RESULT_16: bake-off subset results (CNN1D test 1.261, LSTM-attn
  val 0.978, smoothness < 0.20 across all 4).
- RESULT_13/14: incumbent on full data (val 0.394 / test 0.417 /
  smoothness 0.039).
- Third-party review 2026-05-26 ~04:10 local: bake-off requirement.
- `src/pipeline/fusion/{lstm_attn,tcn,cnn1d_instants}.py` (committed
  by RESULT_16).
- `src/pipeline/training/fusion_trainer.py` (restored RESULT_06,
  arch-agnostic).

## What to report back

In `handoff/results/RESULT_17_phase-b-full-data-retrain-cnn1d-lstm.md`:

1. **Step 0** — implementation choice (--arch flag or runner clones).
2. **Step 1** — pre-test gates × 2.
3. **Step 2** — training summary per candidate (loss curves, val
   + test, params, wall time).
4. **Step 3** — comparison table vs RESULT_13 incumbent; outcome
   label.
5. **Step 4** — subset eval (winner only).
6. **Step 5** — per-trajectory smoothness; **does any candidate
   clear r > 0.20 at full-data scale?**
7. **Step 6** — latency.
8. **Step 7** — verdict + PLAN_18 recommendation.
9. **One open question** for scientist.

## Reversibility

- Step 0: no permanent changes (candidate files committed in
  RESULT_16).
- Step 2: throwaway checkpoints under `runs/overnight/run2_iter_17/`.
- Steps 3–7: documentation.

Files committed: RESULT_17, optionally `--arch` plumbing in the
training wrapper script.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 90 min.
- Step 0: 5 min.
- Step 1: 10 min.
- Step 2: 25 + 35 min (CNN1D simpler than LSTM-attn; sequential
  training).
- Step 3: 5 min.
- Step 4: 5 min.
- Step 5: 5 min.
- Step 6: 5 min.
- Step 7: 5 min.

If overrun: drop LSTM-attn training (Step 2 second pass) and ship
RESULT_17 with CNN1D-only on full data. CNN1D had the best test
MAE on the subset, so it's the more important number; LSTM-attn's
val-tie can be inferred or deferred.

If γ''' fires (both regress), PLAN_18 is Phase C continuation; the
bake-off becomes a documented "tried, regressed at scale" methods-
section footnote rather than a paper claim shift.
