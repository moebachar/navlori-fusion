# Plan 23 — Main results table: RoNIN canonical single-mod IMU (CNN1D + LSTM-attn vs ResNet1D 5.140 m reused)

> Per `handoff/SCIENTIST_NOTE_main-results-table.md`. RoNIN
> canonical is single-modality IMU. RESULT_07 already produced the
> SOTA cell (RoNIN ResNet1D **5.140 m raw ATE** on `list_test_unseen.txt`,
> reproducing the paper number to 0 %). This iteration fills the
> `our CNN1D` and `our LSTM-attn` cells.

## Hypothesis

The fusion architectures (CNN1D, LSTM-attn) are designed for the
(K-instant × M-modality) token regime. With M=1 (IMU only), the
cross-modal attention/aggregation degenerates to a pure-temporal
operator over K=4 instants of IMUCNN's 32-step window embeddings.

Two questions:

1. **Does the temporal aggregator help over pure IMUCNN?**
   RESULT_07's IMUCNN raw ATE was **9.961 m** (canonical unseen,
   no aggregator). If CNN1D / LSTM-attn over K=4 IMUCNN-window
   tokens drops the ATE meaningfully, the temporal lever is real
   on single-modality too.
2. **Is the gap to RoNIN ResNet1D (5.140 m) closeable in our
   architecture?** Our 0.05 M IMUCNN ≈ 9.96 m raw. ResNet1D
   4.6 M ≈ 5.14 m raw. Even at the small CNN1D / LSTM-attn
   aggregator addition (~0.05 M extra), we're not bridging the
   IMUCNN-vs-ResNet1D 9.96→5.14 m structural gap. So expectation:
   our row stays at ~7-10 m, ResNet1D wins by 30-50 %.

Three outcomes:
- **(α6) Surprise: our aggregator closes most of the gap to
  ResNet1D** (test ATE ≤ 6.5 m). Paper claim: "our temporal
  aggregator extracts substantial extra signal from IMUCNN
  windows." Unlikely but informative.
- **(β6) Aggregator drops some MAE but stays at 1.4-2× ResNet1D**
  (~7-10 m). Honest: aggregator helps, but the encoder gap
  dominates the row.
- **(γ6) Aggregator doesn't help / regresses vs IMUCNN-only**.
  The fusion architectures rely on cross-modal information, and
  with M=1 they don't have a useful inductive bias for
  single-modality dead-reckoning.

This is one focused experiment: single-modality main-table row.

## Steps

### Step 0 — Config + smoke (5–10 min)

`configs/data/ronin_a000_intra.yaml` (or similar) might be the
in-domain proxy; for canonical unseen-subjects we use the FRDR
data restored at `data/ronin_frdr/` (RESULT_07). The training
data lives in `data/ronin_frdr/train/` and the test list is
`C:\Users\FabLab\AppData\Local\Temp\ronin\lists\list_test_unseen.txt`
(32 sequences confirmed RESULT_07).

Engineer creates `scripts/_train_ronin_canonical_arch.py` mirroring
`_train_webots_4mod_arch.py` (the `--arch` runner) but:
- modalities = `[imu]` only.
- Uses RoNIN's `GlobSpeedSequence` loader (via vendored
  `Temp/ronin/source/data_glob_speed.py`, Demand #3 honoured).
- 6-channel world-frame IMU input (the RESULT_02 disaster-fix
  preprocessing).
- Window = 200 (RoNIN's standard) — but our IMUCNN window is 32.
  The cleanest implementation: chunk the 200-step RoNIN window
  into K=4 contiguous 50-step sub-windows, feed each as a
  separate IMUCNN instant. K=4 stays consistent with other
  bake-off candidates.

If the chunking choice (50-step sub-windows) is too far from
RESULT_07's preprocessing (which produced 9.961 m at one
window-200 IMUCNN forward, not 4 × 50-step), document the
deviation and explicitly note it shifts the comparison: this
iteration measures CNN1D-with-IMUCNN-tokens, NOT directly
comparable to RESULT_07's IMUCNN-only.

**Acceptance**: 1-epoch synthetic forward at `(B=128, K=4, M=1, D=128)`
produces `(B, 2)` velocity output (or `(B, 2)` position-step output
matching RoNIN's target — engineer matches the canonical loss).

### Step 1 — Pre-test gate (5 min)

5 epochs on 10 train sequences (small subset). Val ATE drops
≥ 10 % across 5 epochs.

### Step 2 — Full training × 2 (~25 min total)

- CNN1D arch on full RoNIN train (69 train + 12 val sequences
  per RESULT_07's coverage probe).
- LSTM-attn arch on the same.

Same protocol as RESULT_17: 20-30 epochs (RoNIN's typical), AdamW
+ OneCycleLR + Huber, B=128, K=4, M=1.

Velocity is the per-step target (RoNIN's standard); cumulative
integration to ATE at eval time.

**Acceptance**: both trained; test ATE on `list_test_unseen.txt`
reported.

### Step 3 — RoNIN canonical row populated

| method | params | raw ATE (m) | Umeyama ATE (m) | RTE | source |
|---|---|---|---|---|---|
| **RoNIN ResNet1D** (SOTA) | 4.24 M | **5.140** | 5.140 (anchor at GT[0]) | 4.377 | RESULT_07 |
| **IMUCNN-only** (our encoder) | 0.05 M | 9.961 | 7.876 | n/a | RESULT_07 |
| **CNN1D aggregator** (this iter) | ~0.10 M | ? | ? | ? | this iter |
| **LSTM-attn aggregator** (this iter) | ~0.12 M | ? | ? | ? | this iter |

**Acceptance**: outcome label (α6 / β6 / γ6) + verdict on whether
the temporal aggregator helps over pure IMUCNN.

### Step 4 — Per-sequence ATE distribution (criterion (d))

Per the 32-sequence canonical test set: median, p25, p75, p90,
max ATE. Compare distribution shape across all 4 methods
(ResNet1D, IMUCNN, CNN1D, LSTM-attn).

Note: smoothness r is not the relevant metric for IMU
dead-reckoning evaluation (no WiFi anchor); the meaningful
robustness measure on RoNIN is RTE (relative trajectory error)
which RESULT_07's `compute_ate_rte` provides.

### Step 5 — Decision + PLAN_24

Three-sentence verdict:
- RoNIN row populated; outcome label.
- Does the aggregator help over pure IMUCNN, and by how much?
- PLAN_24 = UJI K=1 degenerate (final main-table row before
  PLAN_25 SUMMARY + table assembly).

## Sources

- `handoff/SCIENTIST_NOTE_main-results-table.md` (directive).
- RESULT_07: canonical RoNIN unseen-subjects data extraction
  + ResNet1D 5.140 m + IMUCNN 9.961 m raw / 7.876 m Umeyama.
- RESULT_02 / RESULT_07: `scripts/eval_ronin_ate_fixed.py`
  pattern (window=200, GlobSpeedSequence loader).
- Vendored RoNIN repo: `C:\Users\FabLab\AppData\Local\Temp\ronin\`.
- `data/ronin_frdr/{train,unseen,Pretrained_Models}` (RESULT_07
  extraction).
- `src/pipeline/fusion/{cnn1d_instants,lstm_attn}.py` (RESULT_16/17).
- `src/pipeline/encoders/imu.py` (IMUCNN, untouched since
  public-restructure).

## What to report back

In `handoff/results/RESULT_23_main-results-ronin-single-mod.md`:

1. **Step 0** — runner choice; 50-step sub-window justification.
2. **Step 1** — pre-test gate outcome.
3. **Step 2** — training summary per arch.
4. **Step 3** — 4-row table; outcome label.
5. **Step 4** — per-sequence ATE distribution; RTE.
6. **Step 5** — PLAN_24 recommendation (default = UJI).
7. **One open question** for scientist.

## Reversibility

- Step 0: NEW `scripts/_train_ronin_canonical_arch.py` —
  permanent, engineer commits.
- Step 2: throwaway checkpoints under
  `runs/overnight/run2_iter_23/`.
- Steps 3–5: documentation.

Files committed: RESULT_23 + new runner script.

**Demand #3**: vendored RoNIN source untouched. RoNIN's
`GlobSpeedSequence` + `compute_ate_rte` imported pure.

**Compute budget**: ≤ 50 min.
- Step 0: 10 min (runner script + window chunking).
- Step 1: 5 min.
- Step 2: 25 min (2 trainings; RoNIN datasets smaller per RESULT_07's
  ~14 min IMUCNN training in canonical eval).
- Step 3: 3 min.
- Step 4: 5 min.
- Step 5: 5 min.

If overrun: drop Step 4's per-sequence distribution to median +
max (3 stats instead of 5); keep the headline raw ATE numbers in
Step 3.

If the 50-step sub-window decomposition makes the comparison
incoherent (e.g. our aggregator + chunked windows performs
much worse than pure IMUCNN on 200-step windows), document
explicitly: the apples-to-apples comparison is window-aligned, and
report the chunked-window number as a methods-section note about
fusion architecture limitations on single-modality data.
