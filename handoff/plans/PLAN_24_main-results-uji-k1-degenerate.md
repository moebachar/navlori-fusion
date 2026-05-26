# Plan 24 — Main results table: UJI K=1 degenerate (CNN1D + LSTM-attn on per-scan WiFi)

> Final main-table row before PLAN_25 SUMMARY + table assembly.
> Per `handoff/SCIENTIST_NOTE_main-results-table.md`. UJI is the
> *degenerate* row: per-scan dataset, no time axis. Our fusion
> aggregators (CNN1D + LSTM-attn) expect `(B, K=1, M=1, D=128)`
> tokens — a single instant, single modality — which structurally
> collapses the temporal aggregator to an MLP-on-encoder-output.
> Whether the cell numbers are meaningful, or whether the row
> simply documents "fusion architecture not applicable on per-scan
> WiFi," is the iteration's primary question.

## Hypothesis

At K=1 + M=1, the CNN1D's 1D conv over the K axis becomes a no-op
(length-1 sequence + kernel=3 → trivial); the LSTM-attn's BiLSTM
over K=1 sees one step and the readout attention has one token to
attend to. Effectively both aggregators degenerate to **encoder
(Anchor2Vec) + thin MLP head** — the same shape as RESULT_01's
Anchor2Vec UJI runner.

Expected: CNN1D ≈ LSTM-attn ≈ Anchor2Vec val MAE 8.69 m
(within ±5 %; the only difference being the head MLP architecture).
SOTA reference wlan_localization 15.17 m (RESULT_01).

Two outcomes that matter for the paper:
- **(α7) Aggregator collapse confirmed**: both archs land within
  ±5 % of Anchor2Vec's 8.69 m, indistinguishable from each other,
  ALL beat SOTA's 15.17 m by ~43 %. Paper interpretation: "on
  per-scan WiFi-only datasets, our temporal/cross-modal fusion
  architecture isn't structurally applicable; the WiFi encoder
  alone matches SOTA." Honest negative-result row for the table.
- **(β7) Unexpected divergence**: CNN1D and LSTM-attn produce
  meaningfully different numbers (> 5 %) despite the degeneracy.
  Suggests the small architectural differences (BiLSTM vs conv
  vs incumbent) DO matter even at K=1, M=1 — worth probing.

This is the smallest possible iteration in the main-table chain
(everything except training is already settled). One focused
experiment: do the aggregators degenerate cleanly?

## Steps

### Step 0 — UJI dataset + adapter probe (5 min)

UJI data is already in `data/uji_indoorloc/{trainingData,validationData}.csv`
(RESULT_01). The dataloader needs to emit per-scan tokens shaped
`(B, K=1, M=1, D_input)` where D_input is the UJI 520-AP RSSI
vector. The Anchor2Vec encoder converts that to 128-d.

Verify:
- `configs/data/` has a UJI config (engineer checks; if not, write a
  thin one or use the existing `scripts/eval_uji_wifi.py` pattern).
- The `_train_webots_4mod_arch.py` runner (the `--arch` flag pattern)
  can be adapted to UJI's per-scan / single-modality input — the
  dataloader is the surface change.

If the path through the existing `bakeoff.py` registry +
FusionTrainer doesn't fit UJI's per-scan format trivially (e.g.
the FusionTrainer expects a temporal-windowed dataloader),
engineer either:
- Writes a thin `scripts/_train_uji_arch.py` that bypasses
  FusionTrainer and uses a simpler per-scan training loop (CNN1D
  + LSTM-attn as drop-in encoders-over-WiFi-token).
- OR adapts the existing dataloader to emit (B, 1, 1, 520)
  → encoder → (B, 1, 1, 128) → aggregator → (B, 128) → head →
  (B, 2).

**Acceptance**: per-scan UJI dataloader produces correct shapes
for both archs; smoke fwd no NaN.

### Step 1 — Train CNN1D + LSTM-attn on UJI per-scan (sequential, ~20 min total)

Protocol:
- 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5), B=128.
- K=1, M=1 (degenerate).
- Anchor2Vec encoder (RESULT_01's keep verdict).
- `modality_dropout=0`, `instant_dropout=0` (no temporal/cross-modal
  axis to dropout).
- lr=1.3e-3 baseline; if degenerate the lr probably matters less
  than at K=4 K-mod.

Each training: a few minutes on UJI's 19937 train scans / 1111 val
scans. Single-modality + 0.05 M Anchor2Vec params + 0.1-0.2 M
aggregator = lightweight.

**Acceptance**: both candidates train; val MAE reported. RESULT
includes Anchor2Vec's 8.69 m number reused from RESULT_01.

### Step 2 — UJI main-table row populated

| method | params | val MAE | source |
|---|---|---|---|
| **wlan_localization** (SOTA) | (kNN) | **15.17** | RESULT_01 (reused) |
| Anchor2Vec encoder + linear head | 0.075 M | **8.69** | RESULT_01 (reused) |
| **CNN1D aggregator** (this iter) | ~0.1 M | ? | this iter |
| **LSTM-attn aggregator** (this iter) | ~0.15 M | ? | this iter |

(UJI has no canonical test split — `validationData.csv` is the
benchmark. No "test" column. The main table row is val-only,
documented.)

**Acceptance**: 4-row table populated; outcome label
(α7 / β7).

### Step 3 — Per-scan distribution + verdict

Per-sample p25 / p50 / p75 / p90 / max for all 4 methods on
UJI val. (No per-trajectory smoothness — UJI is per-scan, no
time axis; document explicitly per the criterion (d) gate
wording, this measurement is undefined here.)

**Acceptance**: distribution table; verdict on whether the
fusion architectures meaningfully differ from each other and from
Anchor2Vec.

### Step 4 — Decision + PLAN_25

Three-sentence verdict:
- UJI row populated; aggregators degenerate (α7) or differ (β7)?
- Implication for the main-table interpretation: "fusion
  architectures collapse to encoder + head on per-scan" or
  "fusion architecture choice matters even degenerate."
- PLAN_25 = SUMMARY + main-table assembly (the final scientist
  deliverable). All 6 rows now populated:
  - Webots ✓ (CNN1D 0.339 m / LSTM-attn 0.340 m)
  - IMUWiFine ✓ (CNN1D val 1.40 / LSTM-attn val 1.26; test caveat)
  - IPIN floor 0 ✓ (CNN1D 21.61 / LSTM-attn 22.45; β5 outcome)
  - RoNIN canonical ✓ (CNN1D raw 7.59 / LSTM-attn raw 7.50; β6)
  - TartanAir hospital ✓ (DPVOMotion only; not fusion-arch tested)
  - UJI ✓ (this iter populates)

## Sources

- `handoff/SCIENTIST_NOTE_main-results-table.md` (directive).
- RESULT_01: wlan_localization 15.17 m + Anchor2Vec 8.69 m on UJI
  val.
- RESULT_17: CNN1D + LSTM-attn full-data Webots reference for
  the architecture configs.
- `src/pipeline/fusion/{cnn1d_instants,lstm_attn}.py` (bake-off
  candidates committed RESULT_16/17).
- `src/pipeline/encoders/wifi.py` (Anchor2Vec).
- `data/uji_indoorloc/{trainingData,validationData}.csv`.
- `scripts/eval_uji_wifi.py` (the existing per-scan runner; serves
  as the dataloader template).

## What to report back

In `handoff/results/RESULT_24_main-results-uji-k1-degenerate.md`:

1. **Step 0** — UJI adapter approach (FusionTrainer path or
   thin custom runner); any restored configs.
2. **Step 1** — training summary per arch (val MAE, params,
   wall, peak GPU).
3. **Step 2** — 4-row table; outcome label.
4. **Step 3** — per-scan distribution; verdict on degeneracy.
5. **Step 4** — PLAN_25 SUMMARY-prep notes — list of anchored
   numbers + open paper-framing questions.
6. **One open question** for scientist.

## Reversibility

- Step 0: thin runner script `scripts/_train_uji_arch.py` if
  needed — permanent.
- Step 1: throwaway checkpoints.
- Steps 2–4: documentation.

Files committed: RESULT_24 + thin UJI runner (if added).

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 30 min.
- Step 0: 10 min.
- Step 1: 10 min (2 short trainings on small UJI data).
- Step 2: 3 min.
- Step 3: 3 min.
- Step 4: 5 min (SUMMARY-prep notes).

If overrun: drop Step 3's full distribution to median + p90 only;
the load-bearing data is in Step 1's val MAE numbers.

If the UJI dataloader adaptation surfaces a structural issue
(e.g. the bakeoff candidates ASSUME K>1 in their forward), the
honest report is "fusion architectures structurally don't fit
per-scan; UJI row populates with the encoder-only number
(Anchor2Vec 8.69 m) and a methodological note." PLAN_25 still
proceeds.
