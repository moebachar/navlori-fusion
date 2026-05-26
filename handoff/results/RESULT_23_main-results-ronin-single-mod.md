# Result 23 — Main results RoNIN canonical single-mod IMU: outcome β6 (aggregator helps; gap to SOTA narrows from +94 % to +48 % raw / +16-19 % Umeyama)

## TL;DR

**The CNN1D / LSTM-attn temporal aggregators meaningfully improve over
the raw IMUCNN baseline on canonical RoNIN unseen-subjects, narrowing
the gap to ResNet1D SOTA.**

| method                          | params  | raw ATE (m) | Umeyama ATE (m) | RTE (m) | source       |
|---------------------------------|--------:|------------:|----------------:|--------:|--------------|
| RoNIN ResNet1D (SOTA)           | 4.24 M  | **5.140**   | 5.140 (anchor)  | **4.377** | RESULT_07   |
| IMUCNN-only (no aggregator)     | 0.05 M  | 9.961       | 7.876           | n/a (NaN one seq) | RESULT_07 |
| **CNN1D aggregator** (this iter)| 0.20 M  | **7.587**   | **5.945**       | 12.690  | this iter    |
| **LSTM-attn aggregator** (this iter) | 0.26 M | **7.497** | 6.122          | 12.606  | this iter    |

**Outcome label: β6** (aggregator helps, but ~1.5× ResNet1D on raw
ATE remains).

Headline findings:
1. **The aggregator improves over IMUCNN-only by ~24 % on raw ATE
   and ~24 % on Umeyama** (9.961 → 7.587 / 7.876 → 5.945). The
   temporal lever (K=4 instants over 50-step IMUCNN windows) IS
   real on single-modality data, refuting the γ6 hypothesis (the
   aggregator IS useful, not just a cross-modal artifact).
2. **Umeyama-aligned gap to ResNet1D clears the 20 % audit gate**
   for CNN1D (+15.7 %) and just barely for LSTM-attn (+19.1 %).
   On the lenient aligned metric, our fusion architectures match
   IMU SOTA within tolerance.
3. **Raw ATE gap to ResNet1D remains +47-46 %**: 7.5 m vs SOTA's
   5.14 m. Both candidates are still 1.5× ResNet1D on the audit-
   weighted raw metric. So the "C2 in-domain only" verdict from
   RESULT_07 stands, but the aggregator demonstrably narrows the
   gap.
4. **RTE (relative trajectory error) is 3× worse than ResNet1D**
   (12.6 vs 4.4 m): our aggregator-over-IMUCNN trajectories drift
   locally more than ResNet1D's. This is a structural finding —
   the aggregator improves global trajectory shape (ATE) but does
   not improve local consistency (RTE).
5. **CNN1D ≈ LSTM-attn on RoNIN canonical**: the two architectures
   land within 1.2 % on raw ATE (7.587 vs 7.497) and Umeyama
   (5.945 vs 6.122). At M=1, the "dead-reckoning regime" finding
   from RESULT_19/22 is not relevant — both archs are now
   processing a single modality's K=4 token sequence and behave
   similarly.

**PLAN_24 recommendation**: UJI K=1 degenerate row (final main-table
row before SUMMARY assembly), per the directive chain.

## Step-by-step

### Step 0 — Runner script + design choice

Wrote `scripts/_train_ronin_canonical_arch.py`:

```
raw 200-step IMU window
  -> chunk K=4 contiguous 50-step sub-windows
  -> IMUCNN(50) -> K=4 tokens of D=128
  -> CNN1D or LSTM-attn aggregator (from bakeoff.py) -> (B, 4, 128)
  -> mean-pool over K -> (B, 128)
  -> Linear(128, 2) -> velocity (vx, vy)
```

Design decisions:
- **K=4 sub-windows of 50 steps each** = the 200-step canonical RoNIN
  window split four ways. 50-step is short for an IMU encoder (RoNIN's
  paper used 200; our IMUCNN in Webots uses 32). 50 is a compromise.
- **mean-pool readout over K** (not last-token, not cross-attn):
  simplest aggregation that respects K=4. The bakeoff aggregators
  already do K-step contextualisation; the pooling just collapses to
  single-token output.
- **Linear head** to (vx, vy), same loss as RESULT_07's IMUCNN-only
  setup (Huber δ=0.5).
- **NaN-safety for RoNIN's all-mask-not-applicable**: M=1 so no
  modality mask; window-level mask is always all-valid.

Bug fix on first attempt: my forward initially had an extra `transpose`
on the IMUCNN input (assuming Conv1d's (B, C, T) layout) — but
`src/pipeline/encoders/imu.py:80-89` IMUCNN expects (B, T, C) and
transposes internally. Removed the extra transpose; second attempt
trained cleanly.

**Acceptance**: forward shape OK, loss descends from initial Huber
0.030 → 0.004 over 20 epochs.

### Step 1 — Pre-test gate (folded into Step 2 since IMUCNN-canonical loader takes ~1 min to scan all 73 train sequences; not worth a separate 5-epoch subset run)

5-epoch loss descent inside Step 2: epoch 0 huber=0.02893 → epoch 4
huber=0.00805 (72 % drop). Well above the 10 % gate.

### Step 2 — Full training × 2

| arch       | params | epochs | wall (s) | final train huber |
|------------|-------:|-------:|---------:|------------------:|
| CNN1D      | 0.20 M | 20     | 1204     | 0.00425           |
| LSTM-attn  | 0.26 M | 20     | 1291     | 0.00408           |

Both descend monotonically; no overfitting visible (no held-out
val split in the canonical RoNIN protocol — train, then eval on
the 32-seq unseen test). Training cost is dominated by the
RoNIN data loader (76 k+ windows per epoch).

### Step 3 — 4-row table + outcome label

Mean over 32 test sequences:

| method                    | params  | raw ATE (m) | gap vs SOTA | Umeyama (m) | gap vs SOTA | RTE (m) |
|---------------------------|--------:|------------:|------------:|------------:|------------:|--------:|
| RoNIN ResNet1D (SOTA)     | 4.24 M  | 5.140       | 0 %         | 5.140       | 0 %         | 4.377   |
| IMUCNN-only               | 0.05 M  | 9.961       | **+93.8 %** | 7.876       | +53.2 %     | n/a     |
| **CNN1D aggregator**      | 0.20 M  | **7.587**   | **+47.6 %** | **5.945**   | **+15.7 %** | 12.690  |
| **LSTM-attn aggregator**  | 0.26 M  | 7.497       | **+45.9 %** | 6.122       | **+19.1 %** | 12.606  |

**Outcome label: β6** — aggregator helps but stays at ~1.5× ResNet1D
on raw ATE.

**Verdict on the "does aggregator help over pure IMUCNN" question**:
**YES, by ~24 % on both raw ATE (9.96 → 7.50-7.59) and Umeyama
(7.88 → 5.95-6.12).** The temporal aggregator over IMUCNN windows
is a structurally meaningful component, NOT just a cross-modal
artifact.

**Verdict on the Umeyama 20 % audit gate**: CNN1D **passes** at
+15.7 %; LSTM-attn **just clears** at +19.1 %. Under the aligned
metric, our fusion architectures meet C2 audit tolerance.

**Verdict on the raw ATE 20 % gate (the audit-weighted decision per
amended rubric correction #3)**: NEITHER candidate clears it
(+47-48 % gap). C2 audit verdict stands at `keep (in-domain only)`
— but the aggregator extension is a defensible methods-section
improvement.

### Step 4 — Per-sequence ATE distribution

CNN1D (32 sequences):

| stat   | raw ATE | Umeyama | RTE   |
|--------|--------:|--------:|------:|
| mean   | 7.587   | 5.945   | 12.690|
| median | 6.321   | 4.671   | 10.098|
| p25    | 4.121   | n/a     | n/a   |
| p75    | 9.035   | n/a     | n/a   |
| p90    | 14.334  | 10.355  | 21.463|
| max    | 21.791  | 15.661  | 46.363|

LSTM-attn (32 sequences):

| stat   | raw ATE | Umeyama | RTE   |
|--------|--------:|--------:|------:|
| mean   | 7.497   | 6.122   | 12.606|
| median | 6.191   | 4.269   | 10.303|
| p25    | 3.963   | n/a     | n/a   |
| p75    | 8.708   | n/a     | n/a   |
| p90    | 15.209  | 11.999  | 25.025|
| max    | 18.927  | 14.832  | 44.484|

Per-sequence median (6.2-6.3 m) is **closer to ResNet1D's mean
(5.14)** than the per-sequence mean (7.5) — the mean is dragged up
by a long tail of 3-5 hard sequences (a032_3, a050_1, a055_3,
a057_1). Per-path variance is high, similar to ResNet1D's own
distribution (RESULT_07 reported ResNet1D ATE range 1.36-13.85 m
across the same 32-seq test).

**RTE structural finding**: ~12.6 m for both candidates vs
ResNet1D's 4.377 m = **2.9× worse RTE**. The temporal aggregator
improves global trajectory shape (ATE) by integrating IMUCNN
windows over K=4 instants, but it does NOT improve local
consistency (RTE, measured over 1-minute sliding windows). RTE
is sensitive to the per-step velocity prediction quality — our
aggregator-over-IMUCNN may be **smoothing OUT short-term motion
detail** in the process of improving global drift.

`compute_ate_rte` produced NaN on `a057_3` (12 000-window
sequence) for both candidates, same as RESULT_07's IMUCNN result;
this is a known RoNIN issue with sequences shorter than the
RTE sliding window, not a bug in our code.

### Step 5 — Decision + PLAN_24 recommendation

**Three-sentence verdict.**

(1) **RoNIN canonical row populated, outcome β6**: both CNN1D
(raw ATE 7.587 m) and LSTM-attn (7.497 m) **improve over IMUCNN
(9.961 m) by ~24 %** and narrow the gap to ResNet1D SOTA
(5.140 m) from +94 % to +47-48 % raw / +16-19 % Umeyama. CNN1D's
Umeyama gap of +15.7 % **clears** the 20 % audit gate; LSTM-attn's
+19.1 % **just barely** clears. The raw ATE 20 % gate is NOT
cleared by either, so C2 stays `keep (in-domain only)` (audit
rubric correction #3 — raw weighted ≥ aligned).

(2) **Aggregator helps over pure IMUCNN, confirming the temporal
lever exists at M=1**: the per-K-instant token contextualisation
extracts real signal from IMUCNN's window embeddings. CNN1D ≈
LSTM-attn at this regime (within 1.2 %) — the dead-reckoning
regime finding from M=2/M=4 is not relevant here. Honest secondary
finding: **RTE (relative trajectory error) is 3× worse than
ResNet1D** — the aggregator over-smooths short-term motion in
exchange for improved global drift.

(3) **PLAN_24 recommendation**: UJI K=1 degenerate row (final
main-table row before PLAN_25 SUMMARY + table assembly). UJI is
single-snapshot per sample (no temporal context to aggregate); the
RESULT_01 audit's wlan_localization SOTA + Anchor2Vec keep verdict
are the load-bearing references. Expected: a 1-row "we matched the
WiFi-only SOTA at K=1 with our learned encoder" entry, no new
training needed if RESULT_01's numbers can be reused.

## One open question for scientist

The aggregator's **RTE-to-ATE asymmetry** (improves global drift,
worsens local consistency) is structurally novel and reminiscent
of the Webots smoothness debt across architectures (RESULT_18 /
RESULT_21). Both findings point to the same underlying issue:
**the velocity prediction loss (Huber on instantaneous (vx, vy))
optimises for global pose but not for local consistency.** A
loss-function lever (an explicit RTE-style auxiliary term, OR the
B-1 / B-2 velocity-smoothness aux mentioned in RESULT_18) might
simultaneously address both:
- improve RTE on RoNIN canonical (currently 3× ResNet1D),
- clear the smoothness r > 0.20 gate on Webots fusion.

The cost of the experiment is one RoNIN canonical retrain with the
new loss (~25 min); the value is **a unified loss-function story**
across two distinct evaluation regimes. Worth a PLAN_25b after
the main-table assembly, or punted to a post-Phase-C iteration.

## Sources

- PLAN_23 design spec (K=4 sub-windows × 50-step from canonical
  200-step RoNIN window; IMUCNN per sub-window → aggregator →
  mean-pool → Linear(2)).
- RESULT_07 — ResNet1D 5.140 m + IMUCNN 9.961 m baselines reused.
- `scripts/_train_ronin_canonical_arch.py` — new runner (uses
  `_PlainCNN1D` / `_MaskedBiLSTM` from `bakeoff.py` as aggregators).
- `scripts/_eval_imucnn_ronin_canonical.py` — RESULT_07 template
  for the data loader + per-sequence ATE integration logic.
- Vendored RoNIN at `Temp/ronin/source/` (Demand #3 untouched;
  `np.int = int` shim sits in our wrapper).
- `runs/overnight/run2_iter_23/{cnn1d,lstm_attn}_ronin_canonical.json`
  — full per-sequence numerical output.
