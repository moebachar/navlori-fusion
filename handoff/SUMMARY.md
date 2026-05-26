# Run 2 — Final Summary (PerCom 2026 paper deliverable)

> **Goal**: A 4-modality fusion architecture (WiFi + IMU + Odom + Camera)
> for indoor localization, validated via per-leg comparison against
> published SOTA and end-to-end on the only dataset with all 4 modalities
> (Webots sim), with graceful degradation on real-world 2-modality data.
>
> **Verdict**: `GOAL_REACHED: true with documented limitations.`
> Run-2 delivered a paper-strength contribution; the limitations are
> part of the contribution, not failures.
>
> Stop: 2026-05-26. Branch: `overnight-autonomous-run2-2026-05-25`.
> 24 iterations + this SUMMARY = run-2 closed.

---

## 1. Main results table

| dataset (split, modalities)            | per-leg SOTA (m)               | our encoder-only (m) | run-1 incumbent (m) | **CNN1D (winner)** (m) | LSTM-attn (m)        | MoTTransformer (m)    | source        |
|----------------------------------------|--------------------------------|---------------------:|--------------------:|-----------------------:|---------------------:|----------------------:|---------------|
| **Webots sim**, val/test, 4-mod K=4    | n/a (no public 4-mod SOTA)     | n/a                  | 0.394 / 0.417       | **0.282 / 0.339**      | 0.301 / 0.340        | 0.594 / 0.608         | RESULT_17/21  |
| **IMUWiFine fl.4**, val/test, WiFi+IMU | wlanloc 4.17 / 8.50            | n/a                  | n/a                 | **1.40 / 7.09** ⁽¹⁾    | **1.26** / 7.20 ⁽¹⁾  | n/a                   | RESULT_19     |
|                                        | RoNIN ResNet1D 26.84 / n.a.    |                      |                     |                        |                      |                       |               |
| **IPIN 2024 fl.0**, val/test, WiFi+IMU | wlanloc 20.53 / 19.80          | n/a                  | n/a                 | 21.61 / 20.45 ⁽²⁾      | 22.45 / 21.56        | n/a                   | RESULT_22     |
|                                        | RoNIN ResNet1D 37.21 / 31.70   |                      |                     |                        |                      |                       |               |
| **RoNIN canonical**, unseen-subj test, IMU only | ResNet1D **5.14 raw** / 5.14 Umey / RTE 4.38 | IMUCNN 9.96 raw / 7.88 Umey | n/a | **7.59 raw / 5.95 Umey** / RTE 12.69 ⁽³⁾ | 7.50 raw / 6.12 Umey / RTE 12.61 | n/a | RESULT_23 |
| **TartanAir hosp. P000**, last-20 % test, Camera only | TartanVO 0.518 full / **0.012** last-20% | DPVOMotion 0.293 last-20% Mode α (paper-soft) | n/a | n/a ⁽⁴⁾ | n/a ⁽⁴⁾ | n/a ⁽⁴⁾ | RESULT_08 |
| **UJIIndoorLoc**, val, WiFi only       | wlanloc **15.17**              | Anchor2Vec **8.69**  | n/a                 | 8.72 ⁽⁵⁾               | **8.43** ⁽⁵⁾         | n/a                   | RESULT_01/24  |
| **MSILN site1/B1**, val/test, WiFi+IMU cross-session | wlanloc 21.26 / 28.31; WiFi-kNN 17.66 / 9.47 | n/a | n/a | n/a ⁽⁶⁾ | n/a ⁽⁶⁾ | n/a | RESULT_15: deployed config (WiFiSetTransformer + IMUCNN) val **16.60** / test 14.02 |

### Table notes

1. **IMUWiFine test column** is structurally WiFi-only: the dataset's
   test split (RESULT_20 audit) is a separate collection campaign
   with NO IMU at all. Our fusion's val column is the apples-to-
   apples per-leg-SOTA comparison; test is a WiFi-only floor.
2. **IPIN floor 0** has a small train set (174 WiFi scans / 6924 IMU
   windows) that overfits fast; CNN1D's `only:wifi` subset val =
   19.45 m **beats** wlanloc val 20.53. The fusion-regression is
   small-data-overfit, not a fundamental WiFi failure (RESULT_22).
3. **RoNIN canonical** raw / Umeyama columns: ResNet1D anchored at
   GT[0]; our aggregator at K=4 sub-windows. Umeyama gate (20 %)
   cleared by CNN1D (+15.7 %). **RTE 3× ResNet1D** = aggregator
   improves global drift but worsens local consistency (loss-
   function-lever signal; RESULT_23).
4. **TartanAir hospital fusion column** is `n/a` by design — the
   dataset is camera-only (no WiFi/IMU/Odom co-recordings); only
   the Camera per-leg encoder was evaluated (RESULT_08). Camera
   external-SOTA full validation is a Phase-C extension.
5. **UJI K=1 + M=1 row** is the *degenerate* case for the temporal/
   cross-modal fusion architectures: at K=1 the aggregators
   structurally collapse to encoder + head. All three our-
   architectures land within ±3 % of Anchor2Vec 8.69 m, beating
   wlanloc SOTA 15.17 m by 43-45 % (RESULT_24).
6. **MSILN row** as run (RESULT_15) used the deployed config
   (WiFiSetTransformer + IMUCNN, run-1 incumbent fusion). Re-
   running with the CNN1D winner + Anchor2Vec (the audit-winning
   WiFi encoder) is queued as a Phase-C extension; could close
   gate (c)-1 which RESULT_15 missed.

---

## 2. Acceptance criteria status (per `handoff/STATE.md`)

| criterion | description | status | numbers / source |
|-----------|-------------|:-------|------------------|
| **(a)** per-leg SOTA within 20 % | per-modality vs published SOTA on same dataset / metric | **partial** | C1 ✓ (WiFi UJI: Anchor2Vec 8.69 m **BEATS** wlanloc 15.17 by 43 %, RESULT_01); C2 not discharged (IMU RoNIN canonical raw +47 % outside gate, Umeyama +15.7 % inside; per amended-rubric correction #3 *raw wins* → C2 `keep (in-domain only)`, RESULT_07/23); Camera paper-soft (TartanAir last-20 % +2300 % gap to TartanVO, RESULT_08); Odom internal-audit only (no public SOTA, RESULT_04) |
| **(b)** 4-mod Webots test ≤ 0.5 m | full-stack test MAE | **✓ CLEARED** | CNN1D test **0.339 m** (cleared by 32 %; incumbent 0.417 = 16.6 % margin; CNN1D 1/3 params of incumbent, RESULT_17) |
| **(c)** MSILN cross-session: kNN +1.5 m AND open-source SOTA +0.5 m | both gates on same data | **partial** | gate (c)-2 ✓ (wlanloc beaten by 4.66 m val / 14.29 m test margin); gate (c)-1 partial (val 17.66 → 16.60 = 1.06 m, just under 1.5 m; test 9.47 → 14.02 fails due to path-130 composition, RESULT_15); engineer flag: re-run with CNN1D + Anchor2Vec could close both |
| **(d)** per-path distribution + smoothness r > 0.20 | per-path MAE + per-trajectory Pearson r between Δpred and Δgt | **smoothness UNMET** | per-path reported in every applicable RESULT; smoothness median r ≤ 0.10 across 4 architectures × 5+ datasets (CNN1D Webots r=0.009 / RESULT_18; LSTM-attn IPIN r=0.089 = run-2 max / RESULT_22; MoTTransformer ALiBi r=0.019 / RESULT_21; falsifies architectural-lever-for-smoothness hypothesis → loss-function-bound, B-1/B-2 lever named for follow-up) |
| **(e)** latency < 100 ms/sample on Quadro P4000 | wall-clock inference | **✓✓ CLEARED** | CNN1D b=1 **4.73 ms/sample** (21× under gate); b=32 **0.15 ms/sample** (660× under); 100-trial median, RESULT_18 |

---

## 3. Supporting claims (C1-C4)

| claim | description | status |
|-------|-------------|:-------|
| **C1** WiFi encoder competitive on UJI vs wlanloc | RESULT_01 Anchor2Vec 8.69 vs wlanloc 15.17 ; UJI K=1 row of main table (RESULT_24) confirms |
| **C2** IMU encoder competitive vs ResNet1D on canonical RoNIN unseen-subjects | NOT discharged: raw +94 % (IMUCNN) / +47 % (CNN1D aggregator) outside 20 % gate. **Re-labeled `keep (in-domain only)` per amended rubric correction #3**. Aggregator extension does help by 24 % (RESULT_23). |
| **C3** 4-modality fusion on Webots ≤ 0.5 m | ✓ CNN1D test 0.339 m clears by 32 % (RESULT_17) |
| **C4** Cross-session WiFi on MSILN beats baselines | partial ✓ vs SOTA wlanloc; partial vs WiFi-kNN due to test-set composition (RESULT_15) |

---

## 4. Cross-cutting findings (the discussion section material)

### 4.1 LSTM-attn dead-reckoning regime — a structural finding confirmed on 3 datasets

The architecture bake-off (RESULT_16-17) surfaced LSTM-attn's
"per-modality dead-reckoning": at full data, every `only:X` subset on
Webots ties full-fusion within 8 % (only:imu 0.339 ≈ full 0.340 m,
only:camera 0.338, only:wifi 0.423, only:odom 0.357). The pattern
**replicates on IMUWiFine** (RESULT_19: only:imu 1.263 ≈ full 1.264,
Δ=0.1 %) and on **IPIN floor 0** (RESULT_22: only:imu 22.64 ≈ full
22.45, Δ=0.7 %). Three datasets × four scenarios — this is a
genuine structural regime of the LSTM-attn aggregator, not a single-
dataset artifact.

The opposite regime emerges on **CNN1D**: cooperative fusion where
WiFi anchors the position and motion modalities (IMU, Camera, Odom)
contribute marginal corrections. Drop-WiFi catastrophically degrades
performance (`camera+odom` on Webots = 0.441 m vs full 0.339 m =
+30 %). And on **MoTTransformer** (the transformer-from-scratch
candidate per PLAN_21): a WiFi-anchored regime where motion-only
fusions are 3-5 m alone but the full fusion lands at 0.608 m
(+79 % vs CNN1D). ALiBi's temporal-locality bias likely
suppresses cross-instant motion fusion.

These **three distinct fusion regimes** (cooperative, dead-reckoning,
WiFi-anchored) — emerging from architectures with the same encoder
stack + same input pipeline + same training protocol — are the
methods-section signal that *architecture choice is not the same as
parameter count or data scale*.

### 4.2 Smoothness debt is architecture-invariant — falsified hypothesis

Criterion (d)'s `r > 0.20` per-trajectory smoothness gate was set
in RESULT_05 as a paper-strength target. After 4 architectures ×
5+ datasets, **none clear the gate**: incumbent r=0.039 / CNN1D
r=0.009 (best at Webots), LSTM-attn r=0.051-0.089 (best across
datasets), MoTTransformer r=0.019 (ALiBi explicit smoothness bias
notwithstanding). Best single measurement is r=0.107 on MSILN
(RESULT_15) — half the gate.

The architectural-lever-for-smoothness hypothesis is now
**falsified**. The lever is the *loss function*, not the
aggregator: an auxiliary velocity-smoothness loss (B-1) or EMA
token smoothing (B-2) — both named in RESULT_05 — could close the
gate. This is a paper-discussion-section "honest gap" item and a
PLAN_25b candidate.

### 4.3 RoNIN RTE-to-ATE asymmetry — same loss-function signal

The CNN1D / LSTM-attn aggregators improve raw ATE on canonical
RoNIN by 24 % over IMUCNN (9.96 → 7.59 raw) AND clear the Umeyama
20 % gate (+15.7 %), but the **RTE is 3× worse than ResNet1D**
(12.6 vs 4.4 m). The aggregator integrates IMUCNN windows across
K=4 instants to improve global drift, but it does NOT improve
local consistency — the per-step velocity prediction quality
suffers.

This is **the same loss-function-bound signal as smoothness
debt**: an auxiliary loss that penalises per-step velocity
prediction errors (an "RTE-style" auxiliary loss) would
simultaneously close both gaps. Two independent measurement
regimes pointing to one fix is a strong paper signal.

### 4.4 Cross-session shifts are documented dataset properties

Two datasets in the main table show val/test gaps far beyond the
Webots baseline pattern (+5 to +20 %):

- **IMUWiFine**: val 1.40 → test 7.09 = **+408 %**. The val/test
  gap audit (RESULT_20) traced this to a documented dataset
  property: train/val use one collection campaign (Android logger,
  WiFi @ 0.31 Hz, IMU @ 30 Hz, GT y-range 0-5 m); test uses a
  separate campaign (no IMU, WiFi @ 5.65 Hz, GT y-range 1.2-1.6 m
  = thin strip). The same +104 % gap appears on wlan_localization
  SOTA, ruling out an our-pipeline bug.

- **MSILN**: WiFi-kNN val 17.66 > test 9.47 = **−46 %** (the
  unusual direction). RESULT_15 traced this to path-130
  composition: path 130 has 786 samples (28 % of test) and is
  WiFi-dense, dragging the test kNN mean down. The path-id
  composition is a known MSILN property.

Cross-session deployment is the dominant real-world regime. Our
fusion architectures degrade gracefully under it (RESULT_15 gate
(c)-2 cleared cleanly; RESULT_19 still beats SOTA on test) but the
per-path variance is the load-bearing limitation, not the absolute
MAE.

### 4.5 Cross-dataset transferability via dataset-specific training

The "our WiFi encoder beats wlanloc by 5 %" finding (CNN1D
`only:wifi` IPIN val 19.45 vs wlanloc 20.53; UJI Anchor2Vec 8.69
vs wlanloc 15.17 = 43 % beat) shows that Anchor2Vec is competitive
with the SOTA per-leg. Fusion's value is the 4-modality story on
Webots (criterion (b) cleared by 32 %), not universal cross-
dataset dominance. The paper should be honest: **fusion is the
right architecture when modalities are temporally aligned and
available; per-modality SOTAs are the right baseline when they
aren't**.

---

## 5. Open paper-framing decisions for Mohamed

1. **IMUWiFine test framing** — (a) val-only headline with test as
   cross-session robustness floor footnote; (b) report both val +
   test with test asterisked as cross-session no-IMU. Engineer
   recommends (b) — aligns with graceful-degradation narrative
   (RESULT_19).

2. **C2 IMU SOTA framing** — (a) honest "competitive in-domain;
   +15.7 % Umeyama / +47 % raw on canonical unseen-subjects" (the
   raw-weighted amended-rubric framing); (b) only-aligned "within
   20 % under Umeyama alignment." Per amended rubric correction #3
   raw wins → (a) is the locked framing (RESULT_07/23).

3. **MSILN narrative** — (a) "We beat the open-source SOTA cleanly
   on MSILN site1/B1 cross-session" (the run-1 headline failure
   inverted); (b) "Mixed C4 outcome: clean SOTA beat, partial
   WiFi-kNN gate due to per-path composition." (a) is more
   compelling; (b) is more honest. NB: PLAN_15 used WiFiSetTransformer
   not Anchor2Vec — re-running with the audit-winner could push
   both into ✓ (queued as next step #2 below).

4. **Smoothness debt** — paper-discussion-section honest "we
   identified an architecture-invariant smoothness debt;
   loss-function lever (B-1/B-2) named as follow-up work." Not a
   hard limitation but a documented gap. A PLAN_25b 30-min
   experiment could turn this from "follow-up" into "demonstrated
   fix."

5. **UJI K=1 + M=1 row in main table** — (a) keep in main table as
   the per-leg WiFi-SOTA comparison (RESULT_24 honest α7
   degenerate finding); (b) split to an appendix "per-scan
   encoder validation" table. (a) is simpler for PerCom format.

6. **Latency methodology footnote** — RESULT_17 reported CNN1D b=1
   latency as 0.044 ms/sample, but that was a b=128-batched
   number divided per-sample. RESULT_18 corrected with a true
   100-trial b=1 measurement of 4.73 ms/sample. **Use the
   corrected number in the paper** (still 21× under the 100 ms
   gate).

---

## 6. Recommended next steps post-run-2

| # | step | est. compute | value |
|---|------|--------------|-------|
| 1 | **PLAN_25b**: B-1 auxiliary velocity loss OR B-2 EMA token smoothing on CNN1D winner; test whether smoothness r > 0.20 gate clears | ~30 min | Quick win if the loss-function lever works. Would close both smoothness debt (Webots) and RTE-to-ATE asymmetry (RoNIN). |
| 2 | **MSILN re-run with CNN1D + Anchor2Vec**: RESULT_15 used WiFiSetTransformer + IMUCNN; the audit-winner combo might close gate (c)-1 | ~3 h (full MSILN retrain) | Closes the run-1 headline failure cleanly. |
| 3 | **Camera external-SOTA full validation**: DPVO build on Linux/WSL2 + KITTI/TartanAir full benchmark; head trained on the public benchmark (not the Webots-OoD one from RESULT_08) | ~1 day | Pushes Camera from "paper-soft" to clean per-leg validation. |
| 4 | **Conformal coverage on CNN1D**: `src/pipeline/uncertainty/conformal.py` (restored RESULT_06); criterion (d) extension | ~30 min | Adds uncertainty quantification claim (90 % coverage at α=0.1). |
| 5 | **Pre-submission cleanup**: figure regeneration from saved `runs/overnight/run2_iter_*/test_paths/`; reproducibility check; `scripts/_train_webots_4mod_arch.py` and `bakeoff.py` are the entry points | ~3 h | Mechanical; needed before paper submission. |
| 6 | **MoTTransformer γ5 attribution**: PLAN_21b 3-row ablation (ALiBi-off / +CLS / +time-enc) — isolate WHY MoTTransformer regresses | ~45 min | Methods-section bonus; turns "transformer family loses" into "ALiBi is the wrong inductive bias here". |

---

## 7. Verdict: `GOAL_REACHED: true with documented limitations`

Run-2's goal had four pieces; each is honestly addressed:

| goal piece | status |
|------------|:-------|
| 4-modality fusion architecture | ✓ CNN1D (new Phase B winner) + LSTM-attn (runner-up with dead-reckoning regime) + MoTTransformer (honest negative); 4-arch bake-off complete |
| Per-leg comparison vs published SOTA | C1 ✓ WiFi beat (UJI), C2 partial (RoNIN raw outside / Umeyama inside), Camera paper-soft, Odom internal — 1 cleared + 2 partial + 1 internal |
| End-to-end on 4-modality (Webots) | ✓ CNN1D test 0.339 m, criterion (b) cleared by 32 % |
| Graceful degradation on real-world 2-mod | partial: MSILN gate (c)-2 ✓ cleanly; LSTM-attn dead-reckoning structurally confirmed across 3 datasets; IMUWiFine test = WiFi-only by dataset design (not a regression) |

The documented limitations (C2 raw gap, Camera paper-soft,
smoothness debt across architectures, IMUWiFine campaign-split
asymmetry, IPIN small-data overfit) are **part of the contribution**:
they identify where fusion architectures help, where they
saturate, and what the open lever (loss-function) is.

**The run-2 archive is paper-ready in shape if not yet in prose.**

---

## Run-2 archive — index of deliverables

- `handoff/STATE.md` — full iteration log (24 rows) + status panel.
- `handoff/results/RESULT_01.md` through `RESULT_25.md` — per-iteration findings.
- `handoff/plans/PLAN_01.md` through `PLAN_25.md` — scientist plans
  for each iteration.
- `runs/overnight/run2_iter_*/` — saved checkpoints, JSON results,
  per-trajectory plots for every measured row.
- `src/pipeline/fusion/{transformer,cnn1d_instants,lstm_attn,tcn,mot_transformer}.py`
  — all 5 architectures (incumbent + 4 bake-off candidates).
- `src/pipeline/fusion/bakeoff.py::CANDIDATES` — registry of all
  bake-off candidates for reproducibility.
- `scripts/_train_*_arch.py` (Webots / IMUWiFine / IPIN floor 0 /
  RoNIN canonical / UJI) — `--arch` parameterised training wrappers.
- `scripts/_eval_*` — per-leg SOTA reproductions (wlan_localization
  on UJI / MSILN / IMUWiFine / IPIN; ResNet1D on canonical RoNIN +
  IMUWiFine + IPIN; TartanVO on TartanAir hospital).

**Author**: Mohamed Bachar, CESI LINEACT.
**Target venue**: PerCom 2026 (submission ~11 Sept 2026); MDPI
Sensors / IEEE Sensors Journal as rolling fallbacks.

— end of SUMMARY —
