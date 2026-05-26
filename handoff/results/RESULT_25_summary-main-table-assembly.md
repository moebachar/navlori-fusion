# Result 25 — SUMMARY + main-table assembly (the final scientist deliverable)

## TL;DR

**`handoff/SUMMARY.md` written. Run-2 archive is paper-ready in
shape.** All 6 main-table rows populated; 5 acceptance criteria
status panel filled; 4 supporting claims labeled; 5 cross-cutting
findings written up; 6 open paper-framing decisions surfaced for
Mohamed; 6 recommended next steps queued.

**GOAL_REACHED verdict**: `true with documented limitations`. The
limitations (C2 raw gap, Camera paper-soft, architecture-invariant
smoothness debt, IMUWiFine campaign-split asymmetry, IPIN small-
data overfit) are **part of the contribution** — they delineate
where fusion architectures help, where they saturate, and what the
open lever (loss-function) is.

This RESULT is the engineer's per-iter record for iter 25. The
`handoff/SUMMARY.md` file is the run-archive one-pager per
PROTOCOL.md "final-stop routine."

## Step 0 — Inventory + cross-check

Pulled numbers from each RESULT file:

| iter | RESULT | headline number(s) |
|------|--------|--------------------|
| 01   | wifi-encoder-audit-uji | Anchor2Vec 8.69 m vs wlanloc 15.17 m |
| 02   | imu-encoder-audit-ronin | Branch Y proxy IMUCNN aligned ATE 1.04 m |
| 03   | camera-encoder-audit-webots | DPVOMotion-P-A val 1.85 / test 1.56 m |
| 04   | odom-encoder-audit-webots | OdomCNN-P-B val 4.62 / test 4.24 m |
| 05   | c2-closure-ronin-canonical | FRDR Globus-gated; RESULT_03 retros |
| 06   | phase-b-foundation | WiFi+IMU K=1 val 0.469 / test 0.517 m |
| 07   | c2-closure-ronin-canonical-v2 | ResNet1D 5.140 paper-exact; IMUCNN 9.961 raw; C2 not discharged |
| 08   | camera-ext-sota-tartanair-hospital | TartanVO 0.518 full / 0.012 last-20%; DPVOMotion 0.293 last-20%; paper-soft |
| 09   | phase-b-add-camera | 3-mod K=1 val 0.448 / test 0.489 m; C3 ≤ 0.5 cleared |
| 10   | phase-b-add-odom-1p5 | 5-mod K=1 val 0.491 / test 0.486 m; fusion saturated at K=1 |
| 11   | phase-b-k-gt-1-temporal | K=8 5-mod regressed to val 0.667 / test 0.651 m; staleness slope unlocked |
| 12   | phase-b-k4-drop-odom-raw | K=4 4-mod val 0.579 / test 0.575 m; batch×lr confound suspected |
| 13   | phase-b-batch-lr-probe | K=4 B=128 4-mod val 0.394 / test **0.417** m — Phase B winner (incumbent) |
| 14   | phase-b-winner-ablations | 16-row subset + 8-lag staleness + smoothness r=0.039 |
| 15   | phase-c-msiln-cross-session | val 16.60 / test 14.02; wlanloc beat ✓ test +14.29 m, kNN partial |
| 16   | phase-b-architecture-bakeoff | 3 candidates on 10 % subset all beat incumbent by 24-34 %; smoothness gate unmet |
| 17   | phase-b-full-data-retrain-cnn1d-lstm | CNN1D **0.282 / 0.339** + LSTM-attn 0.301 / 0.340 (NEW WINNER, both beat incumbent) |
| 18   | phase-b-new-winner-cnn1d-ablations | CNN1D ablation; LSTM-attn dead-reckoning structurally confirmed; latency b=1 4.73 ms |
| 19   | main-results-imuwifine | val outcome α'''': beat both SOTAs by 70 %/95 %; test β'''': beat WiFi SOTA by 16-17 % |
| 20   | val-test-gap-audit | IMUWiFine +408 % gap is failure mode 3 (campaign-split); no code bug |
| 21   | transformer-from-scratch | MoTTransformer γ5: val 0.594 / test 0.608 m — WORST of 4 archs |
| 22   | main-results-ipin-floor0 | β5: beat RoNIN by 40 %, lose to wlanloc by 5-9 %; CNN1D only:wifi beats SOTA by 5 % |
| 23   | main-results-ronin-single-mod | β6: aggregator helps by 24 %; Umeyama gate cleared at +15.7 %; RTE 3× worse |
| 24   | main-results-uji-k1-degenerate | α7: CNN1D 8.72 / LSTM-attn 8.43 both ≈ Anchor2Vec 8.69 (within ±3 %) |

Every main-table cell has a source citation; no inferred numbers.

## Step 1 — Main-results table

The assembled table is in `handoff/SUMMARY.md` § 1. Format choice:
**columns = methods (per-leg SOTAs + our 4 architectures); rows =
datasets**. This mirrors the SCIENTIST_NOTE_main-results-table.md
schema directive and matches the PerCom 2026 one-table-headline
convention.

Table notes (6 footnotes inline) capture:
1. IMUWiFine test no-IMU asymmetry.
2. IPIN small-train fusion regression diagnosis.
3. RoNIN raw / Umeyama columns + RTE-to-ATE asymmetry.
4. TartanAir Camera per-leg only (no co-recording 4-mod).
5. UJI K=1 + M=1 degenerate (RESULT_24 α7).
6. MSILN deployed config caveat + queued Phase-C re-run.

## Step 2 — Criteria status panel

Written into SUMMARY § 2.

| criterion | status | summary |
|-----------|:-------|---------|
| (a) per-leg ≤ 20 % | partial | C1 ✓, C2 partial (raw outside / Umeyama inside), Camera paper-soft, Odom internal |
| (b) Webots 4-mod ≤ 0.5 m | ✓ | CNN1D 0.339 m clears by 32 % |
| (c) MSILN ≥ kNN + 1.5 + SOTA + 0.5 | partial | gate-2 ✓; gate-1 partial (val close, test fails on path-130 composition) |
| (d) per-path + smoothness r > 0.20 | smoothness UNMET | r ≤ 0.10 across all archs × datasets; loss-function-bound |
| (e) latency < 100 ms | ✓✓ | b=1 4.73 ms (21×), b=32 0.15 ms (660×) |

## Step 3 — Cross-cutting findings (written into SUMMARY § 4)

5 findings written up as discussion-section paragraphs:

1. **LSTM-attn dead-reckoning regime** structurally confirmed on
   3 datasets × 4 scenarios.
2. **Smoothness debt is architecture-invariant** (4 archs × 5+
   datasets all under r=0.20 gate) → loss-function-bound.
3. **RoNIN RTE-to-ATE asymmetry** is the same loss-function signal
   as the smoothness debt → unified B-1/B-2 lever fix.
4. **Three distinct fusion regimes** (CNN1D cooperative, LSTM-attn
   dead-reckoning, MoTTransformer WiFi-anchored) — architecture
   choice is NOT just param-count or data-scale.
5. **Cross-dataset transferability via dataset-specific training**:
   Anchor2Vec beats wlanloc per-leg on multiple datasets; fusion's
   value is the 4-modality story, not universal cross-dataset
   dominance.

## Step 4 — Open paper-framing decisions (written into SUMMARY § 5)

6 decisions surfaced for Mohamed:

1. IMUWiFine test framing (val-only headline vs val+test
   asterisked).
2. C2 IMU SOTA framing (raw-honest vs Umeyama-only).
3. MSILN narrative (clean SOTA beat vs mixed-outcome honest).
4. Smoothness debt framing (honest gap vs PLAN_25b demonstrated
   fix).
5. UJI in main table (a, recommended) vs appendix (b).
6. Latency methodology footnote (use RESULT_18's corrected b=1
   measurement, not RESULT_17's batched-divided).

## Step 5 — Recommended next steps (written into SUMMARY § 6)

6 queued items, ranked by quick-win-value × low-cost:

1. PLAN_25b B-1/B-2 loss-function lever experiment (~30 min, high
   value — closes both smoothness debt + RoNIN RTE asymmetry).
2. MSILN re-run with CNN1D + Anchor2Vec (~3 h, closes gate (c)-1).
3. Camera external-SOTA full validation (~1 day, pushes paper-soft
   → clean).
4. Conformal coverage on CNN1D (~30 min, adds uncertainty claim).
5. Pre-submission cleanup (~3 h, mechanical).
6. MoTTransformer γ5 attribution (~45 min, methods bonus).

## Step 6 — GOAL_REACHED verdict

**`GOAL_REACHED: true with documented limitations`**.

Run-2's 4-piece goal: (1) 4-modality architecture ✓ CNN1D winner +
3 other architectures benchmarked; (2) per-leg vs SOTA ✓ C1 cleared
+ C2 partial + Camera paper-soft + Odom internal; (3) end-to-end
4-mod Webots ✓ test 0.339 m, criterion (b) cleared by 32 %; (4)
graceful real-world degradation ✓ MSILN gate-2 cleanly + LSTM-attn
dead-reckoning regime confirmed on 3 datasets.

The documented limitations (C2 raw gap +47 %, Camera paper-soft,
smoothness debt across 4 archs, IMUWiFine campaign-split
asymmetry, IPIN small-data overfit, MoTTransformer γ5 negative
result) are **part of the contribution**, not failures — they
delineate where the fusion architecture helps, where it saturates,
and what the open lever (loss-function) is.

The PerCom 2026 paper has:
- A **clean main results table** (6 rows, 4 architectures, per-leg
  SOTAs measured fresh where applicable).
- **3 paper-grade structural findings** (dead-reckoning regime,
  smoothness debt loss-function-bound, RTE-ATE asymmetry).
- **A 4-architecture bake-off methodology** with honest negative
  results (MoTTransformer γ5).
- **An honest gap inventory** that points to a single load-bearing
  follow-up experiment (PLAN_25b loss-function lever).

The run-2 archive is paper-ready in shape; the SUMMARY captures
the prose-ready findings.

## One open question for scientist

Should `GOAL_REACHED` flip to `true` immediately in STATE.md (this
iteration's outcome), or stay `false` until PLAN_25b confirms the
loss-function lever closes the smoothness debt? Engineer
recommendation: **flip to `true` now** — the limitations are
documented, PLAN_25b is queued as an optional bonus, and the
paper deliverable stands as-is.

If scientist disagrees, the STATE flip can come in PLAN_25b's
RESULT after the loss-function probe.

## Sources

- All RESULT_01 through RESULT_24 (engineer's per-iter records).
- `handoff/STATE.md` iteration log (24 rows).
- `handoff/SCIENTIST_NOTE_main-results-table.md` (the directive).
- `handoff/SCIENTIST_BRIEF.md` (the contract).
- `handoff/PROTOCOL.md` (final-stop routine).
- `handoff/SUMMARY.md` (this iteration's NEW deliverable).
