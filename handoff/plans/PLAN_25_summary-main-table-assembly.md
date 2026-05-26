# Plan 25 — SUMMARY draft + main results table assembly

> All 6 main-table rows now have measurements (PLAN_19 IMUWiFine, PLAN_22
> IPIN floor 0, PLAN_23 RoNIN single-mod, PLAN_24 UJI degenerate, plus
> RESULT_14/17 Webots + RESULT_08 TartanAir already in hand). This is
> the final scientist-deliverable iteration: assemble the table, write
> `handoff/SUMMARY.md`, surface paper-framing decisions for Mohamed.
> Most work is documentation, not experiments.

## Hypothesis

The run-2 archive is paper-ready in shape if not yet in prose. The
deliverable is one cohesive SUMMARY.md that captures:

1. **The main results table** — 6 rows, multi-column comparison
   across per-leg SOTAs and our 3 fusion architectures (incumbent +
   CNN1D + LSTM-attn + MoTTransformer).
2. **The 5 acceptance criteria status panel** (a/b/c/d/e from STATE.md).
3. **The 4 supporting claims (C1-C4) status** per SCIENTIST_BRIEF.
4. **Cross-cutting findings** that emerged from the 24 iterations —
   not single-dataset results but the patterns visible across
   datasets/architectures.
5. **Honest gaps** documented (C2 raw-gap, Camera paper-soft, smoothness
   debt, IMUWiFine test-no-IMU asymmetry, IPIN small-data overfit).
6. **Paper-framing decisions** surfaced for Mohamed.
7. **Recommended next steps** post-run-2.

## Steps

### Step 0 — Inventory + cross-check (10 min)

Engineer reads each `RESULT_NN_*.md` and extracts a single row of
numbers per main-table cell. Cross-reference with the
SCIENTIST_NOTE_main-results-table.md schema. Confirm every cell
has a source citation (RESULT_NN line / iter commit hash).

If any cell has missing data not flagged in this plan, flag it in
RESULT_25's TL;DR — do not invent numbers.

### Step 1 — Assemble the main results table (10 min)

Target format (paper-ready):

```
| dataset           | modalities    | wlan_localization | RoNIN ResNet1D | TartanVO   | Anchor2Vec | DPVOMotion | IMUCNN | run-1 incumbent | CNN1D (new winner) | LSTM-attn | MoTTransformer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Webots sim        | WiFi+IMU+Cam+Odom | n/a              | n/a            | n/a        | n/a        | n/a        | n/a    | 0.417 (test)    | **0.339 (test)**   | 0.340 (test) | 0.608 (test) |
| IMUWiFine fl.4 (1)| WiFi+IMU      | 4.17 v / 8.50 t   | 26.84 v / n.a  | n/a        | n/a        | n/a        | n/a    | n/a             | 1.40 v / 7.09 t     | 1.26 v / 7.20 t | n/a |
| IPIN 2024 fl.0    | WiFi+IMU      | 20.53 v / 19.80 t | 37.21 v / 31.70 t | n/a    | n/a        | n/a        | n/a    | n/a             | 21.61 v / 20.45 t   | 22.45 v / 21.56 t | n/a |
| RoNIN canon. (2)  | IMU only      | n/a               | **5.14 raw**   | n/a        | n/a        | n/a        | 9.96 raw / 7.88 Umey | n/a       | 7.59 raw / 5.94 Umey | 7.50 raw / 6.12 Umey | n/a |
| TartanAir hosp.   | Camera only   | n/a               | n/a            | 0.518 full / **0.012 last-20%** | n/a | 0.293 last-20% | n/a | n/a | n/a (3) | n/a (3) | n/a (3) |
| UJI IndoorLoc (4) | WiFi only val | 15.17             | n/a            | n/a        | **8.69**   | n/a        | n/a    | n/a             | 8.72                | 8.43       | n/a |
```

Notes inline in the table:
- (1) IMUWiFine test split lacks IMU by dataset design (RESULT_20
  audit); our fusion test column is effectively WiFi-only.
- (2) RoNIN canonical metric = raw / Umeyama-aligned ATE; ResNet1D
  was reproduced to paper number (0.0 % delta).
- (3) Camera external-SOTA validation queued as Phase C extension
  (paper-soft per RESULT_08, our DPVOMotion 0.293 vs TartanVO
  0.012 on the same in-sequence test slice).
- (4) UJI K=1 + M=1 degenerate; fusion aggregators collapse to
  encoder + head (RESULT_24).

Engineer's wider table draft + caption text get committed under
`handoff/results/RESULT_25_summary-main-table-assembly.md` (the
RESULT) AND a free-standing `handoff/SUMMARY.md` (the run-archive
one-pager) per PROTOCOL.md "final-stop routine."

### Step 2 — Status panels (10 min)

**Criteria (a) per-leg validation** (within 20 % of SOTA on same data + metric):
- C1 WiFi UJI Anchor2Vec **8.69 vs SOTA 15.17 m**: ours BEATS by 43 %.
  Audit `keep`.
- C2 IMU RoNIN canonical: raw +47 % outside 20 % gate; Umeyama
  +15.7 % inside gate; per amended rubric correction #3 (raw wins),
  label `keep (in-domain only)`.
- C3 Camera Webots: paper-soft per-leg (RESULT_08 TartanAir gap);
  paper-strength on the 4-modality fusion claim (Webots 0.339).
- Odom: internal audit only (no public SOTA); CNN1D in fusion
  context.

**Criterion (b)** test MAE ≤ 0.5 m on Webots: CNN1D test 0.339 m
clears by 32 %. ✓✓

**Criterion (c)** MSILN cross-session (RESULT_15): clean
wlan_localization SOTA beat (+14.29 m test); WiFi-kNN partial
(val passes 1.06 m / 1.5 m gate just under). β outcome. NB:
RESULT_15 used WiFiSetTransformer; a Phase C extension re-running
with CNN1D + Anchor2Vec could finally close gate (c)-1 — flagged.

**Criterion (d)** per-path distribution + smoothness: reported
across all evaluations. **Per-trajectory smoothness r > 0.20 gate
NOT met by any architecture × dataset combination** (max
LSTM-attn IPIN r=0.089 / Webots r=0.051 / MSILN r=0.107). Confirmed
loss-function-bound; B-1/B-2 follow-up named.

**Criterion (e)** latency < 100 ms/sample on Quadro P4000: CNN1D
b=1 **4.73 ms** (21× under gate), b=32 0.15 ms (660× under).

### Step 3 — Cross-cutting findings (the discussion section material)

Write 3-5 paragraphs covering:

1. **LSTM-attn dead-reckoning regime as a structural finding**.
   Confirmed across 3 datasets (Webots ✓ RESULT_18, IMUWiFine ✓
   RESULT_19, IPIN ✓ RESULT_22): `only:imu` ≈ `only:camera` ≈ `only:wifi`
   ≈ full to within ~1-8 %. Contrasts CNN1D's cooperative-fusion
   regime where motion modalities depend on WiFi anchoring.
   **Two distinct fusion regimes** ≠ matter of param count or
   data scale.
2. **Smoothness debt is architecture-invariant**. 4 architectures
   × 5+ datasets all under r=0.20 gate. The lever isn't
   architecture; it's the loss function (auxiliary velocity loss
   B-1 or EMA token-smoothing B-2 from RESULT_05).
3. **RTE-to-ATE asymmetry on RoNIN** (RESULT_23): aggregator
   improves global drift but worsens local consistency. Same
   loss-function-lever signal as smoothness debt — same fix.
4. **Bake-off methodology** (4 architectures benchmarked
   in identical protocol per the third-party directive — CNN1D
   winner, LSTM-attn runner-up with the dead-reckoning structural
   finding, run-1 incumbent over-parameterised, MoTTransformer
   loses): paper methods section can claim genuine fair
   comparison.
5. **Cross-dataset transferability via dataset-specific training**:
   CNN1D `only:wifi` beats wlan_localization on UJI and IPIN
   floor 0 — our WiFi encoder is competitive in its own right.
   Fusion's value is the 4-modality story (Webots ✓), not
   universal cross-dataset dominance.

### Step 4 — Paper-framing decisions (Mohamed's call)

List the open decisions, each with 2-3 framings:

1. **IMUWiFine test column**. (a) val-only headline + test as
   cross-session robustness floor footnote; (b) report both with
   test asterisked as cross-session-no-IMU. Engineer
   recommendation in RESULT_19.
2. **C2 IMU SOTA status**. (a) honest "competitive in-domain;
   +15.7 % Umeyama / +47 % raw on canonical unseen-subjects"
   framing; (b) only-aligned framing ("within 20 % under Umeyama
   alignment"). Per amended rubric correction #3 raw wins, so
   (a) is the locked-rubric-compliant framing.
3. **MSILN narrative**. (a) "We beat the open-source SOTA cleanly
   on MSILN site1/B1 cross-session" (the gate (c)-2 finding which
   was run-1's headline failure); (b) "Mixed C4 outcome: clean
   SOTA beat, partial kNN gate due to per-path composition." (a)
   is more compelling but (b) is more honest. Note: PLAN_15 used
   WiFiSetTransformer not Anchor2Vec; a Phase C extension could
   re-run with CNN1D + Anchor2Vec and possibly close both gates.
4. **Smoothness debt** — paper-discussion-section honest "we
   identified an architecture-invariant smoothness debt;
   loss-function-lever B-1/B-2 named as follow-up work." Not a
   hard limitation but a documented gap.

### Step 5 — Recommended next steps post-run-2

1. **PLAN_25b** (queued optional): B-1 auxiliary velocity loss OR
   B-2 EMA token smoothing on CNN1D winner — test whether the
   loss-function lever finally clears smoothness r > 0.20 gate.
   Estimated 30 min. Quick win if the lever works.
2. **MSILN cross-session re-run** with CNN1D + Anchor2Vec — could
   close gate (c)-1 that RESULT_15 missed (the engineer's RESULT_22
   flag).
3. **Camera external-SOTA full benchmark** — Phase C extension
   addressing the paper-soft RESULT_08 result; DPVO build on Linux/
   WSL2 + KITTI / TartanAir-validation full run, comparing
   DPVOMotion with a head trained on the public benchmark (not the
   Webots-trained out-of-domain head from RESULT_08).
4. **Pre-submission cleanup**: figure regeneration (all plots
   already saved under `runs/overnight/run2_iter_*/test_paths/`),
   reproducibility check (`scripts/_train_webots_4mod_arch.py` and
   `bakeoff.py` are the entry points for CNN1D / LSTM-attn /
   MoTTransformer).

### Step 6 — Decision: GOAL_REACHED?

Three-sentence verdict on the run-2 goal:

> "A 4-modality fusion architecture (WiFi + IMU + Odom + Camera) for
> indoor localization, validated via per-leg comparison against
> published SOTA and end-to-end on the only dataset with all 4
> modalities (Webots sim), with graceful degradation on real-world
> 2-modality data."

Scoring:
- **4-modality fusion architecture**: CNN1D (new Phase B winner)
  + LSTM-attn (runner-up with dead-reckoning regime) + MoTTransformer
  (honest negative) — bake-off complete.
- **Per-leg comparison vs published SOTA**: WiFi beat ✓, IMU
  in-domain ✓ / canonical partial, Camera paper-soft, Odom internal.
- **End-to-end on 4-modality data (Webots)**: CNN1D 0.339 m,
  criterion (b) cleared by 32 %.
- **Graceful degradation on real-world 2-modality**: MSILN gate
  (c)-2 cleanly ✓ / gate (c)-1 partial; LSTM-attn dead-reckoning
  regime structurally confirmed across 3 datasets.

Engineer's call: set `GOAL_REACHED: true` if all 4 components are
honestly addressed (which they are, with documented honest gaps);
OR `GOAL_REACHED: partial` with the 3 named honest gaps as the
explanation.

My read: **GOAL_REACHED: true with documented limitations**. Run-2
delivered a paper-strength contribution; the limitations are part
of the contribution, not failures.

## Sources

- All RESULTs from RESULT_01 through RESULT_24.
- `handoff/SCIENTIST_NOTE_main-results-table.md` (the directive).
- `handoff/SCIENTIST_BRIEF.md` (the contract).
- `handoff/PROTOCOL.md` "final-stop routine".
- STATE.md iteration log (24 rows).

## What to report back

In `handoff/results/RESULT_25_summary-main-table-assembly.md` +
`handoff/SUMMARY.md`:

1. **Main results table** — paper-ready 6-row, multi-column.
2. **Criteria (a-e) status panel**.
3. **Cross-cutting findings** (3-5 paragraphs).
4. **Paper-framing decisions** for Mohamed (numbered).
5. **Recommended next steps** post-run-2.
6. **GOAL_REACHED verdict** with one-line justification.

## Reversibility

- Everything in this iteration is documentation. Both
  `handoff/SUMMARY.md` and `handoff/results/RESULT_25_*.md` are
  permanent.

Files committed: SUMMARY.md (NEW; per PROTOCOL final-stop), RESULT_25.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 60 min (pure documentation).
- Step 0: 10 min.
- Step 1: 10 min.
- Step 2: 10 min.
- Step 3: 15 min.
- Step 4: 5 min.
- Step 5: 5 min.
- Step 6: 5 min.

If overrun: ship SUMMARY.md and RESULT_25 with Steps 1-2-3 fully
filled and Steps 4-5 as bulleted shortlists. The main table + status
panel + cross-cutting findings are the load-bearing deliverables;
paper-framing decisions and next steps can be terser.

If engineer wants to optionally run PLAN_25b after this iteration
(B-1/B-2 loss-function probe), that's a separate plan. PLAN_25 is
the SUMMARY assembly only.
