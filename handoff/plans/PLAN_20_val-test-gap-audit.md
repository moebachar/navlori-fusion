# Plan 20 — Val/test gap audit across all datasets + methods (methodology check)

> **User flag 2026-05-26 ~08:30 local.** RESULT_19's CNN1D on
> IMUWiFine reports val **1.397 m** / test **7.094 m** — a **5×
> gap** in the wrong direction (test ≫ val). This contradicts the
> consistent pattern in Webots (val ≈ test, ±10–20 %) and inverts
> MSILN (RESULT_15 val 16.60 > test 14.02 — val > test). Before
> populating more rows of the main-results table with potentially
> flawed methodology, audit the split methodology + val/test
> behaviour across every dataset we've measured.
>
> If a real methodology bug surfaces (e.g. leak between
> train/val, or test-format mismatch on IMUWiFine per CLAUDE.md's
> "two raw formats coexist (train/val vs test)" note), the fix
> propagates to the entire main-results table. Better to find it
> on row 2 than row 6.

## Hypothesis

Three failure-mode candidates for the IMUWiFine 5× val/test gap:

1. **Train+val leak.** Val sequences overlap with train (paths
   sharing samples or time windows). Train and val share a
   distribution; test is genuinely OOD. Webots wouldn't show this
   because the canonical split is by path-id (no overlap).
2. **Test-format mismatch.** CLAUDE.md's IMUWiFine integration
   memory entry explicitly notes "two raw formats coexist
   (train/val vs test)". If our converter handles them
   differently — e.g. different RSSI normalisation, different IMU
   units, different sample-rate downsampling — test inputs are
   genuinely out-of-distribution at the *encoder* level, not the
   geometry level.
3. **Distribution shift (legitimate cross-session)**. IMUWiFine's
   test paths may be a different floor or recording session. Real,
   not a bug. MSILN's val > test (RESULT_15) is similar
   structurally — cross-session test could happen to be easier
   than within-session val.

Different remedies per failure mode; the audit must distinguish.

## Steps

### Step 0 — Tabulate val/test gaps across all measured rows (10 min)

Collect from existing RESULTs:

| dataset | run | method | val MAE | test MAE | gap (test−val)/val | source |
|---|---|---|---|---|---|---|
| Webots | iter 06 | FusionTransformer K=1 2-mod | 0.469 | 0.517 | +10 % | RESULT_06 |
| Webots | iter 09 | FusionTransformer K=1 3-mod | 0.448 | 0.489 | +9 % | RESULT_09 |
| Webots | iter 10 | FusionTransformer K=1 5-mod | 0.491 | 0.486 | −1 % | RESULT_10 |
| Webots | iter 13 | FusionTransformer K=4 4-mod | 0.394 | 0.417 | +6 % | RESULT_13 |
| Webots | iter 17 | CNN1D K=4 4-mod | 0.282 | 0.339 | +20 % | RESULT_17 |
| Webots | iter 17 | LSTM-attn K=4 4-mod | 0.301 | 0.340 | +13 % | RESULT_17 |
| MSILN | iter 15 | FusionTransformer + WiFiSetTransformer 2-mod | 16.60 | 14.02 | **−16 %** | RESULT_15 |
| IMUWiFine | iter 19 | wlan_localization | 4.17 | 8.50 | **+104 %** | RESULT_19 |
| IMUWiFine | iter 19 | RoNIN ResNet1D | 26.84 | n/a | n/a (test paths lack IMU?) | RESULT_19 |
| IMUWiFine | iter 19 | CNN1D | 1.397 | 7.094 | **+408 %** | RESULT_19 |
| IMUWiFine | iter 19 | LSTM-attn | 1.264 | 7.196 | **+469 %** | RESULT_19 |

Engineer fills in / verifies the table from each RESULT's saved
JSON. **Acceptance**: table populated; rows ranked by absolute
gap percentage; the IMUWiFine 4-row outlier is visually obvious.

Two patterns to surface:
- **Webots pattern**: small positive gap (+5 to +20 %); val < test;
  consistent across all 6 Webots runs.
- **MSILN pattern**: small NEGATIVE gap (−16 %); val > test.
  Unusual but only one data point.
- **IMUWiFine pattern**: huge positive gap (+100 to +470 %); val
  ≪ test; all 4 methods affected including SOTA — so it's NOT
  our fusion code.

The "all 4 methods affected including wlan_localization SOTA"
finding is load-bearing: it rules out failure-mode (1) above
narrowly (a train+val leak in OUR converter wouldn't affect
wlan_localization, which doesn't use our train/val split — it just
reads `imuwifine_floor4/` per path). So if wlan_localization shows
the same +104 % gap, the underlying val/test partition itself is
the issue, not our model code.

### Step 1 — Inspect IMUWiFine split metadata (15 min)

Engineer reads:
1. `configs/data/imuwifine.yaml` — which paths are in train/val/test?
2. `scripts/convert_imuwifine.py` — does the converter split by
   path-id, by session, by RSSI-availability, or by the documented
   "two raw formats"?
3. `data/imuwifine_floor4/path_*/metadata.json` (if it exists per
   the async_collection schema) — does each path carry a "session
   tag" or "format tag"?

The CLAUDE.md memory entry for IMUWiFine
(`memory/project_imuwifine.md` would be relevant if it's in the
project; otherwise the converter source) should say WHICH paths
are train/val vs test, and whether the "two formats" distinction
aligns with the train/val vs test partition.

**Acceptance**: explicit answer to "are IMUWiFine test paths in a
different raw format from train/val paths?"

### Step 2 — Distribution probe on IMUWiFine train/val vs test (10 min)

For one path in val and one path in test (engineer picks
representatives), report:
- WiFi RSSI distribution: mean, std, fraction of non-detected APs
  (the −100 sentinel).
- IMU acceleration channel means + stds (gravity-aligned check).
- Position trajectory range (m).

If val and test paths have materially different RSSI distributions
or IMU channel stats, that's failure-mode (2) — test inputs are
out-of-domain at the encoder.

**Acceptance**: 2-column distribution table (val sample vs test
sample); engineer notes which channels look mismatched.

### Step 3 — Per-path test MAE distribution on IMUWiFine CNN1D (5 min)

The single test_MAE=7.094 m number aggregates over all test paths.
Is one path dragging the mean? Engineer dumps per-path test MAE
from RESULT_19's saved JSON. If 1-2 paths dominate the test mean
(e.g. >15 m on one path while others land near val's 1.4 m), the
"big test MAE" is concentrated, not pervasive.

**Acceptance**: per-path test MAE distribution (mean, median, p25,
p75, p90, max); a verdict on whether the test set is
"systematically harder" or "anomaly-driven."

### Step 4 — Diagnosis + remedy

Based on Steps 1–3, classify the IMUWiFine gap as:

- **Failure mode (1) leak**: would be inconsistent with
  wlan_localization also showing +104 %. Reject this for IMUWiFine.
- **Failure mode (2) format mismatch**: if Step 2 surfaces
  distribution shifts AND Step 1 confirms different formats.
  Remedy: paper documents IMUWiFine's known train/val vs test
  format split; the gap is a known dataset artifact, not a fusion
  failure. Engineer's RESULT_19 row gets an "honest" footnote.
- **Failure mode (3) legitimate distribution shift**: if Step 2
  surfaces shifts but Step 1 says the converter handles them.
  Same remedy: documented dataset property; paper acknowledges.

For MSILN's val>test (one data point, −16 %): probably just a
test-easier-than-val coincidence on a small split; flag for
revisit if more cross-session iterations surface the same pattern.

### Step 5 — Decision + PLAN_21

Three-sentence verdict:
- Failure mode label for the IMUWiFine gap.
- Implication for the main results table (IMUWiFine row gets
  footnote / re-run / drop?).
- PLAN_21 = IPIN 2024 floor 0 (continue main-results table) IF
  the audit reveals a documented dataset property (no code fix
  needed); ELSE PLAN_21 = fix the converter / split (if a real
  bug surfaced) BEFORE adding IPIN.

## Sources

- RESULT_19 IMUWiFine numbers.
- RESULT_15 MSILN numbers.
- RESULT_06/09/10/13/14/17 Webots numbers.
- `configs/data/imuwifine.yaml` (in repo or restorable from run-1).
- `scripts/convert_imuwifine.py` (in repo or restorable from
  run-1).
- `memory/project_imuwifine.md` per CLAUDE.md memory ref ("two
  raw formats coexist (train/val vs test)").

## What to report back

In `handoff/results/RESULT_20_val-test-gap-audit.md`:

1. **Step 0** — populated gap table; outlier rows highlighted.
2. **Step 1** — IMUWiFine split methodology answer.
3. **Step 2** — distribution probe table (val vs test sample).
4. **Step 3** — per-path test MAE distribution for IMUWiFine
   CNN1D.
5. **Step 4** — failure-mode label.
6. **Step 5** — verdict + PLAN_21 recommendation (likely IPIN
   continuation, but if the audit surfaces a real bug, PLAN_21 is
   the fix).
7. **One open question** for scientist.

## Reversibility

- Steps 0–4: pure diagnostic; no code changes.
- Step 5: documentation.

Files committed: RESULT_20 + any small inspector scripts.

**Demand #3**: no vendored sources touched.

**Compute budget**: ≤ 40 min (no training, pure inspection).
- Step 0: 10 min.
- Step 1: 15 min (read config + converter + memory note).
- Step 2: 10 min (numpy + pandas summary on raw CSV).
- Step 3: 5 min (per-path table dump).
- Step 4–5: documentation.

If overrun: cut Step 2's distribution probe to 1 channel each
(WiFi RSSI mean + IMU accel-z mean) instead of full stats.
Don't skip Step 1 — the converter/config question is the key
diagnostic.
