# Scientist Note — Main results table (multi-iteration Phase C deliverable)

Logged 2026-05-26 ~08:00 local, after third-party review of
RESULT_17 (new CNN1D winner declared). Pinned as a directive on
PLAN_19 and following.

## The main table — required schema

| dataset        | modalities         | our CNN1D | our LSTM-attn | per-leg SOTA (method)            |
|----------------|---------------------|-----------|---------------|----------------------------------|
| Webots sim     | WiFi+IMU+Cam+Odom  | 0.339 m   | 0.340 m       | 0.417 incumbent + per-leg SOTAs  |
| IMUWiFine      | WiFi+IMU           | ?         | ?             | wlan_localization + RoNIN ResNet1D|
| IPIN 2024      | WiFi+IMU (floor 0) | ?         | ?             | wlan_localization + RoNIN ResNet1D|
| RoNIN canon.   | IMU                | ?         | ?             | RoNIN ResNet1D (5.140 m ✓)        |
| TartanAir hosp.| Camera             | ?         | ?             | TartanVO (0.518 m full / 0.012 last-20% ✓)|
| UJI IndoorLoc  | WiFi               | ?         | ?             | wlan_localization (15.17 m ✓)    |

Numbers in hand (reuse, don't re-run):
- UJI: wlan_localization **15.17 m** (RESULT_01).
- RoNIN canonical: ResNet1D **5.140 m raw ATE** (RESULT_07).
- TartanAir hospital P000: TartanVO **0.518 m full / 0.012 m last-20 % slice** (RESULT_08).
- Webots sim: incumbent FusionTransformer **0.417 m test** (RESULT_13/14).
- Webots sim: CNN1D **0.339 m test**, LSTM-attn **0.340 m test** (RESULT_17).
- Webots sim per-leg: Anchor2Vec **8.69 m val Euclid** on UJI (RESULT_01), IMUCNN 9.961 m raw ATE / 7.876 m Umeyama on canonical RoNIN (RESULT_07), DPVOMotionEncoder 0.293 m on TartanAir last-20% (RESULT_08).
- MSILN cross-session (NOT in this main table schema — paper handles MSILN as criterion (c) separately): 16.60 / 14.02 m K=4 2-mod B=128 (RESULT_15). Note: that run used `WiFiSetTransformer`, not Anchor2Vec — divergence flagged.

Numbers MISSING (must be measured in PLAN_19 → PLAN_22):
1. **CNN1D + LSTM-attn on IMUWiFine** + per-leg SOTA repro on
   IMUWiFine (wlan_localization + RoNIN ResNet1D — both NEW
   measurements; never run on this dataset).
2. **CNN1D + LSTM-attn on IPIN 2024 floor 0** + per-leg SOTA
   repro on IPIN 2024 floor 0 (wlan_localization + RoNIN
   ResNet1D — both NEW measurements). Use floor 0 only per
   directive; the multi-floor expansion can become a Phase C
   extension if time permits.
3. **CNN1D + LSTM-attn on RoNIN canonical** (single-modality IMU).
   The fusion stack is multi-mod-shaped; needs an encoder-only
   1-modality mode OR a degenerate 1-modality fusion run.
   Engineer's call on implementation.
4. **CNN1D + LSTM-attn on UJI** (per-scan; K=1 degenerate). UJI
   is not temporally ordered; the K-axis collapses. Run as a
   K=1 single-instant fusion on WiFi-only modality; document
   that the temporal architecture is degenerate here. May not be
   a meaningful comparison — engineer's call on whether to
   report or document the degeneracy.

## Iteration scoping

- **PLAN_18 (in flight)** — CNN1D + LSTM-attn ablations on
  Webots winner. NOT disrupted by this note. Step 7 PLAN_19
  recommendation should explicitly cite this directive.
- **PLAN_19** — IMUWiFine: CNN1D + LSTM-attn + wlan_localization
  + RoNIN ResNet1D on the same WiFi+IMU dataset. Per-leg SOTAs
  are the new measurements; never run on IMUWiFine before.
- **PLAN_20** — IPIN 2024 floor 0: same shape as PLAN_19.
- **PLAN_21** — RoNIN single-mod: CNN1D + LSTM-attn with IMU-only
  input. ResNet1D number already in hand (RESULT_07's 5.140 m);
  reuse.
- **PLAN_22** — UJI: CNN1D + LSTM-attn at K=1 (degenerate temporal).
  wlan_localization 15.17 m number already in hand (RESULT_01);
  reuse. Anchor2Vec 8.69 m number already in hand (RESULT_01);
  reuse.
- **PLAN_23** — SUMMARY draft + main results table assembly.
  Cross-dataset table populated; paper's claims (a)/(b)/(c)/(d)/(e)
  status panel; honest gaps (C2 cross-subject, Camera per-leg
  paper-soft, smoothness debt) documented; PLAN_24 = manual
  pre-submission cleanup OR run-2 closes.

## Budget reality

Each PLAN_19-22 iteration: ~60 min (one or two trainings + one or
two SOTA reproductions + eval). 4 iterations × 60 min = 4 hours.
PLAN_23 SUMMARY draft = ~30 min. Total ~4.5 hours after PLAN_18
closes. Stop at 18:00 local; PLAN_18 expected ~09:00; total
runway to 18:00 = ~9 hours; 4.5 hours of work fits with buffer for
the MSILN re-run (RESULT_15 used the wrong WiFi encoder — could
be quick to redo with Anchor2Vec) and conformal coverage if time
allows.

## Iteration-order rationale

IMUWiFine and IPIN before RoNIN/UJI because:
- They're 2-modality WiFi+IMU — same shape as our Phase B winner's
  MSILN runner; minimal new code needed.
- RoNIN and UJI are 1-modality (IMU only / WiFi only at K=1
  degenerate); each requires a new code path. Risk that they
  surface implementation blockers; better to land the 2-mod
  cross-dataset table first.
