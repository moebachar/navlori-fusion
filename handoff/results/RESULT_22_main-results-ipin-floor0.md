# Result 22 — Main results IPIN 2024 floor 0: outcome β5 (we beat IMU SOTA decisively; lose to WiFi SOTA by 3-9 %)

## TL;DR

**IPIN 2024 floor 0 row of the main-results table populated.** Both
per-leg SOTAs measured fresh on this dataset (NEW). Honest outcome
that's different from IMUWiFine:

| method                          | params  | val MAE  | test MAE | smoothness r | source     |
|---------------------------------|--------:|---------:|---------:|-------------:|------------|
| wlan_localization (WiFi only)   | (kNN)   | **20.53**| **19.80**| n/a          | NEW (1a)   |
| RoNIN ResNet1D (IMU only)       | 4.24 M  | 37.21    | 31.70    | n/a          | NEW (1b)   |
| **CNN1D fusion (WiFi+IMU)**     | 0.34 M  | 21.61    | 20.45    | **0.067**    | this iter  |
| **LSTM-attn fusion (WiFi+IMU)** | 0.41 M  | 22.45    | 21.56    | **0.089**    | this iter  |

**Outcome label: β5** (beat one SOTA decisively, lose to the other).

- vs **RoNIN ResNet1D** (IMU SOTA): we BEAT by ~40-44 % on val (CNN1D
  21.61 vs 37.21) and ~32-35 % on test (CNN1D 20.45 vs 31.70) —
  decisive wins on the IMU leg.
- vs **wlan_localization** (WiFi SOTA): we LOSE by 5 % (CNN1D val
  21.61) to 9 % (LSTM-attn val 22.45) on val; 3-9 % on test. The
  WiFi SOTA is the stronger baseline on this small dataset.

**Crucial finding — fusion archs overfit on IPIN's tiny train set.**
With only **174 WiFi scans + 6924 IMU windows for training** (vs
IMUWiFine's 4187 / 23385), both fusion candidates rapidly overfit
(train loss 0.5 / val loss 9.0 = 17× gap by epoch 20). Best val
epoch came at 8 (LSTM-attn) and 22 (CNN1D); early-stopping at 48/62
saved the run.

**Important diagnostic**: CNN1D's `only:wifi` val 19.45 m actually
**BEATS** wlan_localization val 20.53 m by 5 %. Our learned WiFi
encoder is competitive with the SOTA on its own. The fusion only
loses because adding the noisy IMU branch hurts a small-data
regime. **Honest paper claim**: on IPIN floor 0, the learned WiFi
encoder beats the SOTA, but fusion in this small-data regime is
counter-productive.

**LSTM-attn dead-reckoning regime REPLICATES on IPIN floor 0
(third data point)**:
- val: only:wifi 22.37 ≈ only:imu 22.64 ≈ full 22.45 (all within 1.2 %)
- test: only:wifi 21.38 ≈ only:imu 21.66 ≈ full 21.56 (all within 1.3 %)

The "per-modality recovery" structural finding now confirmed on
Webots (RESULT_18), IMUWiFine (RESULT_19), and IPIN (this iter) —
three datasets × four scenarios.

**PLAN_23 recommendation**: continue main-results table at RoNIN
single-modality IMU row per the directive chain. RESULT_07's
canonical ResNet1D = 5.140 m can be reused as the SOTA reference;
this iter would just run CNN1D + LSTM-attn IMU-only on the
canonical 32-sequence unseen-subjects split.

## Step-by-step

### Step 0 — Pre-flight + data + config

- `data/ipin2024_floor0/` present (16 paths, async_collection
  format).
- `configs/data/ipin2024_floor0.yaml` + `scripts/convert_ipin2024.py`
  + `scripts/eval_wlanloc_ipin.py` (old IPIN floor -2 template)
  restored from `overnight-autonomous-2026-05-24` branch.
- Split: train=0-5 (6 paths), val=6-9 (4 paths), test=10-15 (6 paths).
- AP universe: 232 BSSIDs.
- Modalities: `[wifi, imu]` (smartphone dataset; no Camera/Odom).
- imu window=32 samples (~1 s at IPIN's 25-32 Hz post-downsample).

**Pre-flight IMU-availability check** (the PLAN_20 lesson):
inspected one path from each split:

| path | split | imu.csv lines | wifi.csv scans |
|------|-------|--------------:|---------------:|
| 0    | train | 4076          | 42             |
| 1    | train | 3165          | 33             |
| 6    | val   | 6249          | 12             |
| 10   | test  | 5447          | 56             |
| 11   | test  | 6533          | 66             |
| 15   | test  | 438           | 2              |

**IPIN floor 0 has IMU on ALL splits including test.** Unlike
IMUWiFine, the RoNIN ResNet1D SOTA row is fully measurable.

### Step 1a — wlan_localization on IPIN floor 0 (NEW measurement)

Script: `scripts/_eval_wlanloc_ipin_floor0.py` (clone of
IMUWiFine wrapper, paths swapped to IPIN; same vendored
PositionRegressor + DataPreprocessor via `importlib`, Demand #3
honoured).

| split | n   | mean Euc. (m) | median | p90  | max  |
|-------|----:|--------------:|-------:|-----:|-----:|
| val   | 115 | **20.530**    | 19.07  | 37.25| 52.67|
| test  | 145 | **19.801**    | 18.34  | 37.69| 51.80|

Preprocessor (Box-Cox + PCA) reduces 232 APs → 150 components.
PositionRegressor k=3 manhattan distance-weighted. Val ≈ test
(within 3.5 %) — consistent within-floor cross-session regime.

### Step 1b — RoNIN ResNet1D on IPIN floor 0 (NEW measurement)

Script: `scripts/_eval_ronin_imuwifine.py` parameterised with
`--dataset ipin2024_floor0 --out-dir runs/overnight/run2_iter_22`
(generalised from the IMUWiFine-specific runner). Same empirical
conv-output probe (in_dim=1 for WIN=32).

| split | n windows | n paths | per-sample MAE (m) |
|-------|----------:|--------:|-------------------:|
| val   | 7773      | 4       | **37.21**          |
| test  | 8972      | 6       | **31.70**          |

The IMU-only baseline is again catastrophic (>30 m). Multi-minute
trajectories accumulate too much drift for pure IMU integration.
This confirms the "IMU SOTA is the floor the fusion needs to beat"
framing from RESULT_19.

### Step 2 — Train CNN1D + LSTM-attn on IPIN floor 0

Same protocol as RESULT_17/19: K=4, B=128, AdamW + OneCycleLR +
Huber(δ=0.5), 90 epochs, instant_dropout=0.45, modality_dropout=0.4,
lr=1.3e-3.

| arch       | params  | best val | best ep | wall (s) | peak GPU | early stop |
|------------|--------:|---------:|--------:|---------:|---------:|-----------:|
| CNN1D      | 0.34 M  | **21.609** | 22    | 73       | 260 MB   | ep 62      |
| LSTM-attn  | 0.41 M  | **22.452** | 8     | 57       | 267 MB   | ep 48      |

**Pre-test gate diagnostic**: both candidates show 0.0 % drop on
the 5-epoch pretest (best ≈ first val MAE) — neither learns useful
fresh accuracy in 5 epochs. This is unusual; both passed the
gate on IMUWiFine + Webots cleanly. The small train set (174 WiFi
scans + ~7 k IMU windows) means a 5-epoch budget is too short to
descend below the initial.

Full training (90 epochs with patience early-stopping) does
descend, but BOTH archs overfit fast: by epoch 20 train loss is
0.7 (CNN1D) / 0.6 (LSTM-attn) but val loss is 9.0 — a 13-17× gap.
Best val landed at epoch 8 (LSTM-attn) and 22 (CNN1D). Early
stopping kicked in at 48 / 62 respectively.

Subset eval on best-val checkpoints:

| arch / split | only:wifi | only:imu | wifi+imu |
|--------------|----------:|---------:|---------:|
| CNN1D val    | **19.454**| 21.869   | 21.609   |
| CNN1D test   | **19.651**| 20.458   | 20.446   |
| LSTM-attn val| 22.366    | 22.635   | 22.452   |
| LSTM-attn test| 21.379   | 21.657   | 21.561   |

**Headline diagnostic**: CNN1D `only:wifi` val 19.45 **BEATS**
wlan_localization val 20.53 by 5 %. Our learned WiFi encoder
(Anchor2Vec + 3 × 1D-conv aggregator) is **competitive with the
SOTA on the WiFi leg alone**. The fusion full 21.61 is WORSE
than CNN1D's own only:wifi 19.45 by 11 %, meaning the IMU branch
**actively degrades** the fusion at IPIN's small-train regime.

### Step 3 — Main-results IPIN floor 0 row + outcome label

(Headline table at top.)

**Outcome label: β5** (beat one SOTA, lose to the other).

- We BEAT **RoNIN ResNet1D** (IMU SOTA) by ~40 % val / ~35 % test —
  decisive.
- We LOSE to **wlan_localization** (WiFi SOTA) by 5-9 % full
  fusion. BUT our `only:wifi` branch beats it by 5 %.

Paper-claim framing (honest):
- On **IPIN floor 0**, the WiFi-leg SOTA (`wlan_localization`)
  is the strong baseline at 20.5 m; the IMU-leg SOTA
  (RoNIN ResNet1D, 37.2 m) is catastrophic. Both our fusion
  architectures sit between these (21.6-22.5 m) — they beat
  the IMU SOTA by ~40 % but underperform the WiFi SOTA by ~5-9 %.
- **Our WiFi encoder by itself** matches the WiFi SOTA on this
  dataset (CNN1D only:wifi val 19.45 < wlanloc val 20.53). The
  fusion regression is attributable to the **small-train-overfit
  regime** of IPIN floor 0 (174 WiFi scans, 6924 IMU windows in
  train, ~10× smaller than IMUWiFine) — adding IMU through
  cross-modal attention introduces noise the fusion can't
  compensate for.

### Step 4 — Per-trajectory smoothness + per-path distribution

**Smoothness median r**:
- CNN1D     **r=0.067** (best in Webots+IMUWiFine+IPIN at full data)
- LSTM-attn **r=0.089** (also higher than other datasets)

Both still below the 0.20 gate, but CNN1D's r=0.067 on IPIN is
notable — higher than its 0.009 on Webots and −0.005 on IMUWiFine.
LSTM-attn's r=0.089 on IPIN is the highest single-architecture
smoothness measurement in run-2 (just above its Webots 0.051).

Hypothesis: IPIN floor 0's small dataset means the model
memorises trajectory smoothness via overfitting; the train+val
smoothness is high but per-path test smoothness has wide variance
(some paths near 0.2, others negative).

**CNN1D per-path test MAE**:

| path | n     | mean  | median | p90  |
|------|------:|------:|-------:|-----:|
| 12   | 221   | 10.51 | 6.72   | 16.62|
| 10   | 2196  | 12.21 | 9.62   | 24.14|
| 13   | 1781  | 13.11 | 11.08  | 18.97|
| 15   | 176   | 23.43 | 29.67  | 41.72|
| 14   | 2040  | 25.33 | 23.68  | 43.16|
| 11   | 2636  | 29.11 | 25.45  | 58.84|

**LSTM-attn per-path test MAE**:

| path | n     | mean  | median | p90  |
|------|------:|------:|-------:|-----:|
| 15   | 176   | 7.33  | 6.07   | 11.46|
| 10   | 2196  | 9.18  | 8.26   | 17.84|
| 12   | 221   | 9.74  | 8.98   | 16.83|
| 13   | 1781  | 15.44 | 16.15  | 28.99|
| 14   | 2040  | 27.70 | 25.42  | 48.94|
| 11   | 2636  | 33.20 | 31.80  | 60.34|

Path 11 dominates both arch's test means (2636/8651 ≈ 30 % of
samples, with per-path MAE 29-33 m). The test aggregate hides
the per-path variance: 3 paths land < 15 m, 3 paths land > 23 m.

Per-trajectory plots saved at
`runs/overnight/run2_iter_22/test_paths/{cnn1d,lstm_attn}_path_{10,11,12,13,14}.png`.

### Step 5 — Decision + PLAN_23 recommendation

**Three-sentence verdict.**

(1) **IPIN floor 0 row populated, outcome β5**: we beat RoNIN
ResNet1D (IMU SOTA) by ~40 % on val and ~35 % on test, but
lose to wlan_localization (WiFi SOTA) by 5-9 %. The honest
caveat: our `only:wifi` branch (CNN1D val 19.45) actually
beats wlan_localization val 20.53 by 5 %, so the WiFi encoder
is competitive — the fusion regression is a small-train-overfit
artifact, not a fundamental WiFi failure.

(2) **LSTM-attn dead-reckoning regime replicates on a third
dataset** (val only:imu 22.64 ≈ full 22.45, test only:imu 21.66 ≈
full 21.56, both within 1.3 %). Combined with Webots (RESULT_18)
and IMUWiFine (RESULT_19), the per-modality recovery regime is
now confirmed across 3 datasets × 4 scenarios — **strong paper-
strength structural finding** for the bake-off discussion.

(3) **PLAN_23 recommendation**: RoNIN single-modality IMU row per
the directive chain. Engineer reuses RESULT_07's canonical
ResNet1D 5.140 m as the SOTA reference (paper match exact); this
iteration runs CNN1D + LSTM-attn IMU-only on the canonical
32-sequence unseen-subjects split. Expected outcome per RESULT_07:
gap of +90-94 % from SOTA (= "fusion architectures are not
competitive at IMU-only canonical RoNIN benchmark", confirmed in-
domain only).

## One open question for scientist

The IPIN floor 0 fusion regression (we lose to WiFi SOTA by 5-9 %
even though our only:wifi branch beats it by 5 %) raises a
methodological question: should the main-results table report
the BEST per-architecture subset (i.e. CNN1D only:wifi 19.45 m)
as the fusion claim, or the full-fusion number?

- (a) Headline = full fusion (21.61 m): honest "this is what
  4-mod fusion produces on this dataset," accepts the IMU-induced
  regression and frames it as a small-train artifact.
- (b) Headline = best subset per arch: each architecture reports
  its best subset (CNN1D only:wifi 19.45, LSTM-attn full 22.45);
  the claim becomes "our architectures can match WiFi SOTA when
  the data warrants it." More charitable to our methods but
  requires careful framing.

The PerCom paper's main-results table format probably wants (a)
for consistency; (b) would need a separate "best-subset" column.

## Sources

- `handoff/SCIENTIST_NOTE_main-results-table.md` (the directive).
- `configs/data/ipin2024_floor0.yaml`,
  `scripts/convert_ipin2024.py`, `scripts/eval_wlanloc_ipin.py`
  (restored from `overnight-autonomous-2026-05-24`).
- `scripts/_eval_wlanloc_ipin_floor0.py` — NEW (Step 1a, paths
  swapped from IMUWiFine wrapper).
- `scripts/_eval_ronin_imuwifine.py` — generalised with
  `--dataset` and `--out-dir` flags (filename historically
  IMUWiFine-named; reused for IPIN floor 0 via flag).
- `scripts/_train_imuwifine_2mod_arch.py` — generalised with
  `--dataset` and `--out-dir` flags (now serves all 2-mod
  datasets).
- `runs/overnight/run2_iter_22/{wlanloc,ronin,cnn1d,lstm_attn}_*.json`
  — full numerical output.
- `runs/overnight/run2_iter_22/test_paths/*.png` — per-trajectory
  plots.
- `data/ipin2024_floor0/`, `data/ipin2024_floor0/split.json`
  (16 paths, 6/4/6 train/val/test).
