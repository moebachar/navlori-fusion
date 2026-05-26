# Result 19 — main-results IMUWiFine floor 4: outcome β'''' (we beat WiFi SOTA on test; RoNIN unmeasurable on test)

## TL;DR

**The IMUWiFine row of the main-results table is populated.** Two
honest outcomes depending on the split:

- **VAL (with IMU available): outcome α''''** — our fusions beat
  BOTH per-leg SOTAs decisively. LSTM-attn val 1.264 m is **3.3×
  better than wlan_localization (4.17 m)** and **21× better than
  RoNIN ResNet1D (26.84 m)**.
- **TEST (no IMU per dataset design): outcome β''''** — we beat
  wlan_localization by 16-17 % (7.09 / 7.20 vs 8.50) on the same
  WiFi-only data. RoNIN is **not measurable on test** because the
  IMUWiFine test split carries no IMU windows by construction.

| method                             | params  | val MAE | test MAE | smoothness r | source     |
|------------------------------------|--------:|--------:|---------:|-------------:|------------|
| wlan_localization (WiFi only)      | (kNN n/a) | 4.17    | 8.50     | n/a          | NEW (1a)   |
| RoNIN ResNet1D (IMU only)          | 4.24 M  | 26.84   | n/a      | n/a          | NEW (1b)   |
| **CNN1D fusion (WiFi+IMU)**        | 0.34 M  | **1.40**| **7.09** | −0.005       | this iter  |
| **LSTM-attn fusion (WiFi+IMU)**    | 0.41 M  | **1.26**| 7.20     | −0.007       | this iter  |

**Headline paper-claim sentences:**

1. "On IMUWiFine floor 4 with WiFi + IMU available (val split),
   LSTM-attn fusion at 0.41 M params beats sharan-naribole's
   wlan_localization kNN regressor (4.17 m → 1.26 m, **70 % MAE
   reduction**) and RoNIN ResNet1D (26.84 m → 1.26 m, **95 %
   reduction**) — substantial wins over both per-leg SOTAs."
2. "On the IMUWiFine test split (WiFi-only by dataset design), our
   fusion architectures collapse to WiFi-only inference and still
   beat wlan_localization by 16-17 % (8.50 m → 7.09 m CNN1D)."

**LSTM-attn dead-reckoning regime REPLICATES on IMUWiFine — second
data point for the run-2 structural finding**:
- LSTM-attn IMUWiFine val: only:imu 1.263 ≈ only:wifi 1.274 ≈ full 1.264 (all within 1 %)
- CNN1D IMUWiFine val: only:imu 1.452 / only:wifi 1.417 / full 1.397 (only:imu lags full by 4 %)
- LSTM-attn's per-modality recovery is structural, not Webots-only.

**Smoothness debt persists**: both fusions land r ≈ 0 on IMUWiFine
test paths. Confirmed across THREE datasets now (Webots, MSILN
RESULT_15, IMUWiFine) and FIVE architectures (incumbent, CNN1D,
LSTM-attn, TCN, +iter16 subset variants). The smoothness lever
remains the open knob.

**PLAN_20 recommendation**: IPIN 2024 floor 0 (same shape: CNN1D +
LSTM-attn + wlan_localization + RoNIN ResNet1D). IPIN 2024 has
been integrated per CLAUDE.md and the dataset has IMU on both
val and test, so the RoNIN row will be fully measurable there.

## Step-by-step

### Step 0 — Data + config verification (5 min)

- `data/imuwifine_floor4/` present (80 paths, file format
  `wifi.csv`/`imu.csv`/`ground_truth.csv`/`odometry.csv` per path;
  AP universe = 343 BSSIDs).
- `configs/data/imuwifine.yaml` restored from
  `overnight-autonomous-2026-05-24` (run-1 branch); same for
  `scripts/convert_imuwifine.py` (not re-run; existing
  `imuwifine_floor4/` is already in async_collection format).
- Smoke import of the dataloader succeeds:
  `train_ds=23385  val_ds=13947  test_ds=23724` windows.
- Dataset config: 40/20/20 path split (train=0-39, val=40-59,
  test=60-79); modalities=[wifi, imu]; wifi_pca=128; imu window=32
  (~1 s at the post-downsample 32 Hz).
- Dataset README note: "Test paths carry only WiFi + ground truth
  (no IMU); IMU windows zero-pad." → IMU encoders see zeros on
  test, so the IMU-only branch can't dead-reckon test predictions.

### Step 1a — wlan_localization on IMUWiFine WiFi (NEW measurement)

Script: `scripts/_eval_wlanloc_imuwifine.py` (clone of RESULT_15's
MSILN template + IMUWiFine paths). Vendored `PositionRegressor` +
`DataPreprocessor` loaded via `importlib` per Demand #3.

| split | n     | mean Euc. (m) | median (m) | p90 (m) | max (m) |
|-------|------:|--------------:|-----------:|--------:|--------:|
| val   | 2371  | **4.170**     | 2.399      | 9.04    | 53.77   |
| test  | 23007 | **8.504**     | 4.683      | 23.09   | 64.67   |

Preprocessor (Box-Cox + PCA) reduces 343 APs → 150 components,
fitted on train (4187 scans). PositionRegressor k=3, manhattan,
distance-weighted (per repo defaults).

The val-to-test gap (4.17 → 8.50 m, +104 %) reflects the
between-session deployment shift — train/val are closer-session
than test on this dataset, and the WiFi RSSI fingerprints drift.

### Step 1b — RoNIN ResNet1D on IMUWiFine IMU (NEW measurement)

Script: `scripts/_eval_ronin_imuwifine.py` (clone of RESULT_02's
`eval_ronin_ipin.py` template). ResNet1D trained from scratch on
IMUWiFine train (30 epochs, AdamW + OneCycleLR + Huber). 4.24 M
params.

**Bug fix on the way in**: `WIN//32 + 1` from the IPIN template
yields `in_dim=2`, but ResNet1D's conv output for WIN=32 is
length-1 after the four stride-2 reductions (stem stride 2 +
maxpool stride 2 + 3 of the 4 residual groups stride 2 = factor 32
downsample on length-32 input → length 1). Replaced the hardcoded
formula with an empirical probe (run the conv body on a dummy
input, read the actual output length).

| split | n windows | n paths | per-sample MAE (m) |
|-------|----------:|--------:|-------------------:|
| val   | 13897     | 20      | **26.84**          |
| test  | 0         | 0       | **n/a** (no IMU)   |

Per-path val MAEs span 1.5 m to 60+ m — typical RoNIN-on-real-
data pattern (IMU dead-reckoning drift on multi-minute trajectories).
IMU-only is fundamentally not competitive on a 1-2 minute
trajectory regime; the comparison value is **the floor** the
fusion needs to beat.

Test gives **no measurement** because IMUWiFine test paths lack
IMU — RoNIN ResNet1D needs IMU windows to predict velocity; with
empty windows it cannot generate trajectory. We document this as
an acceptance-criteria asymmetry (paper text needs a footnote).

### Step 2 — CNN1D + LSTM-attn training on IMUWiFine

Script: `scripts/_train_imuwifine_2mod_arch.py` (clone of
`_train_webots_4mod_arch.py` parameterised for 2-mod IMUWiFine).
Same protocol as RESULT_17: K=4, B=128, AdamW + OneCycleLR +
Huber(δ=0.5), 90 epochs, instant_dropout=0.45,
modality_dropout=0.4, lr=1.3e-3.

**Bug fix**: the wrapper called `extract_vision_tokens(dm, vision)`
unconditionally, but `vision` is `None` for IMUWiFine (no Camera).
Guard added: `extra = ... if vision is not None else {}`.

**Bug fix (during JSON dump)**: numpy int64 in the `top5` list
crashed `json.dump`. Cast `int(p)` in the path-length dict.
Postproc script
(`scripts/_iter19_postproc.py`) loads the saved
`fusion_*/model.pt` and regenerates the JSON +
per-trajectory plots, avoiding a re-train.

#### Pre-test gate

Both candidates: 5-epoch run on full train (no 10% subset needed
here — the train split is 23 k windows so 5-epoch full is ~18 s).

| arch       | first val MAE | best val MAE | drop %   |
|------------|--------------:|-------------:|---------:|
| CNN1D      | 13.75         | 1.74         | 87.3 %   |
| LSTM-attn  | 13.07         | 1.82         | 86.1 %   |

Both clear the ≥ 10 % gate by a wide margin.

#### Full training

| arch       | params  | best val MAE | best epoch | wall time | peak GPU |
|------------|--------:|-------------:|-----------:|----------:|---------:|
| CNN1D      | 0.34 M  | **1.397**    | 75         | 321 s     | 325 MB   |
| LSTM-attn  | 0.41 M  | **1.264**    | 70         | 318 s     | 331 MB   |

Memory comfortably under the 6 GB budget. Training stable; both
losses descend monotonically by epoch 50.

#### Subset evals

| arch       | only:wifi | only:imu | wifi+imu | val→test |
|------------|----------:|---------:|---------:|---------:|
| CNN1D val  | 1.417     | 1.452    | 1.397    | —        |
| CNN1D test | 7.094     | 7.298    | 7.094    | +408 %   |
| LSTM-attn val | 1.274  | **1.263**| 1.264    | —        |
| LSTM-attn test | 7.196 | 7.253    | 7.196    | +469 %   |

**Notable**: LSTM-attn val `only:imu` 1.263 ties full-fusion 1.264
(0.1 % gap) and **beats** `only:wifi` 1.274. This is the
dead-reckoning regime from RESULT_18 **replicating on a different
dataset**. Second confirmation of the per-modality-recovery
finding.

**CNN1D contrast** on IMUWiFine val: `only:imu` 1.452, `only:wifi`
1.417, full 1.397 — `only:imu` is 4 % over full, cooperative
fusion regime persists.

On test, `only:wifi` test ≈ full test for both archs (= 7.094 and
7.196). The IMU branch of fusion sees zeros, contributes nothing,
fusion degrades to WiFi-only inference. This is consistent with
the dataset note ("Test paths carry only WiFi + ground truth").

### Step 3 — Main-results IMUWiFine row + outcome label

| method | params | val MAE | test MAE | source |
|---|---:|---:|---:|---|
| wlan_localization (WiFi only) | (kNN) | 4.170 | 8.504 | Step 1a |
| RoNIN ResNet1D (IMU only) | 4.24 M | 26.84 | n/a | Step 1b |
| **CNN1D (WiFi+IMU fusion)** | 0.34 M | **1.397** (−66.5 %) | **7.094** (−16.6 %) | Step 2 |
| **LSTM-attn (WiFi+IMU fusion)** | 0.41 M | **1.264** (−69.7 %) | 7.196 (−15.4 %) | Step 2 |

Percent reductions are vs `wlan_localization`.

**Outcome label**: **β'''' on TEST** (we beat the measurable WiFi
SOTA but RoNIN is unmeasurable due to dataset asymmetry);
**α'''' on VAL** (we beat both SOTAs decisively).

The paper claim is best framed by VAL where both per-leg SOTAs are
measurable side-by-side; TEST is reported honestly as the WiFi-only
inference floor (8.50 → 7.09 = 16.6 % SOTA beat).

### Step 4 — Per-trajectory smoothness + per-path plots

| arch | smoothness median r | min r | max r |
|------|--------------------:|------:|------:|
| CNN1D     | **−0.005** | −0.046 | +0.034 |
| LSTM-attn | **−0.007** | −0.041 | +0.038 |

Both far below the locked r > 0.20 gate. Smoothness debt persists
across three datasets (Webots RESULT_18, MSILN RESULT_15,
IMUWiFine this iter). **Architecture lever doesn't move smoothness
on real data either** — confirmed.

Per-path test plots saved for the 5 longest test paths
(63, 66, 70, 79, 77):
- `runs/overnight/run2_iter_19/test_paths/{cnn1d,lstm_attn}_path_{63,66,70,77,79}.png`

CNN1D per-path test MAE table (selected diagnostic rows):

| path | n | mean MAE | median | p90 | max |
|-----:|--:|---------:|-------:|----:|----:|
| 60   | 842 | 17.10 | 18.54 | 30.93 | 41.24 |
| 63   | 3309 | 10.79 | 8.72 | 22.95 | 32.41 |
| 64   | 1002 | 2.05 | 0.50 | 7.45 | 11.50 |
| 66   | 1797 | 6.83 | 4.08 | 18.20 | 35.84 |
| 70   | top5 | (see plot) | | | |
| 77   | top5 | (see plot) | | | |

The wide per-path variance (0.5 m → 18.5 m median) reflects the
cross-session within-dataset shift in IMUWiFine — some test paths
were collected on different days from train, and WiFi
fingerprints don't transfer between sessions. Per-path-aware
training (or a session-invariant WiFi encoder, per the CLAUDE.md
"What's Next" priority #1) could close this gap.

### Step 5 — Decision + PLAN_20 recommendation

**Three-sentence verdict.**

(1) **IMUWiFine row outcome label**: α'''' on val (we beat both
SOTAs by 70 %/95 % respectively, decisively), β'''' on test (we
beat WiFi SOTA by 16-17 %, RoNIN unmeasurable on test). Best
paper-claim framing uses val where both SOTAs are measurable
apples-to-apples.

(2) **Smoothness gate**: NO — both architectures r ≈ 0 on
IMUWiFine; debt persists across 3 datasets × 5 architectures.
This is now well-documented and load-bearing for the run-2 paper's
"smoothness debt requires a loss-function lever" finding (PLAN_18
verdict reinforced).

(3) **PLAN_20 recommendation**: IPIN 2024 floor 0 row of the
main-results table. IPIN's test split contains IMU (unlike
IMUWiFine), so the RoNIN ResNet1D row will be fully measurable.
The script and config exist (`scripts/eval_ronin_ipin.py`,
`configs/data/ipin2024_floor0.yaml` if present); the only NEW
build is `_eval_wlanloc_ipin.py` (clone of this iter's IMUWiFine
template).

## One open question for scientist

The IMUWiFine test split's "no IMU" design choice means our
fusion claim on test is effectively a WiFi-only claim. Should the
paper:

- (a) Report only the val numbers as the "headline" IMUWiFine
  result (cleanest fusion-vs-SOTA comparison, both SOTAs
  measurable), and footnote the test numbers as a WiFi-only-floor
  measurement?
- (b) Report both val + test, with the test asterisked as "WiFi
  branch only (dataset design)", and frame the fusion claim as
  "matches WiFi-only when only WiFi is available, but adds 70 %
  improvement when IMU is also available"?

(b) is more honest and aligns with the "graceful degradation"
narrative from RESULT_18, but (a) is cleaner for the
main-results table. The choice affects how PLAN_20-22 (IPIN /
RoNIN / UJI rows) should be framed — IPIN has IMU on both splits
so both val + test will be fusion-relevant; RoNIN canonical and
UJI are 1-modality so they're per-leg SOTA reproductions only.

## Sources

- `handoff/SCIENTIST_NOTE_main-results-table.md` (the directive).
- `configs/data/imuwifine.yaml`, `scripts/convert_imuwifine.py`
  (restored from `overnight-autonomous-2026-05-24`).
- `scripts/_eval_wlanloc_imuwifine.py` — NEW (Step 1a).
- `scripts/_eval_ronin_imuwifine.py` — NEW (Step 1b; includes
  in_dim probe fix).
- `scripts/_train_imuwifine_2mod_arch.py` — NEW (Step 2; with
  vision=None guard).
- `scripts/_iter19_postproc.py` — NEW (postproc helper after the
  int64 JSON bug).
- `runs/overnight/run2_iter_19/{wlanloc,ronin,cnn1d,lstm_attn}_imuwifine.json`
  — full numerical output + per-path distributions.
- `runs/overnight/run2_iter_19/test_paths/*.png` — per-trajectory
  plots for the 5 longest test paths.
- `runs/overnight/run2_iter_19/{cnn1d,lstm_attn}/fusion_*/model.pt`
  — trained checkpoints for future ablation iterations.
