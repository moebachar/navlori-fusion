# NavLoRI-Fusion

**Attention-based asynchronous fusion of WiFi, vision, IMU, and odometry for indoor robot localization.**

TIAGO++ robot · Webots simulation · real-world datasets · CESI LINEACT · Author: Mohamed Bachar

---

## Overview

NavLoRI-Fusion predicts the 2-D position `(x, y)` of an indoor mobile robot by fusing four asynchronous sensor streams that arrive at different rates: WiFi RSSI (~1 Hz, absolute), wheel odometry (~15 Hz, motion), IMU (~31 Hz, motion), and RGB camera (~5 Hz, visual). The core mechanism is a single set-transformer that treats every observation as a universal token and uses self-attention to fuse across modalities and time, with a cross-attention `PositionQuery(τ)` reading out `(x, y)` at any instant.

The project is open and honest about negative results: the simulator is generous (≈0.43 m MAE — partly memorisation), but real-world cross-dataset evaluation exposes the WiFi encoder as the bottleneck. The set-transformer itself works — fusion beats both open-source single-modality SOTA baselines on the same data — and the honest reading of how far that takes you sits in [docs/fusion_pipeline.md](docs/fusion_pipeline.md) and [docs/SOTA_BASELINES.md](docs/SOTA_BASELINES.md).

---

## Architecture

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ WiFi RSSI (1 Hz) │  │ IMU (~31 Hz)     │  │ Odom (~15 Hz)    │  │ Camera (~5 Hz)   │
│ → Anchor2Vec     │  │ → IMUCNN         │  │ → OdomCNN        │  │ → DPVOMotion     │
└──────┬───────────┘  └──────┬───────────┘  └──────┬───────────┘  └──────┬───────────┘
       │ 128-d            128-d                 128-d                  128-d
       └──────────────────────┬──────────────────────────────────────────┘
                              │
                  universal token = encoder_embedding
                                  + modality_embedding
                                  + continuous-time_encoding(Δt)
                              │
                ┌─────────────▼──────────────┐
                │   FusionTransformer        │  Stage B + C  (one set-transformer,
                │   self-attention (×depth)  │                three attention roles)
                │   cross-attention readout  │
                └─────────────┬──────────────┘
                              │
                          PositionQuery(τ)  →  (x, y)
                              │
                ┌─────────────▼──────────────┐
                │   ConformalPosition        │  Stage E  (split-conformal,
                │   (x, y) ± r at α=0.10     │            distribution-free)
                └────────────────────────────┘
```

### Stage A — per-modality encoders

| Modality | Encoder | Input shape | Architecture |
|---|---|---|---|
| WiFi RSSI | `Anchor2Vec` | `(B, n_APs)` | k=64 learned anchors + soft-assignment + MLP |
| IMU | `IMUCNN` | `(B, 32, 9)` | 1D-CNN (32→64→128 ch) over a ~1 s window |
| Odometry | `OdomCNN` | `(B, 16, 7)` | 1D-CNN (16→32→64 ch) over a ~1 s window |
| Camera | `DPVOMotionEncoder` | `(B, 2, 3, 480, 640)` | DPVO-style dense optical flow on a frame pair |

All encoders implement `BaseEncoder` and emit `(B, 128)` tokens. ACE and ViT-LoRA vision encoders existed in earlier phases and were removed (2026-05-20) when the audit found they memorised positions on small, single-floor datasets.

### Stage B + C — FusionTransformer

Realised as a **single set-transformer** ([src/pipeline/fusion/transformer.py](src/pipeline/fusion/transformer.py)) rather than a separate mTAN stage plus a 6-pair cross-attention stage. Every observation is one universal token:

```
token = encoder_embedding + modality_embedding + time_encoding(Δt)
```

The same layers serve three roles depending on what they see:

- **cross-modal fusion** — modality tokens of one instant attend to each other (self-attention).
- **temporal fusion** — feed K recent instants and the same self-attention attends across time. The continuous-time encoding folds in mTAN's role.
- **readout** — a `PositionQuery(τ)` cross-attends to the token set and emits `(x, y)` at any timestamp τ.

A boolean padding mask kills absent tokens, so the same model handles 1-, 2-, 3-, or 4-modality streams. Robustness is built by **training**: `modality_dropout` (drop a whole sensor for a sample) and `instant_dropout` (drop individual instant-modality tokens) simulate async arrivals and staleness. A CLS token is always unmasked to avoid the `softmax(-∞)` NaN that fully-padded rows otherwise hit.

Configuration lives in [configs/stage_c/fusion.yaml](configs/stage_c/fusion.yaml); the wiring in [src/pipeline/fusion/builder.py](src/pipeline/fusion/builder.py) (`load_config → build_datamodule → build_encoders → extract_vision_tokens → build_model → build_trainer`) is shared by the notebook, smoke harness, and Optuna search so all three build the identical pipeline.

### Stage E — conformal uncertainty

Split-conformal prediction with `α = 0.10` (90% coverage target). The conformity score is the Euclidean residual; the calibrated radius is its finite-sample `(1−α)` quantile. Implementation: [src/pipeline/uncertainty/conformal.py](src/pipeline/uncertainty/conformal.py) (`ConformalPosition`). Coverage holds only when calibration and inference data are exchangeable — this caveat matters and we measure it explicitly.

The earlier Stage-D KalmanNet design (`configs/stage_d/kalmannet.yaml`) is kept as a stub for future work; temporal self-attention already learns the smoothing that KalmanNet was intended to provide.

---

## Results

### Simulation (Webots TIAGO++)

18 hand-annotated paths on one floor, splits `train=[1,3-12] val=[2,13,14] test=[15-17]`.

| Configuration | Test MAE | Notes |
|---|---|---|
| Fusion, single instant (`K=1`)        | **0.43 m** | All 4 modalities, fresh WiFi |
| Fusion, temporal (`K=8`, stride 9)    | **0.40 m** | Adds graceful staleness handling |
| Single-instant under 2 s stale WiFi   | ~4 m       | Cliff |
| Temporal under 2 s stale WiFi         | 0.8 m      | Graceful slope |

The simulator's WiFi is GPR-synthesised (not measured), so the sub-metre number is **optimistic**. Treat it as a sanity floor, not an SOTA claim.

### Real data (open-source datasets, controlled SOTA comparison)

Validated against open-source baselines run from their unmodified source code (no reimplementations) on each dataset's canonical metric. See [docs/SOTA_BASELINES.md](docs/SOTA_BASELINES.md) for the full protocol.

**Phase A — per-leg validation.**

| Leg | Our encoder | Open-source baseline | Dataset |
|---|---|---|---|
| WiFi | Anchor2Vec — **8.55 m** | `wlan_localization` — 13.92 m | UJIIndoorLoc validation |
| IMU  | IMUCNN — 14.41 m / 8.41 m raw/aligned ATE | RoNIN ResNet — 5.93 m raw ATE | RoNIN unseen subjects |

**Phase B — controlled fusion comparison (IPIN 2024 floor −2, trial-out split).**

| Method | Modalities | Test MAE |
|---|---|---|
| **Our fusion** (decomposed readout, WiFi + world-frame IMU) | WiFi + IMU | **10.05 m** |
| `wlan_localization` (open-source, unmodified) | WiFi only | 23.12 m |
| RoNIN ResNet1D (open-source, unmodified) | IMU only  | 42.87 m |

Fusion beats both open-source single-modality SOTAs **on the same data, with the same metric** (−13.07 m vs WiFi, −32.82 m vs IMU). The win over WiFi-only is real but bounded — the autopsy ([docs/PIPELINE_AUTOPSY.md](docs/PIPELINE_AUTOPSY.md)) quantified IPIN's WiFi sparsity (29% of val samples > 15 s stale) as a ~6–7 m per-leg ceiling.

### Honest findings

These are measured and documented; do not paper over them.

1. **WiFi dominates fresh-data accuracy.** Single-instant fusion ≈ 0.43 m on sim; WiFi-only ≈ 0.46 m. The other modalities add a few cm on fresh data.
2. **Temporal fusion's value is robustness, not fresh accuracy.** Naïve temporal regressed to 0.69 m (overfit); per-instant dropout fixed it back to ≈ 0.44 m *and* unlocked graceful degradation under stale WiFi.
3. **Webots WiFi is GPR-synthesised**, not measured — sub-metre sim numbers are optimistic. Real cross-session splits diverge (train ↓, val ↑); the bottleneck is the WiFi encoder, not the fusion stage.
4. **`drop:wifi` stays ~4 m on sim** — with no absolute reference at any instant, position is genuinely unobservable; fusion cannot invent an anchor.
5. **Conformal coverage holds only under exchangeability** — random halves of one pool give ≈ 90–92%; calibrating on `val` paths and testing on `test` paths under-covers.

---

## Project structure

```
navlori-fusion/
├── src/
│   ├── pipeline/
│   │   ├── encoders/          # Stage A: Anchor2Vec, IMUCNN, OdomCNN, DPVO{Motion,Full}Encoder
│   │   ├── fusion/            # Stage B+C: FusionTransformer, ContinuousTimeEncoding, builder
│   │   ├── data/              # FusionDataset + FusionDataModule (async windows, splits)
│   │   ├── training/          # EncoderTrainer + FusionTrainer (modality/instant dropout, eval)
│   │   ├── evaluation/        # 6-metric encoder evaluation harness
│   │   ├── uncertainty/       # ConformalPosition (split-conformal)
│   │   ├── filters/ temporal/ # Stage D / Stage B stubs (kept for future work)
│   │   └── utils/             # MLflow + TensorBoard + Rich logging
│   ├── simulation/
│   │   ├── worlds/                       # Webots world (TIAGO++ indoor env)
│   │   └── controllers/
│   │       ├── async_collector/          # Event-driven multi-modal data collector
│   │       ├── tiago_unified_collector/  # Unified single-controller variant
│   │       └── wifi_supervisor/          # GPR-based WiFi RSSI predictor
│   └── services/
│       └── grafana/                      # Dashboard JSON + provisioning
├── configs/                              # Hydra YAML configs
│   ├── stage_a/                          # Per-encoder
│   ├── stage_c/fusion.yaml               # FusionTransformer (single source of truth)
│   ├── stage_e/conformal.yaml            # Conformal α
│   ├── data/                             # simulation, imuwifine, ipin2024_*, ronin_a000_*
│   └── training/default.yaml             # AdamW + OneCycleLR + Huber + early stopping
├── dashboard/                            # Streamlit 4-page dashboard
├── scripts/
│   ├── optuna_fusion.py                  # TPE hyperparameter search (FusionTransformer)
│   ├── _smoke_fusion.py                  # 5-phase smoke harness (shape, overfit, train, profile, +DPVO)
│   ├── convert_{imuwifine,ipin2024,ronin}.py   # External-dataset → async_collection format
│   ├── eval_{wlanloc,cnnloc,ronin,uji}_*.py    # SOTA baseline evaluators
│   ├── eval_ronin_ate_fixed.py           # IMU dead-reckoning ATE on RoNIN unseen
│   ├── train_ace_scr.py                  # ACE scene-coord regression (vision research branch)
│   ├── baselines.py                      # Mean / kNN trivial baselines on every dataset
│   ├── inspect_*.py                      # Diagnostic probes (raw data, staleness, encoding, etc.)
│   └── services.ps1, setup_influxdb.ps1, grafana_setup.ps1   # Local InfluxDB/Grafana
├── notebooks/
│   ├── fusion_workbench.ipynb            # End-to-end FusionTransformer demo
│   ├── encoder_workbench.ipynb           # Stage A encoder training + evaluation
│   └── validation.ipynb                  # Reproduces docs/SOTA_BASELINES.md numbers
├── docs/
│   ├── fusion_pipeline.md                # 13-step walkthrough of Stage B+C
│   ├── PIPELINE.md, PIPELINE_AUTOPSY.md  # Architecture overview + audit
│   ├── SOTA_BASELINES.md                 # Open-source baseline comparison protocol
│   ├── EXTERNAL_DEPENDENCIES.md          # DPVO upstream + dataset downloads
│   └── MILESTONES.md                     # Project history
├── handoff/                              # Handoff notes for next phase
├── tests/                                # pytest suite (encoders, fusion, evaluation, dataloader)
└── pyproject.toml
```

---

## Setup

### Prerequisites

- Windows 10/11 or Linux with a CUDA-capable GPU (Pascal sm_61+ supported)
- Python 3.11
- Webots R2025a (only required for collecting new simulation data; not for training)
- Docker (only required for the DPVO research branch; see [docs/EXTERNAL_DEPENDENCIES.md](docs/EXTERNAL_DEPENDENCIES.md))

### Install

```powershell
git clone https://github.com/moebachar/navlori-fusion.git
cd navlori-fusion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
pip install -e ".[dev]"
```

> **PyTorch ≥ 2.7 dropped Pascal (sm_61) binaries.** This machine targets a Quadro P4000, so `pyproject.toml` caps `torch<2.7`. Use the CUDA 12.4 wheels above.

### Environment variables

Copy `.env.example` to `.env` and fill in your local InfluxDB / Grafana credentials before running any controller or service script:

```powershell
Copy-Item .env.example .env
# then edit .env and set INFLUXDB_TOKEN, INFLUXDB_PASSWORD, etc.
```

Nothing in the repo hardcodes credentials; every controller and PowerShell setup script reads them from this environment.

### Webots runtime config

Webots embedded controllers need an absolute path to the Python venv. Copy the template:

```powershell
Copy-Item src\simulation\controllers\async_collector\runtime.ini.example `
          src\simulation\controllers\async_collector\runtime.ini
# then edit and replace <REPO_ROOT> with your absolute checkout path.
```

`runtime.ini` is gitignored so each developer pins their own venv.

### Pull data (DVC)

```powershell
dvc remote add -d local D:\dvc_store   # one-time
dvc pull
```

External datasets (UJI, RoNIN, IPIN 2024, IMUWiFine) are tracked by DVC alongside the Webots `async_collection/`. Raw upstream archives and any datasets you want to convert yourself are documented in [docs/EXTERNAL_DEPENDENCIES.md](docs/EXTERNAL_DEPENDENCIES.md).

### Register Jupyter kernel

```powershell
python -m ipykernel install --user --name navlori-fusion --display-name "NavLoRI Fusion"
```

---

## Running things

The repository is **notebook-driven** for exploration and **script-driven** for repeatable experiments. There is no `train.py` mega-entry; each task has its own script.

### Train and evaluate the FusionTransformer

```powershell
# 1) Smoke test — five phases (shape, overfit, full train, profile, +DPVO)
.\.venv\Scripts\python.exe scripts\_smoke_fusion.py

# 2) Optuna hyperparameter search (TPE, default 20 trials × 30 epochs)
.\.venv\Scripts\python.exe scripts\optuna_fusion.py --dataset simulation

# 3) Interactive end-to-end demo
jupyter lab notebooks\fusion_workbench.ipynb
```

`scripts/optuna_fusion.py --dataset <name>` accepts any of the `configs/data/<name>.yaml` files (`simulation`, `imuwifine`, `ipin2024_floor-2`, `ipin2024_floor-2_intra`, `ipin2024_floor0`, `ronin_a000`, `ronin_a000_intra`). The same Hydra config (`configs/stage_c/fusion.yaml`) drives every dataset; modalities, splits, WiFi-PCA and window sizes come from the per-dataset YAML.

### Reproduce the SOTA baseline numbers

```powershell
.\.venv\Scripts\python.exe scripts\eval_uji_wifi.py            # Anchor2Vec on UJIIndoorLoc
.\.venv\Scripts\python.exe scripts\eval_wlanloc_uji.py         # wlan_localization on UJI
.\.venv\Scripts\python.exe scripts\eval_ronin_ate_fixed.py     # IMUCNN on RoNIN unseen
.\.venv\Scripts\python.exe scripts\eval_ronin_ipin.py          # RoNIN ResNet1D on IPIN
.\.venv\Scripts\python.exe scripts\eval_wlanloc_ipin.py        # wlan_localization on IPIN
.\.venv\Scripts\python.exe scripts\baselines.py                # Mean / kNN trivial floors
```

Each writes a `runs/<name>_<timestamp>/` directory with `eval.json`, `metrics.jsonl`, and `history.json`. `notebooks/validation.ipynb` reads these and reproduces every number in [docs/SOTA_BASELINES.md](docs/SOTA_BASELINES.md).

### Diagnostic probes

When something looks fishy, the `scripts/inspect_*.py` family pins down the cause without training a new model:

| Probe | What it answers |
|---|---|
| `inspect_01_rawdata.py` | Are GT extents, WiFi scan rate, NaN/inf clean per dataset? |
| `inspect_02_wifi_staleness.py` | What fraction of samples carry stale WiFi, and how stale? |
| `inspect_03_transfer.py` | Does WiFi-kNN coherence hold from train → val (within session)? |
| `inspect_04_wifi_encoding.py` | Which encoding step (covis, PCA, distance) destroys signal? |
| `inspect_05_motion_scale.py` | Can a kNN on IMU windows predict displacement? |
| `inspect_06_model_behavior.py` | Is the model collapsing to the centroid? |
| `inspect_07_stalecap_eval.py` | Split val error by WiFi freshness (fresh vs stale-capped) |
| `inspect_08_worldframe_imu.py` | Does world-frame IMU outperform body-frame? (no training) |
| `inspect_09_staleness_error.py` | Error binned by WiFi staleness — does motion bridge gaps? |

### Local services + dashboard

```powershell
# InfluxDB + Grafana (reads token from .env)
powershell -ExecutionPolicy Bypass -File scripts\services.ps1 start

# Streamlit 4-page dashboard (Overview, Encoders, Dataset, Training Monitor)
streamlit run dashboard\app.py --server.port 8501
```

| Service | URL |
|---|---|
| InfluxDB | http://localhost:8086 |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5000 (`mlflow ui --backend-store-uri mlruns`) |
| Dashboard | http://localhost:8501 |

### Tests + lint

```powershell
pytest                 # encoders, dataloader, evaluation, DPVO motion
ruff check src/ tests/
```

`tests/` expects `data/async_collection/` to exist with paths 1–4 (a DVC pull covers this).

---

## Data collection (Webots simulation)

> **Requires Parsec or a real desktop session.** Webots cameras return `NULL` when run over SSH (no GPU context).

### Sensor schedule

| Modality | Rate | Sim steps (32 ms each) | Features |
|---|---|---|---|
| IMU | ~31 Hz | 1 | accel xyz, gyro xyz, roll/pitch/yaw |
| Odometry | ~15 Hz | 2 | pose xy, heading, velocities, wheel speeds |
| Ground truth | ~10 Hz | 3 | supervisor (x, y) |
| WiFi RSSI | ~1 Hz | 31 | per-AP RSSI fingerprint |
| Camera | ~5 Hz | 6 | 640×480 RGB PNG |

All rates include ±20% jitter (`jitter_fraction=0.2` in metadata). Navigation is proportional steering `ω = 1.5 × angle_error` with no obstacle avoidance — paths are pre-validated with a 0.855 m safety margin from all walls.

### Collecting new paths

1. Annotate waypoints with the web tool: `python src\simulation\controllers\async_collector\annotate_paths.py`
2. Set `BATCH_START` and `BATCH_END` in `async_collector.py`
3. Open `src\simulation\worlds\Tiago++'s world.wbt` in Webots and press **Play**
4. Data saves to `data\async_collection\path_XX\`
5. Version with DVC: `dvc add data\async_collection && git add data\async_collection.dvc && git commit -m "data: add path XX-YY"`

---

## External datasets

Three public datasets are converted to the project's `async_collection` format for cross-dataset evaluation. Download links, citations, and conversion steps in [docs/EXTERNAL_DEPENDENCIES.md](docs/EXTERNAL_DEPENDENCIES.md).

| Dataset | Modalities | Paths | Notes |
|---|---|---|---|
| [UJIIndoorLoc](https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc) | WiFi | — | 520 APs, canonical WiFi-fingerprint benchmark |
| [RoNIN (FRDR)](https://ronin.cs.sfu.ca/) subject a000 | WiFi + IMU | 215 | within-trial split, 15 s chunks |
| [IPIN 2024 Track 3](https://evarilos.eu/ipin2024) floors −2 / −1 / 0 | WiFi + IMU | 40+ | GPS-anchored GT |
| [IMUWiFine](https://github.com/lixufa/IMUWiFine) floor 4 | WiFi + IMU | 80 | 343 APs |

---

## Hardware & environment

| Item | Value |
|---|---|
| GPU | Quadro P4000, 8 GB VRAM, sm_61 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11 |
| Webots | R2025a |
| DVC remote | local at `D:\dvc_store` |

---

## References

- Brachmann et al., *Accelerated Coordinate Encoding (ACE)*, CVPR 2023
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023
- Shukla & Marlin, *Multi-Time Attention Networks (mTAN)*, ICLR 2021
- Teed et al., *Deep Patch Visual Odometry (DPVO)*, NeurIPS 2023
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Vovk et al., *Algorithmic Learning in a Random World* (conformal prediction)
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity*, ICML 2020
- Yan et al., *RoNIN: Robust Neural Inertial Navigation in the Wild*, ICRA 2020
- Naribole, `wlan_localization` — github.com/sharan-naribole/wlan_localization
