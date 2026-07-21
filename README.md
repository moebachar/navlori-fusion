# NavLoRI — Fusion

**Attention-Based Asynchronous Fusion of WiFi, Vision, IMU, and Odometry for Indoor Robot Localization**

TIAGO++ robot, CESI LINEACT. Author: Mohamed Bachar.

---

## Project Overview

Indoor position prediction (x, y) using 4-modality fusion: **WiFi RSSI + Vision + IMU + Odometry**.

### Architecture (as built)

| Stage | Module | Description |
|-------|--------|-------------|
| **A** | `src/pipeline/encoders/` | Per-modality encoders → 128-d tokens: WiFi-Net, IMUCNN, OdomCNN, ACEVision / DPVOMotion (vision) |
| **B+C** | `src/pipeline/fusion/` | **One** continuous-time set-transformer: self-attention = cross-modal fusion, K instants = temporal fusion, cross-attention query = readout. CNN1D / LSTM-attn bake-off candidates in `bakeoff.py`. |
| **D** | *(subsumed)* | Temporal self-attention learns the smoothing a filter would; `src/pipeline/filters/` is a stub for future work |
| **E** | `src/pipeline/uncertainty/` | Split-conformal `(x, y) ± r` intervals (α=0.1) |
| ext | `src/pipeline/baselines/` | Centralised loaders for the 6 SOTA submodules under `external_methods/` |
| data | `src/pipeline/data/` | `load_dataset(name)` factory across all datasets (Webots, MSILN, UJI, RoNIN, IMUWiFine, IPIN) |
| viz | `src/pipeline/visualization/` | Paper-figure plotters |

### Contributions

- **C0**: Multi-modal async simulation dataset (Webots, TIAGO++)
- **C1**: Continuous-time set-transformer fusion of asynchronous WiFi+IMU (ICINCO 2026 submission)
- **C2**: Async robustness via modality dropout + per-instant dropout (no rate-resampling)
- **C3**: Conformal position intervals

Full pipeline walkthrough: [docs/fusion_pipeline.md](docs/fusion_pipeline.md).

---

## Project Structure

```
navlori-fusion/
├── src/
│   ├── simulation/                 # Webots simulation
│   │   ├── worlds/                 # .wbt world file (TIAGO++ indoor env)
│   │   └── controllers/
│   │       ├── async_collector/    # Main data collection controller
│   │       │   ├── async_collector.py   # Event-driven multi-modal collector
│   │       │   ├── dwa_planner.py       # DWA obstacle avoidance (from PythonRobotics)
│   │       │   ├── paths.py             # Path loader
│   │       │   ├── paths.json           # 30 collision-free paths
│   │       │   ├── fix_paths.py         # Safety zone violation fixer
│   │       │   ├── viz_paths.py         # Path visualization generator
│   │       │   └── viz/                 # Generated path images
│   │       ├── wifi_supervisor/    # WiFi RSSI predictor (GPR-based)
│   │       └── tiago_unified_collector/ # Legacy collector (reference)
│   ├── pipeline/                   # ML pipeline
│   │   ├── encoders/               # Stage A: per-modality encoders
│   │   ├── fusion/                 # Stage B+C: set-transformer + bake-off + builder
│   │   ├── baselines/              # Loaders for the 6 SOTA submodules
│   │   ├── data/                   # Dataset factory (all datasets)
│   │   ├── evaluation/             # 6-metric harness + MainResultsTable
│   │   ├── training/               # EncoderTrainer / FusionTrainer
│   │   ├── visualization/          # Paper-figure plotters
│   │   ├── uncertainty/            # Stage E: conformal prediction
│   │   ├── filters/                # Stage D stub (subsumed by temporal attention)
│   │   ├── utils/
│   │   └── pipeline.py
│   └── services/
│       └── grafana/                # Dashboard configs + provisioning
├── external_methods/               # 6 SOTA baseline git submodules
├── configs/                        # Hydra YAML configs
│   ├── config.yaml                 # Root config
│   ├── stage_a/                    # Encoder configs
│   ├── stage_b/ ... stage_e/       # Per-stage configs
│   ├── data/                       # Dataset configs
│   ├── training/                   # Training configs
│   └── experiment/                 # Full experiment compositions
├── notebooks/
│   ├── run2_walkthrough.ipynb      # Full experiment-campaign walkthrough (live numbers)
│   ├── paper_results.ipynb         # Paper-scoped results (WiFi+IMU, set-transformer)
│   ├── reproduce_paper.ipynb       # Public reproducibility notebook
│   ├── encoder_workbench.ipynb     # Stage-A encoder exploration
│   └── data_exploration.ipynb      # Data analysis + visualizations
├── scripts/
│   ├── services.ps1                # Start/stop InfluxDB + Grafana
│   ├── launch_webots.ps1           # Launch Webots in interactive session
│   ├── convert_*.py                # External-dataset converters
│   ├── eval_*.py                   # Per-dataset / per-baseline evaluation
│   └── optuna_fusion.py            # Hyperparameter search
├── data/                           # DVC-tracked data directory
│   └── async_collection/           # Collected sensor data (per path)
├── tests/
├── pyproject.toml
└── .gitignore
```

---

## Setup (one-time)

### 1. Clone and create environment

```powershell
git clone https://github.com/moebachar/navlori-fusion.git
cd navlori-fusion
git submodule update --init --recursive
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

The `git submodule update --init --recursive` step pulls the six
SOTA baseline repositories (wlan_localization, ronin, tartanvo,
dpvo, imuwifine, indoor_location_competition_20) under
`external_methods/`. See
[docs/EXTERNAL_DEPENDENCIES.md](docs/EXTERNAL_DEPENDENCIES.md) for
the per-submodule setup notes (TartanVO weights, RoNIN pretrained
ResNet1D, DPVO Windows-build limitation).

### 2. Register Jupyter kernel

```powershell
python -m ipykernel install --user --name navlori-fusion --display-name "NavLoRI Fusion"
```

### 3. Initialize DVC

```powershell
dvc init
dvc remote add -d local D:\dvc_store
mkdir D:\dvc_store
```

### 4. Install services (InfluxDB + Grafana)

Download and extract into the project:

- **InfluxDB v2.7**: extract to `src/services/influxdb/` (needs `influxd.exe`)
- **Grafana v11**: extract to `src/services/grafana/` (needs `bin/grafana-server.exe`)

These binaries are gitignored. First-time InfluxDB setup:

```powershell
# Start InfluxDB
.\src\services\influxdb\influxd.exe

# In another terminal, set up org/bucket/token (pick your own secrets,
# then mirror them in the local .env file — never commit them):
.\src\services\influxdb\influx.exe setup `
  --org navlori --bucket async_data `
  --username <user> --password <password> `
  --token <token> --force
```

### 5. Configure Webots

Open the world file in Webots:
```
src/simulation/worlds/Tiago++'s world.wbt
```

Set the TIAGO++ robot's `controller` field to `async_collector` and the Python command to:
```
C:\Users\Administrateur\navlori-fusion\.venv\Scripts\python.exe
```

---

## Daily Usage

### Start services (InfluxDB + Grafana)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\services.ps1 start
```

- InfluxDB: http://localhost:8086
- Grafana: http://localhost:3000 (admin / admin)

### Stop services

```powershell
powershell -ExecutionPolicy Bypass -File scripts\services.ps1 stop
```

### Run data collection

1. Edit batch range in `src/simulation/controllers/async_collector/async_collector.py`:
   ```python
   BATCH_START = 0    # First path (inclusive)
   BATCH_END = 9      # Last path (inclusive)
   ```

2. Open the world in Webots and press **Play** (or **>>** for fast mode).

3. Data saves to `data/async_collection/path_XX/` with separate CSVs per modality:
   - `imu.csv` — accelerometer + gyroscope + orientation (~31 Hz)
   - `odometry.csv` — wheel encoders + dead reckoning (~15 Hz)
   - `wifi.csv` — per-AP RSSI fingerprints (~1 Hz)
   - `ground_truth.csv` — supervisor position (~10 Hz)
   - `camera/` — RGB + depth PNGs (~0.5 Hz)

4. After collection, version with DVC:
   ```powershell
   dvc add data/async_collection
   git add data/async_collection.dvc data/.gitignore
   git commit -m "data: batch 0-9 collection"
   ```

### Run training

```powershell
python scripts/train.py
```

### Run MLflow UI

```powershell
mlflow ui --backend-store-uri mlruns
```
Open http://localhost:5000

### Run TensorBoard

```powershell
tensorboard --logdir tb_logs
```
Open http://localhost:6006

### Run tests

```powershell
pytest
```

### Run linter

```powershell
ruff check src/ tests/
```

---

## Data Collection Details

### Sensor rates (with ±20% jitter)

| Modality | Nominal Rate | Interval (sim steps) |
|----------|-------------|---------------------|
| IMU | ~31 Hz | 1 step (32ms) |
| Odometry | ~15 Hz | 2 steps |
| Ground Truth | ~10 Hz | 3 steps |
| WiFi | ~1 Hz | 31 steps |
| Camera | ~0.5 Hz | 62 steps |

### Navigation

Uses the **Dynamic Window Approach (DWA)** for obstacle-aware navigation. The planner (adapted from [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics), MIT license) samples velocity commands, simulates trajectories, and picks the one that best balances goal-seeking and obstacle clearance.

Obstacle detection uses the TIAGO++ depth camera (Astra depth sensor).

### Paths

30 predefined paths in `paths.json`. Waypoints are corrected with `fix_paths.py` to enforce a safety clearance of **0.855m** (robot_radius=0.55 × 1.1 + 0.25m extra margin) from all walls. Visualize with `viz_paths.py` which outputs to `viz/`.

The robot's actual collision radius is measured dynamically after `tuck_arms()` by reading the world positions of `ARM_LEFT_4` and `ARM_RIGHT_4` joints, plus a safety margin.

---

## Remote Access (from laptop)

Everything runs on the desktop. From the laptop:

1. **VS Code Remote SSH** → connect to `navlori-gpu` (100.126.253.37 via Tailscline)
2. Open folder: `C:\Users\Administrateur\navlori-fusion`
3. **Parsec** for Webots GUI (cameras need a real GPU session — SSH has no display context)
4. Access services via browser:
   - Grafana: http://100.126.253.37:3000
   - InfluxDB: http://100.126.253.37:8086
   - MLflow: http://100.126.253.37:5000
   - TensorBoard: http://100.126.253.37:6006

---

## Hardware

- **GPU**: Quadro P4000 (8GB VRAM, sm_61)
- **PyTorch**: 2.4.1+cu124
- **Webots**: R2025a
- **Python**: 3.11

---

## SSH to GitHub

Port 22 is blocked on this network. GitHub SSH goes through port 443:

```
# ~/.ssh/config
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```
