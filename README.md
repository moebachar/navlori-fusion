# NavLoRI — Fusion

**Attention-Based Asynchronous Fusion of WiFi, Vision, IMU, and Odometry for Indoor Robot Localization**

TIAGO++ robot, CESI LINEACT. Author: Mohamed Bachar.

---

## Project Overview

Indoor position prediction (x, y) using 4-modality fusion: **WiFi RSSI + Vision + IMU + Odometry**.

### Architecture (5-stage pipeline)

| Stage | Name | Description |
|-------|------|-------------|
| **A** | Encode | Per-modality encoders: Anchor2Vec (WiFi), ViT (Vision), 1D-CNN (IMU), Linear (Odom) |
| **B** | Align | Temporal alignment of async streams: mTAN / GRU-D |
| **C** | Fuse | Cross-modal fusion: 6-pair cross-attention |
| **D** | Filter | State estimation: KalmanNet with attention-conditioned gain |
| **E** | Uncertainty | Conformal Prediction for calibrated intervals |

### Contributions

- **C0**: Multi-modal async dataset (this simulation)
- **C1**: 4-modal Transformer encoder
- **C2**: Continuous-time alignment
- **C3**: Hybrid neural-Kalman filter

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
│   │       │   └── paths.json           # 30 collision-free paths
│   │       ├── wifi_supervisor/    # WiFi RSSI predictor (GPR-based)
│   │       └── tiago_unified_collector/ # Legacy collector (reference)
│   ├── pipeline/                   # ML pipeline
│   │   ├── encoders/               # Stage A: per-modality encoders
│   │   ├── temporal/               # Stage B: temporal alignment
│   │   ├── fusion/                 # Stage C: cross-modal fusion
│   │   ├── filters/                # Stage D: state estimation
│   │   ├── uncertainty/            # Stage E: conformal prediction
│   │   ├── utils/
│   │   └── pipeline.py
│   └── services/
│       └── grafana/                # Dashboard configs + provisioning
├── configs/                        # Hydra YAML configs
│   ├── config.yaml                 # Root config
│   ├── stage_a/                    # Encoder configs
│   ├── stage_b/ ... stage_e/       # Per-stage configs
│   ├── data/                       # Dataset configs
│   ├── training/                   # Training configs
│   └── experiment/                 # Full experiment compositions
├── notebooks/
│   └── data_exploration.ipynb      # Data analysis + visualizations
├── scripts/
│   ├── services.ps1                # Start/stop InfluxDB + Grafana
│   ├── train.py
│   └── evaluate.py
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
cd C:\Users\Administrateur
git clone git@github.com:moebachar/navlori-fusion.git
cd navlori-fusion
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Register Jupyter kernel

```powershell
python -m ipykernel install --user --name navlori-fusion --display-name "NavLoRI Fusion"
```

### 3. Initialize DVC

```powershell
dvc init
dvc remote add -d local C:\Users\Administrateur\dvc_store
mkdir C:\Users\Administrateur\dvc_store
```

### 4. Install services (InfluxDB + Grafana)

Download and extract into the project:

- **InfluxDB v2.7**: extract to `src/services/influxdb/` (needs `influxd.exe`)
- **Grafana v11**: extract to `src/services/grafana/` (needs `bin/grafana-server.exe`)

These binaries are gitignored. First-time InfluxDB setup:

```powershell
# Start InfluxDB
.\src\services\influxdb\influxd.exe

# In another terminal, set up org/bucket/token:
.\src\services\influxdb\influx.exe setup `
  --org navlori --bucket async_data `
  --username navlori --password navlori2026 `
  --token navlori-influx-token-2026 --force
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

30 predefined paths in `paths.json`, generated with collision-free constraints (robot radius × 1.5 safety margin). Each path has 4-10 waypoints covering the indoor environment.

---

## Remote Access (from laptop)

Everything runs on the desktop. From the laptop, connect via VS Code Remote SSH:

1. Open VS Code → Remote SSH → connect to `navlori-gpu` (100.126.253.37 via Tailscale)
2. Open folder: `C:\Users\Administrateur\navlori-fusion`
3. Access services via browser:
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
