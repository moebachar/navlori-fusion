# NavLoRI Fusion

Indoor localization for a mobile robot by fusing WiFi signal strength, an IMU, wheel odometry and a camera.

WiFi tells you roughly where you are in a building. The motion sensors tell you precisely how you moved, but they drift. This repository learns to combine them, even though they report at very different rates (WiFi about once every few seconds, the IMU over a hundred times a second) and one of them is sometimes missing.

Mohamed Bachar, CESI LINEACT.

Paper: *Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization*, ICINCO 2026.

## How it works

Each sensor stream is cut into a short window ending at the moment we want a position. A small encoder turns each window into a 128-d token:

| Sensor | Encoder |
|---|---|
| WiFi | WiFi-Net, or a per-access-point set transformer |
| IMU | 1-D CNN |
| Wheel odometry | 1-D CNN |
| Camera | DPVO motion tokens (experimental) |

The tokens from the last K time instants go into a single set transformer. Self-attention fuses the sensors and the instants, and a learned query reads out `(x, y)`. Every token carries its own timestamp, encoded in continuous time, so nothing is resampled to a common rate.

At training time, whole sensors and whole instants are randomly dropped. This teaches the model to keep working when WiFi is stale or a sensor is down. An optional split-conformal step turns the prediction into `(x, y) ± r`.

[docs/fusion_pipeline.md](docs/fusion_pipeline.md) walks through every step, with the equations and the code locations.

## Repository layout

```
src/pipeline/       encoders, fusion transformer, trainers, dataset loaders, evaluation, conformal intervals
src/simulation/     Webots worlds and controllers (TIAGo++ data collection, replay of real walks)
configs/            OmegaConf configs: data/ (one file per dataset), stage_a/ (encoders), stage_c/ (fusion)
scripts/            training and evaluation entry points, dataset converters
scripts/dataset/    pipeline that turns TurtleBot3 rosbags into a dataset with ground truth
external_methods/   published baselines as git submodules (RoNIN, DPVO, TartanVO, wlan_localization, IMUWiFine, ILC 2.0)
notebooks/, colab/  analysis and paper-result notebooks
docs/               pipeline walkthrough, notes on the external baselines
```

## Data

The data is not in this repository. It lives in a separate data vault (git + DVC). `data/` is expected to point to that vault's `datasets/` folder: on the lab machine it is a link to `X:\navlori-data\datasets`. Configs also honour the `NAVLORI_DATA_ROOT` environment variable.

Every dataset uses the same per-path layout (`path_XX/` holding `ground_truth.csv`, `imu.csv`, `odometry.csv`, `wifi.csv`, `camera.csv`), so a model trains on any of them by changing one config name:

| Config (`configs/data/`) | Data |
|---|---|
| `side_golden` | 12 real TurtleBot3 runs in our building: WiFi, IMU, wheel odometry, camera. Ground truth from lidar SLAM + gyroscope, placed on 26 surveyed AprilTags. |
| `simulation` | Webots collection with a TIAGo++ robot, 18 paths, all four sensors |
| `msiln_site1_b1` | Microsoft Indoor Location Competition 2.0, site 1 floor B1 (smartphone WiFi + IMU) |
| `imuwifine` | IMUWiFine, floor 4 (smartphone WiFi + IMU) |
| `ipin2024_floor0` | IPIN 2024 competition, floor 0 |
| `iln20_5d27099f_F2_replay` | real ILC 2.0 walks re-driven in a Webots model of the floor, which adds camera and odometry |

The converters for the public datasets are `scripts/convert_*.py`.

## Setup

Python 3.11+ and a CUDA GPU are recommended; the code has run on a GTX 1080 and a Quadro P4000.

```bash
git clone https://github.com/moebachar/navlori-fusion.git
cd navlori-fusion
git submodule update --init --recursive      # the external baselines
python -m venv .venv
.venv/Scripts/activate                        # Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Point `data/` at the dataset folder (a link, or `NAVLORI_DATA_ROOT`). The baselines need a few extras such as pretrained weights and build steps, listed in [docs/EXTERNAL_DEPENDENCIES.md](docs/EXTERNAL_DEPENDENCIES.md).

## Running

Train the fusion model on the real robot runs and evaluate every sensor subset on the test runs:

```bash
python scripts/_train_side_golden.py --epochs 60 --no-camera --wifi wifi_net
```

This writes `runs/side_golden/fusion_3mod/`: `results.json` (test error per sensor subset), a training curve, per-run trajectory plots and a video.

One published baseline per sensor, on the same split:

```bash
python scripts/_eval_wlanloc_side_golden.py   # WiFi: wlan_localization kNN
python scripts/_eval_ronin_side_golden.py     # IMU: RoNIN ResNet-1D
python scripts/_eval_odom_dr_side_golden.py   # wheel-odometry dead reckoning
```

The other `scripts/_train_*.py` and `scripts/_eval_*.py` do the same for the other datasets. `colab/` holds the multi-seed runs behind the paper tables. Tests: `pytest`.

## Building the robot dataset

`scripts/dataset/` rebuilds `side_golden` from the raw rosbags:

- **`export/`**: exporters and the loader. Rosbag goes to per-sensor CSVs, camera frames and lidar scans; a lidar scan-to-map SLAM gives the trajectory.
- **`golden/`**: the ground-truth pipeline:
  - AprilTag detection with OpenCV;
  - gyroscope heading with bias estimation;
  - a joint alignment of all runs on the tag network, with a soft prior toward the floor-plan coordinates;
  - map matching for runs that see a single tag;
  - packaging, and plots on the floor plan.

  The shell scripts are meant for WSL/Linux.

Then `scripts/convert_side_golden.py` turns the packaged runs into the format above.

## What to expect

- **WiFi carries the absolute position.** Without it, no combination of motion sensors can say where the robot is, only how it moved.
- **Simulated WiFi is optimistic.** On Webots data the model reaches sub-metre error, but that WiFi is synthesized. On real, cross-session data the errors are metres, and the WiFi encoder is the bottleneck.
- **Simple baselines are strong on real data.** The main strength of the method is graceful behaviour when sensors are late or missing, not absolute accuracy on every dataset.
