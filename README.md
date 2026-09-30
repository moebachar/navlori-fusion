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

## Common tasks

All commands run from the repository root with the virtual environment active.

### See which datasets are available

```bash
python -c "from src.pipeline.fusion.builder import available_datasets; print(available_datasets())"
```

A name is available when there is a `configs/data/<name>.yaml`. That file says which folder under `data/` to read, which sensors to use, and which paths go to train / val / test.

### Convert raw data into the common format

Each converter writes `data/<name>/path_XX/` and prints a per-path summary.

```bash
# our TurtleBot3 golden runs (from the vault's robot/ folder) -> data/side_golden
python scripts/convert_side_golden.py --src X:/navlori-data/robot --out data/side_golden
python scripts/convert_side_golden.py --no-camera          # same, without copying camera frames

# public datasets
python scripts/convert_msiln.py --msiln-root <ILC 2.0 repo with data/> --site site1 --floor B1
python scripts/convert_imuwifine.py --floor 4 --raw-root <IMUWiFine download>
```

The remaining `scripts/convert_*.py` (IPIN 2024, RoNIN, UJIIndoorLoc) follow the same pattern.

### Add your own dataset

1. Write one folder per trajectory, `data/<name>/path_00`, `path_01`, … with these CSVs. `sim_time` is in seconds from the start of the path.

   | File | Columns |
   |---|---|
   | `ground_truth.csv` | `sim_time, gt_x, gt_y` (the targets, metres) |
   | `imu.csv` | `sim_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, roll_deg, pitch_deg, yaw_deg` |
   | `odometry.csv` | `sim_time, odom_x, odom_y, odom_theta_deg, odom_linear_vel, odom_angular_vel, wheel_left_vel, wheel_right_vel` |
   | `wifi.csv` | `sim_time, wifi_visible_count, wifi_strongest_rssi, wifi_strongest_mac`, then one `wifi_rssi_<bssid>` column per access point (empty = not heard). Every path must have the same access-point columns in the same order. |
   | `camera.csv` | `sim_time, frame_id, rgb_path, depth_path, cam_x, cam_y, cam_z` (`rgb_path` relative to the path folder) |

   Leave out the files of sensors you don't have.

2. Copy `configs/data/side_golden.yaml` to `configs/data/<name>.yaml`. Set `collection_dir`, `modalities` and the `split` path ids. Keep `wifi_norm: raw`: the default whitening destroys the WiFi signal.

### Load data in Python

```python
from src.pipeline.fusion.builder import load_config, build_datamodule

cfg = load_config("side_golden")
cfg.dataset.modalities = ["wifi", "imu", "odom"]     # pick the sensors
dm = build_datamodule(cfg)                           # train / val / test datasets + loaders

batch = next(iter(dm.train_dataloader()))
# batch["wifi"]   (128, 1, 173)   the latest scan, one value per access point
# batch["imu"]    (128, 32, 5)    last 32 IMU samples (world-frame accel xy + gyro)
# batch["odom"]   (128, 16, 5)    last 16 odometry samples
# batch["target"] (128, 2)        ground-truth x, y
```

There is one sample per ground-truth timestamp. Each sensor contributes a window of its most recent readings before that time, so the sensors never need a common rate.

### Train and evaluate the fusion model

The ready-made script trains, evaluates every sensor subset, and writes plots, a video and `results.json` to `runs/side_golden/fusion_3mod/`:

```bash
python scripts/_train_side_golden.py --epochs 60 --no-camera --wifi wifi_net
```

Useful options: `--k` (number of past instants fused, default 4), `--arch` (fusion architecture), `--tag` (suffix for the output folder). Without `--no-camera` it adds DPVO camera tokens, which is slow and needs a lot of GPU memory. The other `scripts/_train_*.py` do the same for the other datasets.

The same thing from Python, for your own experiments:

```python
from src.pipeline.fusion.builder import build_encoders, build_model, build_trainer

encoders, _ = build_encoders(cfg, dm)
model = build_model(cfg, encoders)
trainer = build_trainer(cfg, model, dm)
trainer.fit(epochs=60)

pred, gt = trainer.predict("test")                   # (N, 2) tensors, metres
print((pred - gt).norm(dim=1).mean())                # mean position error
print(trainer.evaluate_all_subsets("test"))          # error with each subset of sensors
```

Hyper-parameters (learning rate, dropout rates, K, model size) come from `configs/stage_c/fusion.yaml`.

### Add error bars

```python
from src.pipeline.uncertainty.conformal import ConformalPosition

cp = ConformalPosition(alpha=0.1)                    # target 90 % coverage
cp.calibrate(*trainer.predict("val"))                # radius from validation errors
print(cp.radius, cp.coverage(pred, gt))              # radius in metres, achieved test coverage
```

Coverage is only guaranteed when validation and test paths are alike. On new areas of a building it under-covers.

### Run the baselines

One published method per sensor, trained and tested on the same split:

```bash
python scripts/_eval_wlanloc_side_golden.py                    # WiFi: wlan_localization kNN fingerprinting
python scripts/_eval_ronin_side_golden.py --dataset side_golden # IMU: RoNIN ResNet-1D (any dataset with IMU)
python scripts/_eval_odom_dr_side_golden.py                    # wheel-odometry dead reckoning
```

Results land in `runs/side_golden/<method>/` as JSON plus trajectories. The baseline code itself is the upstream repositories under `external_methods/`.

### Run the tests

```bash
pytest
```

## Building the robot dataset

`scripts/dataset/` rebuilds the TurtleBot3 runs from the raw rosbags:

- **`export/`**: exporters and the loader. Rosbag goes to per-sensor CSVs, camera frames and lidar scans; a lidar scan-to-map SLAM gives the trajectory. `export/load_dataset.py` loads a packaged run directly.
- **`golden/`**: the ground-truth pipeline, run in this order:
  1. `batch_golden.sh`: export every bag and run the lidar SLAM;
  2. `tags_detect.sh`: AprilTag detection;
  3. `gyro_bias_lidar.py`: gyroscope bias;
  4. `tag_ba2.sh`: joint alignment of all runs on the tag network, with a soft prior toward the floor-plan coordinates;
  5. `gt_icp2.py`: map matching for runs that see a single tag;
  6. `golden_package.sh`: packaging;
  7. `plan_register.py` / `plan_plots.py`: plots on the floor plan.

  `alternatives/` holds the variants that were tested and not kept, and `diagnostics/` the one-off checks. The shell scripts are meant for WSL/Linux.

Then `scripts/convert_side_golden.py` (above) turns the packaged runs into the training format.

## What to expect

- **WiFi carries the absolute position.** Without it, no combination of motion sensors can say where the robot is, only how it moved.
- **Simulated WiFi is optimistic.** On Webots data the model reaches sub-metre error, but that WiFi is synthesized. On real, cross-session data the errors are metres, and the WiFi encoder is the bottleneck.
- **Simple baselines are strong on real data.** The main strength of the method is graceful behaviour when sensors are late or missing, not absolute accuracy on every dataset.
