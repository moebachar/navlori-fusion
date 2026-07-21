# configs/ — layout

Every file in this directory is loaded by code; there is no unused scaffold.
Plain OmegaConf (no Hydra composition).

## `stage_a/` — per-modality encoder configs (Stage A)

One yaml per encoder variant, grouped by modality. Loaded by the Stage-A
training path (`EncoderTrainer` in `src/pipeline/training/trainer.py` and the
`scripts/` training/eval entry points).

| File | Encoder |
|------|---------|
| `wifi/wifi_net.yaml` | WiFiNet (renamed from Anchor2Vec, PLAN_39) |
| `imu/cnn1d.yaml` | IMUCNN |
| `odom/linear.yaml` | OdomCNN head config |
| `vision/vit.yaml` | VisionViT (DINOv2 + LoRA) |
| `vision/dpvo.yaml`, `vision/dpvo_full.yaml` | DPVOMotionEncoder variants |

## `stage_c/fusion.yaml` — the fusion pipeline (Stage B+C)

Single source of truth for the FusionTransformer: model dims, training
hyperparameters, temporal (K instants × stride), Optuna search space, and
`data.default_dataset`. Loaded by
`src.pipeline.fusion.builder.load_config(dataset)`, which every consumer
(notebooks, smoke harness, Optuna, eval scripts) goes through.

## `data/` — dataset selection

One yaml per dataset; **the filename is the dataset name** passed to
`load_config("<name>")`. A dataset is selectable when its yaml declares
`split` + `modalities` AND its converted directory exists on disk
(`builder.available_datasets()` enforces both).

| File | Dataset |
|------|---------|
| `simulation.yaml` | Webots TIAGO++ sim, 4 modalities |
| `simulation_2mod.yaml` | Webots sim, WiFi+IMU only (paper scope) |
| `msiln_site1_b1.yaml` | Microsoft Indoor Localization 2.0, cross-session (paper headline) |
| `imuwifine.yaml` | IMUWiFine floor 4 |
| `ipin2024_floor0.yaml` | IPIN 2024 Track 3 floor 0 (diagnostic; excluded from paper) |

## History note

The original 5-stage Hydra scaffold — `config.yaml` defaults list,
`stage_b/` (mTAN/GRU-D), `stage_d/` (KalmanNet), `stage_e/conformal.yaml`,
`training/`, `experiment/`, `data/{silva,tiago}.yaml`, and the
`scripts/{train,evaluate}.py` TODO skeletons — was removed on 2026-07-21.
Those stages were subsumed (temporal attention replaced B and D) or never
built; conformal prediction lives in `src/pipeline/uncertainty/conformal.py`
with parameters set at call sites. Recover any of it from git history.
