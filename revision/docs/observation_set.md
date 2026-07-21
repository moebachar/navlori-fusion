# Observation-set construction (reproducibility note)

This document specifies how a raw async-collection trajectory is turned into
the multi-modal observation sets consumed by the fusion model. The goal is
to make every windowing and indexing choice in our experiments verifiable
from the source.

Source files (all paths relative to the repository root):

* `src/pipeline/data/dataset.py` — per-sample window extraction (`FusionDataset`).
* `src/pipeline/data/datamodule.py` — train/val/test wiring and normalisation sharing.
* `src/pipeline/training/fusion_trainer.py` — K-instant temporal indexing and query-time sampling.
* `configs/stage_c/fusion.yaml` — training defaults (`n_instants`, `instant_stride`, batch size, camera stride).
* `configs/data/<dataset>.yaml` — per-dataset modalities, splits, window sizes and Wi-Fi normalisation mode.

## 1. Anchor grid: the ground-truth timeline

Each trajectory `path_XX/` contains a `ground_truth.csv` sampled at the
dataset's GT rate (Webots: 10 Hz; MS-ILN site1/B1: 10 Hz waypoint
interpolation; IMUWiFine: 10 Hz; IPIN 2024: 10 Hz). One sample is emitted
**per GT row**. Splits are *by path index* (no time-shuffling inside a
path), so train and validation never share trajectory time. Path-level
splits are listed in `configs/data/<dataset>.yaml` (`split.train_paths`,
`split.val_paths`, `split.test_paths`).

The full per-sample bookkeeping is in
`FusionDataset.__init__` (`dataset.py` lines 232-241): each entry stores
`{time, target, path_idx, path_id, path_dir}` where `time` is the GT
timestamp `t` and `target` is the GT `(x, y)`.

## 2. Per-modality windowing at a single instant

For one GT anchor at time `t`, `FusionDataset._get_window` (`dataset.py`
line 480) builds the modality token by taking *the most recent `W_m`
observations of modality `m` at or before `t`*, where `W_m` comes from
the per-dataset `windows` block and defaults to:

| Modality | Window `W_m` | Native rate | Span |
| --- | --- | --- | --- |
| WiFi | 1 scan | ~1 Hz | one scan |
| IMU | 32 samples | ~31 Hz (Webots) / ~32-50 Hz (real) | ~1 s |
| Odometry | 16 samples | ~15 Hz | ~1 s |
| Camera (still frame) | 1 frame | ~5 Hz (Webots) | one frame |
| Camera pair (DPVO) | `camera_window` frames at stride `camera_stride` | -- | see below |

Windows that fall before the start of the trajectory are zero-front-padded
(`dataset.py` lines 543-547). If a modality has *no* observation at or
before `t`, the window is all zeros and the modality is flagged unavailable
in the next step.

**Camera windows for motion encoders.** When the camera modality serves a
motion encoder (DPVO), `configs/stage_c/fusion.yaml` sets
`camera_window=2` and `camera_stride=5`, so a pair spans ~1 s of motion at
the ~5 Hz Webots camera rate. This is implemented in
`FusionDataset._get_camera` (`dataset.py` line 589): the frame at index
`end` and the frame at index `end - camera_stride` are stacked into a
`(2, 3, H, W)` clip.

**IMU stride.** For the IMU we always take the trailing `W_imu`
consecutive raw rows (no sub-sampling): the native rate already matches
the ~1 s window we want, so `imu_stride` is effectively 1. For Webots,
`configs/data/simulation_2mod.yaml` keeps `imu: 32` and the IMU world-frame
projection (yaw-rotated horizontal accel) is selected via `imu_frame:
world` where requested (`dataset.py` lines 40-51).

**Availability flags.** A modality is marked **available** for instant `t`
iff at least one raw observation has `time <= t`. Empty / missing-file
paths and the WiFi staleness cap (`wifi_max_stale_s`; treats a carried-over
scan older than the cap as absent, `dataset.py` lines 514-522) feed the
unavailability mask, which the FusionTransformer respects via padding-mask
attention.

## 3. WiFi normalisation modes

The Wi-Fi feature is treated differently per dataset, controlled by
`preprocessing.wifi_norm` in `configs/data/<dataset>.yaml`:

* `wifi_norm: raw` — missing values filled with -100 dBm, then a fixed
  affine `(rssi + 100) / 100` mapping into `[0, 0.7]`. **No PCA, no
  z-score.** Used for MS-ILN site1/B1 and IPIN 2024 because per-component
  z-scoring of PCA components destroys signal on real Wi-Fi (a probe
  showed Wi-Fi-only kNN regressing from ~5 m to ~21 m on IPIN under
  whitening). See `dataset.py` lines 152-159, 529-535.
* `wifi_pca: 128` + default normalisation — PCA-rotate the raw RSSI vector
  to 128 dimensions, then z-score per component using stats from the
  **training set only**. Used for Webots and IMUWiFine (343-AP) datasets;
  the PCA is fit in `FusionDataset._fit_wifi_pca` and shared between
  train / val / test in `FusionDataModule.setup` (`datamodule.py` lines
  126-128).

## 4. K-instant temporal indexing

After per-instant windows exist, `FusionTrainer._build_temporal_index`
(`fusion_trainer.py` lines 198-236) assembles the *temporal index*. For
each sample (GT row `L` of its path) it grabs the `K = n_instants` most
recent in-path rows
`[L - (K-1) * step, ..., L - step, L]`, where `step = instant_stride`.
The anchor / query instant is always the last (`k = K-1`). Instants whose
position falls before the start of the path are masked out (not clamped).

Defaults from `configs/stage_c/fusion.yaml` for the main results in the
paper:

| Field | Value |
| --- | --- |
| `n_instants` | **4** in published checkpoints; the config default is 8 (also reported) |
| `instant_stride` | **9** GT rows |
| `batch_size` | 128 |

With the 10 Hz GT grid, `instant_stride=9` corresponds to ~**0.9 s**
between adjacent instants, so `K=4` covers roughly the last ~2.7 s of
trajectory at the query anchor.

The temporal index returns three tensors of shape `(N, K)`:

* `inst_idx[n, k]` — global cache index of the `k`-th instant of sample `n`.
* `inst_av[n, k]`  — boolean availability (False for instants before path start).
* `inst_dt[n, k]`  — `t_instant - t_anchor` in seconds, used by the
  continuous-time encoding on each token.

`FusionTrainer._batch` (`fusion_trainer.py` line 238) gathers the per-
modality windows for those `K` indices, producing modality tensors of
shape `(B, K, *window)` plus a `(B, K)` availability mask per modality.

## 5. Query-time sampling

The fusion readout is a cross-attention `PositionQuery` parameterised by a
target time `tau` (`configs/stage_c/fusion.yaml`: `readout: query`).
`FusionTrainer._resolve_query` (`fusion_trainer.py` lines 257-273)
samples `tau` differently for training and evaluation:

* **Training.** For each sample, draw `q ~ Uniform{0,...,K-1}` and set
  `tau` to the timestamp of instant `q`. The target is the GT position
  at instant `q`, taken from the pre-stacked `y_inst[B, K, 2]`. Randomising
  `q` forces the query to learn time-conditional routing.
* **Validation / test / subset eval.** `tau` is fixed at the anchor
  (`query_dt = 0`), and the supervision target is `y_anchor`, i.e. the GT
  at the latest instant. All `evaluate_*` paths (`evaluate_subsets`,
  `evaluate_staleness`, `predict`) go through this branch and so report
  numbers conditioned on querying at the most recent GT instant.

## 6. Augmentation: modality and instant dropout

Two independent dropout mechanisms are applied **only at training time**
inside `FusionTrainer._apply_dropout` (`fusion_trainer.py` lines 275-314):

* `modality_dropout` zeros the availability mask of an entire modality
  across all K instants for a random subset of samples (`p = 0.4` in
  `fusion.yaml`).
* `instant_dropout` zeros individual `(instant, modality)` tokens
  independently (`p = 0.45`).

A rescue pass guarantees that at least one anchor-instant token survives
per sample (otherwise the FusionTransformer's CLS attention row would be
fully masked, producing NaN softmax). Evaluation never drops tokens; the
masks come only from sensor availability and (for Wi-Fi) the optional
`wifi_max_stale_s` staleness cap.

## 7. Normalisation discipline

`FusionDataModule.setup` (`datamodule.py` lines 107-163) computes per-
modality `(mean, std)` and (when used) the Wi-Fi PCA basis from the
**training paths only**, then reuses the same statistics for val and
test. This is what `stats=shared_stats` and `wifi_pca_model=shared_pca`
enforce when constructing the val and test `FusionDataset` instances.
There is no leakage of validation or test statistics into training.
