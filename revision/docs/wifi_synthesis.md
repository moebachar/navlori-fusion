# Webots WiFi-field synthesis (reproducibility note)

This document specifies how Wi-Fi RSSI observations are produced in our
Webots simulation environment, so that the sub-metre Wi-Fi MAE we report on
the simulated dataset can be interpreted correctly.

Source files (all under `src/simulation/controllers/wifi_supervisor/webots_export/`):

* `metadata.json` — area extents, AP list, per-AP RSSI normalisation statistics.
* `hyperparams.json` — fitted Gaussian-process (GP) hyperparameters per AP.
* `wifi_predictor.py` — runtime predictor used by the Webots controller.
* `rssi_grid.npz` — pre-evaluated `(y, x, AP)` mean / std grid (25 cm spacing).
* `gp_models.pkl` — the trained GP regressors (used only by the exact-GP mode).

## What is synthesised

The Webots Wi-Fi field is *not* a closed-form path-loss model. It is a
**calibration to a real survey**: for each access point (AP), an independent
Gaussian-process regression is fit on real RSSI measurements indexed by
metric `(x, y)` positions, and the field at any robot position is then either
(a) read from a pre-computed grid by bilinear interpolation or (b) sampled
exactly from the GP posterior. This reproduces the spatial structure of the
calibration survey, including survey-specific shadowing and AP-power
asymmetries, but it does *not* model channel-specific multipath, AP duty
cycle, or session-to-session drift. The numbers in the table below should
therefore be read as a faithful replay of the calibration set rather than as
generic Wi-Fi propagation.

## Geometry and AP layout

| Field | Value |
| --- | --- |
| Number of APs (`num_aps`) | **120** |
| AP MACs (vendor blocks) | 75 with prefix `34:15:93:5c:**`, 45 with `34:15:93:9c:**` |
| Bounding box `x` | `[-11.76, 9.29]` m  (extent **21.04 m**) |
| Bounding box `y` | `[-18.07, 1.39]` m  (extent **19.46 m**) |
| Floor area (bounding box) | **409.5 m^2** |
| Pre-evaluated grid | `79 x 86` cells at **0.25 m** spacing |

## Per-AP GP model

* **Kernel family.** Per-AP RBF (squared-exponential) with anisotropic
  lengthscales `(lengthscale_x, lengthscale_y)`, scalar `outputscale`,
  homoscedastic `noise` and a learned constant `mean`. One GP is trained
  per AP, in the normalised `(x, y)` cube `[0, 1]^2`, on the RSSI target
  z-scored by the per-AP `(mean, std)` in `metadata.json["normalization"]["targets"]`.
* **Training set per AP.** Up to 5 000 randomly subsampled survey points
  (the cap was hit for 115/120 APs; the five low-coverage APs in vendor
  block `34:15:93:5c:84:**` saw 3 529 to 4 198 points).
* **Inference modes.** `wifi_predictor.py` exposes two modes:
  * `method="grid"` (default in Webots): bilinear lookup against the
    `(79, 86, 120)` `rssi_mean` / `rssi_std` cube, then Gaussian noise of
    the locally-interpolated std is added. ~0.1 ms per query.
  * `method="gp"`: exact GP posterior from `gp_models.pkl` (slower; only
    used for offline accuracy checks).
* In both modes the controller clamps outputs to `[-100, -20]` dBm and
  marks AP responses below `-150` dBm as missing (out-of-coverage).

### Fitted hyperparameter statistics (across 120 APs)

Lengthscales in the table are reported both in normalised units (as stored
in `hyperparams.json`) and converted to **metres** by multiplying through
the per-axis extent of the bounding box.

| Hyperparameter | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- |
| `noise` (z-scored) | 0.238 | 0.185 | 0.168 | 0.029 | 0.705 |
| `outputscale` | 2.55 | 0.50 | 2.64 | 0.50 | 4.00 |
| `mean` (z-scored offset) | -0.046 | 0.339 | -0.105 | -0.836 | 0.801 |
| `lengthscale_x` (norm) | 0.0597 | 0.0283 | 0.0540 | 0.0009 | 0.164 |
| `lengthscale_y` (norm) | 0.0584 | 0.0264 | 0.0522 | 0.0011 | 0.139 |
| `lengthscale_x` (**m**) | **1.26** | **0.60** | 1.14 | 0.02 | 3.45 |
| `lengthscale_y` (**m**) | **1.14** | **0.51** | 1.02 | 0.02 | 2.70 |
| `n_train` per AP | 4 954 | 228 | 5 000 | 3 529 | 5 000 |

### Per-AP target statistics (RSSI in dBm)

Across the 120 APs the survey RSSI distribution has:

| Quantity | mean across APs | std across APs | min | max |
| --- | --- | --- | --- | --- |
| Per-AP `mean(RSSI)` | -73.5 dBm | 9.4 dBm | -88.2 | -55.7 |
| Per-AP `std(RSSI)`  |  7.0 dBm  | 2.6 dBm | 2.1  | 11.3 |

Interpretation: typical AP coverage is ~1 m correlation length on each
axis with ~7 dB amplitude variation around a ~-73 dBm mean. APs with
narrow lengthscales (`< 0.05` m normalised, i.e. ~1 m) carry sharp
spatial signature; APs with wide lengthscales (`> 0.1` m, i.e. ~2 m)
behave closer to "global level only".

## Honest caveat (calibration vs. propagation)

The Webots field is GPR-synthesised from a single real calibration sweep,
not a propagation simulation. Two consequences are relevant for the paper:

1. **Sub-metre simulation Wi-Fi MAE is optimistic.** Because the same
   spatial process produced the train, val and test trajectories in our
   Webots split, no session-to-session calibration drift is present. Real
   Wi-Fi fingerprints exhibit non-trivial inter-session drift (we observe
   this directly on IMUWiFine and IPIN 2024).
2. **The bottleneck is the encoder, not fusion.** On real cross-session
   splits (`ipin2024_floor0`, `imuwifine`) the Wi-Fi encoder does not
   transfer, and the fusion network cannot recover information that the
   encoder did not extract. The sub-metre Webots numbers should therefore
   be cited as a *controlled-conditions ceiling*, not as a deployment
   forecast. We make this point explicit when reporting simulation results.
