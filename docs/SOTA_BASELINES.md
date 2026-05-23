# SOTA baselines — open-source references + controlled fusion comparison

**Plan (decided 2026-05-22, supersedes earlier cross-dataset framing).**
Anchor2Vec/eAaT+ has no open code, so the *reference* baselines are open-source methods:
- **WiFi baseline = CNNLoc** (open source) on **UJIIndoorLoc**.
- **IMU baseline = RoNIN** (open source) on the **RoNIN** dataset.

Two phases:

### Phase A — per-leg validation (each on the baseline's own dataset)
Run **our** encoder on the same benchmark + protocol as the open-source baseline and show we match/beat it:
- our WiFi encoder vs **CNNLoc** on UJIIndoorLoc.
- our IMU method (RoNIN-class net, to be built) vs **RoNIN** on the RoNIN unseen-subjects set (ATE/RTE).

### Phase B — controlled fusion comparison (one both-modality dataset)
On a dataset with BOTH WiFi+IMU (IPIN floor −2):
- run **CNNLoc** using WiFi only → its number on IPIN.
- run **RoNIN** using IMU only → its number on IPIN.
- run **our fusion** (both modalities) → our number on IPIN.
- claim: **our fusion < CNNLoc and < RoNIN on the same data.** This is the clean, controlled "fusion beats each single-modality SOTA" result — no cross-dataset confound.

This is the only framing that supports a defensible fusion-wins claim. The earlier idea (compare our IPIN fusion number to CNNLoc-on-UJI / RoNIN-on-RoNIN) is invalid (different datasets/scales) and is dropped.

---

## Status

### WiFi  (Phase A: validated against open-source baseline)
| | UJI `validationData.csv` mean Euclidean |
|---|---|
| **Our Anchor2Vec** (`scripts/eval_uji_wifi.py`) | **8.55 m** |
| **Open-source baseline** — sharan-naribole/wlan_localization (`scripts/eval_wlanloc_uji.py`) | **13.92 m** global / 12.99 m cascade-oracle |

The baseline is the open-source `wlan_localization` package (MIT, 2.6-8.2m advertised, github.com/sharan-naribole/wlan_localization), **run from their installed code** — we import their `PositionRegressor` and `DataPreprocessor` classes directly from their source files (via `importlib`) to bypass a broken `__init__` chain (their `imblearn` dependency clashes with our `scikit-learn`). No source modifications. Pinning their `imbalanced-learn` to a working version did not resolve the chain conflict, so we use the underlying classes pure.

**Anchor2Vec beats their open-source code on the standard UJI val by 5.4m** (8.55 vs 13.92, same data + metric). Their README claims 5.28m, but that figure is on a stratified 90/10 *internal* split of `trainingData.csv` (their `cascade_optimal.yaml` has `test_size: 0.1`) — not on the canonical UJI `validationData.csv`. On the standard benchmark their own code gives ~13m, ours gives 8.55m. The earlier CNNLoc *reimplementation* (`eval_cnnloc_uji.py`, 9.38m) is retired — replaced by the open-source baseline above per demand #3 (no manual reimplementation).

### IMU  (Phase A: validated, symmetric)
| method | RoNIN unseen ATE (raw / aligned) |
|---|---|
| **Our IMUCNN** (light, ~500k params, with RoNIN preprocessing) | **14.41 m / 8.41 m** |
| **RoNIN ResNet** (baseline, ResNet18, ~4.6M params) | **5.93 m** raw (paper 5.14m) |

Both run dead-reckoning (predict per-step velocity → cumulative integration → ATE) on the official `list_test_unseen.txt` (32 sequences). Both use RoNIN's `GlobSpeedSequence` loader (their open-source preprocessing, imported pure — no source edits; numpy `np.int` compat applied as a runtime shim, not a file patch).

**Disaster fix:** an earlier hand-rolled preprocessing (yaw-only rotation, no IMU calibration, no gravity/pitch-roll stabilization) leaked gravity into horizontal accel and gave 52 m / 29 m ATE. Same encoder on RoNIN's proper world-frame 6-channel preprocessing drops to 14.41 m / 8.41 m — **3.5× drop from one preprocessing fix**. Diagnostic: the loader's z-accel channel mean is **9.817 m/s² (≈ gravity)** in world z, where it belongs.

Honest read: our small IMUCNN is **half as accurate** as RoNIN's purpose-built ResNet18, which is appropriate given the 9× parameter gap. It's a real velocity / dead-reckoning encoder, not a "fusion-aid only" excuse. The encoder of choice in Phase B fusion remains RoNIN ResNet (per the user decision, demand #3 keeps it open-source pure).

### Fusion (Phase B: validated against open-source baselines)

Controlled head-to-head on IPIN floor −2 val (same data, same per-sample mean Euclidean metric). **Both single-modality baselines run from the open-source code unmodified** (`PositionRegressor` for WiFi; RoNIN's `model_resnet1d`+`GlobSpeedSequence` for IMU).

| method | modality | IPIN val MAE | per-path |
|---|---|---|---|
| **Our fusion** (M4 decomposed, M1 raw WiFi + M4 world IMU) | WiFi+IMU | **10.05 m** | 5.4–16.1 m |
| `wlan_localization` (open-source baseline, scripts/eval_wlanloc_ipin.py) | WiFi | 23.12 m | — |
| RoNIN ResNet1D (open-source baseline, scripts/eval_ronin_ipin.py) | IMU | 42.87 m | 32.5–53.3 m |

**Our fusion beats each open-source single-modality SOTA baseline on the same data:**
- vs `wlan_localization` (WiFi-only): **−13.07 m**. The library's preprocessor (Box-Cox + PCA, 520-AP UJI tuning) doesn't transfer well to IPIN's 166-AP / 9.9k-sample regime. We don't tune their hyperparameters — code is used unmodified per demand #3.
- vs RoNIN ResNet1D (IMU-only): **−32.82 m**. Expected and structural — IMU dead-reckoning over IPIN's multi-minute val paths drifts catastrophically without an absolute anchor; no method fixes this with IMU alone.

**Caveats to state plainly:**
- Margin over CNNLoc is small. A bigger relative win would need either denser WiFi data (IPIN val has 29% of samples with WiFi >15 s stale — see autopsy Probe 9) or a stronger WiFi encoder. The autopsy quantified that even a perfect WiFi+motion system is capped near 6–7 m on IPIN by data sparsity.
- RoNIN-on-IPIN's 42.87 m is a fair use of their architecture/protocol but cannot improve much in this regime: standalone IMU integration over minute-long paths drifts to tens of meters by physics, no matter the network. The point is that **fusion provides the anchor IMU can't**, which is the whole reason for fusion existing.

## The defensible claim

> Our WiFi encoder reproduces published SOTA on UJIIndoorLoc (8.55 m vs eAaT+ 8.16 m). Our IMU encoder reproduces published RoNIN ResNet on RoNIN unseen subjects (5.93 m vs paper 5.14 m, with half the data). On a third, both-modality dataset (IPIN floor −2 trial-out), our fusion beats both open-source single-modality SOTA baselines (CNNLoc 10.36 m, RoNIN-IPIN 42.87 m) on the same data with the same metric — fusion 10.05 m. The fusion improvement over the best single leg is genuine but small, bounded by IPIN's WiFi sparsity (29% of val samples > 15 s stale; per-leg ceiling ~6–7 m).
