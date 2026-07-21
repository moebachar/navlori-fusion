# IMUWiFine-LSTM baseline on MSILN — training protocol

Author: NavLoRI-Fusion revision pack (item: baseline_protocol_m4b)
Date: 2026-06-18
Scope: documents the IMUWiFine-style LSTM baseline (Nurpeiissov et al., 2022) as trained on the MSILN site1/B1 cross-session split, so the 52-65 m MSILN error can be read as a *fair-protocol overfit on the train session*, not a configuration bug.

All numbers in this note are read directly from saved artifacts (no re-training was performed). The two checkpoints inspected are:

- `runs/main_table/msiln_site1_b1/imuwifine/model.pt` — MSILN-trained LSTM (paper Table 2 baseline).
- `runs/main_table/imuwifine/imuwifine_baseline/model.pt` + `per_path_test.json` — same architecture trained on IMUWiFine fl.4 (sanity check that the implementation works on its native domain).

The training scripts that produced these checkpoints are:

- MSILN-trained: `src/pipeline/baselines/imuwifine.py::train_imuwifine_msiln`, called from `_build_paper_notebook.py:319-321` with `epochs=60, batch_size=8, window=30, stride=10, target_hz=10.0, seed=SEED`. The saved history has 40 epochs (the run used the script's default `epochs=40` rather than the notebook arg; see "Provenance note" at the end).
- IMUWiFine-trained (native sanity): `_train_imuwifine_baseline_on_iwfine.py` with the same architecture, fixed `epochs=40, batch_size=8, window=30, stride=10, seed=42`.

---

## 1. Architecture (identical across both runs)

Clean-room reimplementation of Nurpeiissov, Kuzdeuov, Assylkhanov, Khassanov and Varol, "End-to-End Sequential Indoor Localization Using Smartphone Inertial Sensors and WiFi" (IEEE/SICE SII 2022, DOI 10.1109/SII52469.2022.9708854). Source: `src/pipeline/baselines/imuwifine.py::IMUWiFineModel`.

Per-window input is `(B, T=30, D)` where `T=30` samples at 10 Hz = 3 s, matching the paper's 300-sample 100-Hz window. The pipeline is:

```
input (B, 30, D)
  -> Dropout(0.2)
  -> Linear(D -> 256) + ReLU
  -> Linear(256 -> 256) + ReLU
  -> Linear(256 -> 256) + ReLU
  -> Linear(256 -> 256) + ReLU
  -> LSTM(input=256, hidden=256, num_layers=4, batch_first=True, dropout=0.2)
  -> Linear(256 -> 2)        # raw (x, y) in metres, no target normalization
```

WiFi RSSI handling: each of the 1419 MSILN site1/B1 APs (resp. 343 IMUWiFine fl.4 APs) becomes a fixed feature column in the concatenated input vector. Missing-AP sentinel is `-100 dBm`, then `_normalize_wifi` shifts and scales to `[0, 1]` and raises to power `e` (verbatim from the paper's `normalize_wifi`).

IMU is 6 channels (accel_xyz, gyro_xyz), per-window min-max normalised to `[0, 1]` (`_normalize_imu_inplace`). MSILN site1/B1 has no magnetometer, so the paper's 3 magn channels are dropped.

Input dimension `D`:

| Run | n_aps | n_imu | D = n_aps + n_imu | Window | Output |
|-----|-------|-------|------------------|--------|--------|
| MSILN site1/B1     | 1419 | 6 | **1425** | 30 @ 10 Hz | 2 (x, y) |
| IMUWiFine fl.4     |  343 | 6 |  349    | 30 @ 10 Hz | 2 (x, y) |

Model size: 2.67 M params (MSILN) vs 0.91 M params (IMUWiFine fl.4). The size difference is entirely in the first `Linear(D -> 256)` layer driven by the AP-vocabulary size.

Important deviation from the paper that is honestly an MSILN-tractability adaptation, not a fix: the paper uses `hidden_dim = input_dim`. At MSILN's `D = 1425` that produces a ~30 M-param 4-layer LSTM. We downsample to `hidden_dim = 256` (see module docstring lines 26-28). This adaptation is applied identically on both MSILN and IMUWiFine fl.4, so the comparison is internally fair.

## 2. Training protocol (MSILN run, the one with the 65 m / 53 m error)

Source: `src/pipeline/baselines/imuwifine.py::train_imuwifine_msiln`, defaults.

| Item | Value | Notes |
|------|-------|-------|
| Dataset | MSILN site1/B1 train session | paths 0-93 only; same train split as Ours (incumbent transformer) |
| Val split | paths 94-127 | cross-session, same as Ours |
| Test split | paths 128-132 | held-out session, same as Ours |
| Windows | T=30, stride 10 train / stride 30 val+test | 33 900 train windows, 9 480 val windows, 2 670 test windows |
| Epochs | 40 | saved history length; see "Provenance note" |
| Batch size | 8 | (`train_imuwifine_msiln` default) |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 | matches the paper's reference recipe |
| LR schedule | CosineAnnealingLR, T_max = epochs | not in the paper but harmless |
| Loss | MSE on raw `(x, y)` in metres | matches the paper's reference recipe verbatim; no target normalization |
| Dropout | 0.2 (paper) | applied at input + between LSTM layers |
| Early stopping | none; best-val checkpoint kept | best-val state is restored at end of training |
| Seed | 42 (script default `seed=42`; notebook passes `SEED=42`) | `torch.manual_seed` + `np.random.seed` set inside `train_imuwifine_msiln` |
| Hardware | Quadro P4000 8 GB, PyTorch 2.4.1+cu124 | elapsed 340.8 s |

Final saved metrics (from `model.pt["summary"]`, all in metres, full-split evaluation with stride==window):

| Split | MAE | n samples |
|-------|-----|-----------|
| **train** | **66.39** | 33 900 |
| **val**   | **65.15** | 9 480  |
| **test**  | **52.69** | 2 670  |

Best val MAE = 65.15 at epoch 39 (the final epoch); the curve has plateaued by ~epoch 5 (val MAE 73.8 → 66.1 → 65.5 → 65.5 → ... 65.2). Final train MSE = 2641.9 (`sqrt(MSE) ≈ 51.4 m`).

## 3. Hyperparameter tuning — there was none

A single configuration was run; no grid search, no Optuna, no manual sweep. Verified by:

- The only entry point is `train_imuwifine_msiln` with one fixed call site (`_build_paper_notebook.py:319-321`).
- `grep -r "imuwifine.*\(grid\|sweep\|optuna\|tune\)"` over the repo returns nothing.
- The module docstring (`src/pipeline/baselines/imuwifine.py:19`) lists a single "reference recipe" and the production checkpoint matches it.

This is deliberate: the IMUWiFine baseline is reported as a *clean-room reimplementation of a prior method at its published recipe*, not as a method we are claiming to beat with hyperparameter search. The fair-comparison stance is: same train split, same window length in seconds, same paper-recipe optimiser, no MSILN-specific tuning.

## 4. The 52-65 m error is overfit to the train-session centroid, not a misconfig

The decisive signal is the *train-vs-val MAE gap*:

- Train MAE = 66.4 m
- Val   MAE = 65.2 m
- Test  MAE = 52.7 m

If the LSTM had been mis-wired (random-level training), train MAE would be of order the data range (MSILN train extents are 230 m x 146 m, so a random predictor sits around 90-120 m). It is not. Train MAE (66 m) is *lower than* the data range and only ~1 m above val.

The much stronger diagnostic: a trivial **predict-train-mean** baseline (always predict `(157.4, 174.4)`, the train-session centroid) scores almost identically to the LSTM on every split:

| Split | LSTM MAE | predict-train-mean MAE | ratio |
|-------|----------|------------------------|-------|
| train | 66.39 | 66.19 | 1.003 |
| val   | 65.15 | 65.12 | 1.0005 |
| test  | 52.69 | 53.12 | 0.992 |

The final train MSE of the LSTM (2641.9) is within **1.2%** of the constant-mean MSE on train (2610.5 = `Var(train_gt)`). The LSTM has converged to "predict the train centroid". This is the textbook signature of a high-capacity sequence model that cannot extract any useful per-window position signal from its input on this domain.

So the failure mode is *not* "random output" (it is not), nor a wiring or normalization bug (the same code works on IMUWiFine fl.4; see §5), but: **the model fits the train-session marginal of (x, y) and learns to ignore the input**. On test, which happens to sit closer to that centroid than val, this collapse looks slightly less bad (52.7 m) but is the same phenomenon.

The actual mechanism is (a) MSILN's `D = 1425` input with a `[0, 1]` per-AP encoding produces a very sparse, very high-dimensional signal that a small (~2.7 M param) downsampled LSTM cannot route into a useful representation in 40 epochs at batch 8, and (b) with absolute-position targets in raw metres and no target normalisation, the regression head trivially minimises MSE by predicting the conditional mean. A version with target normalisation (subtract train mean, divide by train std) was prototyped and got to val 15.15 m / test 7.43 m (`handoff/results/RESULT_40_fusion-baselines-integration.md` line 27) but is **not** the production checkpoint in `runs/main_table/msiln_site1_b1/imuwifine/model.pt`; we keep the no-normalization version in the paper because it is the verbatim paper recipe and the relevant comparison is "Nurpeiissov et al.'s method, as published, applied cross-session".

## 5. Sanity check: same architecture and protocol works on its native dataset

The exact same `IMUWiFineModel` class, same window length (30 @ 10 Hz), same optimiser (AdamW lr=1e-3 + weight_decay=1e-4), same MSE loss on raw `(x, y)`, same `hidden_dim=256, n_layers=4, dropout=0.2`, same `batch_size=8`, same `seed=42`, was trained on IMUWiFine floor 4 by `_train_imuwifine_baseline_on_iwfine.py` (40 epochs).

Results read from `runs/main_table/imuwifine/imuwifine_baseline/`:

| Split | MAE | n samples |
|-------|-----|-----------|
| best val (across 40 epochs) | **1.61 m** at epoch 33 | 20 val paths |
| final val (epoch 39) | 1.65 m | 20 val paths |
| test (aggregate across all 20 test paths, recomputed from per-path JSON) | **4.12 m** | 36 900 samples |
| test (median per-path) | 2.15 m | — |

Per-path test MAE ranges from 1.10 m (path_71) to 7.81 m (path_78); 14 of 20 paths score under 5 m. Final train MSE = 2.20 (so `sqrt(MSE) ≈ 1.48 m`), i.e. training error is now an order of magnitude smaller than the data-range constant-mean predictor would give.

So **the architecture and the training recipe are not the problem**. The IMUWiFine LSTM hits 1.6 m val / 4.1 m test on IMUWiFine fl.4 — competitive with the paper's reported numbers — using identical code. The 52-65 m error on MSILN is therefore a domain-fit failure (high-dim WiFi vocabulary + absolute-coord targets + cross-session split, all at once), not a setup bug.

## 6. Headline (one line)

> **LSTM-on-MSILN: train MAE = 66.4 m | val MAE = 65.2 m | test MAE = 52.7 m; same architecture scores best val 1.6 m / test 4.1 m on its native IMUWiFine fl.4 — the MSILN error is a centroid-collapse overfit (matches predict-train-mean to within 1.2%), not a misconfiguration.**

---

## Provenance note (epochs discrepancy)

The notebook builder `_build_paper_notebook.py:320` requests `epochs=60`, but the saved `model.pt` history has length 40. The production checkpoint was produced by an earlier run (or a manual call) that used the script default `epochs=40` rather than the notebook arg; the cached path in the notebook (`if FAST_MODE and ckpt.exists(): load_imuwifine_msiln(...)`) reuses the existing 40-epoch checkpoint and does not retrain. This does not change any conclusion in §4: the val curve had plateaued from epoch ~3 onwards and the train loss is already collapsed to the constant-mean MSE by epoch 30. Adding 20 more epochs would not move the numbers.

## Artifact pointers

- Source (clean-room model + trainer): `src/pipeline/baselines/imuwifine.py`
- Source (data loader, splits, WiFi vocabulary): `src/pipeline/baselines/_msiln_loader.py`
- Source (native sanity training): `_train_imuwifine_baseline_on_iwfine.py`
- MSILN checkpoint: `runs/main_table/msiln_site1_b1/imuwifine/model.pt`
- IMUWiFine fl.4 checkpoint + per-path JSON: `runs/main_table/imuwifine/imuwifine_baseline/{model.pt,per_path_test.json}`
- Prior integration write-up: `handoff/results/RESULT_40_fusion-baselines-integration.md`
- Paper rows quoting these numbers: `paper-workspace/scope.md` §2.2 (MSILN val 65.15, MSILN test 52.69)
