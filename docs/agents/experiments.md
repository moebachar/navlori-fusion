# Brief: experiments agent (training and benchmarks)

## Mission

Run the experiments that turn the datasets into results: baselines per sensor, the fusion model, ablations and multi-seed numbers, on every dataset. You also own the model code (`src/pipeline`). Report results honestly, including when a simple baseline wins.

## What exists

- **Model:** per-sensor encoders → one continuous-time set transformer over the last K instants → `(x, y)`; modality and instant dropout for robustness; split-conformal intervals. Walkthrough: `docs/fusion_pipeline.md`. The API is shown in the README ("Common tasks").
- **Datasets:** `configs/data/*.yaml`, all readable through `data/` (the vault):
  - `side_golden`: 12 real TurtleBot3 runs, 4 sensors;
  - `simulation`: Webots;
  - `msiln_site1_b1`, `imuwifine`, `ipin2024_floor0`;
  - `iln20_5d27099f_F2_replay`.
- **Paper:** "Continuous-Time Set-Transformers for Asynchronous WiFi-IMU Indoor Localization", ICINCO 2026. The paper text lives on branch `paper-icinco-2026`; the multi-seed runs are in `colab/`.
- **First benchmark on `side_golden`** (2026-09-28, test runs 2/6/12, mean / median error):

  | Method | Mean | Median |
  |---|---|---|
  | WiFi, wlan_localization kNN | 13.9 m | 6.2 m |
  | IMU, RoNIN ResNet-1D, from the true start pose | 4.7 m | 4.2 m |
  | Wheel odometry dead reckoning, from the true start pose | 0.13 m | 0.08 m |
  | Fusion WiFi + IMU + odometry | 3.9 m | 2.3 m |

  Notes on these numbers:
  - The same fusion network with **WiFi alone gets 2.7 m**, so IMU and odometry currently hurt it.
  - The wheel-odometry error is below the ground-truth accuracy (~15 cm), and odometry shares the gyro heading with the ground truth. Don't rank it against the others.
  - Scripts: `scripts/_train_side_golden.py`, `scripts/_eval_{wlanloc,ronin,odom_dr}_side_golden.py`. Results are in `X:\navlori-fusion\runs\side_golden\` (main checkout).
- **The camera (4-sensor) run froze the whole PC.** DPVO token extraction (`extract_vision_tokens_fast`, `flush_batch=128`) filled the 8 GB GPU that also drives the display. Before retrying, make extraction memory-safe (small `flush_batch`, copy tensors out of batches, a memory watchdog), run it under the GPU lock, and tell the user first.

## Backlog, in priority order

1. **Why sensors hurt fusion on real data.** Why is WiFi-only (2.7 m) better than WiFi + IMU + odometry (3.9 m) on `side_golden`? Check normalisation, window lengths at the real rates (IMU 50 Hz, odometry 15 Hz after conversion), overfitting on 7 short training runs, and the dropout settings. Report per-subset numbers over several seeds.
2. **A robust camera baseline** on `side_golden` that fits in memory, so all four sensors are covered.
3. **Session-invariant WiFi**, the known bottleneck on real data. The best lead from June is "idea1: WiFi as place context in the IMU sequence" (MSILN 9.93 ± 0.25 m over 4 seeds) plus WiFi-JEPA pretraining. Code: `X:\navlori-data\archive\experiments\idea1_wifi_pe_imu.py`; see the "What's Next" list in `CLAUDE.md`.
4. **Replay datasets:** when the replay agent merges a new dataset, benchmark it with the same protocol.

## Rules specific to you

- Other agents build their data through `src/pipeline` (the dataset classes and builder). Keep its public API stable: add options with defaults that preserve current behaviour.
- Every GPU job goes through `scripts/gpu_lock.py`. Train in your worktree; outputs go to your worktree's `runs/`.
- Multi-seed (at least 3) before any claim, mean ± std, per-path numbers next to the averages.
