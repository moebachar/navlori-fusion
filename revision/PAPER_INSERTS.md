

<!-- M1_M2_TABLE_START -->

## M1 + M2 — Time-encoding ablation × seed variance

All cells report `mean ± std` over the seeds listed in the n column. Seeds={42, 7, 123}. Headline = test MAE in metres.

### Webots

| Time encoding | n | val MAE (m) | test MAE (m) |
|---|---:|---:|---:|
| Learned continuous (ours) | 3 | 0.412 ± 0.054 | 0.448 ± 0.044 |
| No time encoding | 3 | 0.560 ± 0.020 | 0.585 ± 0.024 |
| Binned (log-quantized) | 3 | 0.436 ± 0.038 | 0.465 ± 0.033 |
| Positional index (rank) | 3 | 0.523 ± 0.017 | 0.553 ± 0.043 |

### MSILN site1/B1

| Time encoding | n | val MAE (m) | test MAE (m) |
|---|---:|---:|---:|
| Learned continuous (ours) | 3 | 15.324 ± 0.250 | 11.527 ± 3.154 |
| No time encoding | 3 | 15.368 ± 0.709 | 13.115 ± 2.846 |
| Binned (log-quantized) | 3 | 16.724 ± 1.733 | 10.179 ± 0.754 |
| Positional index (rank) | 3 | 16.130 ± 1.004 | 12.490 ± 1.544 |

### Take-aways (auto)

- **Webots / No time encoding**: test MAE 0.585 m (Δ vs learned-continuous: +0.138 m, +30.8 %).
- **Webots / Binned (log-quantized)**: test MAE 0.465 m (Δ vs learned-continuous: +0.017 m, +3.8 %).
- **Webots / Positional index (rank)**: test MAE 0.553 m (Δ vs learned-continuous: +0.105 m, +23.6 %).

- **MSILN site1/B1 / No time encoding**: test MAE 13.115 m (Δ vs learned-continuous: +1.588 m, +13.8 %).
- **MSILN site1/B1 / Binned (log-quantized)**: test MAE 10.179 m (Δ vs learned-continuous: -1.349 m, -11.7 %).
- **MSILN site1/B1 / Positional index (rank)**: test MAE 12.490 m (Δ vs learned-continuous: +0.963 m, +8.4 %).

<!-- M1_M2_TABLE_END -->

<!-- D3_PERIOD_TABLE_START -->

## D3 — Period-range sensitivity

Single-seed (42) sweep of the continuous-time encoding's period range. Default is `(0.05, 120) s`; deltas are vs the M1+M2 baseline at the same seed.

### Webots

| Period range | val MAE (m) | test MAE (m) | Δ vs default |
|---|---:|---:|---:|
| **default** (0.05, 120) s | — | **0.444** | — |
| narrow (0.5, 10) s | 0.391 | 0.451 | +0.006 m |
| wide (0.01, 600) s | 0.380 | 0.448 | +0.003 m |
| shifted (0.1, 30) s | 0.404 | 0.418 | -0.026 m |

### MSILN site1/B1

| Period range | val MAE (m) | test MAE (m) | Δ vs default |
|---|---:|---:|---:|
| **default** (0.05, 120) s | — | **13.600** | — |
| narrow (0.5, 10) s | 15.509 | 17.216 | +3.616 m |
| wide (0.01, 600) s | 15.282 | 10.334 | -3.266 m |
| shifted (0.1, 30) s | 16.571 | 11.397 | -2.203 m |

### Take-aways (auto)

- **Webots**: best = `shifted` (0.418 m), worst = `narrow` (0.451 m), spread = 0.033 m.
- **MSILN site1/B1**: best = `wide` (10.334 m), worst = `narrow` (17.216 m), spread = 6.883 m.

<!-- D3_PERIOD_TABLE_END -->

<!-- D2_IMU_TABLE_START -->

## D2 — Larger IMU backbone (reviewer Moderate concern)

Reviewer asked whether a modestly larger IMU backbone (still ≪ 4.6 M params) would change the fusion conclusions. We bump IMUCNN channels from `(32, 64, 128)` (≈0.05 M params) to `(64, 128, 256)` (≈0.16 M params, 3.3 × larger). Same K=4 / 40 epochs / MBL=false / seed 42 as the M1+M2 baseline.

| Dataset | Variant | IMU params | val MAE (m) | test MAE (m) | Δ vs base test |
|---|---|---:|---:|---:|---:|
| Webots | baseline IMUCNN (32, 64, 128) | ~50 k | 0.409 | **0.444** | — |
| Webots | larger IMUCNN (64, 128, 256) | 159,104 | 0.395 | 0.488 | +0.043 m |
| MSILN site1/B1 | baseline IMUCNN (32, 64, 128) | ~50 k | 15.437 | **13.600** | — |
| MSILN site1/B1 | larger IMUCNN (64, 128, 256) | 158,336 | 15.403 | 12.976 | -0.624 m |

### Take-aways (auto)

- **Webots**: bigger IMU test 0.488 m vs baseline 0.444 m (Δ = +0.043 m, +9.7 %).
- **MSILN site1/B1**: bigger IMU test 12.976 m vs baseline 13.600 m (Δ = -0.624 m, -4.6 %).

Conclusion: the larger IMU backbone changes test MAE by at most 0.624 m — within seed-level noise on both datasets. The reviewer's hypothesis (IMU encoder size caps the fusion ceiling) is not supported: bumping IMU capacity by 3.3 × yields no significant improvement, suggesting the bottleneck is elsewhere (WiFi-encoder cross-session transfer for MSILN; fusion already saturates Webots).

<!-- D2_IMU_TABLE_END -->
