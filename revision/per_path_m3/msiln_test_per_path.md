# MSILN site1/B1 — Per-path test MAE

Checkpoint: `runs/main_table/msiln_site1_b1/transformer` (arch=transformer, K=4)
Modalities: wifi, imu
Total test samples: 2767

| path_id | n | fraction | MAE (m) | RMSE (m) |
|--------:|--:|---------:|--------:|---------:|
| 128 | 367 | 0.133 | 16.918 | 24.245 |
| 129 | 360 | 0.130 | 14.718 | 23.793 |
| 130 | 786 | 0.284 | 10.411 | 13.624 |
| 131 | 659 | 0.238 | 9.919 | 12.574 |
| 132 | 595 | 0.215 | 6.595 | 11.565 |

| **macro avg** | — | — | **11.712** | — |
| sample-weighted (aggregate sanity) | 2767 | 1.000 | 10.897 | — |

Dominant path: **130** (28.4% of test samples, n=786).  Macro MAE = 11.712 m vs sample-weighted 10.897 m.