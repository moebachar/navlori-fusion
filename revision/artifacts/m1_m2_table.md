# M1 + M2 results: time-encoding ablation x seed variance

Each cell is `val MAE (m) | test MAE (m)` reported as `mean +/- std`
over the seeds listed in the next column. Cells with `n_seeds < 3`
are still pending (the batch is sequential).

## Webots

| Time encoding | n seeds | val MAE (m) | test MAE (m) |
|---|---:|---:|---:|
| Learned continuous (ours) | 3 | 0.412 +/- 0.054 | 0.448 +/- 0.044 |
| No time encoding | 3 | 0.560 +/- 0.020 | 0.585 +/- 0.024 |
| Binned (log-quantized) | 3 | 0.436 +/- 0.038 | 0.465 +/- 0.033 |
| Positional index (rank) | 3 | 0.523 +/- 0.017 | 0.553 +/- 0.043 |

## MSILN site1/B1

| Time encoding | n seeds | val MAE (m) | test MAE (m) |
|---|---:|---:|---:|
| Learned continuous (ours) | 3 | 15.324 +/- 0.250 | 11.527 +/- 3.154 |
| No time encoding | 3 | 15.368 +/- 0.709 | 13.115 +/- 2.846 |
| Binned (log-quantized) | 3 | 16.724 +/- 1.733 | 10.179 +/- 0.754 |
| Positional index (rank) | 3 | 16.130 +/- 1.004 | 12.490 +/- 1.544 |
