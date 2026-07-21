# MSILN site1/B1 - test-time K sweep

Checkpoint: `runs/main_table/msiln_site1_b1/transformer` (trained at K=4)
Dataset: `msiln_site1_b1`  -  arch: `transformer`

| K | test MAE (m) | n samples |
|---|--------------|-----------|
| 1 | 11.506 | 2767 |
| 2 | 11.096 | 2767 |
| 4 | 10.897 | 2767 |
| 8 | 11.761 | 2767 |

Headline: MSILN test MAE: K=1 11.51 | K=2 11.10 | K=4 10.90 | K=8 11.76 (trained at K=4).
