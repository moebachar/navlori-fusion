# SOTA baselines + run-2 main results

> Run-2 closed 2026-05-26 (`GOAL_REACHED: true with documented
> limitations`). Source of truth for the paper-facing numbers is
> [`handoff/SUMMARY.md`](../handoff/SUMMARY.md). This doc mirrors
> the headline table for reproducibility + cross-references the
> consolidated APIs that produce each cell.

## Paper-facing main results (6 rows × 9 columns)

Exclusions per
[`handoff/SCIENTIST_NOTE_notebook-exclusions.md`](../handoff/SCIENTIST_NOTE_notebook-exclusions.md):
- **IPIN 2024 floor 0** dropped from paper-facing rows (RESULT_22 β5).
- **MoTTransformer** dropped from paper-facing columns (RESULT_21 γ5).

Both remain in the repo for reproducibility (see
`src/pipeline/data/ipin2024.py`, `src/pipeline/fusion/mot_transformer.py`,
`runs/overnight/run2_iter_{21,22}/`).

Render the live table via:

```python
from src.pipeline.evaluation import MainResultsTable
print(MainResultsTable.from_archive().to_markdown())
```

| dataset            | modalities         | wlan_localization | RoNIN ResNet1D  | TartanVO     | Anchor2Vec | DPVOMotion  | IMUCNN      | incumbent       | cnn1d (winner)        | lstm_attn          |
|--------------------|--------------------|-------------------|-----------------|--------------|------------|-------------|-------------|-----------------|-----------------------|--------------------|
| Webots sim         | WiFi+IMU+Cam+Odom  | n/a               | n/a             | n/a          | n/a        | n/a         | n/a         | 0.394 v/0.417 t | **0.282 v / 0.339 t** | 0.301 v / 0.340 t  |
| IMUWiFine fl.4 (1) | WiFi+IMU           | 4.17 v / 8.50 t   | 26.84 v / n.a.  | n/a          | n/a        | n/a         | n/a         | n/a             | 1.40 v / 7.09 t       | **1.26 v** / 7.20 t|
| MSILN site1/B1     | WiFi+IMU x-session | 21.26 v / 28.31 t | n/a             | n/a          | n/a        | n/a         | n/a         | 16.60 v / 14.02 t | (PLAN_15 incumbent)  | n/a                |
| RoNIN canonical (2)| IMU only           | n/a               | **5.140 t**     | n/a          | n/a        | n/a         | 9.961 t     | n/a             | 7.59 t (Umey 5.95)    | 7.50 t (Umey 6.12) |
| TartanAir hosp.    | Camera only        | n/a               | n/a             | **0.012 t-20%** | n/a     | 0.293 t-20% | n/a         | n/a             | n/a (3)               | n/a (3)            |
| UJI IndoorLoc      | WiFi only val      | 15.17 v           | n/a             | n/a          | **8.69 v** | n/a         | n/a         | n/a             | 8.72 v                | **8.43 v**         |

Notes:
1. IMUWiFine test split lacks IMU per dataset design (RESULT_20 audit) →
   fusion test = WiFi-only inference floor.
2. RoNIN raw / Umeyama-aligned ATE; reuses RESULT_07 pretrained ResNet1D
   (paper-exact 5.140). CNN1D's **Umeyama gap +15.7 % clears the 20 %
   audit gate** (RESULT_23).
3. Camera external-SOTA validation queued as Phase C extension
   (paper-soft per RESULT_08); fusion test n/a — image-only sequence,
   no co-recording multi-mod data.

## Reproducing each cell

Each canonical script is built on the consolidated APIs from
PLAN_26-28 (`src.pipeline.baselines`, `src.pipeline.data`,
`src.pipeline.fusion`, `src.pipeline.training`):

| script | reproduces | numbers |
|--------|-----------|---------|
| `scripts/eval_uji.py` | RESULT_01 | wlanloc 15.17 m + Anchor2Vec 8.69 m on UJI val |
| `scripts/eval_ronin_canonical.py` | RESULT_07 | ResNet1D pretrained 5.140 m on canonical unseen |
| `scripts/eval_tartanair_hospital.py` | RESULT_08 | TartanVO 0.012 m + DPVOMotion 0.293 m last-20% |
| `scripts/_eval_wlanloc_imuwifine.py` | RESULT_19 | wlanloc on IMUWiFine fl.4 (val 4.17 / test 8.50) |
| `scripts/_eval_wlanloc_msiln.py` | RESULT_15 | wlanloc on MSILN (val 21.26 / test 28.31) |

The `scripts/_eval_*.py` underscore-prefix variants are
iteration-scoped historical runners. The non-underscore
`scripts/eval_*.py` versions are the consolidated canonical
wrappers PLAN_29 promoted.

## Cross-cutting findings

See [`handoff/SUMMARY.md`](../handoff/SUMMARY.md) §4 (5 paragraphs):

1. **LSTM-attn dead-reckoning regime** confirmed across 3 datasets ×
   4 scenarios (Webots, IMUWiFine, IPIN floor 0).
2. **Smoothness debt is architecture-invariant** — falsified the
   architectural-lever hypothesis (4 archs × 5+ datasets all under
   r=0.20 gate).
3. **RoNIN RTE-to-ATE asymmetry** is the same loss-function signal
   as the smoothness debt — same fix.
4. **Three distinct fusion regimes** emerged: CNN1D cooperative,
   LSTM-attn dead-reckoning, MoTTransformer WiFi-anchored. Same
   encoders + protocol; only the aggregator differs.
5. **Cross-dataset transferability**: Anchor2Vec beats wlanloc on
   UJI + IPIN per-leg. Fusion's value is the 4-modality story on
   Webots, not universal cross-dataset dominance.

## Criterion verdicts (a-e)

| crit. | description                                | status | source       |
|-------|--------------------------------------------|:-------|--------------|
| (a)   | per-leg SOTA ≤ 20 %                        | partial | RESULT_01/07/08 |
| (b)   | 4-mod Webots test ≤ 0.5 m                  | ✓ 32 % margin | RESULT_17    |
| (c)   | MSILN cross-session                        | partial | RESULT_15    |
| (d)   | per-path + smoothness r > 0.20             | smoothness UNMET | run-2 4-arch falsification |
| (e)   | latency < 100 ms / sample                  | ✓✓ 21×-660× under | RESULT_18    |

## Open follow-ups (post-run-2)

Per SUMMARY.md §6:

1. **PLAN_25b**: B-1 auxiliary velocity loss / B-2 EMA token
   smoothing on CNN1D winner — close smoothness debt + RoNIN RTE
   asymmetry in one experiment (~30 min).
2. **MSILN re-run with CNN1D + Anchor2Vec** (~3 h) — may close
   gate (c)-1.
3. **Camera external-SOTA full validation** (~1 day) — paper-soft → clean.
4. **Conformal coverage on CNN1D** (~30 min).
5. **Pre-submission cleanup**: figure regeneration; mechanical.

## Historical context (pre-run-2 framing)

The original SOTA-baselines plan from 2026-05-22 named:
- WiFi baseline = CNNLoc on UJI.
- IMU baseline = RoNIN on the RoNIN unseen-subjects set.

Run-2 retained the structure but updated the WiFi baseline to
`sharan-naribole/wlan_localization` (CNNLoc had no working open
release at the time; RESULT_01 reproduces wlanloc 15.17 m as the
SOTA reference). The IMU side stuck with RoNIN ResNet1D
(RESULT_07 paper-exact reproduction at 5.140 m).
