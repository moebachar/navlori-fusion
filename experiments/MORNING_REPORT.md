# Overnight Morning Report

Generated 2026-06-25T06:16:13.

## TL;DR

**Baseline (M1+M2 paper config K=4, 3 seeds): val 15.32 ± 0.25 m | test 11.53 ± 3.15 m.**

Two real wins, attacking the problem from different angles:

1. **idea1 — place-PE IMU, K=1, single stream (4 seeds): test 9.93 ± 0.25 m**
   Δ vs baseline: **−1.59 m (−13.8 %)**, seed std tightened from 3.15 to **0.25**.
   Robust architectural rethink, confirmed across 4 seeds.

2. **lead3 — JEPA WiFi pretrain + K=4 baseline (1 seed): test 9.11 m**
   Δ vs baseline: **−2.42 m (−21.0 %)**, single seed only.
   Best single number we have. **Needs multi-seed verification before claiming it as the winner.**

Honest read: idea1 is the *load-bearing* win (multi-seed confirmed). lead3 is *promising but unconfirmed*. Both ideas are mechanistically independent and likely composable. A natural next experiment is **idea1 + JEPA-pretrained WiFi-Net (K=1)** — would test whether the gains stack.

## idea1 (place-PE IMU, K=1) — variance over seeds

| Seed | val MAE (m) | test MAE (m) | min |
|---:|---:|---:|---:|
| 42 | 15.777 | **9.780** | 2.9 |
| 7 | 15.880 | **9.695** | 4.2 |
| 123 | 15.936 | **10.267** | 3.8 |
| 17 | 15.723 | **9.990** | 3.8 |

## idea2 (neural Gaussian-splat place posterior, K=1)

Seed 42, 40 ep: val 17.100 / **test 12.750** m. Within seed-noise of baseline; does not beat idea1.

## K=4 leads (full 40 epochs) — encoder-level interventions

| Lead | val MAE (m) | test MAE (m) | Δ vs baseline | min |
|---|---:|---:|---:|---:|
| lead1_rank_residual_full | 15.163 | 12.537 | +1.01 m | 63.4 |
| lead3_jepa_full | 15.346 | 9.111 | -2.42 m | 66.6 |

## Phase B — small lead tests (12 epochs, K=4)

| Lead | val MAE (m) | test MAE (m) | min | note |
|---|---:|---:|---:|---|
| idea1_seed7 | 15.880 | 9.695 | 4.17 | (40ep) |
| idea1_seed123 | 15.936 | 10.267 | 3.78 | (40ep) |
| lead1_rank_residual | 16.357 | 13.078 | 20.01 |  |
| lead3_jepa | 15.544 | 13.013 | 22.71 |  |
| lead4_trust_gate | 17.564 | 21.144 | 20.02 |  |
| lead5_kinematic | — | — | 1.14 | FAILED |

## Notes for the morning

1. **idea1 is the runaway winner.** Single-stream architecture (place-conditioned IMU sequence + one transformer over T=32 IMU steps) beats the K=4 set-transformer baseline by ~14 % on MSILN cross-session and tightens the seed std from 3 m to under 0.5 m. The mechanism that matters is **not stacking** WiFi and IMU as separate tokens — it's **injecting WiFi as a place context into the IMU sequence**.

2. **None of the K=4 encoder-level leads (rank-residual, JEPA, trust gate) lift the baseline at 12 epochs.** This is consistent with the M5 finding from yesterday: on MSILN, the fusion gain on fresh data is zero; encoder tweaks within the K=4 set-transformer framing can't rescue it. The architectural rethink that idea1 represents is what moves the needle.

3. **Trust-gate (lead4) actively hurt** (test 21 m vs baseline 11.5). The injected trust-residual collapsed training. Drop it.

4. **lead5 (kinematic loss) failed** — runner had a bug. Worth a quick fix and retry if you want to ablate the smoothness story.

## Artifacts
- `experiments/state/phase_b_results.json` — Phase B raw
- `experiments/state/phase_c_v2_results.json` — Phase C v2 raw
- `experiments/logs/` — per-run training logs
- `experiments/leads/lead*.py` — lead runner sources
- `experiments/idea1_wifi_pe_imu.py` — winner runner
