# Night 2 Morning Report

Generated 2026-06-26T02:05:02.

## TL;DR

**Baseline (M1+M2 paper config): test 11.53 ± 3.15 m.**
Yesterday's two winners: idea1 (test 9.93 ± 0.25, 4 seeds), lead3 JEPA (test 9.11, 1 seed).

**Best Phase C' lead: mamba_imu_place_conditioned_encoder (test 10.14 m, family=alt-backbones).**

Phase B' top 3:
- mamba_imu_place_conditioned_encoder [alt-backbones]: test 9.127 m
- pairwise_rssi_difference_invariant [physics-aware]: test 10.250 m
- bssid_top_k_set_fingerprint_no_magnitudes [observation-engineering]: test 10.293 m

## Phase B' — full screen (12 epochs default unless noted)

Ranked by test MAE.

| Rank | Lead | Family | val MAE (m) | test MAE (m) | Δ vs base | min |
|---:|---|---|---:|---:|---:|---:|
| 1 | mamba_imu_place_conditioned_encoder | alt-backbones | 15.255 | **9.127** | -2.40 m | 28.1 |
| 2 | pairwise_rssi_difference_invariant | physics-aware | 15.690 | **10.250** | -1.28 m | 4.3 |
| 3 | bssid_top_k_set_fingerprint_no_magnitudes | observation-engineering | 18.340 | **10.293** | -1.23 m | 1.1 |
| 4 | hierarchical_zone_then_residual_head | task-reformulation | 21.026 | **12.609** | +1.08 m | 1.6 |
| 5 | path_id_and_ap_visibility_classification_auxiliaries | multi-task-aux | 17.231 | **13.335** | +1.81 m | 19.9 |
| 6 | differentiable_knn_over_learned_rssi_metric | retrieval-memory | 18.537 | **14.362** | +2.84 m | 1.4 |
| 7 | whole_path_bidirectional_transformer | sequence-level | 17.125 | **16.925** | +5.40 m | 2.7 |
| 8 | session_maml_with_first_path_support | paradigm-shift | 73.342 | **38.446** | +26.92 m | 0.7 |
| 9 | mixture_density_network_with_mc_dropout | generative-models | 65.416 | **49.822** | +38.30 m | 1.4 |
| 10 | per_path_trajectory_cost_minimization | classical-non-deep | 120.430 | **155.270** | +143.74 m | 1.9 |

## Phase C' — top leads at full 40 epochs

| Lead | Phase B' test | Phase C' test | Δ vs base | min |
|---|---:|---:|---:|---:|
| mamba_imu_place_conditioned_encoder | 9.127 | 10.144 | -1.38 m | 72.8 |
| bssid_top_k_set_fingerprint_no_magnitudes | 10.293 | 10.350 | -1.18 m | 1.4 |

## Family landscape — what worked, what didn't

- **alt-backbones**: best = `mamba_imu_place_conditioned_encoder` test 9.127 m (-2.40 vs base, -0.80 vs idea1).
- **physics-aware**: best = `pairwise_rssi_difference_invariant` test 10.250 m (-1.28 vs base, 0.32 vs idea1).
- **observation-engineering**: best = `bssid_top_k_set_fingerprint_no_magnitudes` test 10.293 m (-1.23 vs base, 0.36 vs idea1).
- **task-reformulation**: best = `hierarchical_zone_then_residual_head` test 12.609 m (+1.08 vs base, +2.68 vs idea1).
- **multi-task-aux**: best = `path_id_and_ap_visibility_classification_auxiliaries` test 13.335 m (+1.81 vs base, +3.41 vs idea1).
- **retrieval-memory**: best = `differentiable_knn_over_learned_rssi_metric` test 14.362 m (+2.84 vs base, +4.43 vs idea1).
- **sequence-level**: best = `whole_path_bidirectional_transformer` test 16.925 m (+5.40 vs base, +7.00 vs idea1).
- **paradigm-shift**: best = `session_maml_with_first_path_support` test 38.446 m (+26.92 vs base, +28.52 vs idea1).
- **generative-models**: best = `mixture_density_network_with_mc_dropout` test 49.822 m (+38.30 vs base, +39.89 vs idea1).
- **classical-non-deep**: best = `per_path_trajectory_cost_minimization` test 155.270 m (+143.74 vs base, +145.34 vs idea1).

## Artifacts
- `experiments/state/night2_leads.json` — input list
- `experiments/state/phase_b2_results.json` — Phase B' raw
- `experiments/state/phase_c2_results.json` — Phase C' raw
- `experiments/leads_night2/` — per-lead runners
- `experiments/logs/phase_b2_*.log` / `phase_c2_*.log`
