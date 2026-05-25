# Overnight Run — Coordination State

Started: 2026-05-24 23:49 local
Stop at: 2026-05-25 10:00 local  (OR sooner if `GOAL_REACHED: true` below)
Branch: `overnight-autonomous-2026-05-24`
Push policy: **commit locally each iteration; NO push. User pushes manually on wake.**

## Status

- `CURRENT_ITERATION:` 6
- `LAST_PLAN:` PLAN_06_wifi-set-encoder-sparse-observed.md
- `LAST_RESULT:` RESULT_06_wifi-set-encoder-sparse-observed.md
- `GOAL_REACHED:` false
- `STOP_REASON:` **FINAL STOP at 2026-05-25 09:57 local** (3 min before the 10:00 budget gate). Reached end of allotted time with 6 iterations committed (PLAN_07 not written by scientist before stop). Engineer resumed at 08:36 after laptop sleep cycle and executed PLAN_06 cleanly: sparse-observed encoder rewrite landed (commit `96567dc`), peak GPU 435 MB at B=128 (vs iter_05 dense OOM), bar=NO-PASS strict but smoothness collapsed 12.9→3.4 (4× tighter trajectories — goal criterion (d) win). Set-transformer is 15× slower per fwd-bwd than Anchor2Vec → forced K=8→1 to fit budget; PLAN_07 should re-run K=8 with SDPA/FlashAttention or lighter encoder. **6 commits on branch `overnight-autonomous-2026-05-24`; user pushes manually on wake.**

## Goal

**Publish a conference paper showing our WiFi+IMU fusion (FusionTransformer) beats
open-source SOTA single-modality baselines on a public benchmark where 1–3 m
positioning is physically reachable, with both per-sample and per-trajectory metrics
suitable for real-time use.**

**Target venue:** PerCom 2026 (submission deadline ~11 Sept 2026) as primary;
IEEE Sensors Journal / MDPI Sensors as rolling-deadline fallback; IPIN 2027 follow-up.

### Acceptance criteria (all must hold, on a public WiFi+IMU benchmark)

(a) **Per-sample mean Euclidean 2D position error (MAE) ≤ 3.0 m** on the
    test/val split.
(b) **Our fusion beats the best open-source single-modality baseline by ≥ 1.5 m**
    on the **same data, same metric, same protocol**, using **unmodified
    baseline code** (Demand #3). Candidate baselines: Locaris (arXiv 2510.11926,
    code at sachini.github.io/niloc) for WiFi; RoNIN ResNet1D for IMU; optionally
    Fusion-DHL (Sachini, ICRA 2021) as a published WiFi+IMU+floorplan reference.
(c) **Per-path MAE distribution reported** (median, p25, p75, p90, max), not
    only the aggregate mean. (Probe 6 of the autopsy showed 2.3× per-path
    variance — never report a single mean again.)
(d) **Per-trajectory ATE** (Absolute Trajectory Error) reported alongside MAE
    for the top 5 longest test paths — addresses the user's "good path prediction
    in real time" requirement.
(e) **Inference latency < 100 ms per sample** on the project GPU (Quadro P4000,
    8 GB) — real-time capable.

### Strategic context (this is a course-shift)

The prior strategy iterated fixes on the IPIN 2024 floor −2 benchmark. The
autopsy (Probe 2.1) measured a **~4 m centroid floor** on IPIN val under
WiFi carry-forward; even an oracle WiFi-only model cannot beat that, and
the brief inflates this to a 6–7 m ceiling once IMU drift between scans is
included. **IPIN cannot deliver a publishable 1–3 m result by construction.**

Two parallel directions emerge from the literature scan:
1. **Switch the primary benchmark to Microsoft Indoor Location & Navigation
   (Kaggle ILN 2.0, 2021)** — denser WiFi, multi-day cross-session splits,
   multi-floor, public SOTA leaderboard at **1.3–1.6 m**
   ([H2O.ai writeup](https://h2o.ai/blog/2021/what-does-it-take-to-win-a-kaggle-competition-lets-hear-it-from-the-winner-himself/),
   [MobiCom 2023 retrospective](https://feng-qian.github.io/paper/localization_competition_mobicom23.pdf)).
2. **Replace `Anchor2Vec` with a per-AP/BSSID set-transformer encoder
   pretrained with contrastive SSL** (AP-dropout + RSSI jitter augmentations).
   This directly attacks the known cross-session WiFi drift, the bottleneck
   the autopsy identified for the existing IPIN runs
   ([Lazaro et al. 2025, arXiv 2506.00656](https://arxiv.org/abs/2506.00656);
   [SelfLoc, MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/13/2675)).

PLAN_01 starts with the dataset switch as a feasibility probe — cheap, no
existing code touched, and a single GO/NO-GO decision feeds PLAN_02. IPIN
remains as a secondary benchmark for ablation/transfer claims.

## Iteration log

| # | plan file | result file | engineer commit | scientist note |
|---|---|---|---|---|
| 1 | PLAN_01_msiln-feasibility-probe.md | RESULT_01_msiln-feasibility-probe.md | 301c80e iter 01: msiln-feasibility-probe (GO — site1/B1 recommended) | feasibility probe for Microsoft ILN 2.0 dataset switch |
| 2 | PLAN_02_msiln-convert-and-baselines.md | RESULT_02_msiln-convert-and-baselines.md | 3cb454b iter 02: msiln-convert-and-baselines (wifi-kNN floor 17.7m val / 9.5m test; fusion target <=6m test) | converter + cross-session split + trivial baselines on site1/B1 (no training yet) |
| 3 | PLAN_03_msiln-fusion-baseline-run.md | RESULT_03_msiln-fusion-baseline-run.md | bae6e06 iter 03: msiln-fusion-baseline-run (val 15.7m / test 8.99m; wifi-only ~ full-fusion -> encoder is bottleneck; PLAN_04 = encoder_swap) | first FusionTransformer training on msiln_b1; smoke + 90-epoch + eval + per-traj + latency |
| 4 | PLAN_04_wifi-encoder-capacity-probe.md | RESULT_04_wifi-encoder-capacity-probe.md | ffd5253 iter 04: wifi-encoder-capacity-probe (NO-PASS; structurally bound; PLAN_05 = swap_committed) | embed_dim 128->256 probe (1419 BSSIDs vs IPIN's ~125); gates whether PLAN_05 = polish or full per-AP set-transformer rebuild |
| 5 | PLAN_05_wifi-set-transformer-encoder.md | (blocked — no formal RESULT_05) | (none — engineer offline) | encoder built but training OOM'd at all 3 batch sizes (dense-masked 1419 tokens × O(N²) attn). See handoff/SCIENTIST_NOTE_iter05.md. Superseded by PLAN_06. |
| 6 | PLAN_06_wifi-set-encoder-sparse-observed.md | RESULT_06_wifi-set-encoder-sparse-observed.md | 96567dc iter 06: wifi-set-encoder-sparse-observed (NO-PASS strict; smoothness 12.9->3.4; encoder 15x slower so K=8 forced down to K=1; PLAN_07 = redesign_or_pivot) | rewrite WiFiSetTransformer.forward() to use sparse-observed (~127 tokens, not 1419); fits 8 GB at bs=128; same bar rubric as PLAN_05 |

(Both sides update this table — append a row when you finish your half.)
