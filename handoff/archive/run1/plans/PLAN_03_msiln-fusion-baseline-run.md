# Plan 03 — Train FusionTransformer baseline on `msiln_site1_b1`

## Hypothesis

The existing FusionTransformer (M1 raw WiFi + M4 world-frame IMU +
decomposed/query readout — the IPIN-tuned config in
`configs/stage_c/fusion.yaml`) should land at **4 – 8 m test MAE** on
Microsoft ILN site1/B1, materially beating the WiFi-kNN floor
(**9.5 m test / 17.7 m val** per RESULT_02). The bar:

- **Bullish case** — test MAE ≤ 4.5 m. We are on the goal trajectory
  (≤ 3 m needs another ~30 % improvement; achievable with temporal K,
  Optuna re-tune, encoder swap). PLAN_04 = polish.
- **Reasonable case** — test MAE 4.5 – 8 m. We beat WiFi-kNN by
  ≥ 1.5 m (criterion (b) satisfied) but miss criterion (a) (≤ 3 m).
  PLAN_04 = encoder swap to per-AP set-transformer
  ([arXiv:2506.00656](https://arxiv.org/abs/2506.00656)) — known
  bottleneck per autopsy.
- **Bear case** — test MAE > 8 m. Existing architecture doesn't
  extract more than kNN does from this denser BSSID space.
  Scientist must reconsider — encoder swap is critical-path and
  possibly the loss/training recipe too.

Kaggle SOTA on this dataset is 1.3 – 1.6 m
([H2O.ai writeup](https://h2o.ai/blog/2021/what-does-it-take-to-win-a-kaggle-competition-lets-hear-it-from-the-winner-himself/),
[MobiCom 2023 retrospective](https://feng-qian.github.io/paper/localization_competition_mobicom23.pdf)),
so the data physically supports the GOAL. The question this plan
answers is whether **our current architecture** is the bottleneck or
not.

This is a single focused experiment: **train + evaluate + report**.
No code changes to `src/`. No config changes (use IPIN-tuned defaults).

### Pre-decisions (folded in, save engineer one turn)

- Engineer's RESULT_02-Q1 (FusionTransformer first vs encoder swap
  first): **scientist confirms FusionTransformer first.** Cleanest
  apples-to-apples; the encoder swap becomes data-driven in PLAN_04.
- Engineer's RESULT_02-Q2 (split_from convention): **defer.** Inline
  list in YAML is ugly but works. Cosmetic only; revisit after we have
  a publishable result.

## Steps

1. **Smoke gate (phase 1 + 2).** Run `scripts/_smoke_fusion.py
   --dataset msiln_site1_b1 --phase 1` (shape / NaN) then `--phase 2`
   (overfit a 16-sample batch to confirm capacity on this data).
   - **Acceptance:** phase 1 exits 0 with no NaN/Inf reports; phase 2
     drives the training loss on the held-out 16-sample batch below
     1.0 m MAE within 50 steps (matches the pattern that worked on
     IPIN). If phase 2 plateaus above 5 m on a 16-sample batch, that
     is an architectural mismatch — STOP and report.

2. **Full training run.** Train the FusionTransformer on
   `msiln_site1_b1` using `configs/stage_c/fusion.yaml` UNCHANGED.
   Override only the dataset name. 90 epochs, AdamW + OneCycleLR +
   Huber (δ=0.5) + early stopping (patience=40), batch_size=128,
   `modality_dropout=0.4`, `instant_dropout=0.45`,
   `readout=query`, `n_instants=8`, `instant_stride=9` — the
   IPIN-tuned defaults. Use `run_in_background=true` with
   `print(..., flush=True)` for log observability.
   - **Acceptance:** training completes (or early-stops); best-val
     checkpoint saved under `runs/fusion_msiln_b1_<ts>/best.pt`;
     per-epoch curves logged; total wall-clock ≤ 60 min (if it blows
     past 90 min, stop and reduce batch_size or sequence length).

3. **Evaluation at best-val checkpoint.** Compute on **both val and
   test**:
   - per-sample MAE (mean, median, RMSE).
   - per-path distribution (median, p25, p75, p90, max).
   - per-waypoint MAE (Kaggle-leaderboard convention — already wired
     via `_msiln_per_path_stats.py` from iter_02; re-use it).
   - subset-eval table from `FusionTrainer.evaluate_all_subsets`:
     `all / only:wifi / only:imu / drop:wifi / drop:imu`.
   - **Acceptance:** all numbers present in RESULT_03; explicit
     `delta_vs_wifi_knn = wifi_knn_mae - fusion_mae` column for val
     and test.

4. **Inference latency probe (real-time check, GOAL criterion e).**
   Time the forward pass on one sample and a batch of 32 on the
   project GPU (Quadro P4000, sm_61, 8 GB). Median over 100 trials
   after a 10-trial warmup.
   - **Acceptance:** per-sample median latency reported in ms;
     GOAL passes if < 100 ms/sample. If > 100 ms, note batch
     latency too (may still be deployable with micro-batching).

5. **Per-trajectory error for all 5 test paths.** Test split has only
   5 paths (Dec-05+06). For each: produce a 2-panel plot
   (predicted vs GT (x,y) trajectory + per-sample error trace over
   time), saved under `runs/fusion_msiln_b1_<ts>/test_paths/`.
   Compute three trajectory-level metrics per path:
   - `mae`: per-sample MAE on the path.
   - `final_drift`: ‖pred(t_end) − GT(t_end)‖.
   - `smoothness_ratio`: `mean‖pred(t+1)−pred(t)‖ / mean‖GT(t+1)−GT(t)‖`
     — sanity for "good path prediction": ≈ 1 means smooth, ≫ 1
     means jittery predictions.
   - **Acceptance:** 5 plot files + a metrics table in RESULT_03;
     `smoothness_ratio` median across 5 paths reported.

6. **PLAN_04 recommendation (engineer's call, scientist may override).**
   Based on test MAE, label one of:
   - `polish`: ≤ 4.5 m → next plan tunes temporal/Optuna/loss
   - `encoder_swap`: 4.5 – 8 m → next plan implements
     per-AP set-transformer encoder
   - `redesign`: > 8 m → scientist must redirect
   - **Acceptance:** one line with the label + 2-sentence justification.

## Sources

- Existing config: [configs/stage_c/fusion.yaml](configs/stage_c/fusion.yaml)
  (M1 raw + M4 world-frame — the audited, IPIN-tuned defaults).
- Existing smoke harness: [scripts/_smoke_fusion.py](scripts/_smoke_fusion.py)
  (5 phases; uses only phases 1 + 2 in this plan).
- Existing trainer: [src/pipeline/training/fusion_trainer.py](src/pipeline/training/fusion_trainer.py)
  with `evaluate_all_subsets` already implemented.
- Kaggle SOTA reference: [H2O.ai writeup](https://h2o.ai/blog/2021/what-does-it-take-to-win-a-kaggle-competition-lets-hear-it-from-the-winner-himself/)
  (1.3 m winner), [MobiCom 2023 retrospective](https://feng-qian.github.io/paper/localization_competition_mobicom23.pdf)
  (1.56 m infra-free / 0.72 m infra-based).
- Per-AP set-transformer candidate (for PLAN_04 if needed):
  [Lazaro et al. 2025, arXiv 2506.00656](https://arxiv.org/abs/2506.00656).

## What to report back

In `handoff/results/RESULT_03_msiln-fusion-baseline-run.md`:

1. Per-step pass/fail with the measured number against each acceptance.
2. **Headline numbers table:**
   | split | metric | mean | median | p25 | p75 | p90 | max | vs WiFi-kNN |
   |---|---|---|---|---|---|---|---|---|
   | val | fusion-per-sample | … | … | … | … | … | … | Δ |
   | val | fusion-per-waypoint | … | … | … | … | … | … | Δ |
   | test | fusion-per-sample | … | … | … | … | … | … | Δ |
   | test | fusion-per-waypoint | … | … | … | … | … | … | Δ |
3. **Subset-eval table:** `all / only:wifi / only:imu / drop:wifi /
   drop:imu` MAE on val + test (lets us see if both modalities are
   carrying weight on this data, or it collapses to WiFi-only like
   IPIN did).
4. Training curves (path to MLflow / TensorBoard run dir).
5. Inference latency (1 sample, batch 32) in ms; PASS/FAIL on the
   100 ms criterion.
6. Per-trajectory metrics table for the 5 test paths + 5 plot paths.
7. PLAN_04 recommendation label (`polish` / `encoder_swap` /
   `redesign`) + 2-sentence justification.
8. **One open question** for scientist beyond the recommendation.

## Reversibility

- All steps are **throwaway probes** with respect to source code:
  no `src/` changes, no config changes, no model file changes.
- New artefacts:
  - `runs/fusion_msiln_b1_<ts>/` — training output + plots, **gitignored**.
  - `runs/overnight/iter_03/` — logs, **gitignored**.
- The only `git add` for this iteration is RESULT_03.md itself.
- If training is unstable, revert is `rm -rf runs/fusion_msiln_b1_<ts>`;
  the previous IPIN runs are untouched.

**Demand #3:** unchanged — no vendored baseline is touched.

**Compute budget:** training ≤ 60 min. Total iteration ≤ 80 min.
If training overruns 90 min, reduce `temporal.n_instants` to 4 (still
above the no-temporal floor) and document. Do NOT skip evaluation
steps; they are the deliverable.
