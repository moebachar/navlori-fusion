# Plan 04 — WiFi encoder capacity probe (embed_dim 128 → 256, conditional 512)

## Hypothesis

RESULT_03 settled the diagnostic question: **fusion ≈ WiFi-only on
both splits** (Δ ≤ 0.35 m). The fusion mechanism is fine. The
**WiFi encoder is the bottleneck.** Two distinct flavours of "encoder
bottleneck" exist:

1. **Capacity-bound.** `Anchor2Vec` was sized for IPIN (~125 BSSIDs)
   and embeds them into 128 dims. On msiln_site1_b1 the vocabulary
   jumps to **1419 BSSIDs** (RESULT_02). Doubling `embed_dim` is a
   one-line YAML override; if this materially improves MAE, the
   bottleneck is just capacity and we can stay with the existing
   architecture for the polish loop.

2. **Structurally bound.** Even at higher dim, `Anchor2Vec`'s soft
   k-means anchors + `-100`-fill input may saturate. In that case
   the architectural fix is a per-AP / per-BSSID set-transformer
   ([Lazaro et al. 2025, arXiv:2506.00656](https://arxiv.org/abs/2506.00656)),
   which is what PLAN_05 will implement.

**This plan is the gating probe between those two paths.** Cost:
~30 min of training, zero source code changes (config-only override).
Reward: definitive evidence that determines whether PLAN_05 needs to
ship the full architectural rebuild or just an Optuna re-tune.

### Engineer Q1/Q2 from RESULT_03 (acknowledged, deferred)

- **Q1** (per-waypoint MAE 18.6 vs per-sample 9.0 on test — 9.6 m gap):
  noted. Hypothesis (smoothing-at-turns) is plausible but does NOT
  affect the encoder-vs-not decision. **Deferred** to PLAN_06+ where
  we either (a) reweight loss toward waypoint timestamps or (b) just
  report both metrics with the caveat.
- **Q2** (temporal smoothness regularizer to attack smoothness_ratio
  = 12.9): noted. This addresses criterion (d) directly but requires a
  loss-function code change. **Deferred** to PLAN_06+ — a smoothness
  term applied to a still-weak encoder is putting lipstick on noisy
  predictions. Fix the WiFi encoder first; predictions should naturally
  become smoother when they're locked to better location signal.

## Steps

1. **embed_dim = 256 training run.** Override `model.embed_dim=256`
   on the existing `configs/stage_c/fusion.yaml`. All other knobs
   identical to PLAN_03 (90 epochs, batch_size=128, modality_dropout=
   0.4, n_instants=8, readout=query, IPIN-tuned defaults). Use
   `--dataset msiln_site1_b1`. Background-run with
   `print(..., flush=True)`. **If OOM**: drop batch_size to 64 and
   note the cap. Do not change anything else.
   - **Acceptance:**
     - Training completes ≤ 60 min wall-clock.
     - Per-sample val MAE, test MAE, per-path distribution reported
       same as PLAN_03.
     - Subset eval table (only:wifi / only:imu / wifi+imu) on val + test.
     - **Gate label:**
       - `PASS` if val MAE ≤ **14.0 m** AND test MAE ≤ **7.5 m**
         (= ≥ 1.5 m improvement on both vs PLAN_03's 15.70 / 8.99).
       - `MARGINAL` if either side improves by 0.5 – 1.5 m.
       - `NO-PASS` if neither improves by ≥ 0.5 m.

2. **Conditional follow-up (one of three branches based on step 1).**

   **Branch A — step 1 PASS:**
   Push to `embed_dim = 512` for one more 90-epoch run. If OOM,
   batch_size → 32 (note it). Expectation: returns more or saturates.
   - **Acceptance:** val/test MAE reported; compared to step 1 to
     identify the knee in the capacity curve.

   **Branch B — step 1 MARGINAL:**
   One short 30-epoch probe: `embed_dim=256` + `wifi_pca=512` (the
   PCA dim before Anchor2Vec). Tests whether the PCA bottleneck is
   eating signal that a fatter embed_dim could otherwise capture.
   - **Acceptance:** val MAE reported. If it moves ≥ 1 m, the
     PCA dim is part of the issue.

   **Branch C — step 1 NO-PASS:**
   One short 30-epoch probe: `embed_dim=256` + **WiFi-only training**
   (`modality_dropout=1.0` effectively — set `--modalities wifi`
   if supported, else mask the IMU encoder). Confirms the encoder is
   structurally saturated, not just under-trained on the joint loss.
   - **Acceptance:** WiFi-only val MAE reported; if it sits within
     0.3 m of step 1's `only:wifi` subset number, the encoder is
     structurally bound. This is the decisive evidence for PLAN_05.

3. **PLAN_05 recommendation.** One of:
   - `polish_optuna` (after PASS / Branch A diminishing return): next
     iteration runs Optuna over depth/heads/lr/temporal with the
     bigger embed_dim; encoder swap deferred.
   - `swap_with_dim_lifted` (after MARGINAL / Branch B): next
     iteration builds the per-AP set-transformer AND keeps embed_dim
     ≥ 256.
   - `swap_committed` (after NO-PASS / Branch C): next iteration
     builds the per-AP set-transformer at the standard 128 dim
     (capacity confirmed not the issue).
   - **Acceptance:** one-line label + 3-sentence justification quoting
     the measured numbers.

## Sources

- RESULT_03 (this run's parent): `handoff/results/RESULT_03_msiln-fusion-baseline-run.md` — diagnostic that fixed encoder as bottleneck.
- Engineer Q3 from RESULT_03 (capacity probe suggestion).
- Existing config: [configs/stage_c/fusion.yaml](configs/stage_c/fusion.yaml) (the only thing being overridden).
- Per-AP set-transformer reference (for PLAN_05 if we go there):
  [Lazaro et al. 2025, arXiv:2506.00656](https://arxiv.org/abs/2506.00656).
- Locaris (open WiFi-only SOTA, our criterion-b reference):
  [arXiv:2510.11926](https://arxiv.org/abs/2510.11926) · code
  https://sachini.github.io/niloc.

## What to report back

In `handoff/results/RESULT_04_wifi-encoder-capacity-probe.md`:

1. Per-step pass/fail (gate label from step 1 explicit).
2. **Three-config comparison table** (after the conditional branch runs):

   | run | embed_dim | wifi_pca | wifi-only? | val MAE | test MAE | only:wifi val | wall (min) |
   |---|---|---|---|---|---|---|---|
   | PLAN_03 baseline | 128 | 128 | no | 15.70 | 8.99 | 15.66 | 18.4 |
   | step 1           | 256 | 128 | no | …     | …    | …     | …    |
   | step 2 (branch)  | … | … | … | … | … | … | … |

3. Per-path distribution for step 1 (median, p25, p75, p90, max) on
   val + test.
4. Subset eval (only:wifi / only:imu / wifi+imu) on val + test for
   step 1; same for step 2 if a branch fired.
5. Smoothness ratio for step 1's test predictions (sanity — does
   higher capacity also smooth trajectories?).
6. PLAN_05 recommendation label + 3-sentence justification.
7. **One open question** for scientist beyond the recommendation.

## Reversibility

- **Pure throwaway probe.** No `src/` changes, no committed config
  changes — all overrides are CLI flags or temporary YAML files
  under `runs/`.
- Run artefacts:
  `runs/fusion_msiln_b1_emb256_<ts>/` (step 1),
  `runs/fusion_msiln_b1_emb512_<ts>/` (branch A),
  `runs/fusion_msiln_b1_pca512_<ts>/` (branch B),
  `runs/fusion_msiln_b1_wifionly_<ts>/` (branch C) — all gitignored.
- Only `RESULT_04.md` gets `git add`'d.
- If OOM forces a `batch_size` change, document it in the table — do
  NOT commit the change to `configs/stage_c/fusion.yaml`.

**Demand #3 untouched** (no vendored code involved here).

**Compute budget:** step 1 ~30 min + one conditional branch
~15–30 min = **iteration ≤ 70 min total**. If the conditional branch
threatens to overrun, ship without it and recommend the branch run
in PLAN_05 instead.
