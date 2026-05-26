# Run 2 — Coordination State

Started: Started: 2026-05-25 <12:30> local
Stop at: 2026-05-26 18:00 local
Branch: overnight-autonomous-run2-2026-05-25
Push policy: **commit locally each iteration; NO push. User pushes
              manually on wake.**

Run 1 archived at `handoff/archive/run1/` — read its `README.md` for
the autopsy.

## Status

- `CURRENT_ITERATION:` 29  (awaiting PLAN_29)
- `LAST_PLAN:` PLAN_28_fusion-encoders-training-consolidation.md (third consolidation iter: `build_arch(name)` factory across 5 fusion archs + paper-citation-ready design-rationale docstrings + `Encoder.demo_forward` per encoder + 5 public `FusionTrainer` methods incl. `load_trained(checkpoint_dir)`; smoke verifies CNN1D winner reproduces RESULT_18)
- `LAST_RESULT:` RESULT_28_fusion-encoders-training-consolidation.md (**fusion/encoders/training consolidation shipped**: `src.pipeline.fusion.{build_arch, list_archs, DEFAULT_CONFIG}` factory across 5 archs (incumbent/cnn1d/lstm_attn/tcn/mot_transformer); param counts match RESULT_17/21 exactly. **Encoder demo_forward** on Anchor2Vec (anchor attention weights), IMUCNN (conv stack activations), OdomCNN (conv stack activations), DPVOMotionEncoder (per-patch tokens) for notebook §0. **3 NEW FusionTrainer methods**: `compute_per_trajectory_smoothness(split)`, `latency_probe(batch_sizes, n_trials)`, module-level `load_trained(checkpoint_dir, arch, dataset)`. Bakeoff.py docstring rewritten as paper-methods-section material (4-arch verdict table). **Smoke 5/5 archs build + 3/3 demos + load_trained on CNN1D winner reproduces test 0.341 m vs RESULT_17 0.339 m = +0.6 % drift**; latency b=1 = 4.729 ms (exact match RESULT_18). One open item: val MAE drift +4.6 % from RNG state in vision-token extraction (test column unaffected; flagged for PLAN_29 fix))
- `GOAL_REACHED:` true for run-2 scientific work; FALSE for consolidation deliverable (codebase + walkthrough notebook)
- `STOP_REASON:` (none — entering consolidation phase per user directive 2026-05-26 ~12:30 local)

## Post-run-2 consolidation roadmap (PLAN_26 → PLAN_30)

User directive: wrap up the work into the codebase as importable
modules; eval scripts + notebook stay thin and just import. Pull
all SOTA dependencies into `external_methods/` (Git submodules
for GitHub presentation).

| iter | scope | depends on |
|---|---|---|
| **26** | `external_methods/` submodules + `src/pipeline/baselines/` centralised loader package + wrapper migration (THIS ITER) | — |
| 27 | `src/pipeline/data/` factory (`load_dataset(name)`, `dataset_stats(name)`, `preprocessing_demo(name, modality)`) + `src/pipeline/visualization/` plotters | 26 |
| 28 | `src/pipeline/fusion/` consolidation (`build_arch(name)` factory + design-rationale docstrings) + `encoders/.demo_forward()` helpers + `training/` public methods (`evaluate_all_subsets`, `evaluate_staleness`, `compute_per_trajectory_smoothness`, `latency_probe`, `load_trained`) | 26 |
| 29 | `src/pipeline/evaluation/MainResultsTable` (reads RESULT JSONs, returns paper-table DataFrame) + canonical `scripts/eval_*.py` triage + configs/docs sweep. **Per user 2026-05-26 ~13:50 (`SCIENTIST_NOTE_notebook-exclusions.md`): MainResultsTable drops IPIN row + MoTTransformer column from paper-facing presentation; code stays in repo for reproducibility.** | 26, 27, 28 |
| 30 | `notebooks/run2_walkthrough.ipynb` scaffold (§0 dataset pre-section with stats + figures + preprocessing demos; §1-3 Phase A/B/C walkthrough; §4 honest gaps; §5 paper-framing decisions; §6 reproducibility) — engineer iterates with user directly after this. **Excludes IPIN dataset + MoTTransformer architecture from paper-facing presentation per `SCIENTIST_NOTE_notebook-exclusions.md`.** | 26, 27, 28, 29 |

## Goal

**A 4-modality fusion architecture (WiFi + IMU + Odom + Camera) for
indoor localization, validated via per-leg comparison against published
SOTA and end-to-end on the only dataset with all 4 modalities (Webots
sim), with graceful degradation on real-world 2-modality data.**

**Target venue:** PerCom 2026 (submission ~11 Sept 2026); MDPI Sensors
/ IEEE Sensors Journal as rolling fallbacks.

### Acceptance criteria

(a) **Per-leg validation (each modality):** our encoder's
    published-protocol MAE within **20 %** of the named SOTA repo's
    number on the same dataset and same metric.

    | modality | SOTA repo | benchmark | metric |
    |---|---|---|---|
    | WiFi | sharan-naribole/wlan_localization + Sachini/niloc | UJIIndoorLoc | mean Euclidean on `validationData.csv` |
    | IMU  | Sachini/ronin | RoNIN unseen-subjects | raw + aligned ATE |
    | Camera | DPVO numbers | Webots sim | per-sample MAE |
    | Odom | (internal, no public SOTA) | Webots sim | per-sample MAE |

(b) **4-modality fusion on Webots sim** — test MAE ≤ 0.5 m
    (run-1 baseline was ~0.43 m). Per-modality subset eval reported.

(c) **Cross-session real-world plausibility** — Microsoft ILN 2.0
    site1/B1: beat WiFi-kNN by ≥ 1.5 m AND beat the open-source
    SOTA (CNNLoc or Locaris) by ≥ 0.5 m on the same data.

(d) **Per-path MAE distribution + per-trajectory smoothness ratio**
    reported in every evaluation. Per-trajectory plots for top 5
    longest test paths.

(e) **Inference latency < 100 ms / sample** on the Quadro P4000.

### Amended audit rubric (locked 2026-05-25 ~12:55 local, applies PLAN_03+)

Third-party review of RESULT_02 flagged the original rubric as too
narrow. The amendments below apply to PLAN_03, PLAN_04, and any
follow-up audit iteration. RESULT_02 gets a retroactive Umeyama
addendum in PLAN_03 Step 0c (raw ATE was unaffected; the IMUCNN
`keep` verdict may or may not change after re-alignment).

1. **Multi-condition validation** — `keep` requires the encoder
   holds on **≥ 2 evaluation conditions**. For modalities with
   multiple datasets (WiFi, IMU), at least one should be a
   different dataset; for modalities with one dataset (Camera =
   Webots only by project design), the two conditions become
   {nominal split, transfer split} in the same world (e.g. val on
   {2, 13, 14} vs test on {15, 16, 17} for the Webots Tiago paths).
   Document the limitation when only one dataset is available.

2. **Preprocessing is a first-class audited variable** — every
   RESULT names the preprocessing applied and reports **at least
   one preprocessing-variation probe** (or cites a prior one
   directly applicable to the iteration's data, e.g. RESULT_02's
   reuse of the run-1 IMU world-frame disaster fix). Preprocessing
   is the variable with the largest measured downstream effect in
   this project (IMU: 3.5× ATE drop documented in
   `docs/SOTA_BASELINES.md`).

3. **Aligned metrics use Umeyama / a standard library** —
   `scipy.spatial.procrustes`, `evo` (`evo.core.metrics.APE`), or
   RoNIN's own `metric.compute_ate_rte`. NEVER a hand-rolled SVD
   Procrustes fit. **Raw error is weighted at least as heavily as
   aligned in the audit decision** — the fusion's downstream
   consumer (WiFi as the absolute anchor) cares about raw error.
   If raw and aligned disagree on the `keep`/`modify`/`replace`
   label, raw wins.

### C2 placement decision (locked 2026-05-25 ~12:55 local)

C2 ("Our IMU encoder is competitive with IMU SOTA on canonical
RoNIN unseen-subjects") was NOT discharged by RESULT_02 — Branch Y
proxy is single-subject, not the canonical 32-sequence unseen split.

**Decision: queue C2 closure as PLAN_05, AFTER PLAN_04 (Odom audit)
but BEFORE Phase B fusion redesign.** This places paper-defensible
C2 evidence in the run before any architectural choice depends on
it. PLAN_05 is a data-acquisition + canonical-eval iteration: fetch
the FRDR RoNIN archive, verify well-formed, run the canonical
unseen-subjects benchmark with IMUCNN + RoNIN ResNet1D + Umeyama
alignment, label C2 discharged or not.

Phase-A roadmap updated:
- PLAN_01: WiFi audit on UJI ✓
- PLAN_02: IMU audit on RoNIN (Branch Y proxy, C2 not discharged) ✓
- PLAN_03: Camera audit on Webots
- PLAN_04: Odom audit on Webots
- **PLAN_05: C2 closure — canonical RoNIN unseen-subjects re-eval**
- Phase B begins at PLAN_06 (was PLAN_05).

### Strategic context (run-1 archived, why)

Run 1 (24 May → 25 May 2026) is archived under
`handoff/archive/run1/`. Headline failures that drive the run-2
strategy:

- Run 1 never ran an open-source SOTA baseline on its primary
  dataset (Microsoft ILN 2.0) — only trivial kNN floors. So the
  "beats baseline by 1.96 m" claim was against a trivial reference.
- Run 1 collapsed to WiFi+IMU and ignored Odom + Camera, killing
  the 4-modality story.
- Run 1 scaled compute before evidence (no small-subset pre-tests).
- Engineer /loop died with the laptop sleep cycle and lost ~3 hours.

Run 2 fixes these via the new cycle rules in `PROTOCOL.md` (Run 2+
section) and via the **encoder audit ordering**: WiFi → IMU → Camera
→ Odom, one iteration each, with SOTA-baseline reproduction as the
day-1 task of every iteration.

## Phase plan (your starting roadmap, you own it)

**Phase A — Encoder audit (PLAN_01 → PLAN_05; CLOSED 2026-05-25 ~16:05)**
- 01: WiFi encoder audit vs `wlan_localization` on UJIIndoorLoc ✓
- 02: IMU encoder audit on a000 proxy (Branch Y) ✓ — C2 not discharged
- 03: Camera encoder audit (DPVO motion) on Webots sim ✓
  — `keep with smoothness debt` after PLAN_05 Step 0 retros
- 04: Odom encoder internal audit on Webots sim ✓
- 05: C2 closure attempted; FRDR Globus-gated → C2 NOT discharged,
  IMUCNN relabelled `keep (in-domain only)`. RESULT_03 retros done
  in same iteration.

**Phase B — Fusion redesign (PLAN_06 ✓ + PLAN_09 → PLAN_12)**
- 06: Phase B foundation ✓ — WiFi+IMU K=1 baseline reproduced
  (val 0.469 m, test 0.517 m, IMU net-positive, latency 0.044 ms).

**Phase A close-out (formerly MANUAL Phase C; now active after
user-side unblock 2026-05-25 evening — both datasets in `data/`)**
- 07: C2 closure v2 — canonical RoNIN unseen-subjects from FRDR
  ZIP (`data/FRDR_dataset_538_download_606_202605251142.zip`,
  ~14.9 GB); ResNet1D + IMUCNN with RoNIN's own
  `compute_ate_rte` metric AND Umeyama-aligned column.
- 08: Camera external-SOTA validation on TartanAir hospital sample
  (`data/hospital_sample_P000.tar.gz`); pipeline = DPVO if
  `lietorch`/`altcorr` build succeeds, else TartanVO MIT
  pure-PyTorch, else DROID-SLAM. Metric = `evo` ATE RMSE with
  Sim(3) alignment (DPVO's reported convention).

**Phase B (resumes after 07/08)**
- 09: Add Camera (DPVOMotion-P-A) as 3rd modality, same
  FusionTransformer, retrain, evaluate.
- 10: Add Odom 1.5-modality path (OdomCNN-P-B + raw odom_x/y) as
  4th, retrain, evaluate.
- 11: Architecture bake-off on 10 % Webots subset — set-transformer /
  TCN / LSTM-attn / late+gate. Commit to winner.
- 12: Phase B winner full-training + per-modality ablations
  (`only:X`, `drop:X` for all 4 modalities).

**Phase C — Main results table + real-world validation (PLAN_15 ✓ + PLAN_19→23)**

PerCom paper hinges on ONE main results table (per third-party
directive 2026-05-26 ~08:00 local; see
`handoff/SCIENTIST_NOTE_main-results-table.md` for full schema +
in-hand vs missing-number inventory). Phase C deliverables, in
iteration order:

- 15: MSILN site1/B1 cross-session ✓ — gate (c)-2 clean SOTA beat
  (+14.29 m test over wlan_localization); gate (c)-1 partial
  (kNN test-anomaly). NB: used WiFiSetTransformer, divergent
  from RESULT_01 audit-winner Anchor2Vec; re-run flagged.

- **19: IMUWiFine** — CNN1D + LSTM-attn + per-leg SOTA
  reproduction (wlan_localization on WiFi + RoNIN ResNet1D on
  IMU). NEW measurements; never run on this dataset.

- **20: IPIN 2024 floor 0** — same shape as PLAN_19. Floor 0
  per directive (scope-bounded; other floors optional Phase C
  extension if time permits).

- **21: RoNIN single-mod** — CNN1D + LSTM-attn IMU-only.
  ResNet1D 5.140 m reused from RESULT_07.

- **22: UJI** — CNN1D + LSTM-attn at K=1 (degenerate temporal
  axis; UJI is per-scan). wlan_localization 15.17 m and
  Anchor2Vec 8.69 m reused from RESULT_01.

- **23: SUMMARY + main table assembly** — populate the 6-row
  table; cross-dataset comparison; honest gaps documented
  (C2 cross-subject, Camera per-leg paper-soft, smoothness
  debt across architectures).

Optional Phase C extensions IF time remains after PLAN_23:
- Conformal coverage at α=0.1 on the Phase B winner (criterion
  (d) extension; `src/pipeline/uncertainty/conformal.py` restored
  RESULT_06).
- MSILN cross-session re-run with the CNN1D + Anchor2Vec
  combination (RESULT_15's divergence — could close gate (c)-1
  if the new architecture margin holds cross-session).

## Iteration log

| # | plan file | result file | engineer commit | scientist note |
|---|---|---|---|---|
| 01 | PLAN_01_wifi-encoder-audit-uji.md (revised 2026-05-25 scientist first wake) | RESULT_01_wifi-encoder-audit-uji.md | iter 01: wifi-encoder-audit-uji (baf1a61) | Step 0 (recover run-1 audit files) added — files exist only on `overnight-autonomous-2026-05-24`. Locaris bonus dropped (Sachini/niloc = NILoc IMU, not WiFi; real Locaris arXiv:2510.11926 no code yet). Engineer: Anchor2Vec **keep** (8.69 m, +1.6 % vs run-1 ref), WiFiSetTransformer **replace on UJI / defer cross-session to Phase C** (12.95 m, +49 % vs Anchor2Vec). Recommend PLAN_02 = IMU audit, no parallel WiFi track. |
| 02 | PLAN_02_imu-encoder-audit-ronin.md | RESULT_02_imu-encoder-audit-ronin.md | iter 02: imu-encoder-audit-ronin (f494a35) | Branch X (canonical unseen-subjects) preferred; Branch Y (a000 intra-session proxy) fallback if full RoNIN data missing locally (only `data/ronin_a000` confirmed present). Three orthogonal probes for bottleneck: architecture (RoNIN ResNet1D) + capacity (IMUCNN 2× width) + preprocessing (run-1 disaster fix). Engineer: **Branch Y used (full RoNIN data not on machine)**. IMUCNN = **keep** (aligned ATE 1.04 m vs ResNet1D 0.97 m, +7 %; raw 3.55 m vs 2.89 m, +23 % borderline; 95× smaller, 4× faster). Capacity probe **refuted** modify hypothesis (2× width raw ATE 5.81 m, +63 %). C2 NOT discharged — queued as PLAN_05 (locked before Phase B). Verdict subject to Umeyama re-alignment (PLAN_03 Step 0c addendum). |
| 03 | PLAN_03_camera-encoder-audit-webots.md (first plan under amended rubric: multi-condition validation, preprocessing as first-class variable, Umeyama for any alignment, raw weighted ≥ aligned) | RESULT_03_camera-encoder-audit-webots.md | iter 03: camera-encoder-audit-webots (92e9f2c) | Branch P (DPVO importable) preferred; Branch Q (motion-encoder-only fallback). Step 0c retrofits RESULT_02 with Umeyama alignment. Within-sim val→test gap < 20 % is the multi-condition gate (Camera = sim-only by project design). Engineer: **Branch Q for DPVO SLAM (no lietorch/altcorr); Branch P for motion encoder.** DPVOMotionEncoder = **keep** with P-A preprocessing (val 1.85 m, test 1.56 m on canonical CLAUDE.md split; test-val gap −15.7 % = no overfitting). Preprocessing-variation probe: P-A beats P-B by 9 %. Capacity probe (stride 10): neutral. Honest weakness: per-traj smoothness median r ≈ 0.07 (poor). Step 0c Umeyama retro on IMU = all three encoders collapse to ~0.30 m Umeyama-aligned ATE; IMUCNN-keep stands; capacity-refuted claim softened to "not clearly the bottleneck." Recommend PLAN_04 = Odom audit. |
| 04 | PLAN_04_odom-encoder-audit-webots.md | RESULT_04_odom-encoder-audit-webots.md | iter 04: odom-encoder-audit-webots (823b4f9) | Internal audit (no public SOTA). "Day-1" baseline = trivial cumulative-integration of odometry → position MAE (Step 1). Floor gate: OdomCNN must beat trivial integration by ≥ 10 % raw test MAE or label = `replace`. Two preprocessing variants (P-A raw norm / P-B Δ-features). One probe (width OR window). Phase A closes after this RESULT; Phase A summary table required. Engineer: trivial floor val 12.17 m / test 8.27 m / smoothness r=0.999. **OdomCNN P-B = keep** (val 4.62 / test 4.24, +49 % over floor; gap −8.3 %). P-A fails gate (+20.9 %); window32 neutral. Honest weakness: OdomCNN smoothness r ≈ 0 vs trivial 0.999 → Phase B should consider feeding *both* OdomCNN embedding (absolute-MAE) and raw integrated `(odom_x, odom_y)` (smoothness). **Phase A closed (4/4 keep)**; recommend PLAN_05 = C2 closure per locked plan. |
| 05 | PLAN_05_c2-closure-ronin-canonical.md (folds in 3 RESULT_03 review notes: Step 0a difficulty-matched probe, Step 0b smoothness debt re-label, Step 0c PLAN_06 queue) | RESULT_05_c2-closure-ronin-canonical.md | iter 05: c2-closure-ronin-canonical (c9ea806) | Focused experiment = C2 closure on canonical RoNIN unseen-subjects with Umeyama. Step 1 = FRDR archive fetch (gated path; fallback = Branch Y reaffirmation + C2 deferred to Phase C). PLAN_06 newly inserted as Camera external-SOTA validation on a public VO benchmark (TartanAir/EuRoC/KITTI; method = DPVO unblocked or TartanVO/DROID-SLAM). Engineer: **Step 1 BLOCKED — FRDR is Globus-OAuth-gated; no canonical data on disk**. Steps 0a/0b/0c done as RESULT_03 addenda (difficulty-normalised gap +17.5 % at the edge; relabelled `keep with smoothness debt`; PLAN_06 queue confirmed). RESULT_02 IMU verdict updated to **`keep (in-domain only)`** with C2 deferred to manual / Phase C. Phase B can begin at PLAN_06 (Camera external SOTA) or PLAN_07 (bake-off). |
| 06 | PLAN_06_phase-b-foundation.md (**scope flipped from "Camera ext-SOTA" to "Phase B foundation"** — time-value call after 2nd wake-up stall; Camera ext-SOTA + canonical-C2 both bundled as MANUAL Phase C tasks) | RESULT_06_phase-b-foundation.md | iter 06: phase-b-foundation (9133e54) | Restore run-1 fusion stack (`src/pipeline/fusion/*`, `training/fusion_trainer.py`, `configs/stage_c/`, `_smoke_fusion.py`) + reproduce 2-modality (WiFi+IMU) baseline on Webots. Target: val MAE within ±15 % of run-1's 0.43 m. No Camera, no Odom, no architecture change yet — those queue as PLAN_07/08/09. Engineer: fusion stack restored cleanly (33 files + `encoders/__init__.py` extended for `DPVOMotionEncoder`). **WiFi+IMU K=1 baseline: val 0.469 m (+9.1 % vs run-1 ref, well inside ±15 %), test 0.517 m, latency 0.044 ms/sample, IMU adds net-positive 6.6 %/1.3 % val/test over WiFi-only.** Recommend PLAN_07 = add Camera (DPVOMotion P-A) as 3rd modality at K=1, same FusionTransformer config. |
| 07 | PLAN_07_c2-closure-ronin-canonical-v2.md (user-side unblock: FRDR ZIP now in `data/`; Phase A close-out, NOT Phase B continuation as originally planned) | RESULT_07_c2-closure-ronin-canonical-v2.md | iter 07: c2-closure-ronin-canonical-v2 (c3bf2de) | Focused experiment = canonical RoNIN unseen-subjects C2 closure (32-sequence test). Step 0 = extract 14.9 GB FRDR ZIP + verify HDF5 layout (`synced/{gyro_uncalib, acce, time}` + `pose/{tango_pos, tango_ori}`) + per-list coverage. Step 1 = ResNet1D SOTA repro (prefer pretrained if in FRDR archive; else train). Step 2 = IMUCNN with **RoNIN's own `metric.compute_ate_rte`** + Umeyama side-by-side. Step 3 = audit decision, raw-weighted. References: run-1 14.41 m IMUCNN / 5.93 m ResNet1D; paper 5.14 m. Engineer: **32/32 unseen seqs extracted**; ResNet1D pretrained reproduces **5.140 m ATE exact paper match**; IMUCNN canonical raw ATE **9.961 m / Umeyama 7.876 m** — **C2 NOT discharged** (raw gap +93.8 %, Umeyama gap +53.2 %, both fail 20 % gate). IMUCNN label stays `keep (in-domain only)`; Phase B contingency live (swap to ResNet1D unmodified if `only:imu` >= 1.4x `only:wifi` AND `drop:imu` doesn't help). |
| 08 | PLAN_08_camera-ext-sota-tartanair-hospital.md (user-side unblock: TartanAir hospital sample now in `data/`; Phase A close-out continued) | RESULT_08_camera-ext-sota-tartanair-hospital.md | iter 08: camera-ext-sota-tartanair-hospital (f6e427d) | Focused experiment = Camera per-leg validation on TartanAir hospital sample. Step 0 = extract `hospital_sample_P000.tar.gz` (image-only TartanAir v1: NO IMU); verify NED 7-float `pose_left.txt`. Step 1 = pipeline selection: DPVO retry on Windows → TartanVO MIT pure-PyTorch fallback → DROID-SLAM last resort. Step 2 = chosen SOTA on the sequence via `evo` Sim(3)-aligned ATE. Step 3 = our DPVOMotionEncoder (Mode α: Webots-trained head out-of-domain). Step 4 = upgrade-or-keep label at 30 % paper-strength bar. Engineer: **TartanVO ran via python3-branch + 4 wrapper-only shims** (scipy `as_dcm`→`as_matrix`, `cupy.cuda.compile_with_cache` → RawModule, `numpy.linalg.linalg`, in-script Umeyama); full-seq Umeyama ATE **0.518 m**, last-20% slice **0.012 m**. **DPVOMotion** (Mode α infeasible — no saved Webots head; ran trunk-transferability w/ in-domain linear head on first-80% P000): last-20% slice Umeyama ATE **0.293 m**, Umeyama scale 0.698. **Gap +2300 % → paper-soft**; RESULT_03 `keep with smoothness debt` stands, sharpened to "fit-for-purpose as fusion encoder; NOT a standalone VO baseline". **Phase A fully closed** (5 audits + 2 ext-SOTA + Phase B foundation). Recommend PLAN_09 = add Camera (DPVOMotion-P-A) as 3rd modality to WiFi+IMU K=1 fusion (option A from RESULT_06's open Q). |

| 09 | PLAN_09_phase-b-add-camera.md | RESULT_09_phase-b-add-camera.md | iter 09: phase-b-add-camera (c3cd889) | Add Camera (DPVOMotion-P-A) as 3rd modality to WiFi+IMU K=1 fusion. Same FusionTransformer, same canonical Webots split, frozen DPVO trunk + trainable head. Pre-test gate + memory budget. Full subset eval (`only:wifi/imu/camera` + `drop:*`). **Per-trajectory smoothness ratio reported** (enforces RESULT_05's locked Phase B gate). Decision: Camera net-positive vs marginal vs noise (raw-weighted vs RESULT_06 baseline). PLAN_10 default = add Odom 1.5-modality. Engineer: **C3 lower bound CLEARED**. val 0.448 m (−4.5 % vs RESULT_06), test 0.489 m (−5.4 %, < 0.50 m gate). Camera contributes net-positive at aggregate. Surprising K=1 diagnostic: **WiFi+Camera (no IMU) tied with full 3-modality** (val 0.449 / test 0.481) — IMU is redundant at K=1; expected to differ at K>1. Smoothness debt persists (median r=0.029, similar to RESULT_03's r=0.07 standalone). Latency 0.053 ms/sample. Recommend PLAN_10 = Odom 1.5-modality (option B: 2 slots — `odom_cnn` + `odom_raw` projection). |

| 10 | PLAN_10_phase-b-add-odom-1p5.md | RESULT_10_phase-b-add-odom-1p5.md | iter 10: phase-b-add-odom-1p5 (54ec613) | Add 4th modality = Odom 1.5-modality per RESULT_04 option (iii): OdomCNN-P-B (Δ-features) embedding **+** raw integrated `(odom_x, odom_y)` columns projected 2→128. Engineer picked **option (B-variant)**: separate `odom_raw` modality slot in fusion model, served via `extra_inputs` (cleaner than dataset change). 5-mod (WiFi+IMU+Camera+Odom+Odom_raw) K=1: **val 0.491 / test 0.486** (C3 ≤ 0.50 cleared, but val regressed +9.6 % vs RESULT_09 3-mod 0.448). **Headline finding**: `only:wifi` test 0.489 ≈ full 5-mod test 0.486 → **fusion saturated at K=1**; WiFi alone does the job. Smoothness debt persists (r=0.015); `drop:odom_raw` indistinguishable from full → raw column NOT being attended. Recommend PLAN_11 = K>1 temporal fusion (K=8 default per fusion.yaml), NOT a late+gate bake-off (architecture isn't the bottleneck, missing temporal axis is). |

| 11 | PLAN_11_phase-b-k-gt-1-temporal.md | RESULT_11_phase-b-k-gt-1-temporal.md | iter 11: phase-b-k-gt-1-temporal (c4c681e) | Architectural pivot per RESULT_10. K=1→K=8 + per-instant dropout 0.45 (run-1 audit-fix value per CLAUDE.md). Single focused experiment on the K dimension. Three outcomes named: (α) K=8 beats K=1 + graceful staleness → fresh-accuracy headline; (β) ties K=1 + graceful staleness → robustness headline; (γ) regresses → PLAN_11b architecture probe. Staleness sweep (0/5/15/30) + smoothness gate (r > 0.20) + memory budget < 6 GB (K=4 fallback if 6 GB blown). PLAN_12 default = Phase B winner full-train + ablations. Engineer: **outcome γ** — K=8 5-mod val 0.667 / test 0.651, **+33.9 % regression** vs RESULT_10 K=1 (test 0.486). C3 lower bound FAILS at K=8 on fresh. **Staleness slope unlocked** (0.65 → 1.30 m across 18 s WiFi lag, NOT cliff — robustness payoff). Subset eval at K=8: IMU/Camera/Odom **finally contribute** (wifi+imu+camera+odom test 0.594, −19 % vs only:wifi 0.732). odom_raw actively distracts at K=8 (5-mod 0.651 vs 4-mod 0.594). Smoothness r=−0.010 (B-3 hypothesis falsified). Recommend PLAN_12 = K=4 + drop odom_raw. |

| 12 | PLAN_12_phase-b-k4-drop-odom-raw.md | RESULT_12_phase-b-k4-drop-odom-raw.md | iter 12: phase-b-k4-drop-odom-raw (0d873b5) | Per engineer's RESULT_11 recommendation: K=8→K=4 + drop `odom_raw` (RESULT_11 showed K=8 5-mod test 0.651 vs K=8 4-mod (no odom_raw) 0.594 → odom_raw distracts at K>1). Two coupled changes: tests whether K=8 was overshoot (sweet spot K=4) OR whether modality combinatorics drove the regression. Three outcomes: (α') beats K=1, (β') ties K=1 + staleness slope = robustness headline, (γ') still regresses → PLAN_13 = lr/batch/dropout sweep. RESULT_11 surfaced motion-modality contribution at K=8 (4-mod 0.594 vs only:wifi 0.732, −19 %) — that's the structural finding even at the regressed K=8 number. Engineer: **outcome γ'** — 4-mod K=4 val 0.579 / test 0.575 (+17.6 % vs RESULT_09 4-mod K=1's 0.489). K-scale NOT the bottleneck; **batch×lr confound** the suspected cause (RESULT_10 used B=128, this iter B=64 same lr=1.3e-3 as Optuna-tuned for B=128). Staleness slope ×2.1 across 18 s (similar to K=8's ×2.0 but lower absolute). Smoothness r=0.048 (best yet, still under 0.20 gate). Recommend PLAN_13 = isolated batch×lr probe (K=1 5-mod at B=64 AND K=4 4-mod at B=128 with lr×1.41). |

| 13 | PLAN_13_phase-b-batch-lr-probe.md | RESULT_13_phase-b-batch-lr-probe.md | iter 13: phase-b-batch-lr-probe (58a7a0a) | Single-axis batch probe per RESULT_12's recommendation: K=4 + 4-mod + **B=128** (restore RESULT_06's pre-K>1 default), lr=1.3e-3 unchanged. Diagnostic: if test recovers to ≤ 0.51 m (was 0.575 at B=64), batch confound is confirmed → PLAN_14 = full ablations (α''/β'' outcomes). If still regresses → architecture probe (γ''). Engineer's RESULT_12 also suggested "K=4 + lr×1.41" as a parallel branch; PLAN_13 keeps it single-axis per the one-experiment-per-plan rule. Staleness slope expected to PERSIST at B=128 (K-axis property, not batch-axis). Engineer: **outcome α'' — C3 WINNER** val 0.394 / test **0.417** at K=4 B=128 4-mod (beats CLAUDE.md run-1 K=8 ref 0.43 m by 9 %). Batch×lr confound CONFIRMED as RESULT_11/12 regression driver. Staleness slope holds (×2.2 across 18 s). Subset eval surprise: wifi+imu+camera (drop Odom) val 0.381 / test 0.406 BEATS full 4-mod by ~3 % — actual minimal stack is **3-mod K=4 B=128**. Per-path test 15/16/17 means 0.32/0.51/0.47. Smoothness debt persists r=0.039 (B-1/B-2 PLAN_15 candidate). Recommend PLAN_14 = (P1) drop-Odom training + (P2) Phase C kickoff (MSILN C4). |

| 14 | PLAN_14_phase-b-winner-ablations.md | RESULT_14_phase-b-winner-ablations.md | iter 14: phase-b-winner-ablations (056b5c3) | Full ablation suite on the Phase B winner (K=4 + 4-mod + B=128). Single-experiment scope = the ablation study itself. Steps: checkpoint reuse OR fresh re-train w/ seed=42; 16-row subset eval (15 non-empty 4-mod subsets + full) to settle drop-Odom-vs-full question; 8-lag staleness sweep (0–30 instants ≈ 0–27 s) as paper-figure-grade robustness evidence; per-trajectory smoothness on test paths 15/16/17 (criterion (d) gate); latency at b=1 and b=32 (criterion (e) gate). Step 6 = PerCom main-results panel: status per criterion (a)/(b)/(c)/(d)/(e). PLAN_15 default = Phase C MSILN cross-session kickoff; alternative = K-axis sweep at B=128. Engineer: used checkpoint (Step 0A); val/test reproduced exactly (0.394/0.417). 16-row subset eval confirms `wifi+imu+camera` test 0.406 m beats full-4-mod 0.417 m by 2.6 % (Odom marginally net-negative). 8-lag staleness sweep: 0.417 m fresh → 1.197 m at 27 s (linear regression slope 0.029 m/s, R²=0.998 — clean slope, no cliff). Latency b=1 6.41 ms / b=32 0.20 ms (criterion (e) cleared by 500× at b=32). Smoothness median r=0.039 persists (debt documented). **Phase B CLOSED.** Recommend PLAN_15 = Phase C kickoff (MSILN site1/B1 cross-session C4). |

| 15 | PLAN_15_phase-c-msiln-cross-session.md | RESULT_15_phase-c-msiln-cross-session.md | iter 15: phase-c-msiln-cross-session (cce6632) | Phase C kickoff. Criterion (c) / C4: MSILN site1/B1 cross-session. Two gates: (1) beat WiFi-kNN by ≥ 1.5 m; (2) beat open-source SOTA (wlan_localization) by ≥ 0.5 m. **Day-1 wlan_localization on MSILN is the new measurement** run-1 never made (run-1's headline failure). Phase B winner architecture adapted to 2-modality (WiFi+IMU only — MSILN has no Camera/Odom): K=4 + B=128 + same dropout. Three outcomes: (α) both gates → conformal next; (β) kNN-only → WiFiSetTransformer re-eval next; (γ) neither → run-2 SUMMARY draft + honest paper framing. Engineer: **outcome β (partial C4)** — val 16.60 m / test 14.02 m. Gate (c)-2 vs wlan_localization (NEW measurement val 21.26 / test 28.31) **PASSES** with margin (+4.66 / +14.29 m). Gate (c)-1 fails on test (kNN 9.47 → ours 14.02 = +56 % regression) but **val 17.66 → 16.60 only just below 1.5 m gate**. kNN test anomaly traced to per-path composition: path 130 (786 samples, 28 % of test, WiFi-dense) pulls kNN test mean down to 9.47; our paths 128/129 (Dec-session out-of-distribution) suffer most. Smoothness r=0.107 **best in run-2**. Training cost: 12968 s (~3.6 h). Used WiFiSetTransformer per config (deferred RESULT_01 question). Recommend PLAN_16 = SUMMARY draft + optional Anchor2Vec MSILN re-run. |

| 16 | PLAN_16_phase-b-architecture-bakeoff.md (inserted per third-party review 2026-05-26 ~04:10 local; bake-off was wrongly dropped at iter 10 — iter 13 refuted the dropping premise; paper-strength fix for methods section) | RESULT_16_phase-b-architecture-bakeoff.md | iter 16: phase-b-architecture-bakeoff (849378f) | 4 candidates {LSTM-attn / TCN / 1D-CNN / Transformer-from-scratch} on 10 % Webots subset, K=4 B=128, same encoders + readout + protocol; only fusion block changes. Incumbent run-1 FusionTransformer benchmarked on same subset for fair compare. Decision: beat 0.417 m AND r > 0.20 ⇒ new winner; else incumbent stands defensible ("4 architectures compared, kept this one"). PLAN_17 = if new winner → full-data retrain; if not → Phase C continuation. **NB**: engineer's RESULT_15 surfaced that the deployed MSILN config used `WiFiSetTransformer` per their default — that's an unintended divergence from the RESULT_01 audit decision (Anchor2Vec was the keep verdict on UJI); flag for PLAN_17 Phase C follow-up. Engineer: **3 of 4 candidates implemented** (transformer_scratch cut per "overrun" provision). On 2-path subset, all 3 candidates BEAT incumbent by 24-34 % on val/test at 1/3 params (CNN1D val/test 0.978/1.261 vs incumbent 1.493/1.688). **Smoothness gate r > 0.20 NOT met by any** (max LSTM-attn r=0.085) → strict PLAN_16 rule NOT met. Smoothness debt is architecture-invariant (4-of-4 fail). Subset finding may NOT generalise to full data (incumbent has 5.7× more data at full scale). Recommend PLAN_17 = full-data retrain CNN1D + LSTM-attn (~1.5h compute). |

| 17 | PLAN_17_phase-b-full-data-retrain-cnn1d-lstm.md | RESULT_17_phase-b-full-data-retrain-cnn1d-lstm.md | iter 17: phase-b-full-data-retrain-cnn1d-lstm (a81356f) | Settle the bake-off at production data scale. Train CNN1D (test-leader on subset) and LSTM-attn (val-tied) on full Webots data at the K=4 + 4-mod + B=128 + lr=1.3e-3 + 90 epochs incumbent config. Compare against RESULT_13's 0.394/0.417 m. Three outcomes: α''' / β''' / γ'''. Critical secondary question: does smoothness r > 0.20 gate finally clear at full-data scale? Engineer: **outcome α''' fires decisively** — both CNN1D and LSTM-attn beat incumbent at full data. CNN1D val **0.282** / test **0.339** (−28%/−19% vs incumbent 0.394/0.417), LSTM-attn val 0.301 / test 0.340 (−24%/−19%); both at ~0.51-0.57 M params (1/3 of incumbent's 1.55 M); CNN1D peak GPU 775 MB, LSTM-attn 819 MB (well under RESULT_14's 466 MB budget? actually higher — note: B=128 K=4). **CNN1D is the new C3 paper-claim model**, criterion (b) ≤ 0.5 cleared by 32 % margin. **LSTM-attn surfaces structural finding**: only:imu test 0.339 ≈ full fusion 0.340 m; only:camera test 0.338 — LSTM-attn dead-reckons from each motion modality (not WiFi-anchor dependent). Subset eval: `wifi+camera` for CNN1D val 0.276 / test 0.339 = tied with full 4-mod (Odom marginal). Latency: CNN1D 0.044 ms / LSTM-attn 0.047 ms b=1 (well under 100 ms gate). Smoothness debt: CNN1D r=0.009, LSTM-attn r=0.051 (best in run-2), incumbent r=0.039 — architectural lever doesn't clear 0.20 gate; loss-function (B-1 aux velocity / B-2 EMA) remains the open lever. Recommend **PLAN_18 = full ablations on CNN1D** (mirror of RESULT_14: 16-row subset, 8-lag staleness sweep, per-trajectory plots, formal latency b=1 + b=32); LSTM-attn documented as runner-up with the dead-reckoning structural finding in paper discussion. |

| 18 | PLAN_18_phase-b-new-winner-cnn1d-ablations.md | RESULT_18_phase-b-new-winner-cnn1d-ablations.md | iter 18: phase-b-new-winner-cnn1d-ablations (5122835) | Mirror RESULT_14's ablation suite on the new CNN1D winner (val 0.282 / test 0.339); 16-row subset, 8-lag staleness, per-trajectory plots, b=1+b=32 latency. Plus LSTM-attn dead-reckoning probe (16-row + 4-lag). Engineer: **CNN1D ablation holds paper-strength shape**. Sanity reproduction exact (val 0.282 / test 0.339 from cached checkpoint, both archs). CNN1D 16-row subset: `wifi+imu+camera` drop-Odom 0.338 m ≈ full-4-mod 0.339 m (Δ=0.3%); RESULT_14 drop-Odom pattern holds qualitatively but margin shrinks (CNN1D Δ=−0.3% vs incumbent Δ=−2.6%). CNN1D 8-lag staleness: linear slope **0.0280 m/s R²=0.995** (essentially identical to incumbent's 0.029 m/s — K=4 temporal property is architecture-invariant); 0 s 0.339 → 27 s 1.088 m. CNN1D per-trajectory smoothness median r=0.009 (debt persists; loss-function lever B-1/B-2 remains the open knob). CNN1D latency 100-trial: b=1 **4.73 ms/sample** (21× under 100 ms gate), b=32 **0.15 ms/sample** (660× under; ~25% faster than incumbent b=32). RESULT_17's "0.044 ms b=1" was a b=128-divided number; this is the corrected single-sample latency. **LSTM-attn dead-reckoning probe confirmed**: 16-row reveals ALL 4 only:X rows within 8 % of full (only:wifi 0.423, only:imu 0.339, only:camera 0.338, only:odom 0.357 vs full 0.340); only:imu ≈ only:camera ≈ full to noise level; uniform per-modality recovery NOT WiFi-anchor dependent (contrast CNN1D where only:camera=0.422 and only:odom=0.741 lag full by 24-119%). LSTM-attn 4-lag staleness: slope **0.0236 m/s** (R²=0.984) — 16% shallower than CNN1D's 0.028, 19% shallower than incumbent's 0.029 — dead-reckoning hypothesis supported by data (per-modality recovery means LSTM-attn leans on other 3 modalities harder when WiFi degrades). **Two fusion regimes confirmed structurally** = paper-grade discussion finding (cooperative CNN1D vs dead-reckoning LSTM-attn, tie on fresh accuracy, LSTM-attn more robust to staleness). One open question for scientist: LSTM-attn's `only:camera` 0.338 < full 0.340 — modality_dropout=0.4 over-regularising at inference? PLAN_19 recommendation: **option (a) MSILN cross-session re-run with new CNN1D winner** (RESULT_15 used incumbent + WiFiSetTransformer; CNN1D's 19% fresh accuracy gain in-sim may help gate (c)-1 which RESULT_15 missed); fallback option (b) Conformal coverage on CNN1D (~20 min). |

| 19 | PLAN_19_main-results-imuwifine.md | RESULT_19_main-results-imuwifine.md | iter 19: main-results-imuwifine (b5a36b2) | First iteration of the main-results table (per third-party directive 2026-05-26 ~08:00; SCIENTIST_NOTE_main-results-table.md). IMUWiFine row: CNN1D + LSTM-attn + wlan_localization (NEW) + RoNIN ResNet1D (NEW). 2-mod (WiFi+IMU), K=4 + B=128 + lr=1.3e-3, 90 epochs. Engineer: configs restored from `overnight-autonomous-2026-05-24` (imuwifine.yaml + convert_imuwifine.py); dataloader smoke-pass (train 23385 / val 13947 / test 23724 windows). **Step 1a wlan_localization on IMUWiFine** (NEW): val 4.170 m / test 8.504 m (k=3 manhattan dist-weighted, Box-Cox+PCA preprocessor 343→150 dims). **Step 1b RoNIN ResNet1D on IMUWiFine** (NEW): val **26.84 m** / test n/a (IMUWiFine test paths lack IMU per dataset design; ResNet1D needs in_dim probe fix for WIN=32 since conv output length=1 not 2 as the IPIN template assumed). **Step 2 CNN1D + LSTM-attn training** (90 epochs each, ~320 s, peak GPU ~325 MB): CNN1D 0.34 M val **1.397 m** (best ep 75) / test 7.094 m; LSTM-attn 0.41 M val **1.264 m** (best ep 70) / test 7.196 m. **Step 3 main-table outcome**: **α'''' on VAL** (we beat both per-leg SOTAs by 70 % vs wlan_localization and 95 % vs RoNIN — decisive wins); **β'''' on TEST** (we beat WiFi SOTA by 16-17 %, but RoNIN is unmeasurable due to no-IMU test design). **LSTM-attn dead-reckoning regime REPLICATES on IMUWiFine — second data point**: only:imu val 1.263 ≈ full val 1.264 (Δ=0.1 %), only:wifi 1.274 (Δ=0.8 %); CNN1D contrast only:imu 1.452 lags full 1.397 by 4 % (cooperative regime). On test all `only:X` collapse to `only:wifi` ≈ full because the IMU branch sees zeros. **Step 4 smoothness**: CNN1D r=−0.005, LSTM-attn r=−0.007 (debt persists across 3 datasets × 5 archs; loss-function lever B-1/B-2 remains the open knob). Per-path test plots saved for top-5 longest test paths (63, 66, 70, 77, 79). **Honest caveat**: IMUWiFine test split has no IMU by dataset design → fusion test = WiFi-only test; the val column is the apples-to-apples per-leg-SOTA comparison. One open question for scientist: paper framing (a) headline=val only, footnote test as WiFi-floor; or (b) report both with test asterisked as WiFi-only. PLAN_20 recommendation: IPIN 2024 floor 0 row (IMU on both splits so RoNIN fully measurable). |

| 20 | PLAN_20_val-test-gap-audit.md (user-flagged diagnostic 2026-05-26 ~08:30 local) | RESULT_20_val-test-gap-audit.md | iter 20: val-test-gap-audit (e4c5809) | Methodology audit on val/test gap across all measured datasets/methods. Engineer: **failure mode 3 (legitimate cross-session dataset shift), NO code bug**. Step 0 gap table: Webots +5-20 % (consistent generalisation), MSILN cross-session weird patterns (val>test on kNN, val<test on SOTA — RESULT_15 already explained path-130 composition), **IMUWiFine all 4 methods affected by +104 % to +470 %**. **The wlan_localization SOTA +104 % gap is the diagnostic that rules out our-code as the culprit** (SOTA uses separate pipeline, no shared split code with us). Step 1: `convert_imuwifine.py:42-52` explicitly documents "two raw formats coexist" — train/val use Android logger format with IMU + ms-since-epoch; test use header-less format with **no IMU**, different timestamps; converter detects per-file and parses correctly. Step 2 distribution probe surfaces three sample-level shifts: test WiFi @ **5.65 Hz vs train+val 0.31 Hz (18× faster)**; **NO IMU on test**; GT y-range **1.2-1.6 m on test** vs **0-5 m on train+val** (test walks a thin physical strip, not the full corridor). RSSI mean ≈ −74 m both; RSSI std 9-12. Step 3 per-path CNN1D test MAE long-tailed 1.96-17.10 m across 20 paths (7 easy <4 m comparable to val, 8 medium 4-7 m, 5 hard >7 m); aggregate test median 3.62 m close to val 1.40 m but mean 7.09 m dragged by 5 hard paths consistent with WiFi-fingerprint drift across novel sub-regions. Step 4 verdict: no code fix needed; methodology sound. Step 5: continue main-table at IPIN floor 0 (PLAN_21); IPIN per CLAUDE.md is "converted per floor" single-campaign so unlikely to show same campaign-split pattern. **Open question for scientist**: paper framing of IMUWiFine row — (a) val-only headline + test as cross-session robustness floor footnote, or (b) report both val + test with test asterisked as cross-session; choice has downstream effect on IPIN/RoNIN/UJI row framing. |

| 21 | PLAN_21_transformer-from-scratch.md (scientist-designed MoTTransformer; restoration of the bake-off candidate cut in PLAN_16) | RESULT_21_transformer-from-scratch.md | iter 21: transformer-from-scratch (3847864) | Scientist-designed transformer per RESULTs 17/18 evidence: 3-layer 2-head encoder; learnable modality embeddings; ALiBi temporal bias; single-query cross-attention readout (no CLS); 2D FFN; ~0.48 M params target. Engineer: implemented `src/pipeline/fusion/mot_transformer.py` (custom ALiBi self-attention since `nn.MultiheadAttention` doesn't accept per-head bias cleanly), registered in `bakeoff.py::CANDIDATES["mot_transformer"]`. Smoke passed (drop-camera + all-masked NaN-safety verified). Real-encoder param count **0.74 M** (above the 0.48 M body-only estimate; encoders contribute ~0.23 M). 90-epoch training: val **0.594** / test **0.608 m** (best ep 68, 250 s, peak GPU 820 MB). **Outcome γ5 — MoTTransformer is the WORST of 4 architectures** (+79% vs CNN1D test 0.339, +46% vs incumbent test 0.417). Subset eval reveals **WiFi-anchored regime**: only:wifi 0.704 ≈ full 0.608 (16% gap); motion-only 3-5 m alone (only:imu 3.928, only:camera 2.072, only:odom 5.191) — opposite of LSTM-attn dead-reckoning, also worse than CNN1D cooperative fusion. ALiBi likely suppresses cross-instant motion fusion. **Smoothness median r=0.019** — ALiBi does NOT clear 0.20 gate. **Smoothness debt now falsified across 4 architectures (incumbent 0.039, CNN1D 0.009, LSTM-attn 0.051, MoTTransformer 0.019) — confirmed loss-function-bound, not architecturally tractable**. Staleness slope 0.028 m/s R²=1.000 (architecture-invariant across 4 archs). Latency b=1 5.82 ms (23% slower than CNN1D 4.73, faster than incumbent 6.41), b=32 0.20 ms — criterion (e) cleared by 17× / 500×. Outputs at `runs/overnight/run2_iter_21/` (script's hard-coded OUT_DIR was `iter_17` so files were moved after training). **4-architecture bake-off COMPLETE**: CNN1D winner, LSTM-attn runner-up with dead-reckoning structural finding, incumbent over-parameterised, MoTTransformer loses. PLAN_22 recommendation = continue main-results table at IPIN 2024 floor 0; CNN1D remains the Phase B paper-claim model. Open question for scientist: PLAN_21b 3-row ablation (ALiBi-off / +CLS / +time-enc) would isolate WHY MoTTransformer regresses; cost ~45 min compute, optional methods-section bonus. |

| 22 | PLAN_22_main-results-ipin-floor0.md | RESULT_22_main-results-ipin-floor0.md | iter 22: main-results-ipin-floor0 (dbd9ad0) | Second main-table row. IPIN 2024 floor 0: CNN1D + LSTM-attn + wlan_localization (NEW) + RoNIN ResNet1D (NEW). Pre-flight Step 0 IMU-availability check per PLAN_20 lesson. Engineer: IPIN floor 0 has IMU on ALL splits (unlike IMUWiFine). Configs restored from run-1; data 16 paths 6/4/6 train/val/test. **Step 1a wlanloc** (NEW): val **20.530** / test 19.801 m (n=115/145, 232 APs reduced to 150 via Box-Cox+PCA). **Step 1b RoNIN ResNet1D** (NEW; same in_dim=1 fix as IMUWiFine): val **37.21** m / test **31.70** m (IMU dead-reckoning catastrophic on multi-minute paths). **Step 2 CNN1D + LSTM-attn**: CNN1D val 21.61 / test 20.45 m (0.34 M, best ep 22, early stop ep 62, peak GPU 260 MB); LSTM-attn val 22.45 / test 21.56 m (0.41 M, best ep 8, early stop ep 48, peak 267 MB). **Both overfit** (train loss 0.5 vs val loss 9.0 = 17× gap by ep 20) on the tiny training set (174 WiFi scans + 6924 IMU windows = ~10× smaller than IMUWiFine). **Step 3 outcome β5**: we BEAT RoNIN by ~40 % val / ~35 % test (decisive), LOSE to wlanloc by 5-9 % full-fusion. **Crucial diagnostic**: CNN1D `only:wifi` val **19.45** m BEATS wlanloc val 20.53 by 5 % — fusion regression is a small-train-overfit artifact, NOT fundamental WiFi failure; the IMU branch actively degrades fusion in small-data regime. **LSTM-attn dead-reckoning REPLICATES on third dataset**: val only:imu 22.64 ≈ full 22.45 (1 % gap); test only:imu 21.66 ≈ full 21.56 (1.3 % gap). Webots + IMUWiFine + IPIN = 3-dataset confirmation of the per-modality recovery structural finding. Smoothness median r=0.067 (CNN1D) / 0.089 (LSTM-attn) — highest in run-2 but still below 0.20 gate. Honest open question for scientist: main-results headline = full fusion (21.61) or best-subset-per-arch (CNN1D only:wifi 19.45 beats SOTA)? PLAN_23 recommendation = RoNIN single-modality IMU row (reuse RESULT_07 canonical 5.140 m + run CNN1D/LSTM-attn IMU-only on canonical unseen-subjects; expected +90-94 % gap per RESULT_07). |

| 23 | PLAN_23_main-results-ronin-single-mod.md | RESULT_23_main-results-ronin-single-mod.md | iter 23: main-results-ronin-single-mod (085f941) | Third main-table row. RoNIN canonical single-mod IMU: SOTA cells reused (ResNet1D 5.140 m + IMUCNN 9.961 m from RESULT_07); run CNN1D + LSTM-attn aggregators over chunked 50-step IMU sub-windows (K=4 M=1 effective single-modality fusion). Engineer: wrote `scripts/_train_ronin_canonical_arch.py` with `RoninCNN1D` wrapper (IMUCNN per K=4 × 50-step sub-window → aggregator → mean-pool → Linear(2)). Bug fix on first run: extra `transpose` before IMUCNN; IMUCNN expects (B, T, C) and transposes internally. CNN1D 0.20 M / 20 epochs / 1204 s; LSTM-attn 0.26 M / 1291 s. **Outcome β6**: aggregator helps over pure IMUCNN by ~24% on both raw ATE (9.961→7.587) and Umeyama (7.876→5.945). Gap to ResNet1D narrows from +94% to **+47.6% raw / +15.7% Umeyama** (CNN1D); LSTM-attn +45.9% / +19.1% (both clear Umeyama 20% gate; CNN1D more comfortably). Raw 20% gate NOT cleared by either — C2 stays `keep (in-domain only)` per rubric correction #3. **RTE 12.6 vs ResNet1D 4.4 m = 3× worse**: aggregator improves global trajectory drift but worsens local consistency (over-smooths short-term motion in exchange for global accuracy). CNN1D ≈ LSTM-attn at M=1 (within 1.2%); dead-reckoning regime finding from M=2/M=4 not relevant for single-modality. **The aggregator's RTE-to-ATE asymmetry on RoNIN + smoothness debt on Webots both point to loss-function lever** (RESULT_18 B-1/B-2 hypothesis reinforced from a second data type) — paper-strength finding worth a PLAN_25b experiment after main-table assembly. PLAN_24 recommendation = UJI K=1 degenerate row (final main-table row before SUMMARY assembly; expected: reuse RESULT_01 numbers, no new training needed). |

| 24 | PLAN_24_main-results-uji-k1-degenerate.md | RESULT_24_main-results-uji-k1-degenerate.md | iter 24: main-results-uji-k1-degenerate (5293ad2) | Final main-table row before SUMMARY. UJI K=1 + M=1 degenerate single-modality WiFi per-scan. SOTA cells reused (wlan_localization 15.17 / Anchor2Vec 8.69 from RESULT_01). Engineer: wrote `scripts/_train_uji_arch.py` thin runner (not via FusionTrainer; Anchor2Vec → unsqueeze → aggregator at S=1 → squeeze → Linear(2); 120 epochs B=256 lr=1e-3 Huber δ=1.0; ~3 min training each). **Outcome α7 CONFIRMED**: CNN1D val **8.723 m** (+0.4% vs Anchor2Vec 8.69 = essentially identical), LSTM-attn val **8.426 m** (−3.0% vs Anchor2Vec, marginally beats). Both BEAT wlanloc SOTA 15.17 by 43-45%. At K=1 M=1 the aggregators (Conv1d kernel-3 over length-1 sequence; BiLSTM cell over single step) have no temporal/cross-modal axis to operate on — they re-parameterise the head with a few % of noise. **WiFi encoder Anchor2Vec is the load-bearing component on per-scan WiFi-only data.** Per-scan distribution (val n=1111): CNN1D median 6.40 / p90 18.3 / max 80.9; LSTM-attn median 5.89 / p90 18.9 / max 188.6 (heavier tail but lower median). No per-trajectory smoothness — UJI is per-scan, criterion (d) gate structurally N/A. **Main-results table all 6 rows populated**: Webots (RESULT_17 winner), IMUWiFine (R_19 β5), IPIN floor 0 (R_22 β5), RoNIN canonical (R_23 β6), TartanAir hospital (R_08 paper-soft, encoder only), UJI (this iter α7). Plus 4-arch Webots bake-off (R_16/17/18/21). PLAN_25 recommendation = SUMMARY + table assembly. Open scientist question: UJI in main table (a) or split to per-scan-encoder appendix (b)? |

| 25 | PLAN_25_summary-main-table-assembly.md (final scientist deliverable per PROTOCOL.md final-stop routine) | RESULT_25_summary-main-table-assembly.md + handoff/SUMMARY.md (NEW) | iter 25: summary-main-table-assembly (796be3c) | Final scientist deliverable; pure documentation. Engineer: cross-checked all 24 RESULTs; wrote 6-row main-results table (Webots K=4 4-mod / IMUWiFine / IPIN floor 0 / RoNIN canonical / TartanAir hospital / UJI per-scan) with per-leg SOTAs measured fresh where applicable + 3 our-architectures (incumbent + CNN1D winner + LSTM-attn runner-up + MoTTransformer γ5 negative). 5 criteria status: (a) partial — C1 ✓ / C2 partial / Camera paper-soft / Odom internal; (b) ✓ cleared by 32% (CNN1D test 0.339 m); (c) partial — gate-2 ✓ cleanly / gate-1 partial; (d) smoothness UNMET across 4 archs × 5+ datasets (r ≤ 0.10, loss-function-bound); (e) ✓✓ b=1 4.73 ms (21× under) / b=32 0.15 ms (660× under). 4 supporting claims C1-C4 labeled. **5 cross-cutting findings written as discussion paragraphs**: LSTM-attn dead-reckoning regime 3-dataset structural confirmation; smoothness debt architecture-invariant (hypothesis falsified); RoNIN RTE-to-ATE asymmetry as same loss signal; three distinct fusion regimes (cooperative/dead-reckoning/WiFi-anchored); cross-dataset per-leg WiFi competitiveness. 6 paper-framing decisions surfaced for Mohamed (IMUWiFine test column / C2 framing / MSILN narrative / smoothness gap / UJI in main table or appendix / latency methodology footnote). 6 next steps queued: **PLAN_25b loss-function lever (~30 min, highest value)**; MSILN re-run with CNN1D+Anchor2Vec; Camera ext-SOTA full benchmark; Conformal coverage; pre-submission cleanup; MoTTransformer γ5 attribution. **Engineer's verdict: GOAL_REACHED = true with documented limitations** — the limitations (C2 raw gap, Camera paper-soft, smoothness debt, IMUWiFine campaign-split, IPIN small-data overfit, MoTTransformer γ5 negative) are *part of the contribution*, not failures. The PerCom 2026 paper has: clean main table + 3 paper-grade structural findings + 4-arch bake-off methodology + honest gap inventory pointing to single follow-up experiment. Run-2 archive paper-ready in shape if not yet in prose. |

| 26 | PLAN_26_external-methods-relocation-and-baselines.md (post-run-2 consolidation foundation per user directive 2026-05-26 ~12:30) | RESULT_26_external-methods-relocation-and-baselines.md | iter 26: external-methods-relocation-and-baselines (b4c2e91) | Foundational consolidation iter; 4 SOTA repos relocated to `external_methods/` as Git submodules pinned to the commits that produced RESULT_01-24. Engineer: Step 0 inventory completed — wlan_localization `5e1949da` (master), ronin `805b7f0f` (master), tartanvo `ec2ecc38` (python3), DPVO main HEAD. MSILN starter NOT added (data-only ~2.1 GB, documented as data-only reference). Step 1 4× `git submodule add` + pin via `git checkout <hash>`. Step 2 file presence verified for all required imports. Step 3 created `src/pipeline/baselines/` (7 files, ~430 lines) wrapping each baseline + 4 idempotent compat shims (`apply_np_int_shim`, `apply_scipy_as_dcm_shim`, `apply_numpy_linalg_submodule_shim`, `apply_cupy_compat_shim`) — all monkey-patch our process, NEVER edit vendored sources (Demand #3 preserved). Smoke-imported package: `load_position_regressor()`, `ResNet1D`, `load_test_list()` (returns 32 canonical seqs), `BasicEncoder4` (181 k params). Step 4 migrated 12 wrappers + 1 src file (~250 lines of boilerplate deleted). Step 5 **verification: `eval_wlanloc_uji.py` reproduces 15.171 m exactly = RESULT_01's 15.17 m, 0 % drift**. Step 6 documentation: NEW `docs/EXTERNAL_DEPENDENCIES.md` (~140 lines, per-submodule URL/commit/license/loader/used-by-RESULT; compat shim table; reproducibility check), updated `README.md` setup Step 1 to include `git submodule update --init --recursive`, added 7th Critical Rule to `CLAUDE.md` naming the baselines package + external_methods location. Foundation for PLAN_27-30 in place. One open Q: legacy `external/dpvo/` partial vendoring still exists alongside new submodule — recommend delete in PLAN_27 when notebook exercises the new path. |

| 27 | PLAN_27_data-factory-and-visualization.md | RESULT_27_data-factory-and-visualization.md | iter 27: data-factory-and-visualization (d5e9fc7) | Second consolidation iter. Built **7 per-dataset modules** under `src/pipeline/data/` (`webots`/`msiln`/`imuwifine`/`ipin2024`/`ronin_canonical`/`tartanair`/`uji`) each exposing the SAME 3-function API. **All 7 surface honest `known_caveats`** in `stats()` (IMUWiFine test no IMU per RESULT_20; IPIN small-train per RESULT_22; MSILN path-130 composition per RESULT_15; RoNIN raw +94% gate per RESULT_07; TartanAir image-only per RESULT_08; UJI K=1 M=1 degenerate per RESULT_24; Webots GPR-synthesised per CLAUDE.md). Dispatcher `factory.py` registers all 7; `src.pipeline.data.{list_datasets, load_dataset, dataset_stats, preprocessing_demo}` exported. NEW `src/pipeline/visualization/` package (6 plotters: `plot_dataset_overview`/`plot_per_trajectory`/`plot_staleness_curve`/`plot_subset_eval_bar`/`plot_main_results_heatmap`/`plot_preprocessing_demo`) + `_style.py` with stable per-method color palette. **Verification**: `_smoke_data_visualization.py` runs all 7 datasets in ~13 s, saves **19 PNGs** under `runs/overnight/run2_iter_27/dataset_overviews/` (7 overviews + 12 modality preprocessing demos); 7/7 pass. NEW `docs/DATA_AND_VISUALIZATION.md`; `CLAUDE.md` Pipeline Architecture table extended with `ext`/`data`/`viz` rows. Open Q for scientist: Webots Camera preprocessing_demo returns text-only by default (engineer recommendation: keep simple; richer DPVO-trunk activation visualisation is a PLAN_30 notebook-polish item). |

| 28 | PLAN_28_fusion-encoders-training-consolidation.md | RESULT_28_fusion-encoders-training-consolidation.md | iter 28: fusion-encoders-training-consolidation (2ed396f) | Third consolidation iter. Engineer: (1) `src/pipeline/fusion/__init__.py` extended — `build_arch(name, encoders=None, dataset='simulation', **overrides)`, `list_archs()` returns 5 names, `DEFAULT_CONFIG` (K=4/M_max=4/D=128/modality_dropout=0.4/instant_dropout=0.45); module docstring rewritten with run-2 verdict table. (2) `bakeoff.py` docstring rewritten as paper-methods-section material (4-arch params/val/test/smoothness/latency + design rationale per aggregator + smoothness-debt falsification finding). (3) **Encoder demo_forward methods**: Anchor2Vec (anchor attention weights as `intermediate`), IMUCNN (conv stack pre-pooling activations), OdomCNN (same), DPVOMotionEncoder (per-patch tokens (B, n_patches, 132)) — each returns `{raw, preprocessed, intermediate, encoded, description}`. WiFiSetTransformer + VisionViT deferred (parked / legacy). (4) **3 NEW `FusionTrainer` methods + module-level helper**: `compute_per_trajectory_smoothness(split)` Pearson r per path; `latency_probe(batch_sizes, n_trials)` 100-trial wall-clock; `load_trained(checkpoint_dir, arch, dataset)` rebuilds + loads `model.pt` ready-to-evaluate. (5) **Smoke `scripts/_smoke_fusion_consolidation.py`**: archs 5/5 build (incumbent 1.55M / cnn1d 0.51M / lstm_attn 0.57M / tcn 0.51M / mot_transformer 0.74M — exact match RESULT_17/21); 3/3 encoder demos OK; load_trained on RESULT_17 CNN1D winner reproduces test 0.341 m vs RESULT_17 0.339 m = **+0.6 % drift (within tolerance)**; latency b=1 4.729 ms = exact match RESULT_18; smoothness median r=0.012 ≈ RESULT_18 0.009. **One open item**: val MAE drift +4.6 % from `torch.manual_seed(42)` re-init affecting vision-token extraction (test column unaffected — paper claim safe); deferred to PLAN_29 (snapshot vision tokens OR reorder seed before extract_vision_tokens). |

(Both sides update this table — append a row when you finish your half.)
