# Run 2 — Coordination State

Started: Started: 2026-05-25 <12:30> local
Stop at: 2026-05-26 18:00 local
Branch: overnight-autonomous-run2-2026-05-25
Push policy: **commit locally each iteration; NO push. User pushes
              manually on wake.**

Run 1 archived at `handoff/archive/run1/` — read its `README.md` for
the autopsy.

## Status

- `CURRENT_ITERATION:` 3  (RESULT_03 committed; awaiting PLAN_04)
- `LAST_PLAN:` PLAN_03_camera-encoder-audit-webots.md (2026-05-25 ~12:55 local; first plan under amended rubric)
- `LAST_RESULT:` RESULT_03_camera-encoder-audit-webots.md (2026-05-25 ~13:50 local)
- `GOAL_REACHED:` false
- `STOP_REASON:` (none yet)

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

**Phase A — Encoder audit (PLAN_01 → PLAN_05; PLAN_05 added under
amended-rubric review of RESULT_02)**
- 01: WiFi encoder audit vs `wlan_localization` on UJIIndoorLoc ✓
  (Locaris removed — Sachini/niloc is NILoc IMU, not WiFi)
- 02: IMU encoder audit on a000 proxy (Branch Y) ✓ — C2 not discharged
- 03: Camera encoder audit (DPVO motion) on Webots sim
- 04: Odom encoder internal audit on Webots sim
- 05: C2 closure — fetch FRDR RoNIN dataset + canonical
  unseen-subjects re-eval with Umeyama alignment

**Phase B — Fusion redesign (PLAN_06 → PLAN_08ish)**
- Small-bake-off iteration: transformer / TCN / LSTM-attn / late+gate
  on 10 % Webots subset. Then commit to one + iterate.

**Phase C — Validation + ablations (PLAN_09+)**
- Full 4-modality fusion on Webots sim (C3).
- Per-modality subset eval (`only:X`, `drop:X`).
- Cross-session real-world subset on Microsoft ILN 2.0 (C4).
- Conformal coverage, per-trajectory plots, latency.

## Iteration log

| # | plan file | result file | engineer commit | scientist note |
|---|---|---|---|---|
| 01 | PLAN_01_wifi-encoder-audit-uji.md (revised 2026-05-25 scientist first wake) | RESULT_01_wifi-encoder-audit-uji.md | iter 01: wifi-encoder-audit-uji (baf1a61) | Step 0 (recover run-1 audit files) added — files exist only on `overnight-autonomous-2026-05-24`. Locaris bonus dropped (Sachini/niloc = NILoc IMU, not WiFi; real Locaris arXiv:2510.11926 no code yet). Engineer: Anchor2Vec **keep** (8.69 m, +1.6 % vs run-1 ref), WiFiSetTransformer **replace on UJI / defer cross-session to Phase C** (12.95 m, +49 % vs Anchor2Vec). Recommend PLAN_02 = IMU audit, no parallel WiFi track. |
| 02 | PLAN_02_imu-encoder-audit-ronin.md | RESULT_02_imu-encoder-audit-ronin.md | iter 02: imu-encoder-audit-ronin (f494a35) | Branch X (canonical unseen-subjects) preferred; Branch Y (a000 intra-session proxy) fallback if full RoNIN data missing locally (only `data/ronin_a000` confirmed present). Three orthogonal probes for bottleneck: architecture (RoNIN ResNet1D) + capacity (IMUCNN 2× width) + preprocessing (run-1 disaster fix). Engineer: **Branch Y used (full RoNIN data not on machine)**. IMUCNN = **keep** (aligned ATE 1.04 m vs ResNet1D 0.97 m, +7 %; raw 3.55 m vs 2.89 m, +23 % borderline; 95× smaller, 4× faster). Capacity probe **refuted** modify hypothesis (2× width raw ATE 5.81 m, +63 %). C2 NOT discharged — queued as PLAN_05 (locked before Phase B). Verdict subject to Umeyama re-alignment (PLAN_03 Step 0c addendum). |
| 03 | PLAN_03_camera-encoder-audit-webots.md (first plan under amended rubric: multi-condition validation, preprocessing as first-class variable, Umeyama for any alignment, raw weighted ≥ aligned) | RESULT_03_camera-encoder-audit-webots.md | iter 03: camera-encoder-audit-webots (pending hash) | Branch P (DPVO importable) preferred; Branch Q (motion-encoder-only fallback). Step 0c retrofits RESULT_02 with Umeyama alignment. Within-sim val→test gap < 20 % is the multi-condition gate (Camera = sim-only by project design). Engineer: **Branch Q for DPVO SLAM (no lietorch/altcorr); Branch P for motion encoder.** DPVOMotionEncoder = **keep** with P-A preprocessing (val 1.85 m, test 1.56 m on canonical CLAUDE.md split; test-val gap −15.7 % = no overfitting). Preprocessing-variation probe: P-A beats P-B by 9 %. Capacity probe (stride 10): neutral. Honest weakness: per-traj smoothness median r ≈ 0.07 (poor). Step 0c Umeyama retro on IMU = all three encoders collapse to ~0.30 m Umeyama-aligned ATE; IMUCNN-keep stands; capacity-refuted claim softened to "not clearly the bottleneck." Recommend PLAN_04 = Odom audit. |

(Both sides update this table — append a row when you finish your half.)
