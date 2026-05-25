# Run 2 — Coordination State

Started: Started: 2026-05-25 <12:30> local
Stop at: 2026-05-26 18:00 local
Branch: overnight-autonomous-run2-2026-05-25
Push policy: **commit locally each iteration; NO push. User pushes
              manually on wake.**

Run 1 archived at `handoff/archive/run1/` — read its `README.md` for
the autopsy.

## Status

- `CURRENT_ITERATION:` 0  (no iteration started yet)
- `LAST_PLAN:` (none yet — PLAN_01 ready in handoff/plans/)
- `LAST_RESULT:` (none yet)
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

**Phase A — Encoder audit (PLAN_01 → PLAN_04)**
- 01: WiFi encoder audit vs `wlan_localization` + Locaris on UJIIndoorLoc
- 02: IMU encoder audit vs RoNIN ResNet1D on RoNIN unseen
- 03: Camera encoder audit (DPVO motion) on Webots sim
- 04: Odom encoder internal audit on Webots sim

**Phase B — Fusion redesign (PLAN_05 → PLAN_07ish)**
- Small-bake-off iteration: transformer / TCN / LSTM-attn / late+gate
  on 10 % Webots subset. Then commit to one + iterate.

**Phase C — Validation + ablations (PLAN_08+)**
- Full 4-modality fusion on Webots sim (C3).
- Per-modality subset eval (`only:X`, `drop:X`).
- Cross-session real-world subset on Microsoft ILN 2.0 (C4).
- Conformal coverage, per-trajectory plots, latency.

## Iteration log

| # | plan file | result file | engineer commit | scientist note |
|---|---|---|---|---|

(Both sides update this table — append a row when you finish your half.)
