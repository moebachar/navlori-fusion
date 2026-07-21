# Plan 09 — Phase B: add Camera (DPVOMotion-P-A) as 3rd modality on Webots

> Phase A close-out is complete (PLAN_07 ✓ + PLAN_08 ✓). C2 NOT
> discharged on canonical RoNIN (gap +93.8 % raw vs ResNet1D);
> Camera per-leg paper-soft on TartanAir hospital (TartanVO 0.518 m
> vs our encoder 0.293 m on the same in-sequence test slice — gap is
> 24×, fit-for-purpose as a fusion encoder only). Phase B resumes
> here per STATE.md's locked roadmap.

## Hypothesis

RESULT_06 reproduced the run-1 2-modality (WiFi + IMU) Webots
baseline cleanly: **val MAE 0.469 m / test 0.517 m**, IMU adds
+6.6 % val / +1.3 % test over WiFi-only. CLAUDE.md's pre-run-1
single-instant K=1 baseline ≈ 0.43 m on Webots — that number was on
the 2-modality (WiFi + IMU) configuration before vision was wired
in. **Adding Camera (DPVOMotion-P-A) as the 3rd modality is the
next minimal step on the C3 critical path.**

Expected outcomes:
- **C3 lower bound passes** (criterion (b) ≤ 0.50 m test MAE) —
  Camera adds an additional 5–10 % improvement on top of the
  RESULT_06 baseline. Headline becomes "3-modality fusion clears
  C3 with margin"; PLAN_10 adds Odom 1.5-modality for the
  4-modality C3 claim.
- **Camera contributes net-noise or marginal value** — RESULT_03's
  "smoothness debt" (Pearson r ≈ 0.07 on motion-magnitude) leaks
  through the temporal axis. Triggers a smoothness debug as
  PLAN_09b, OR validates the RESULT_03 follow-up entry (B-3) "let
  fusion transformer absorb noise" with concrete evidence.

This is the smallest viable Phase B continuation; one focused
experiment.

## Steps

### Step 0 — Config + smoke (5–10 min)

`configs/stage_c/fusion.yaml` (restored in PLAN_06) currently lists
WiFi + IMU. Engineer updates the modality list to include camera:

```yaml
modalities: [wifi, imu, camera]   # add camera; rest unchanged
```

`configs/data/simulation.yaml` should already have the camera
fields (`camera_csv: camera.csv`, `camera_dir: camera/`,
`camera_stride: 5`, `camera_window: 2`) from run-1's
async_collection layout — verify by reading the file.

Verify the builder finds the DPVOMotionEncoder via the registry —
`src/pipeline/encoders/__init__.py` was extended in RESULT_06 Step 0
to export `DPVOMotionEncoder`. `builder.build_encoders(cfg)`
should construct it when `modalities: camera` is added.

If `configs/data/simulation.yaml` is missing camera fields, restore
from run-1 the same way as PLAN_06 Step 0:

```powershell
git diff overnight-autonomous-2026-05-24..HEAD -- configs/data/simulation.yaml
# inspect the camera-specific lines and add only those
```

DO NOT just blanket-checkout the run-1 version — that could revert
PLAN_06's WiFi+IMU baseline config.

**Smoke**: `scripts/_smoke_fusion.py --phase 1` runs a single forward
with 3 modalities; no NaN, shape-check passes.

**Acceptance**: smoke passes; the camera encoder is constructed at
build time (engineer prints `len(encs)` or similar = 3).

### Step 1 — Pre-test gate (10 % Webots train, 5 epochs)

Same as PLAN_06's pre-test pattern: 10 % of train paths (~1 path of
the 11), 5 epochs, full 3-modality config. Acceptance: val MAE
drops ≥ 10 % across 5 epochs OR clearly descending. RESULT_06's
pre-test on WiFi+IMU dropped from 5.232 → 0.719 m in 5 epochs;
3-modality should pre-test similarly fast because two of the three
modalities have pre-tested already.

**Memory budget check**: forward+backward at full batch (B=32 or
config default), 3 modalities, K=1. Peak GPU MB reported; must be
< 6 GB. The DPVO trunk is frozen and takes ~300 MB peak per
RESULT_03; total should still fit comfortably.

If pre-test FAILS (loss diverges, NaN, or no descent), STOP and
write a partial RESULT documenting the failure mode — do not promote
to full training.

### Step 2 — Full 3-modality training

Same protocol as PLAN_06:
- 90 epochs, patience 25, AdamW + OneCycleLR + Huber(δ=0.5).
- modality_dropout 0.4, instant_dropout 0.45 (config defaults).
- K=1 single-instant.

Run via the builder pattern from RESULT_06 (engineer's wrapper
script if any, else inline builder call).

**Acceptance**: training completes; val MAE + test MAE reported
with per-path distribution (criterion (d)).

### Step 3 — Compare to RESULT_06 baseline

| config | val MAE | test MAE | best epoch | latency (ms/sample) |
|---|---|---|---|---|
| WiFi+IMU (RESULT_06 baseline) | 0.469 m | 0.517 m | 76 | 0.044 |
| **WiFi+IMU+Camera (this iter)** | **?** | **?** | ? | ? (DPVO trunk adds ~10 ms) |

**Acceptance** (raw-weighted per amended-rubric correction #3):
- ANY improvement in test MAE → Camera contributes.
- Test MAE within ±5 % of baseline → Camera contributes marginally;
  C3 still cleared via the 2-modality floor + Camera-as-redundancy.
- Test MAE regresses > +5 % over baseline → Camera adds noise;
  RESULT_03's smoothness debt is leaking through. PLAN_09b
  becomes a smoothness-debug iteration (auxiliary velocity loss
  or EMA on camera tokens — B-1/B-2 from RESULT_03 addendum).

### Step 4 — Per-modality subset eval

Run `FusionTrainer.evaluate_all_subsets` on the best-val checkpoint:

| subset | val MAE | test MAE | Δ vs full-fusion (val) | Δ vs full-fusion (test) |
|---|---|---|---|---|
| `only:wifi` | … | … | … | … |
| `only:imu` | … | … | … | … |
| `only:camera` | … | … | … | … |
| `drop:wifi` | … | … | … | … |
| `drop:imu` | … | … | … | … |
| `drop:camera` | … | … | … | … |
| **`wifi+imu+camera`** | **…** | **…** | — | — |

Two diagnostic signals:
1. **`drop:camera` close to full-fusion** → Camera is net-redundant
   (the other two carry the signal). The 4-modality story still
   benefits because Camera adds robustness when WiFi is stale.
2. **`drop:camera` worse than full-fusion** → Camera is genuinely
   contributing.
3. **`only:camera`** measures Camera's standalone fusion-input
   quality. RESULT_03's 1.56 m test linear-probe MAE is the
   reference — expect `only:camera` to be in that ballpark.

### Step 5 — Per-trajectory smoothness check (criterion (d), RESULT_03 debt)

Compute per-test-path **per-trajectory smoothness ratio** (Pearson r
between ‖Δpredᵢ‖ and ‖Δgtᵢ‖). RESULT_06 didn't report this for
WiFi+IMU; this iteration starts the practice (the locked Phase B
gate from RESULT_05's smoothness-debt reframe says "Phase B
bake-off must report per-modality smoothness in every 4-modality
test run").

Save per-trajectory plots for test paths 15/16/17 under
`runs/overnight/run2_iter_09/test_paths/`.

If the 3-modality smoothness r is materially worse than RESULT_06's
WiFi+IMU smoothness (compute that retro from RESULT_06's saved
checkpoint if possible — same canonical test paths), that's the
RESULT_03 debt leaking. Flag explicitly in the RESULT.

### Step 6 — Decision + PLAN_10 recommendation

Three-sentence verdict:
- Does 3-modality clear C3 lower bound (test MAE ≤ 0.50 m)?
- Camera contribution: net-positive / net-noise / marginal — quote
  numbers from Step 4's `drop:camera` row.
- PLAN_10 = add Odom 1.5-modality path (OdomCNN-P-B embedding +
  raw integrated `odom_x, odom_y` for smoothness per RESULT_04
  finding) — confirm, OR if Step 5 surfaces a smoothness regression,
  insert a PLAN_09b smoothness debug first.

## Sources

- RESULT_06: WiFi+IMU K=1 baseline val 0.469 / test 0.517 m,
  latency 0.044 ms/sample, IMU net-positive +6.6/+1.3 %.
- RESULT_03: DPVOMotionEncoder P-A 1.85 m val / 1.56 m test on
  Webots linear-probe; smoothness debt r ≈ 0.07.
- RESULT_05: smoothness-debt reframe; Phase B follow-ups B-1/B-2/B-3.
- CLAUDE.md "Universal token: encoder_embedding + modality_embedding
  + time_encoding(Δt)" — the fusion contract that handles the WiFi
  1 Hz / IMU 31 Hz / Camera 5 Hz rate mismatch.
- `configs/stage_c/fusion.yaml`, `configs/data/simulation.yaml`,
  `src/pipeline/fusion/{builder.py, transformer.py}`,
  `src/pipeline/training/fusion_trainer.py` — all restored in
  RESULT_06.
- `src/pipeline/encoders/dpvo_motion.py` — restored in RESULT_03 + 06.
- `runs/_weights/dpvo.pth` — restored in RESULT_03.

## What to report back

In `handoff/results/RESULT_09_phase-b-add-camera.md`:

1. **Step 0** — config diff (modality list, any camera-related
   config restoration).
2. **Step 1** — pre-test gate outcome + memory budget peak.
3. **Step 2** — val + test MAE, best epoch, wall, params, latency.
4. **Step 3** — comparison table vs RESULT_06; verdict on Camera
   contribution.
5. **Step 4** — full subset-eval table (6 subset rows + full-fusion).
6. **Step 5** — per-test-path smoothness ratio + plots.
7. **Step 6** — decision + PLAN_10 recommendation.
8. **One open question** for scientist.

## Reversibility

- Step 0 (config edit): permanent (engineer commits the modality
  list change with the result). Reversible by `git revert`.
- Step 2 (training): throwaway model checkpoint under
  `runs/overnight/run2_iter_09/` (gitignored).
- Steps 3–6: documentation.

Files committed: RESULT_09, config change (`configs/stage_c/fusion.yaml`
modality list), any small wrapper script changes.

**Demand #3** — no vendored sources touched.

**Compute budget**: ≤ 60 min.
- Step 0: 10 min (config + smoke).
- Step 1: 5 min (pre-test).
- Step 2: 25 min (90 epochs; DPVO trunk is frozen, only the head
  trains, so this is closer to RESULT_06's 217 s than to a full
  4-modality train).
- Step 3: 5 min (compare).
- Step 4: 5 min (subset eval, no retraining).
- Step 5: 5 min (smoothness + plots).
- Step 6: 5 min writeup.

If overrun: cut Step 5 last (it's the new gate — keep it). Cut
Step 4 second-last (it can be deferred to PLAN_10 if needed but
loses the "Camera contribution" diagnostic).

If Step 1 pre-test FAILS (NaN, divergence), the iteration becomes
"3-modality pre-test diagnostic" — write partial RESULT with the
failure mode + the smallest reproducer; scientist plans the fix
in PLAN_09b.
