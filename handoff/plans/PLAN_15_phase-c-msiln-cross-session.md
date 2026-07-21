# Plan 15 — Phase C kickoff: MSILN site1/B1 cross-session (criterion (c) / C4)

> Phase B is closed (RESULT_14). The Phase B winner is K=4 + 4-mod
> + B=128 on Webots Tiago sim with val 0.394 / test 0.417 m, latency
> < 1 ms/sample. Phase C opens with **C4** — cross-session real-world
> plausibility on Microsoft ILN 2.0 site1/B1.

## Hypothesis

Run-1 archived findings (`handoff/archive/run1/`) show this exact
data is hard: WiFi-kNN test 9.5 m, run-1 fusion (WiFi+IMU K=8) test
9.0 m — run-1 cleared "≥ 1.5 m beat over kNN" on val (17.7→15.7,
2.0 m) but NOT on test (9.5→9.0, 0.5 m). Run-1 also never ran any
open-source SOTA on MSILN — that gap is the run-2 priority.

Criterion (c) — STATE.md gate:
1. **Beat WiFi-kNN by ≥ 1.5 m** on the cross-session metric.
2. **Beat the open-source SOTA (wlan_localization / CNNLoc) by ≥ 0.5 m**
   on the same data.

Three outcomes possible:
- **(α) Both gates clear** → C4 discharged; PLAN_16 = Phase C
  continuation (conformal coverage or per-modality robustness on
  MSILN).
- **(β) WiFi-kNN gate clears, open-source-SOTA gate fails** → C4
  partial; honest paper claim. PLAN_16 explores whether stronger
  WiFi encoder (the run-1 `WiFiSetTransformer`, which RESULT_01
  parked as "replace on UJI / defer cross-session") helps.
- **(γ) Both gates fail** → C4 not discharged; the cross-session
  WiFi anchor is a fundamental limit for the architecture. Paper
  claim shifts to "Webots-sim only + RoNIN partial-C2 + Camera
  paper-soft" — still publishable but the run-2 thesis loses its
  real-world tier.

This is the first cross-session real-world iteration — Phase C
kickoff. One focused experiment.

## Steps

### Step 0 — Verify MSILN integration + Phase B winner ports (10 min)

**Step 0a — data + config presence.** The run-1 archive notes
`scripts/convert_msiln.py`, `configs/data/msiln_site1_b1.yaml`,
`data/msiln_site1_b1/` (untracked, ~133 traces with Nov-train /
Dec-test cross-session split) were preserved. Verify on this branch:

```powershell
ls data/msiln_site1_b1/ | head -5
cat configs/data/msiln_site1_b1.yaml
ls scripts/convert_msiln.py
```

If `configs/data/msiln_site1_b1.yaml` is missing, restore from
run-1:

```powershell
git checkout overnight-autonomous-2026-05-24 -- configs/data/msiln_site1_b1.yaml scripts/convert_msiln.py
```

If `data/msiln_site1_b1/` is missing on disk: that's a feasibility
blocker. Document and STOP — Phase C MSILN can't run without the
converted traces. Run-1's note says they were 2.1 GB; engineer can
re-run `scripts/convert_msiln.py` if the raw MSILN starter is still
at `C:\Users\FabLab\AppData\Local\Temp\msiln20\` (per
SCIENTIST_BRIEF section 6).

**Step 0b — Phase B winner config adaptation.** MSILN is
**WiFi+IMU 2-modality** (no Camera, no Odom on real smartphone
data). The Phase B winner config (K=4 + 4-mod + B=128) needs a
2-modality variant. The simplest path: edit the wrapper or config
to set `modalities: [wifi, imu]` for the MSILN runner, keeping
K=4 + B=128 + same dropout / lr / Huber loss.

The model architecture stays identical — only the encoder set and
modality registry shrinks. `FusionTransformer` handles this via
`build_encoders` returning a 2-encoder list.

Verify the dataset config `configs/data/msiln_site1_b1.yaml`
exposes both `wifi` and `imu` modality keys.

**Acceptance**: data + config present; 2-modality builder builds 2
encoders.

### Step 1 — WiFi-kNN baseline (criterion-(c) lower anchor, 5 min)

Run-1's `runs/baselines/msiln_site1_b1/` may still contain the
WiFi-kNN baseline (run-1 archive `RESULT_02_msiln-convert-and-baselines.md`
reported 17.7 val / 9.5 test). Check:

```powershell
ls runs/baselines/msiln_site1_b1/
cat runs/baselines/msiln_site1_b1/baselines.json
```

If the file exists with WiFi-kNN numbers, reuse. Otherwise re-run
the kNN baseline using the existing run-1 helper (`scripts/baselines.py`
if restored, else a thin sklearn wrapper).

**Acceptance**: WiFi-kNN val + test MAE recorded.

### Step 2 — Day-1 SOTA reproduction: `wlan_localization` on MSILN cross-session (15 min)

PLAN_01's restored `scripts/eval_wlanloc_uji.py` is UJI-specific.
For MSILN, write a small companion `scripts/_eval_wlanloc_msiln.py`
that:

1. Reads MSILN train/test CSVs (per `configs/data/msiln_site1_b1.yaml`
   split keys).
2. Uses the SAME vendored `PositionRegressor` + `DataPreprocessor`
   from `C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src`,
   loaded via the same `importlib` shim as PLAN_01's script
   (Demand #3 honoured — no edits to vendored).
3. Treats MSILN as a single-floor regression (the
   building/floor cascade isn't applicable — site1 is one site).
4. Reports val + test mean Euclidean error.

**Acceptance**: open-source SOTA val + test MAE on MSILN
cross-session reported. This is the **new measurement** run-1 never
made.

If wlan_localization's preprocessor (Box-Cox + PCA) blows up on
MSILN's RSSI distribution, document the obstacle and fall back to
**CNNLoc-from-scratch** (`scripts/eval_cnnloc_uji.py` style;
retired for UJI but might be the only working open-source baseline
for MSILN). If both fail: criterion (c) gate 2 is structurally
unmet; document and ship with the WiFi-kNN comparison only.

### Step 3 — Train Phase B winner architecture (K=4 + 2-mod + B=128) on MSILN

Same protocol as RESULT_13 / RESULT_14 — only the dataset config
changes. Single training run; 90 epochs; AdamW + OneCycleLR +
Huber(δ=0.5); `instant_dropout=0.45`, `modality_dropout=0.4` (for
2-modality this means dropping one modality at random in 40 % of
windows — equivalent to "only:wifi" or "only:imu" eval-time).

**Pre-test gate**: same — 5 epochs on 10 % MSILN train, val MAE
drops ≥ 10 % OR clear descent.

**Memory budget**: K=4 2-mod B=128 should peak at ~200 MB (smaller
than Webots's 4-mod 466 MB). Confirm.

**Acceptance**: training completes; val + test MAE + per-path
distribution reported.

### Step 4 — Compare against gates

| metric | value | gate | passes? |
|---|---|---|---|
| WiFi-kNN test MAE | … (from Step 1) | — | — |
| `wlan_localization` test MAE | … (from Step 2) | — | — |
| **Phase B winner test MAE** | … (from Step 3) | — | — |
| **Gate (c)-1**: ours beats kNN by ≥ 1.5 m | Δ = kNN − ours | ≥ 1.5 m | ? |
| **Gate (c)-2**: ours beats SOTA by ≥ 0.5 m | Δ = SOTA − ours | ≥ 0.5 m | ? |

**Acceptance**: outcome label (α / β / γ); explicit verdict on
criterion (c).

### Step 5 — Per-trajectory smoothness + per-path distribution (criterion (d))

Same gate per RESULT_05 lock. Median Pearson r across the longest
test trajectories. MSILN traces are longer than Webots paths (per
run-1's autopsy, multi-minute sessions); the per-trajectory
smoothness should be informative.

Per-path table with mean / p50 / p90 / max for the top 5 longest
test trajectories. Per-trajectory plots saved under
`runs/overnight/run2_iter_15/test_paths/`.

### Step 6 — Decision + PLAN_16 recommendation

Three-sentence verdict:
- Criterion (c) status (α / β / γ); quote both gate deltas.
- Smoothness + per-path findings worth flagging for Phase C
  continuation.
- PLAN_16 recommendation:
  - **(α)** PLAN_16 = conformal coverage on the trained model
    (criterion-(d)-style uncertainty quantification, run-1
    archive notes `src/pipeline/uncertainty/conformal.py` exists).
  - **(β)** PLAN_16 = WiFiSetTransformer cross-session re-eval on
    MSILN (the parked "replace on UJI / defer cross-session"
    verdict from RESULT_01).
  - **(γ)** PLAN_16 = honest write-up + run-2 SUMMARY.md draft;
    the paper claim shifts to "Webots-sim + paper-soft Camera +
    in-domain IMU" without C4.

## Sources

- Run-1 MSILN integration (preserved per `handoff/archive/run1/README.md`):
  `scripts/convert_msiln.py`, `configs/data/msiln_site1_b1.yaml`,
  `data/msiln_site1_b1/`, `runs/baselines/msiln_site1_b1/`.
- Run-1 MSILN baselines (per `handoff/archive/run1/README.md`):
  WiFi-kNN test 9.5 m; FusionTransformer (Anchor2Vec, K=8) test 9.0 m.
- RESULT_01 `wlan_localization` reproduction on UJI: 15.17 m global
  (8.97 % off run-1 ref). MSILN-specific number is the new
  measurement.
- RESULT_14: Phase B winner — K=4 + 4-mod + B=128 val 0.394 / test
  0.417 on Webots.
- `src/pipeline/uncertainty/conformal.py` (restored RESULT_06).
- `src/pipeline/encoders/wifi_set.py` (restored RESULT_01) — the
  "deferred cross-session" WiFi encoder for PLAN_16(β) branch.

## What to report back

In `handoff/results/RESULT_15_phase-c-msiln-cross-session.md`:

1. **Step 0** — data + config presence; what was restored.
2. **Step 1** — WiFi-kNN baseline.
3. **Step 2** — `wlan_localization` MSILN cross-session — the new
   SOTA number.
4. **Step 3** — Phase B winner trained on MSILN val + test MAE.
5. **Step 4** — gate (c)-1 and gate (c)-2 status table.
6. **Step 5** — per-trajectory smoothness + per-path table + plots.
7. **Step 6** — outcome label + PLAN_16 recommendation.
8. **One open question** for scientist.

## Reversibility

- Step 0: file restores permanent. Engineer commits.
- Step 1: throwaway eval if re-run.
- Step 2: NEW file `scripts/_eval_wlanloc_msiln.py` — permanent.
- Step 3: throwaway checkpoint under `runs/overnight/run2_iter_15/`.
- Steps 4–6: documentation.

Files committed: RESULT_15, `scripts/_eval_wlanloc_msiln.py`, any
restored MSILN-config files.

**Demand #3**: vendored `wlan_localization` source untouched.

**Compute budget**: ≤ 75 min.
- Step 0: 10 min.
- Step 1: 5 min (or instant if already cached).
- Step 2: 15 min (eval-only; `wlan_localization` is non-parametric
  per RESULT_01).
- Step 3: 25 min (90-epoch training; smaller dataset than Webots).
- Step 4: 5 min.
- Step 5: 5 min.
- Step 6: 10 min writeup.

If overrun: drop Step 5 to per-path table only (no plots) — keep
the gate-(c) verdict.

If outcome (γ) fires (both gates fail), prioritise an honest
diagnostic in the RESULT — what's the failure mode (WiFi
saturation? IMU dead-reckoning ineffective? domain shift?). That
diagnostic informs PLAN_16's framing.
