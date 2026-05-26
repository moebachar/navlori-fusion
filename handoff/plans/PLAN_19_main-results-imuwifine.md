# Plan 19 — Main results table: IMUWiFine (CNN1D + LSTM-attn + per-leg SOTAs)

> Per `handoff/SCIENTIST_NOTE_main-results-table.md` (logged
> 2026-05-26 ~08:00 local from third-party directive). PLAN_19 is
> the FIRST iteration filling the multi-iter Phase C main-results
> table. Engineer's RESULT_18 recommended MSILN re-run instead;
> that's queued as an OPTIONAL extension after the main table is
> populated (PLAN_23+). The main table priority is unchanged.

## Hypothesis

IMUWiFine floor 4 is a real-world WiFi+IMU dataset already
integrated (per CLAUDE.md "External Datasets: IMUWiFine floor 4 —
80 paths, WiFi+IMU, `data/imuwifine_floor4/`"). It's the cleanest
2-modality cross-dataset row in the main results schema:

| arch          | IMUWiFine val | IMUWiFine test | source             |
|---------------|---------------|----------------|--------------------|
| wlan_localization | ?         | ?              | this iter (NEW)    |
| RoNIN ResNet1D    | ?         | ?              | this iter (NEW)    |
| **CNN1D**     | ?             | ?              | this iter (winner) |
| **LSTM-attn** | ?             | ?              | this iter (runner-up) |

Three outcomes:
- **(α'''') CNN1D / LSTM-attn beat BOTH per-leg SOTAs**: paper
  claim "our fusion beats both single-modality SOTAs on the same
  data" — the classic per-leg-fusion-win.
- **(β'''') Our fusion beats one SOTA but loses or ties the
  other** (typical run-1 finding: IMU dead-reckoning catastrophic
  on multi-minute paths, our fusion wins by physics; WiFi SOTA
  margin is the real test): paper documents the asymmetry
  honestly.
- **(γ'''') Our fusion is competitive with neither SOTA**:
  surprising; suggests Webots config doesn't transfer; PLAN_20
  reframes.

This is one focused experiment (the IMUWiFine row of the main
table). PLAN_20–22 follow for IPIN / RoNIN / UJI.

## Steps

### Step 0 — Data + config verification (5 min)

`data/imuwifine_floor4/` should be on disk (untracked) per
CLAUDE.md. Verify:

```powershell
ls data/imuwifine_floor4/ | head -5
cat configs/data/imuwifine.yaml
ls scripts/convert_imuwifine.py 2>$null
```

If `configs/data/imuwifine.yaml` is missing, restore from run-1:

```powershell
git checkout overnight-autonomous-2026-05-24 -- configs/data/imuwifine.yaml scripts/convert_imuwifine.py
```

**Acceptance**: data present; config restored; smoke import of
the IMUWiFine dataloader succeeds.

### Step 1 — Day-1 SOTA reproductions on IMUWiFine

NEW measurements; neither SOTA has been run on this dataset before.
Run BOTH unmodified per Demand #3:

**Step 1a — `wlan_localization` on IMUWiFine WiFi.** Write
`scripts/_eval_wlanloc_imuwifine.py` mirroring
`_eval_wlanloc_msiln.py` (created in RESULT_15) but with
IMUWiFine's CSV split. Use the SAME vendored `PositionRegressor`
+ `DataPreprocessor` from
`C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src` —
loaded via `importlib` (Demand #3 honoured). Single-floor
regression (no building/floor cascade). Report val + test mean
Euclidean error.

**Step 1b — RoNIN ResNet1D on IMUWiFine IMU.** Write
`scripts/_eval_ronin_imuwifine.py` mirroring
`scripts/eval_ronin_ipin.py` (restored RESULT_02). Train RoNIN's
ResNet1D from scratch OR load pretrained — RESULT_07 showed the
canonical-pretrained ResNet1D worked cleanly; engineer reuses
that checkpoint here if the IMUWiFine IMU stream is in the
RoNIN-compatible shape. If shapes differ, train fresh on
IMUWiFine's train split.

**Acceptance**: both SOTAs produce val + test MAE numbers on
IMUWiFine. Document any preprocessing alignment work in the
RESULT.

### Step 2 — Train CNN1D + LSTM-attn on IMUWiFine

Same protocol as RESULT_17:
- 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5), B=128, K=4.
- 2-modality: `[wifi, imu]` (IMUWiFine has no Camera/Odom).
- `instant_dropout=0.45`, `modality_dropout=0.4`.
- lr=1.3e-3.

Use the `--arch` flag introduced in RESULT_17. Run CNN1D first
then LSTM-attn (sequential).

**Pre-test gate**: 5-epoch run on 10 % subset; val MAE drops ≥
10 % OR clear descent.

**Memory budget**: < 6 GB. K=4 + 2-mod + B=128 + CNN1D should
peak ~500 MB.

**Acceptance**: both candidates trained; val + test MAE with
per-path distribution.

### Step 3 — Main-results IMUWiFine row

| method | params | val MAE | test MAE | source |
|---|---|---|---|---|
| wlan_localization (WiFi only) | — | ? | ? | step 1a |
| RoNIN ResNet1D (IMU only) | ~4.6 M | ? | ? | step 1b |
| **CNN1D (WiFi+IMU fusion)** | ~0.51 M | ? | ? | step 2 |
| **LSTM-attn (WiFi+IMU fusion)** | ~0.57 M | ? | ? | step 2 |

**Acceptance**: outcome label (α'''' / β'''' / γ'''').

### Step 4 — Smoothness + per-path distribution (criterion (d))

For each of CNN1D / LSTM-attn: per-trajectory r between
‖Δpredᵢ‖ and ‖Δgtᵢ‖; per-path test MAE distribution. The IMUWiFine
test paths may be different from Webots paths; engineer picks
top-5 longest per criterion (d) of STATE.md.

### Step 5 — Decision + PLAN_20

Three-sentence verdict:
- IMUWiFine row populated; outcome label.
- Did either of our fusion architectures clear the smoothness
  gate (r > 0.20) at this new dataset?
- PLAN_20 = IPIN 2024 floor 0 (same shape: CNN1D + LSTM-attn +
  wlan_localization + RoNIN ResNet1D).

## Sources

- `handoff/SCIENTIST_NOTE_main-results-table.md` (the directive
  scoping this multi-iter deliverable).
- `data/imuwifine_floor4/`: 80 paths, WiFi+IMU per CLAUDE.md.
- `configs/data/imuwifine.yaml`, `scripts/convert_imuwifine.py`
  (restored from run-1 in this iter if needed).
- `scripts/_eval_wlanloc_msiln.py` (RESULT_15 template).
- `scripts/eval_ronin_ipin.py` (RESULT_02 template; restored).
- `src/pipeline/fusion/{cnn1d_instants,lstm_attn}.py` (committed
  RESULT_16/17).
- `runs/overnight/run2_iter_17/<arch>/model.pt` (checkpoints for
  reference architectures).
- RESULT_07 pretrained ResNet1D checkpoint at
  `data/ronin_frdr/pretrained_resnet/...`.

## What to report back

In `handoff/results/RESULT_19_main-results-imuwifine.md`:

1. **Step 0** — data + config verification; restored files (if any).
2. **Step 1a** — wlan_localization on IMUWiFine: val + test +
   per-path distribution.
3. **Step 1b** — RoNIN ResNet1D on IMUWiFine: val + test ATE +
   per-sequence distribution.
4. **Step 2** — CNN1D + LSTM-attn training; val + test MAE +
   per-path.
5. **Step 3** — main-table row populated; outcome label.
6. **Step 4** — per-trajectory smoothness r per arch; per-path
   plots for top-5 longest test paths.
7. **Step 5** — PLAN_20 recommendation (default = IPIN 2024 floor 0).
8. **One open question** for scientist.

## Reversibility

- Step 0: file restores permanent.
- Step 1a/1b: NEW wrapper scripts permanent.
- Step 2: throwaway checkpoints under
  `runs/overnight/run2_iter_19/` (gitignored).
- Steps 3–5: documentation.

Files committed: RESULT_19, restored configs/scripts, NEW
`scripts/_eval_wlanloc_imuwifine.py` +
`scripts/_eval_ronin_imuwifine.py`.

**Demand #3**: vendored sources untouched. Compat shims in OUR
wrappers only.

**Compute budget**: ≤ 70 min.
- Step 0: 5 min.
- Step 1a: 10 min (eval-only, non-parametric per RESULT_01/15).
- Step 1b: 20 min (eval if pretrained shape-compatible / train ~30
  min if not).
- Step 2: 25 min (2 trainings sequentially, ~12-15 min each on
  2-modality data).
- Step 3: 5 min.
- Step 4: 5 min.
- Step 5: 5 min.

If overrun: cut Step 4's plots, keep the smoothness median
number. Don't skip Step 1 SOTAs — they are the load-bearing
NEW measurements for the main-results table.

If Step 0 surfaces a missing data path (IMUWiFine not on disk),
write a partial RESULT with the obstacle; either user re-stages
the data OR PLAN_20 swaps to a different dataset row.
