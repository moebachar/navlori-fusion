# Plan 22 — Main results table: IPIN 2024 floor 0 (CNN1D + LSTM-attn + per-leg SOTAs)

> Bake-off finished (RESULT_21 γ5 — MoTTransformer is the worst of
> 4; honest negative result for paper methods). Resuming the
> main-results table per
> `handoff/SCIENTIST_NOTE_main-results-table.md`.

## Hypothesis

IPIN 2024 floor 0 is one of three IPIN floors already converted
per CLAUDE.md ("IPIN 2024 Track 3 — floors -2, -1, 0, WiFi+IMU,
`data/ipin2024_floor*/`"). Per the directive, **floor 0 only**
this iteration; other floors are an optional Phase C extension.

Same shape as RESULT_19 IMUWiFine row:

| arch              | IPIN floor 0 val | IPIN floor 0 test | source         |
|-------------------|------------------|-------------------|----------------|
| wlan_localization | ?                | ?                 | this iter (NEW)|
| RoNIN ResNet1D    | ?                | ?                 | this iter (NEW)|
| **CNN1D**         | ?                | ?                 | this iter      |
| **LSTM-attn**     | ?                | ?                 | this iter      |

Three outcomes (same labels as RESULT_19, fresh data):
- **(α5) CNN1D / LSTM-attn beat BOTH per-leg SOTAs** on val (and
  ideally test).
- **(β5) Win one, lose one** — likely scenario per IMUWiFine
  precedent.
- **(γ5) Beat neither** — surprising; reframe.

Per PLAN_20 lesson: IMUWiFine's 5× val/test gap was a documented
dataset property (test format lacks IMU). **Pre-flight check** in
Step 0: does IPIN floor 0's test split include IMU? If not, the
"main table 2-modality fusion row" framing needs an asterisk
matching IMUWiFine.

## Steps

### Step 0 — Pre-flight + data + config verification (10 min)

`data/ipin2024_floor0/` should be on disk (untracked per git
status). Verify:

```powershell
ls data/ipin2024_floor0/ | head -5
cat configs/data/ipin2024_floor0.yaml
ls scripts/convert_ipin2024.py 2>$null
```

If config/converter missing, restore from run-1:

```powershell
git checkout overnight-autonomous-2026-05-24 -- configs/data/ipin2024_floor0.yaml scripts/convert_ipin2024.py
```

**Pre-flight IMU-availability check** (the lesson from PLAN_20):
inspect a few train + val + test paths' `imu.csv` (or absence):

```powershell
ls data/ipin2024_floor0/path_*/ | head -10
# pick one train path and one test path; check imu.csv presence + row count
```

**Acceptance**: data + config present; document whether IPIN
floor 0 test paths include IMU. If test paths lack IMU, the
fusion test column is WiFi-only by design (same as IMUWiFine);
note in RESULT.

### Step 1 — Day-1 per-leg SOTA reproductions on IPIN floor 0

Both NEW measurements; neither SOTA has been run on this dataset.

**Step 1a — `wlan_localization` on IPIN floor 0 WiFi.** Write
`scripts/_eval_wlanloc_ipin_floor0.py` (or reuse the IMUWiFine
runner if it can be parametrised by dataset config; engineer's
call on minimum-surface implementation). Vendored
`PositionRegressor` + `DataPreprocessor` from
`Temp/wlan_localization/src` via `importlib`. Demand #3 — no
edits to vendored.

**Step 1b — RoNIN ResNet1D on IPIN floor 0 IMU.** Use the
restored `scripts/eval_ronin_ipin.py` if it's already a working
pattern (RESULT_02 restored it but I don't think it ran on
IPIN floor 0 specifically; engineer adapts the path). Prefer
the pretrained checkpoint at
`data/ronin_frdr/pretrained_resnet/ronin_resnet/checkpoint_gsn_latest.pt`
(RESULT_07 confirmed it loads cleanly) and eval on IPIN floor 0
IMU. If IPIN IMU dimensions/shape don't match RoNIN's
expected input, document the obstacle and either train fresh
on IPIN train or report "n/a" with reasoning.

**Acceptance**: both SOTAs produce val + test MAE on IPIN
floor 0 (or "n/a" with documented reason).

### Step 2 — Train CNN1D + LSTM-attn on IPIN floor 0

Same protocol as RESULT_17 / RESULT_19:
- 90 epochs, AdamW + OneCycleLR + Huber(δ=0.5), B=128, K=4.
- 2-modality `[wifi, imu]`.
- `instant_dropout=0.45`, `modality_dropout=0.4`, lr=1.3e-3.

Use the `--arch` flag introduced in RESULT_17. Sequential:
CNN1D first, then LSTM-attn.

**Pre-test gate**: 5 epochs on 10 % train; val MAE drops ≥ 10 %
or clear descent.

**Memory budget**: < 6 GB. 2-mod K=4 B=128 should peak ~300 MB.

**Acceptance**: both candidates trained; val + test MAE + per-path
distribution recorded.

### Step 3 — Main-results IPIN floor 0 row

| method | params | val MAE | test MAE | smoothness r | source |
|---|---|---|---|---|---|
| wlan_localization (WiFi only) | — | ? | ? | n/a | step 1a |
| RoNIN ResNet1D (IMU only) | ~4.6 M | ? | ? | n/a | step 1b |
| CNN1D (WiFi+IMU fusion) | ~0.51 M | ? | ? | ? | step 2 |
| LSTM-attn (WiFi+IMU fusion) | ~0.57 M | ? | ? | ? | step 2 |

**Acceptance**: outcome label (α5 / β5 / γ5) + verdict on
LSTM-attn dead-reckoning replication (third data point after
Webots + IMUWiFine).

### Step 4 — Per-trajectory smoothness + per-path distribution (criterion (d))

Same shape as RESULT_19's Step 4. Median r per arch; per-path
MAE distribution; top-5 longest test paths plotted.

### Step 5 — Decision + PLAN_23

Three-sentence verdict:
- IPIN floor 0 row populated; outcome label.
- LSTM-attn dead-reckoning regime — third data point's verdict
  (replicates? attenuates?).
- PLAN_23 = RoNIN single-mod IMU row (next per the directive
  chain). RESULT_07's canonical ResNet1D number 5.140 m is
  reused; we run CNN1D + LSTM-attn IMU-only.

## Sources

- `handoff/SCIENTIST_NOTE_main-results-table.md` (directive).
- RESULT_19 (IMUWiFine row precedent + scripts pattern).
- RESULT_20 (val/test gap audit lesson — pre-flight IMU check).
- RESULT_15 (MSILN cross-session pattern; wlan_localization
  msiln runner template).
- RESULT_07 (canonical RoNIN ResNet1D checkpoint location).
- `data/ipin2024_floor0/`: per CLAUDE.md memory entry.
- `configs/data/ipin2024_floor0.yaml`,
  `scripts/convert_ipin2024.py`, `scripts/eval_ronin_ipin.py`,
  `scripts/eval_wlanloc_ipin.py` — restorable from run-1 if absent.

## What to report back

In `handoff/results/RESULT_22_main-results-ipin-floor0.md`:

1. **Step 0** — data + config presence; restored files; **IMU
   availability per split** (the pre-flight check).
2. **Step 1a** — wlan_localization on IPIN floor 0.
3. **Step 1b** — RoNIN ResNet1D on IPIN floor 0.
4. **Step 2** — CNN1D + LSTM-attn training; val + test + per-path.
5. **Step 3** — main-table row + outcome label.
6. **Step 4** — smoothness + plots.
7. **Step 5** — PLAN_23 recommendation (default: RoNIN single-mod).
8. **One open question** for scientist.

## Reversibility

- Step 0: file restores permanent.
- Step 1a/1b: NEW wrapper scripts if needed; permanent.
- Step 2: throwaway checkpoints.
- Steps 3–5: documentation.

Files committed: RESULT_22 + restored configs/scripts + NEW
`_eval_wlanloc_ipin_floor0.py` if dataset-specific.

**Demand #3**: vendored sources untouched. Shims in OUR wrappers.

**Compute budget**: ≤ 70 min.
- Step 0: 10 min (pre-flight + restoration).
- Step 1a: 10 min (eval-only).
- Step 1b: 20 min (pretrained eval if shape-compatible, train if
  not).
- Step 2: 25 min (2 trainings).
- Step 3: 3 min.
- Step 4: 5 min.
- Step 5: 5 min.

If overrun: drop Step 4 plots, keep median r number. Don't skip
Step 1 — the per-leg SOTAs are the load-bearing main-table
measurements.

If pre-flight Step 0 surfaces an IPIN floor 0 IMU-availability
issue analogous to IMUWiFine, document explicitly and continue
— this iteration's outcome will be (β5) on test with the
per-leg-SOTA val-only comparison being the load-bearing finding.
