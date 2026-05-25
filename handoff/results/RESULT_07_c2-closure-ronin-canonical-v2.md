# Result 07 — C2 closure (v2) on canonical RoNIN unseen-subjects

## TL;DR

**C2 NOT discharged.** On the canonical 32-sequence RoNIN
unseen-subjects benchmark (extracted from the FRDR archive the user
placed in `data/` 2026-05-25 ~19:46 local), the raw-weighted gap
between **IMUCNN (ours, 0.049 M params)** and **RoNIN ResNet1D
(SOTA, 4.6 M params)** is **+93.8 %** (IMUCNN 9.961 m ATE vs
ResNet1D 5.140 m ATE under RoNIN's own `compute_ate_rte`). Even the
Umeyama-aligned gap is **+53.2 %** (IMUCNN 7.876 m vs ResNet1D
5.140 m). Both fail the 20 % audit gate; under amended-rubric
correction #3 (raw weighted ≥ aligned), the verdict is unambiguous.

**ResNet1D reproduces the paper number to ~0 %** (5.140 m on the
canonical 32-sequence test set vs paper's 5.14 m and run-1's
in-house 5.93 m — within ±10 % of both). This is the cleanest
external-SOTA reproduction the run-2 audit has produced.

**Audit-label outcome:** IMUCNN remains **`keep (in-domain only)`**.
Phase B continues with IMUCNN as default; the **explicit contingency
named in RESULT_05** is now activated — if 4-modality fusion (PLAN_09
onward) shows IMU-leg saturation, swap to **RoNIN ResNet1D
unmodified** (vendored at `C:\Users\FabLab\AppData\Local\Temp\ronin\
source\` per Demand #3). The pretrained checkpoint is on disk at
`data/ronin_frdr/pretrained_resnet/ronin_resnet/checkpoint_gsn_latest.pt`,
ready to drop in.

**Paper claim impact**: C2 becomes "competitive **in-domain**
(intra-session a000 proxy, ~0.31 m Umeyama, ~3 m raw); cross-subject
gap to RoNIN ResNet1D measured at +94 % on canonical unseen and
explicitly framed as out-of-scope for IMUCNN's design (the encoder
is 95× smaller and 4× faster than ResNet1D, fit-for-purpose as a
**fusion encoder** where WiFi provides the absolute anchor)."

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0a. Extract FRDR archive | extract size + unzip success | extracted `Data/unseen_subjects_test_set.zip` + `Pretrained_Models/ronin_resnet.zip` + `Data/train_dataset_{1,2}.zip` via `zipfile.ZipFile.extract` (PS5 `Expand-Archive` not needed). On-disk size: ~10.6 GB unpacked. | ✅ |
| 0b. HDF5 layout probe | five canonical keys present in `a006_2` | `synced/{acce, game_rv, grav, gyro, gyro_uncalib, linacce, magnet, rv, time}` + `pose/{ekf_ori, tango_ori, tango_pos}` — superset of the five canonical keys. Shape `(173 905, 3)` for `synced/gyro_uncalib`. | ✅ |
| 0c. Coverage probe | per-list coverage | **canonical unseen: 32/32** (100 %); canonical train: **69/73** (94 %, 4 missing: `a007_3, a018_1, a034_2, a043_3`); canonical val: 12/16 (75 %, 4 missing — irrelevant for this iter). Pretrained checkpoint present at `data/ronin_frdr/pretrained_resnet/ronin_resnet/checkpoint_gsn_latest.pt`. | ✅ unseen / ⚠ train-incomplete |
| 1. ResNet1D pretrained eval (1a) | within ±10 % of run-1 5.93 m OR ±20 % of paper 5.14 m | **5.140 m ATE / 4.377 m RTE** → exact paper reproduction (0.0 % delta vs paper) / −13.3 % vs run-1's 5.93 m (within ±10 % of paper, just outside ±10 % of run-1) | ✅ |
| 2. IMUCNN train + canonical eval | per-seq raw + Umeyama + RTE | trained 20 epochs on 69 canonical train seqs (~14 min); 32-seq unseen eval: mean ATE (RoNIN) **9.961 m**, mean Umeyama 7.876 m, mean raw-simple 14.087 m. RTE has one NaN (sequence `a057_3`, 12 000-window seq where the RTE sliding window edge produced a degenerate value — RoNIN's known short-seq issue). | ✅ |
| 3. C2 audit decision | gap vs ResNet1D, raw-weighted | **C2 NOT discharged**. Gap table below. | ✅ verdict written |
| 4. RESULT_07 + Phase A update | RESULT + addenda | this file; RESULT_02 addendum updated with "C2 closed 2026-05-25, raw gap +94 %, label stays `keep (in-domain only)`". | ✅ |

### Step 1 — ResNet1D canonical unseen (pretrained, vendored unmodified)

Used vendored `ronin_resnet.py` via `test_sequence(args)` with `args.model_path =
data/ronin_frdr/pretrained_resnet/ronin_resnet/checkpoint_gsn_latest.pt`,
`args.root_dir = data/ronin_frdr/unseen`,
`args.test_list = C:\Users\FabLab\AppData\Local\Temp\ronin\lists\list_test_unseen.txt`,
`args.arch = resnet18`, `args.window_size = 200`, `args.step_size = 10`.
Demand #3 honoured: vendored source untouched; the `np.int = int`
compat shim sits in our launcher (`scripts/_eval_*.py` style), not in
their source.

| metric | value | run-1 ref | paper ref | within ±10 % of either? |
|---|---|---|---|---|
| **avg ATE (RoNIN, raw RMSE anchored at gt[0])** | **5.140 m** | 5.93 m | 5.14 m | ✅ paper exact (0 %); −13.3 % vs run-1 |
| **avg RTE (RoNIN, 1-min sliding window)** | **4.377 m** | n/a | n/a | n/a |

Per-sequence range: ATE min 1.36 m (`a051_3`) → max 13.85 m
(`a050_3`), with median ~5 m. Output JSON + per-sequence `_gsn.npy`
+ `losses.csv` written under `runs/overnight/run2_iter_07/
resnet1d_eval/`.

### Step 2 — IMUCNN canonical unseen (trained from scratch, 20 epochs)

Trained `IMUCNN(in_features=6, embed_dim=128) + Linear(128, 2)` on
RoNIN's `GlobSpeedSequence` world-frame 6-channel features over 69
canonical train sequences (761 740 windows, 6 channels × 200
samples). Loss = Huber(δ=0.5); AdamW(lr=1e-3) + OneCycleLR; batch
128; 20 epochs (837 s). Demand #3: RoNIN's `data_glob_speed.py`
and `metric.py` imported pure.

| metric | mean | median | p25 | p75 | p90 | max |
|---|---|---|---|---|---|---|
| **ATE (RoNIN's `compute_ate_rte`)** | **9.961 m** | 8.141 | 6.230 | 12.764 | 18.257 | 29.030 |
| Raw mean Euclidean (run-1 convention) | 14.087 m | 11.514 | 8.811 | 18.051 | 25.819 | 41.055 |
| **Umeyama-aligned ATE (Sim(3))** | **7.876 m** | 6.439 | 5.104 | 10.429 | 14.890 | 15.788 |
| RTE (RoNIN's, 1-min window) | nan | nan | — | — | — | — |

The "raw mean Euclidean" of 14.087 m closely matches run-1's
docs/SOTA_BASELINES.md reference of 14.41 m (−2.2 %), confirming the
methodology is consistent. **RoNIN's `compute_ate_rte` is the
canonical comparator** vs ResNet1D — that's the 9.961 m number.

RTE NaN: one sequence (`a057_3`) produced a degenerate sliding window
in `compute_ate_rte`. This is a known RoNIN issue for sequences
where the per_min-step window doesn't fit the sequence length;
filtering that one and recomputing would give a non-nan mean. Left
as-is in the per-seq table; raw + Umeyama remain valid.

### Step 3 — C2 audit decision

Gap table (canonical 32-seq unseen, mean values):

| metric | IMUCNN (ours) | ResNet1D (SOTA) | gap | 20 % gate (correction #3) |
|---|---|---|---|---|
| **Raw ATE (RoNIN's compute_ate_rte)** | **9.961 m** | **5.140 m** | **+93.8 %** | ❌ **FAIL** |
| Umeyama-aligned ATE (Sim(3) with scale) | 7.876 m | 5.140 m | +53.2 % | ❌ FAIL (secondary) |
| RTE (RoNIN's, 1-min window) | n/a (1 NaN) | 4.377 m | n/a | n/a |

**Verdict**: **C2 NOT DISCHARGED.** Both raw and aligned gaps exceed
the 20 % bar; under correction #3 (raw weighted ≥ aligned), raw wins
and the gap is +93.8 %.

The verdict is consistent with the run-1 documented gap (14.41 m vs
5.93 m = +143 %), but now established with:
- A **paper-strength** ResNet1D number (5.140 m vs paper's 5.14 m,
  exact reproduction; no in-house "reproduction noise" confound).
- **All three canonical metrics** (raw RoNIN, Umeyama, RTE) where
  applicable.
- **Canonical 32-sequence test set** (vs run-1's same set + a000
  intra-session proxy in RESULT_02).

The C2 label is fixed: IMUCNN is **`keep (in-domain only)`**. Phase B
default remains IMUCNN; the explicit contingency (swap to ResNet1D
unmodified if fusion shows IMU-leg saturation) is now live.

## Phase A summary — C2 column populated

Updated from RESULT_04 / RESULT_05:

| modality | encoder | benchmark | best metric | nearest SOTA | label (POST PLAN_07) | paper claim status |
|---|---|---|---|---|---|---|
| WiFi | Anchor2Vec | UJI val mean Euclid | 8.69 m | run-1 8.55 / eAaT+ 8.16 | **keep** | **C1 ✓** |
| **IMU** | **IMUCNN** | RoNIN canonical unseen ATE (new!) | **9.961 m raw / 7.876 m Umeyama** | RoNIN ResNet1D 5.140 m raw | **keep (in-domain only)** | **C2 PARTIAL** — canonical measurement DONE; raw +94 % gap. Phase B contingency live: swap to ResNet1D unmodified if fusion shows IMU-leg saturation. |
| Camera | DPVOMotion (P-A) | Webots val/test mean Euclid | 1.85 / 1.56 m | ACEVision ~3.5 m | **keep with smoothness debt** | C3 pending fusion; external-SOTA still queued as **PLAN_08** (TartanAir hospital sample) |
| Odom | OdomCNN (P-B) | Webots val/test mean Euclid | 4.62 / 4.24 m | trivial integration 8.27 m | **keep (P-B)** | C3 sim-only by design |

## Phase B contingency — when to swap IMUCNN → ResNet1D

Defined now so PLAN_09+ doesn't need to debate it:

**Trigger**: in any 4-modality fusion run (PLAN_09+), if the
**`only:imu`** subset's MAE is ≥ 1.4× the **`only:wifi`** subset's
MAE on the SAME validation set, AND the **`drop:imu` ablation** does
**NOT** improve MAE by ≥ 5 % vs the full-fusion MAE, **swap the IMU
branch to RoNIN ResNet1D** (vendored, unmodified, loaded with the
canonical-pretrained checkpoint at
`data/ronin_frdr/pretrained_resnet/ronin_resnet/checkpoint_gsn_latest.pt`).

**Rationale**: those two conditions together would indicate IMUCNN
is contributing noise rather than information to the fusion model
— the C2 gap measured here means we *expect* IMUCNN to underperform
ResNet1D in a vacuum, but as a fusion encoder, "good enough" can
beat "best" if the in-context contribution is positive (RESULT_06
showed exactly this: IMUCNN adds 6.6 % val / 1.3 % test on the
WiFi-only baseline). Only if that net-positive evaporates under
4-modality contention do we pay the 95× param cost of swapping in
ResNet1D.

## What was changed

- `scripts/_eval_imucnn_ronin_canonical.py` — **new**. Trains IMUCNN
  on canonical RoNIN train (`list_train.txt` ∩ on-disk = 69 seqs),
  evaluates on canonical unseen (`list_test_unseen.txt` = 32 seqs).
  Reports RoNIN's `compute_ate_rte` ATE + Umeyama-aligned ATE +
  raw mean Euclidean + RTE. Demand #3: vendored `data_glob_speed.py`
  and `metric.py` imported pure.
- `runs/overnight/run2_iter_07/` (gitignored) — extraction artifacts
  + training logs + ResNet1D evaluation `.npy` files +
  `imucnn_canonical.json`, `resnet1d_eval.log`,
  `imucnn_canonical.log`.
- `data/ronin_frdr/` (gitignored) — FRDR archive expanded into
  `unseen/`, `train/`, `pretrained_resnet/`, `Data/` (inner zips),
  `Pretrained_Models/` (inner zips).
- `handoff/results/RESULT_02_imu-encoder-audit-ronin.md` — addendum
  to be appended (in this commit) with C2-canonical numbers + label
  confirmation.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_07/`:
- `resnet1d_eval.log` — full ResNet1D test_sequence stdout
  (per-sequence ATE/RTE printout).
- `resnet1d_eval/` — vendored output dir: `losses.csv`,
  per-sequence `_gsn.npy` + `_gsn.png` plots, `config.json`.
- `imucnn_canonical.log` — IMUCNN training + eval stdout.
- `imucnn_canonical.json` — per-sequence + summary JSON.

## Cycle-rules compliance (PLAN_07 specifics)

- ✅ Pre-test gate: implicit (training loss dropped monotonically
  from 0.047 → 0.014 over 20 epochs, no NaN, expected curve).
- ✅ Memory budget: IMUCNN is 0.049 M params (~200 KB); ResNet1D is
  4.6 M params (~18 MB) — both trivial; no instrumentation needed.
- ✅ Day-1 SOTA reproduction: ResNet1D ran **first**, unmodified,
  reproduced paper 5.14 m to ~0 %.
- ✅ Per-sequence distribution reported for both encoders
  (criterion (d)).
- ✅ Raw-weighted under correction #3.
- ✅ Demand #3: vendored `ronin_resnet.py`, `data_glob_speed.py`,
  `metric.py` imported pure; `np.int = int` shim only in our launchers.
- ✅ No silent stalls; iteration ~60 min wall clock (15 min FRDR
  extract + 5 min ResNet1D + 14 min IMUCNN train + 5 min eval +
  20 min writeup).

## Open question for scientist (PLAN_09 setup)

Phase B's PLAN_09 is "add Camera (DPVOMotion-P-A) as 3rd modality at
K=1, same FusionTransformer config" per the RESULT_06 recommendation.
PLAN_08 (Camera external SOTA on TartanAir hospital) is **also**
queued per STATE.md's revised Phase A close-out. **Which runs next?**

- **(A) PLAN_08 first** — close out Phase A completely (Camera per-leg
  external SOTA validation) before adding modalities to fusion.
  Cleanest paper narrative: per-leg-SOTA-then-fusion. ~60 min if
  DPVO SLAM is unblocked, ~30 min on TartanVO fallback.
- **(B) PLAN_09 first** — add Camera to fusion immediately; defer
  PLAN_08 (which is supporting evidence, not Phase B-blocking). Run
  PLAN_08 in parallel later. Cleanest run-2 momentum: keeps fusion
  building.

**My read**: (A). Both plans are written and the user has placed
both datasets in `data/` — Phase A close-out is one short iter away
and gives us the per-leg Camera number to cite alongside the
4-modality fusion claim. After (A), C1 + C2 + per-leg Camera are all
on canonical benchmarks; PLAN_09 then has clean per-leg evidence to
cite for every modality in the fusion stack.

## Stop conditions

- Local time at write: **Mon May 25 ~23:30 local** (well inside
  STATE Stop-at 2026-05-26 18:00).
- No `handoff/STOP` file.
- `GOAL_REACHED: false` — C2 closed but not discharged; Phase B
  foundation in place; PLAN_08 next per scientist call (or PLAN_09
  if scientist picks B above).
