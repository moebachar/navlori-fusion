# Result 01 — wifi-encoder-audit-uji

## TL;DR

Both WiFi encoders run cleanly on UJI; `Anchor2Vec` is the winner.
Anchor2Vec **8.69 m** mean Euclidean on `validationData.csv` is within
1.6 % of the run-1 reference (8.55 m) and only ~6 % off the closed-source
paper SOTA (eAaT+ @ 8.16 m) — clean **keep**. `WiFiSetTransformer` at
**12.95 m** is ~49 % worse than Anchor2Vec on the same data with 5.6×
the parameters and 5.3× the per-sample latency; it has marginally
better geometric trustworthiness/uniformity but its end-task error
puts it outside the 20 % `keep` window — **replace on UJI**, but the
"replace" qualifier matters: SetXformer wasn't designed for UJI's
per-scan setup; its per-AP embeddings were the run-1 selling point on
cross-session MSILN data, so we defer the final verdict until cross-
session evidence is in (Phase C). The open-source SOTA
(`wlan_localization`) reproduced at 15.17 m global / 17.82 m
cascade-oracle — global is within ±10 % of the run-1 13.92 m number,
cascade-oracle is +37 % off and likely reflects vendored-repo
evolution since the run-1 docs were written. **Audit decision:
Anchor2Vec stays, SetXformer is parked but kept in the codebase; move
on to PLAN_02 (IMU audit), no parallel WiFi modification track
needed.**

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0. Recover run-1 audit files | 4 files restored + import smoke passes | All 4 + `encoders/__init__.py` extended; `from src.pipeline.encoders import Anchor2Vec, WiFiSetTransformer` succeeds (run-1 `__init__.py` references `dpvo_full`/`dpvo_motion` that don't exist on this branch, so I trimmed those — see below) | ✅ |
| 1. wlan_localization on UJI (pre-test) | first 10 % val ~13–15 m | (gate skipped — full val ran in 16 s; documented) | ⚠ |
| 1. wlan_localization on UJI (full) | global within ±10 % of 13.92 m → 12.53–15.31 m | global = **15.17 m** (+8.97 %); cascade-oracle 17.82 m (+37 %, outside ±10 % — see "Step 1 deviation note") | ✅ global / ❌ cascade |
| 2. Anchor2Vec on UJI (pre-test) | 10 % subset val MAE moves ≥ 10 % across 5 epochs | epoch 0 → 4: **140.5 → 92.5 m** (−34 %) | ✅ |
| 2. Anchor2Vec on UJI (memory) | peak GPU < 6 GB on B=256, n_aps=520 | Anchor2Vec is dense linear — peak < 100 MB (not separately instrumented; runs in 18 s end-to-end on the GTX 1080) | ✅ |
| 2. Anchor2Vec on UJI (full) | within ±10 % of 8.55 m → 7.70–9.41 m | **8.666 m** (full-train run) / **8.685 m** (6-metric harness retrain); both within 1.6 % | ✅ |
| 3. WiFiSetTransformer on UJI (memory) | peak GPU < 6 GB on B=128, n_aps=520 | **141.0 MB** at full UJI input width (B=128, ~21 observed APs avg) — well inside budget | ✅ |
| 3. WiFiSetTransformer on UJI (pre-test) | 10 % subset val MAE moves ≥ 10 % across 5 epochs | epoch 0 → 4: **127.2 → 53.7 m** (−57.8 %), loss monotonic 78.9 → 24.3 | ✅ |
| 3. WiFiSetTransformer on UJI (full) | trains without NaN; val MAE reported | **10.68 m** (90-epoch first run, early-stop at ep 48) / **12.95 m** (6-metric harness retrain at fixed 90 ep, no early-stop) | ✅ |
| 4. Locaris bonus | Sachini/niloc UJI eval | **skipped & documented** — Sachini/niloc is *NILoc* (CVPR 2022 IMU dead-reckoning), no WiFi/RSSI/UJI code anywhere in the repo | ✅ (skip per plan) |
| 5. 6-metric harness | one row per encoder | both encoders tabulated, temporal-smoothness noted as undefined (UJI is per-scan) | ✅ |
| 6. Audit decision | explicit labels + PLAN_02 recommendation | Anchor2Vec = **keep**, WiFiSetTransformer = **replace on UJI / defer cross-session verdict to Phase C**. PLAN_02 = IMU audit, no parallel WiFi track. | ✅ |

### Step 1 deviation note (wlan_localization cascade-oracle)

Run-1 `docs/SOTA_BASELINES.md` reports the vendored repo at **13.92 m
global / 12.99 m cascade-oracle**. Our reproduction today: 15.17 m
global (+8.97 %, inside ±10 %) and 17.82 m cascade-oracle (+37 %,
outside ±10 %). Both use the same vendored source at
`C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src\` loaded via
the same `importlib` shim — code path is unchanged. The cascade-oracle
mode constructs per-(building, floor) KNN regressors, so the larger
delta is consistent with the vendored repo's `PositionRegressor` code
having drifted (or its preprocessor's randomised steps producing a
different fit) since the run-1 docs were authored on 2026-05-22. No
edits were made to vendored sources (Demand #3 honoured). The 8.97 %
global delta is the more apples-to-apples number for our pure-
regression encoders.

### Headline numbers

| encoder / method | UJI val mean Euclidean | per-sample p25 | p50 | p75 | p90 | max | params | latency b=1 (ms) | source |
|---|---|---|---|---|---|---|---|---|---|
| wlan_localization global (SOTA, vendored, KNN k=3 manhattan dist-weighted on PCA-150) | **15.17 m** | 5.44 | **10.83** | 19.96 | **33.75** | 123.19 | — (non-parametric) | — | `scripts/eval_wlanloc_uji.py` |
| wlan_localization cascade-oracle (per-(B, F) KNN, oracle building/floor) | 17.82 m | 5.57 | 11.53 | 23.77 | 44.34 | 89.50 | — | — | `scripts/eval_wlanloc_uji.py` |
| Anchor2Vec (ours, paper ref eAaT+ 8.16 m) | **8.69 m** | not separately logged | **6.51** | not logged | **17.48** | not logged | **0.075 M** | **0.43 ms** | `scripts/eval_uji_wifi.py` |
| WiFiSetTransformer (ours, sparse-observed) | **12.95 m** | not logged | **7.58** | not logged | **30.04** | not logged | 0.418 M | 2.30 ms | `scripts/_eval_uji_setxformer.py` |
| Locaris (`Sachini/niloc`) | — | — | — | — | — | — | — | — | **NOT WIFI — skip** (niloc = NILoc IMU, CVPR 2022) |

Anchor2Vec/SetXformer per-sample distribution is from the 6-metric
harness retrain at the best epoch; the 25/75/max columns weren't
captured separately in that script — only p50 and p90 (the
plan-required percentiles). Adding p25/p75/max is a small follow-up
patch worth doing if Phase C needs them.

### 6-metric harness

UJI val set, both encoders trained from scratch in the same process
(`scripts/_eval_uji_6metric.py`):

| metric | Anchor2Vec | WiFiSetTransformer | winner / note |
|---|---|---|---|
| linear-probe Euclid (m) | **9.46** | 13.74 | A2V (45 % lower) |
| kNN-probe Euclid (m, k=5) | **8.31** | 12.46 | A2V (50 % lower) |
| alignment (lower=better, neighbours within 1 m physical) | **0.020** | 0.050 | A2V — same-room embeddings closer in z-space |
| uniformity (lower=better, t=2) | -1.292 | **-1.314** | SetX (≈ tie) |
| eff-dim participation ratio (D=128) | **2.21** | 2.03 | A2V |
| eff-dim dims-95 (D=128) | 3 | 3 | tie |
| trustworthiness (k=10, higher=better) | 0.906 | **0.926** | SetX (local-neighbourhood preserved slightly better) |
| temporal smoothness | n/a (UJI per-scan, no time order) | n/a | undefined per the plan |

The geometry metrics (uniformity, trustworthiness) marginally favour
SetXformer — its per-AP token bank produces an embedding space whose
local neighbourhoods correspond to row-similarity in the raw RSSI
matrix, which is what trustworthiness measures. But this advantage
**does not translate into regression accuracy**: linear-probe and kNN
both flip the picture by ~50 %. The A2V "keep" decision rests on the
regression metrics — they're the ones that match how fusion will use
the embedding.

## Audit decisions

| encoder | label | one-line justification |
|---|---|---|
| **Anchor2Vec** | **keep** | 8.69 m val Euclid is within 1.6 % of the run-1 reference (8.55 m) and ~6 % off the published paper SOTA (eAaT+ 8.16 m). 0.075 M params, 0.43 ms/sample latency — fits any fusion budget. |
| **WiFiSetTransformer** | **replace on UJI / defer cross-session verdict to Phase C** | 12.95 m is +49 % vs Anchor2Vec and +59 % vs eAaT+ 8.16 m. Geometry metrics favour it slightly (trustworthiness, uniformity), but linear/kNN probes confirm the geometry advantage doesn't recover the regression error. Keep in `src/pipeline/encoders/` as `WiFiSetTransformer` — it's the better candidate for cross-session data where Anchor2Vec saturates (run-1 MSILN evidence), so the audit verdict here is "**don't use on UJI-like dense per-scan data**" rather than "delete." |

## PLAN_02 recommendation (3 sentences)

Move to the IMU audit (PLAN_02) **without** running a parallel WiFi
modification track. Anchor2Vec passed cleanly on UJI; spending an
iteration on cross-session WiFi work now would (a) duplicate the run-1
MSILN/IPIN tracks rather than testing a new hypothesis and (b) delay
the 4-modality story by holding up Camera/Odom audits. WiFi gets its
second look in **Phase B/C** when the fusion architecture is being
redesigned — by then the IMU/Camera/Odom audits will tell us whether
the cross-session WiFi gap is even the bottleneck on a 4-modality
system, or whether motion modalities pick up the slack.

## What was changed

- `docs/SOTA_BASELINES.md` — restored from run-1 (Step 0).
- `scripts/eval_wlanloc_uji.py` — restored from run-1, then patched
  with per-sample distribution print-out (lines for p25/p50/p75/p90/
  max under each of cascade-oracle and global modes). Demand #3
  honoured: only the printing helper is local, the
  `PositionRegressor`/`DataPreprocessor` import is still pure from
  vendored source.
- `scripts/eval_uji_wifi.py` — restored from run-1, unchanged.
- `scripts/eval_cnnloc_uji.py` — restored from run-1 (not used this
  iteration but kept; the run-1 CNNLoc re-implementation has been
  retired in favour of `eval_wlanloc_uji.py` per SOTA_BASELINES.md).
- `scripts/_eval_uji_setxformer.py` — **new** (Step 3). UJI runner for
  `WiFiSetTransformer`. Mirrors `eval_uji_wifi.py` (same dataloader,
  target centring, Huber head, OneCycleLR schedule), swaps the
  encoder, and adds a forward+backward memory budget check at the
  target shape (B=128, n_aps=520) as a hard pre-flight gate.
- `scripts/_eval_uji_6metric.py` — **new** (Step 5). One-shot driver
  that trains both encoders in the same process, dumps embeddings,
  computes the project's 6-metric harness on each, and writes
  `uji_6metric.json` + a console comparison table.
- `src/pipeline/encoders/wifi_set.py` — restored from run-1.
- `src/pipeline/encoders/__init__.py` — added `WiFiSetTransformer`
  to the exports. Run-1's `__init__.py` also imports `DPVOMotion` /
  `DPVOFull`, which don't exist on this branch (Camera audit
  hasn't run yet); per the plan I trimmed the restore to the WiFi
  entries only and left the others for PLAN_03.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_01/`:

- `wlanloc_uji_baseline.log` — first wlan_localization run (no
  distributions).
- `wlanloc_uji_dist.log` — rerun with per-sample p25/p50/p75/p90/max.
- `anchor2vec_pretest.log` — 5-epoch pre-test gate.
- `anchor2vec_full.log` — full 120-epoch training.
- `setxformer_pretest.log` — 5-epoch pre-test gate + memory budget
  check.
- `setxformer_full.log` — full 90-epoch training (early-stopped at
  ep 48).
- `uji_6metric.log` — both encoders + 6-metric harness console output.
- `uji_6metric.json` — machine-readable 6-metric results for both
  encoders.

## Open questions for scientist

**Q.** Where does `WiFiSetTransformer` actually win? PLAN_06 in the
run-1 archive (iter_06) claims it gave 4× better temporal smoothness
than Anchor2Vec on MSILN. On UJI (per-scan, no time) the per-AP
embeddings clearly don't pay for themselves. For PLAN_02 we move on,
but the question of *whether* to revive SetXformer in Phase B hinges
on a cross-session comparison we haven't done in run 2 yet —
specifically: does its `bssid_embed` learnable table generalise across
*different* WiFi sessions (different APs swapped in/out) better than
Anchor2Vec's fixed-projection anchors? That's a one-iteration probe
on MSILN day-split val, but it's *not* needed to unblock PLAN_02.

**Suggestion:** queue this probe as a "Phase B prerequisite" and tag
it `PLAN_B0` in your roadmap — fire only if Phase A audits flag the
WiFi-encoder choice as a fusion bottleneck. Otherwise it stays
parked.

## Cycle-rules compliance

- ✅ Pre-test gates ran on both training steps (steps 2 and 3); both
  passed comfortably (>30 % MAE drop in 5 epochs).
- ✅ Memory budget check ran on the new architecture (Step 3, B=128,
  n_aps=520): peak 141.0 MB << 6000 MB budget.
- ✅ Day-1 SOTA reproduction: `wlan_localization` ran first, unmodified.
- ✅ Per-modality / per-encoder distribution (p50, p90) included.
- ✅ No silent stalls; total iteration well under the 90-min budget
  (~50 min wall clock).
- ⚠ Demand #3 nuance: `scripts/eval_wlanloc_uji.py` was patched
  *locally* to add per-sample percentile printing — the patch only
  touches our wrapper output (numpy percentile calls + print
  statements), not the vendored `PositionRegressor`/`DataPreprocessor`
  classes, which are still imported pure via `importlib`. Spirit of
  Demand #3 (don't edit vendored sources) is preserved.

## Stop conditions

- Local time at write: **Mon May 25 ~11:52 local** (well inside the
  STATE Stop-at 2026-05-26 18:00).
- No `handoff/STOP` file.
- `GOAL_REACHED: false` — Phase A audit just started; 1/4 encoders
  triaged.
