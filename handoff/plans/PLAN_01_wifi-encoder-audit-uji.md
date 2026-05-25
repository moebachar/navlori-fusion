# Plan 01 — WiFi encoder audit on UJIIndoorLoc (vs `wlan_localization`)

> **Plan revision 2026-05-25 (scientist, first wake).** The original
> PLAN_01 referenced files (`scripts/eval_wlanloc_uji.py`,
> `scripts/eval_uji_wifi.py`, `src/pipeline/encoders/wifi_set.py`,
> `docs/SOTA_BASELINES.md`) that exist on branch
> `overnight-autonomous-2026-05-24` (run 1) but NOT on this run-2
> branch (which was cut from `main`). It also named `Sachini/niloc`
> ("Locaris") as a WiFi SOTA — `Sachini/niloc` is **NILoc, an IMU
> method** (CVPR 2022), not WiFi (verified by `WebFetch`). Steps below
> are revised: (i) add an explicit recovery step before any
> evaluation, (ii) drop the Locaris-clone step, (iii) keep the
> WiFiSetTransformer-on-UJI work since run 1 only ever ran it on
> MSILN, never on UJI.

## Hypothesis

The WiFi encoder is the cross-session bottleneck (run-1 archive
evidence: `handoff/archive/run1/results/RESULT_04` showed Anchor2Vec
structurally saturated at ~15.7 m val MAE on MSILN regardless of
capacity). Run 1's `docs/SOTA_BASELINES.md` reported Anchor2Vec at
**8.55 m** on UJI vs `wlan_localization`'s **13.92 m** — i.e. Anchor2Vec
already passed the 20 % gate on UJI, but the number was never
reproduced after the run-1/run-2 branch split and was never
cross-checked against `WiFiSetTransformer`. This plan does the clean
side-by-side from a reproducible state.

Expected outcome (acceptance criterion (a) from STATE.md):

- `keep` — encoder within **20 %** of SOTA on same data + protocol →
  move on to IMU audit (PLAN_02).
- `modify` — within 20–50 % → name the specific bottleneck (capacity /
  regularisation / pretraining) for a Phase-B follow-up.
- `replace` — gap > 50 % → name the alternative; gate it in Phase B.

No fusion training. No MSILN training. Pure WiFi-encoder-on-UJI
benchmark.

## Steps

### Step 0 — Recover run-1 WiFi-audit artifacts (NEW)

Restore the files that run-1's WiFi audit produced. They are
already on the run-1 branch — copy them in without merging:

```powershell
git checkout overnight-autonomous-2026-05-24 -- `
  scripts/eval_wlanloc_uji.py `
  scripts/eval_uji_wifi.py `
  src/pipeline/encoders/wifi_set.py `
  src/pipeline/encoders/__init__.py `
  docs/SOTA_BASELINES.md
```

**Sanity check:** `python -c "from src.pipeline.encoders import
Anchor2Vec, WiFiSetTransformer"` must succeed.

If `src/pipeline/encoders/__init__.py` pulls in modules that also
don't exist on this branch (e.g. `dpvo_full`, `dpvo_motion`), only
copy the WiFi imports across — `Anchor2Vec` + `WiFiSetTransformer` —
and leave the other registrations to subsequent encoder-audit plans.
Document any such trimming in the RESULT.

- **Acceptance:** four files restored + import smoke passes.
- **Reversibility:** all four are permanent; engineer commits them
  with the result.

### Step 1 — Day-1 SOTA reproduction: `wlan_localization` on UJI

Use the just-restored `scripts/eval_wlanloc_uji.py`. The script imports
`PositionRegressor` and `DataPreprocessor` from the vendored repo at
`C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src` via
`importlib` (Demand #3 honoured — no edits to vendored source). The
run-1 reference number is **13.92 m global / 12.99 m cascade-oracle**
on `validationData.csv`.

- **Pre-test gate:** run on the first 10 % of `validationData.csv`
  only — expect ~13–15 m mean Euclidean on ~110 samples. If wildly
  off, the vendored repo or import shim is broken; STOP and write a
  partial RESULT documenting the obstacle.
- **Acceptance:** full-val mean Euclidean within ±10 % of 13.92 m
  (12.53–15.31 m) for the global mode. If outside ±10 % document
  precisely why (numpy compat shim, vendored repo evolved, etc.).

### Step 2 — Reproduce `Anchor2Vec` on UJI

Use the just-restored `scripts/eval_uji_wifi.py`. The run-1 reference
is **8.55 m** mean Euclidean on `validationData.csv`. Default
hyperparameters: `embed_dim=128`, `n_anchors=64`, 120 epochs, batch
256, AdamW + OneCycleLR + Huber(δ=1.0).

- **Pre-test gate (training):** first run 10 % of `trainingData.csv`
  for 5 epochs (kill if not converging). Acceptance: subset val MAE
  drops ≥ 10 % across the 5 epochs.
- If a saved Anchor2Vec checkpoint exists under
  `runs/encoder_wifi_*/` or anywhere on the run-1 branch
  (`git log overnight-autonomous-2026-05-24 -- runs/encoder_wifi*`),
  load and eval-only — skip training.
- **Memory budget check:** forward+backward on a synthetic batch at
  UJI's full input width (`n_aps=520`, batch=256). Peak GPU MB
  reported; must be < 6 GB. (This will pass trivially for the dense
  Anchor2Vec.)
- **Acceptance:** Anchor2Vec val mean Euclidean within ±10 % of
  8.55 m (7.70–9.41 m). Report per-sample distribution (p25 / p50 /
  p75 / p90 / max).

### Step 3 — Build the UJI runner for `WiFiSetTransformer`

`WiFiSetTransformer` (per-AP / per-BSSID set-transformer + CLS
readout, sparse-observed forward — arXiv:2506.00656 inspiration) was
built in run-1 iter_06 but only evaluated on MSILN. UJI evaluation is
genuinely new.

Create `scripts/_eval_uji_setxformer.py` (underscore prefix marks
iteration-scoped; can be promoted later if the encoder wins the
audit). Mirror `scripts/eval_uji_wifi.py` structurally — same data
loading, same target centring, same train/val split, same metric
(mean Euclidean on `validationData.csv`), same training schedule —
only the encoder class changes.

- **Memory budget check (mandatory pre-flight):** forward+backward
  on a synthetic batch at UJI's full input width (`n_aps=520`,
  batch=128), `max_observed_per_scan` per the encoder's default
  (256, but UJI has ≤ 60 detected APs per scan, so the effective
  cap is well under that). Peak GPU MB reported; must be < 6 GB. If
  OOM at batch=128, drop to batch=64 and note.
- **Pre-test gate:** 10 % of `trainingData.csv` for 5 epochs first;
  loss drops monotonically, subset val MAE moves ≥ 10 %.
- Train on full UJI `trainingData.csv` (no time-window — UJI is
  per-scan), 120 epochs max with patience=15.
- **Acceptance:** training completes without NaN; full-val mean
  Euclidean reported with per-sample distribution.

### Step 4 — Six-metric harness on each encoder

Run `src/pipeline/evaluation/encoder_eval.py` on each of
{`Anchor2Vec`, `WiFiSetTransformer`} using UJI val embeddings:
linear probe, kNN (k=5), alignment, uniformity, eff. dim
(participation ratio + dims for 95 % variance), trustworthiness.
Skip temporal smoothness — UJI is a per-scan dataset with no
temporal structure, so this metric is undefined; document the skip
in the result.

If `src/pipeline/evaluation/encoder_eval.py` itself is missing on
this branch, restore it the same way as Step 0 (single
`git checkout overnight-autonomous-2026-05-24 -- <path>`). If
recovery is non-trivial (the harness pulls a chain of missing
modules), write a partial RESULT with the obstacle and skip Step 4
— do not let this step block the audit decision.

- **Acceptance:** one row per encoder for the six computable
  metrics. Whichever metric is most informative is whichever
  separates the two encoders by the largest fractional gap.

### Step 5 — Audit decision (PLAN_02 input)

Label each of {Anchor2Vec, WiFiSetTransformer} with `keep` / `modify` /
`replace` per the hypothesis rubric. One-line justification per label
quoting the numbers from steps 1–4.

- **Acceptance:** explicit table; recommendation for PLAN_02
  (default = IMU audit, the next encoder in the audit order).
  Explicitly answer: does PLAN_02 need a parallel WiFi-encoder
  modification track running alongside the IMU audit, or is WiFi
  finished for now?

## Sources

- WiFi SOTA repo (vendored): `C:\Users\FabLab\AppData\Local\Temp\wlan_localization\`
  (sharan-naribole, MIT). Existing runner restored in Step 0.
- UJI benchmark: https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc
  — already DVC-tracked under `data/uji_indoorloc/`
  (`trainingData.csv`, `validationData.csv` confirmed present
  2026-05-25).
- WiFiSetTransformer inspiration: Lazaro et al. 2025,
  https://arxiv.org/abs/2506.00656 (per-AP / per-BSSID
  set-transformer for WiFi fingerprinting).
- Existing run-1 audit doc (to be restored): see
  `overnight-autonomous-2026-05-24:docs/SOTA_BASELINES.md` —
  reports Anchor2Vec 8.55 m, wlan_localization 13.92 m global /
  12.99 m cascade-oracle.

**Literature context (NOT to be reproduced this iteration — citation
only for the audit writeup):**
- **Locaris** (Bhatia et al., Oct 2025) — decoder-only transformer,
  RSSI/FTM as tokens, claims sub-metre but **no public code yet**:
  https://arxiv.org/abs/2510.11926.
- **WiFiGPT** (May 2025, same group) — same family,
  https://arxiv.org/abs/2505.15835. No public code.
- **Hierarchical Stage-Wise Linked DNN** (Li, Kim et al., 2024):
  https://arxiv.org/abs/2407.13288 — 8.19 m 3-D error on UJI, CNN
  not transformer, no public code.
- **`Sachini/niloc`** is **NILoc / Neural Inertial Localization**
  (CVPR 2022) — IMU dead-reckoning, NOT WiFi. Dropped from the
  baseline list. (NILoc may resurface in PLAN_03 as the IMU
  audit's reference baseline candidate alongside RoNIN.)

## What to report back

In `handoff/results/RESULT_01_wifi-encoder-audit-uji.md`:

1. **Step 0 recovery log:** which files restored, what failed to
   restore (if any), what was trimmed in `encoders/__init__.py`.
2. **Pre-test gate outcomes** for Steps 2 and 3: did the 10 %
   subset move ≥ 10 % MAE? Full-training only ran if yes.
3. **Memory budget checks** for Steps 2 (Anchor2Vec) and 3
   (WiFiSetTransformer): peak GPU MB at the UJI-full
   forward+backward.
4. **Headline numbers table:**

   | encoder / method | UJI val mean Euclidean | per-sample p50 | per-sample p90 | params | latency (ms / sample) | source |
   |---|---|---|---|---|---|---|
   | wlan_localization (SOTA, vendored) | …  (target 13.92 ±10 %) | … | … | — | — | scripts/eval_wlanloc_uji.py |
   | Anchor2Vec (ours) | … (target 8.55 ±10 %) | … | … | … | … | scripts/eval_uji_wifi.py |
   | WiFiSetTransformer (ours, sparse-observed) | … | … | … | … | … | NEW scripts/_eval_uji_setxformer.py |

5. **6-metric harness table** (Anchor2Vec vs WiFiSetTransformer):
   linear probe / kNN / alignment / uniformity / eff. dim /
   trustworthiness. (Temporal smoothness skipped — UJI per-scan;
   document.)
6. **Audit labels:** Anchor2Vec = `keep|modify|replace`,
   WiFiSetTransformer = `keep|modify|replace`, each with a one-line
   justification quoting the numbers.
7. **PLAN_02 recommendation:** continue to IMU audit as planned, OR
   pause and fix the WiFi encoder if `WiFiSetTransformer` shows a
   structural advantage that argues for parallel encoder work
   during the IMU iteration. 3-sentence justification.
8. **One open question** for scientist.

## Reversibility

- Step 0 (file recovery): permanent (engineer commits the restored
  files with the result). Reversible by `git revert`.
- Step 1 (run vendored baseline): throwaway probe, eval-only.
- Step 2 (Anchor2Vec eval / retrain): throwaway; if a new training
  run produces a saved checkpoint, save it under
  `runs/encoder_audit_wifi_<ts>/anchor2vec.pt` (gitignored).
- Step 3 (NEW `scripts/_eval_uji_setxformer.py`): permanent; engineer
  commits.
- Step 4 (6-metric harness): throwaway.
- Step 5 (audit labels): documentation in RESULT_01.

All run artefacts under `runs/encoder_audit_wifi_<ts>/`, gitignored.
Files committed: restored files from Step 0, RESULT_01,
`scripts/_eval_uji_setxformer.py`, any small fix to
`src/pipeline/encoders/__init__.py` needed to make the WiFi imports
work in isolation.

**Demand #3 untouched.** Vendored SOTA repos at `Temp/` are not
edited; shims live in our `scripts/` only.

**Compute budget:** total iteration ≤ 90 min.
- Step 0: 5 min (file recovery + import smoke).
- Step 1: 5–10 min (eval-only script, runs in seconds; majority of
  time is the cascade-oracle mode reload).
- Step 2: 5 min eval-only if a checkpoint is recoverable, ~25 min
  if retraining from scratch (Anchor2Vec is small).
- Step 3: 30–45 min (write new runner + memory check + train).
- Step 4: 5 min (eval-only, no training).
- Step 5: 5 min writeup.

If the iteration overruns 90 min, ship RESULT_01 with whatever is
complete from Steps 0–3 + 5; Step 4 (6-metric harness) is the
soonest droppable.

If Step 0's recovery fails (e.g. `src/pipeline/encoders/__init__.py`
on run-1 imports an unbounded chain of also-missing modules), write
a partial RESULT after 20 min explaining the blockage and revise
PLAN_01 next iteration to clone-from-scratch rather than checkout.
