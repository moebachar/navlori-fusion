# Plan 01 — WiFi encoder audit on UJIIndoorLoc (vs `wlan_localization` + Locaris)

## Hypothesis

The WiFi encoder is the cross-session bottleneck (run-1 evidence;
archive run1/RESULT_03 + RESULT_04). Run 1 ran two WiFi architectures
(`Anchor2Vec`, `WiFiSetTransformer`) but never produced a clean
side-by-side against the **published WiFi-fingerprinting SOTA on
UJIIndoorLoc** — the canonical benchmark. This iteration produces
that comparison from scratch so the audit decision (keep / modify /
replace) is evidence-based.

Expected outcome (acceptance criterion (a) from STATE.md):

- `keep` — our encoder lands within **20 %** of the SOTA's published
  number on the same data + protocol → architecture is good, move on
  to next audit.
- `modify` — within 20-50 % → identify the specific weakness
  (capacity / regularisation / pretraining) and propose a targeted
  modification for PLAN_02 follow-up after the IMU audit.
- `replace` — gap > 50 % → name the alternative
  (Locaris / contrastive SSL / per-AP attention) and gate it as the
  fusion-prep iteration in Phase B.

No fusion training in this plan. No msiln training. Pure encoder-on-UJI
benchmark.

## Steps

1. **Day-1 rule: reproduce the SOTA baseline on UJI.** Use the
   existing `scripts/eval_wlanloc_uji.py` (vendored at
   `C:\Users\FabLab\AppData\Local\Temp\wlan_localization\` per
   SCIENTIST_BRIEF). Expected number per `docs/SOTA_BASELINES.md`:
   **13.92 m global / 12.99 m cascade-oracle** on
   `validationData.csv`.
   - **Pre-test gate:** run on the first 10 % of `validationData.csv`
     only — should report ~13-15 m mean Euclidean on 110 samples.
     If wildly off, the vendored repo or our shim is broken; STOP
     and diagnose.
   - **Acceptance:** full-val mean Euclidean within ±10 % of 13.92 m
     OR document a precise reason it differs (vendored repo
     evolved, numpy compat shim, etc.).

2. **Reproduce `Anchor2Vec` on UJI.** Use `scripts/eval_uji_wifi.py`.
   Expected per docs: **8.55 m**.
   - **Pre-test gate:** load the existing trained checkpoint if one
     exists (look under `runs/encoder_wifi_*/`); skip the training
     step if a saved checkpoint reproduces the number.
   - If no checkpoint exists, train on 10 % of `trainingData.csv` for
     5 epochs first. Acceptance: subset val MAE drops by ≥ 10 %
     across the 5 epochs. Only then promote to full training.
   - **Acceptance:** Anchor2Vec val mean Euclidean within ±10 % of
     8.55 m. Per-path / per-sample distribution reported (UJI uses
     per-sample but the 6-metric harness can give us groupings).

3. **Build a UJI runner for `WiFiSetTransformer`.** This encoder was
   built in run-1 iter 06 but never evaluated on UJI. Adapt
   `scripts/eval_uji_wifi.py` into a new `scripts/_eval_uji_setxformer.py`
   that swaps `Anchor2Vec` → `WiFiSetTransformer` and reuses the
   exact same dataloader / metric / split. Underscore prefix marks it
   as iteration-scoped (can be promoted later if the encoder wins the
   audit).
   - **Memory budget check (mandatory pre-flight):** forward + backward
     on a synthetic batch at UJI's full input width (520 APs, batch=128)
     — peak GPU MB reported; must be < 6 GB.
   - **Pre-test gate:** 10 % of `trainingData.csv` for 5 epochs first;
     loss drops monotonically, val MAE moves ≥ 10 %.
   - Train on full UJI `trainingData.csv` (no time-window — UJI is
     per-scan), 90 epochs max with patience=15.
   - **Acceptance:** training completes; val mean Euclidean reported.

4. **Try Locaris (Sachini/niloc) — bonus baseline.** Clone
   `https://github.com/Sachini/niloc` to
   `C:\Users\FabLab\AppData\Local\Temp\niloc\`. Look for a UJI
   eval script; if one ships, run it unmodified per Demand #3 and
   record the number. If Locaris doesn't ship a UJI eval, **skip and
   document** — do not spend an iteration writing one. Locaris's
   number, if obtainable, becomes a stronger SOTA reference than
   `wlan_localization` (it's 2025, transformer-based).
   - **Acceptance:** either a number is reported OR the obstacle is
     documented in one paragraph; iteration not blocked.

5. **Six-metric harness on each encoder.** Run
   `src/pipeline/evaluation/encoder_eval.py` on each of (Anchor2Vec,
   WiFiSetTransformer) using UJI val embeddings: linear probe, kNN,
   alignment, uniformity, eff. dim (participation ratio + 95 %
   variance), temporal smoothness, trustworthiness.
   - **Acceptance:** one row per encoder in the headline table; the
     four geometry metrics (alignment / uniformity / eff. dim /
     trustworthiness) inform the "structural saturation" claim that
     run-1 made without evidence — confirm or refute it.

6. **Audit decision (PLAN_02 input).** Based on steps 1-5, label each
   of {Anchor2Vec, WiFiSetTransformer} with `keep` / `modify` /
   `replace` per the hypothesis rubric. One-line justification per
   label quoting numbers.
   - **Acceptance:** explicit table with the labels; recommendation
     for PLAN_02 = IMU audit (next encoder in the audit order); does
     PLAN_02 need a parallel WiFi-encoder modification track or not?

## Sources

- WiFi SOTA repo (vendored): `C:\Users\FabLab\AppData\Local\Temp\wlan_localization\`
  (sharan-naribole, MIT). Existing runner:
  `scripts/eval_wlanloc_uji.py`.
- Locaris (NEW for run 2): https://github.com/Sachini/niloc — clone to
  `C:\Users\FabLab\AppData\Local\Temp\niloc\`.
- UJI benchmark: https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc
  — already DVC-tracked under `data/uji_indoorloc/`.
- Existing eval scripts: `scripts/eval_uji_wifi.py`,
  `scripts/eval_wlanloc_uji.py`.
- Existing encoders: `src/pipeline/encoders/wifi.py` (Anchor2Vec),
  `src/pipeline/encoders/wifi_set.py` (WiFiSetTransformer, sparse-
  observed after run-1 iter_06).
- Previously reported numbers (to reproduce within ±10 %):
  `docs/SOTA_BASELINES.md` — wlan_localization 13.92 m, Anchor2Vec
  8.55 m.
- Six-metric harness: `src/pipeline/evaluation/encoder_eval.py`.

## What to report back

In `handoff/results/RESULT_01_wifi-encoder-audit-uji.md`:

1. Per-step pass/fail with the measured number against each
   acceptance.
2. **Memory budget check** (step 3): peak GPU MB at the UJI-full
   forward+backward.
3. **Pre-test gate outcomes** for each training step (steps 2 and 3):
   did the 10 % subset move ≥ 10 % MAE? Full-training only ran if
   yes.
4. **Headline numbers table:**

   | encoder / method | UJI val mean Euclidean | per-sample p50 | per-sample p90 | params | latency (ms) | source |
   |---|---|---|---|---|---|---|
   | wlan_localization (SOTA, vendored) | …  (target 13.92 ±10 %) | … | … | — | — | docs/SOTA_BASELINES.md |
   | Locaris (NEW, if cloned) | … or `not run, reason: ...` | … | … | … | … | sachini/niloc |
   | Anchor2Vec (ours) | … (target 8.55 ±10 %) | … | … | … | … | eval_uji_wifi.py |
   | WiFiSetTransformer (ours, sparse-observed) | … | … | … | … | … | NEW _eval_uji_setxformer.py |

5. **6-metric harness table** (Anchor2Vec vs WiFiSetTransformer):
   linear probe / kNN / alignment / uniformity / eff. dim /
   trustworthiness / temporal smoothness.
6. **Audit labels:** Anchor2Vec = `keep|modify|replace`,
   WiFiSetTransformer = `keep|modify|replace`, each with a one-line
   justification quoting the numbers.
7. **PLAN_02 recommendation:** Continue to IMU audit as planned, OR
   pause-and-fix the WiFi encoder if the gap to SOTA is structural
   (label `replace` for both). 3-sentence justification.
8. **One open question** for scientist.

## Reversibility

- Step 1 (run vendored baseline): throwaway probe, no code changes.
- Step 2 (existing Anchor2Vec runner): throwaway, runs existing script.
- Step 3 (NEW `scripts/_eval_uji_setxformer.py`): permanent (joins
  the `_eval_*.py` underscore family); engineer commits.
- Step 4 (Locaris clone + eval): vendored under `Temp/`, throwaway.
- Step 5 (6-metric harness on existing encoders): throwaway.
- Step 6 (audit labels): documentation in RESULT_01.

All run artefacts under `runs/encoder_audit_wifi_<ts>/`, gitignored.
Only `RESULT_01.md` + `scripts/_eval_uji_setxformer.py` get
`git add`'d.

**Demand #3 untouched.** Vendored SOTA repos at `Temp/` are not
edited; shims live in our `scripts/` only.

**Compute budget:** total iteration ≤ 90 min.
- Step 1: 5 min (already-runnable script).
- Step 2: 30 min if retraining Anchor2Vec, 5 min if checkpoint exists.
- Step 3: 30-45 min (new runner + train).
- Step 4: 15 min if Locaris cooperates; skip otherwise.
- Step 5: 5 min (eval-only, no training).
- Step 6: 5 min writeup.

If the iteration overruns 90 min, skip step 4 (Locaris is bonus) and
ship RESULT_01 with whatever is complete from steps 1-3 + 5-6.
