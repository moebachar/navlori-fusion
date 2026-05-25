# Plan 01 — Microsoft Indoor Location & Navigation (ILN 2.0): downloadability + schema feasibility probe

## Hypothesis

The IPIN 2024 benchmark caps useful WiFi+IMU fusion error near 4 m
(autopsy Probe 2.1: centroid floor 4.18 m on IPIN val under WiFi
carry-forward). To reach the publishable 1–3 m target we need a denser
WiFi benchmark with a known public SOTA in that range.

**Microsoft Indoor Location & Navigation (ILN 2.0, Kaggle 2021)** is the
strongest candidate identified in the literature scan:

- Both WiFi RSSI scans **and** full smartphone IMU streams (accel/gyro/mag,
  ~50–200 Hz on Android).
- Multi-day cross-session train/test (the publishable axis).
- Multi-floor, 204 buildings, surveyor-clicked waypoint ground truth.
- **Documented public SOTA of 1.3–1.6 m** (Kaggle private-LB winner
  "Track me if you can"; MobiCom 2023 retrospective reports 0.72 m
  infrastructure-based / 1.56 m infrastructure-free).
- Full dataset ~24 GB, but a single-site subset fits well under 5 GB.

Before committing to converting it to `async_collection` format and
porting the FusionTransformer pipeline (PLAN_02+), we probe whether:
(a) we can actually obtain the data on this Windows machine,
(b) the schema is compatible with our existing converters,
(c) the WiFi/IMU rates are as advertised, and
(d) the per-site disk footprint fits.

This plan touches **no existing code or configs**. All artefacts land in
new files; if the probe says NO-GO, nothing in the repo is dirtied.

## Steps

1. **Vendor the starter repo.** Clone
   `https://github.com/location-competition/indoor-location-competition-20`
   to `C:\Users\FabLab\AppData\Local\Temp\msiln20` (matches the
   demand-#3 vendored-baseline pattern used for `wlan_localization` and
   `ronin`). The repo ships <100 MB of sample data sufficient for schema
   inspection if step 2 fails.
   - **Acceptance:** clone succeeds; the `data-sample/` directory (or
     equivalent name in the repo) exists and contains at least one
     `*.txt` waypoint trace.

2. **Full-dataset acquisition (1 site only, smallest possible).** Try
   each of the following until one works; do **not** download the full
   24 GB.
   - 2a. Look for a HuggingFace mirror first (
     `huggingface.co/datasets?search=indoor-location-navigation` or
     `aka.ms/location20dataset`).
   - 2b. If unavailable, attempt `kaggle competitions download -c
     indoor-location-navigation -f <smallest-site>.zip` — requires
     `~/.kaggle/kaggle.json`; if missing, stop and report.
   - 2c. If both fail, fall back to using only the github sample data
     from step 1 and **note this clearly in RESULT_01** — the probe
     can still partially proceed.
   - **Acceptance:** at least one site (or the github sample) is
     extracted under `data_raw/msiln_<site>/`. If none of 2a/2b/2c
     succeeds, STOP at step 2 and document the obstacle (this is a
     valid result, not a failure to plan).

3. **Schema inspection script.** Write `scripts/inspect_msiln.py` that
   loads ONE site's data and reports the same metrics as the autopsy
   Probe 1 table:
   - GT extent (m × m) — floor bounding box from waypoint annotations.
   - GT rate (Hz) and step distribution (cm: med / p90 / max).
   - WiFi scan rate (Hz), WiFi NaN/missing %, APs visible / total.
   - IMU rate (Hz) and any obvious NaN/Inf.
   - Number of sessions / paths in the site, ideally split by day.
   - **Acceptance:** `runs/overnight/iter_01/inspect_msiln.txt` written
     with all of these fields plus a one-paragraph human-readable
     summary at the top. Script committed; report file
     gitignored (consistent with `runs/`).

4. **Comparability assessment (judgement call, in RESULT_01).** Answer
   these four yes/no questions explicitly with one-sentence evidence:
   - (a) Is WiFi scan rate ≥ 0.5 Hz? (target: yes — denser than IPIN's
     0.15–0.25 Hz, the ceiling driver per autopsy Probe 2.)
   - (b) Is per-site disk usage < 5 GB? (target: yes — fits the budget.)
   - (c) Does the schema map naturally to async_collection (timestamped
     per-modality files + GT)? (target: yes; minor wrangling OK, major
     redesign = NO-GO.)
   - (d) Are cross-session train/test splits available (different days,
     same site)? (target: yes — this is the publishable transfer axis.)
   - **Acceptance:** all four answered with evidence in RESULT_01.

5. **GO / NO-GO recommendation.** Based on (4), one paragraph:
   - If GO: name the site for the first conversion (PLAN_02 will write
     `scripts/convert_msiln.py`); estimate disk + conversion time.
   - If NO-GO: identify the specific blocker and recommend the
     next-best candidate from the literature scan:
     [Fusion-DHL](https://github.com/Sachini/Fusion-DHL) (smaller, has
     open code) or [PerfLoc/NIST](https://www.nist.gov/ctl/pscr/perfloc-data)
     (free download, no Kaggle auth).

## Sources

- Microsoft Indoor Location Competition 2.0 starter:
  https://github.com/location-competition/indoor-location-competition-20
- Kaggle competition (data + leaderboard):
  https://www.kaggle.com/competitions/indoor-location-navigation/data
- Kaggle winning solution (~1.3 m), H2O.ai writeup:
  https://h2o.ai/blog/2021/what-does-it-take-to-win-a-kaggle-competition-lets-hear-it-from-the-winner-himself/
- MobiCom 2023 retrospective on this dataset:
  https://feng-qian.github.io/paper/localization_competition_mobicom23.pdf
- Backup dataset 1 — Fusion-DHL (Henniges et al., ICRA 2021):
  arXiv https://arxiv.org/abs/2105.08837 · code https://github.com/Sachini/Fusion-DHL
- Backup dataset 2 — NIST PerfLoc:
  https://www.nist.gov/ctl/pscr/perfloc-data

## What to report back

In `handoff/results/RESULT_01_msiln-feasibility-probe.md`:

1. Per-step pass / fail with the measured number against each acceptance.
2. **Verbatim contents of `inspect_msiln.txt`** (paste it in, plus a
   path link to the live file).
3. The four yes/no answers from step 4 with evidence sentences.
4. The GO / NO-GO recommendation with one-paragraph justification.
5. Total disk used by the downloaded site (or by the sample).
6. Wall-clock time for steps 1–3 (calibrates engineer iteration speed
   for subsequent plans).
7. Any auth obstacles, schema surprises, or NaN/Inf weirdness.
8. **One open question** for the scientist that the next plan should
   answer.

## Reversibility

- **Step 1** (vendored clone under `Temp/`): throwaway — re-cloneable.
- **Step 2** (downloaded site under `data_raw/`): throwaway — untracked,
  can be deleted any time.
- **Step 3** (`scripts/inspect_msiln.py`): **permanent** — joins the
  `scripts/inspect_*.py` family of diagnostic probes. Engineer commits
  it. Has no runtime dependency on the dataset existing (graceful
  error on missing data).
- **Steps 4–5** (RESULT_01 narrative): documentation only.

**Nothing in this plan touches** `src/`, `configs/`, model weights,
existing baselines, or the IPIN evaluation. IPIN remains the secondary
benchmark untouched.
