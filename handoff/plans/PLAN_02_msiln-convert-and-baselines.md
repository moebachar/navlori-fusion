# Plan 02 — Convert Microsoft ILN 2.0 site1/B1 to async_collection + measure trivial baselines

## Hypothesis

Site1/B1 (160 traces, 4 distinct survey days, 0.51 Hz WiFi, 50 Hz IMU,
280 MB on disk — per RESULT_01) is a clean drop-in for the existing
FusionTransformer pipeline once converted to async_collection format.
Before any model training, we need the **trivial baseline floor numbers**
(centroid, WiFi-kNN, IMU dead-reckoning) on this new benchmark so we
know what "decent" means. A meaningful run later must clear those floors
by a clear margin — without this measurement the eventual "fusion wins"
claim has no published reference on this dataset (the autopsy already
showed how easy it is to ship a model that only beats the centroid by
a few metres and call it progress).

This plan does **not** train any model. Training is PLAN_03, gated on
the baseline numbers from this iteration.

### Scientist answer to RESULT_01 Q1 (waypoint GT interpolation)

**Use option (a): linearly time-interpolate waypoint anchors at IMU rate.**
Justification: this is what the Kaggle leaderboard scoring assumes
(predict at every timestamp, scored at the evaluation waypoints), what
RoNIN's `GlobSpeedSequence` does, and what `convert_ipin2024.py` already
does for POSI anchors. Keeps us SOTA-comparable on this dataset's
public metric. Also report the **waypoint-only** metric in step 6 as a
sanity check that the interpolation doesn't materially bias the score.

## Steps

1. **Freeze the cross-session split.** Group the 160 B1 traces by survey
   day (4 days, per RESULT_01: `Nov-24 / Nov-25 / Dec-05 / Dec-06`
   approximately, split 73/75/7/5). Build a deterministic
   `data/msiln_site1_b1/split.json` (mirroring
   `data/ipin2024_floor-2/split.json`) with:
   - `train`: Nov-24 + Nov-25 traces (~148 paths)
   - `val`: Dec-05 traces (~7 paths)
   - `test`: Dec-06 traces (~5 paths)
   Drop traces with < 3 waypoint anchors or duration < 5 s
   (matches `MIN_POSI_PER_SEGMENT=3 / MIN_SEGMENT_SECONDS=5.0` in
   `convert_ipin2024.py`).
   - **Acceptance:** `split.json` written; every retained trace is in
     exactly one split; per-split counts ≥ (50, 3, 3); the day
     boundaries match the surveyor metadata exactly (no day crosses
     splits).

2. **Write `scripts/convert_msiln.py`.** Use `scripts/convert_ipin2024.py`
   as the template — match its output layout, CLI surface, and
   `metadata.json` schema. Per trace:
   - Call vendored `io_f.read_data_file` (imported from
     `C:\Users\FabLab\AppData\Local\Temp\msiln20\io_f.py` via
     `importlib`; Demand #3 — no upstream edits).
   - Linearly interpolate the waypoint `(x, y)` anchors against IMU
     timestamps to produce GT at IMU rate (50 Hz) → resample to 10 Hz
     for `ground_truth.csv` (mirrors IPIN convention `GT_INTERP_HZ=10`).
   - Emit `wifi.csv` rows tagged by `bssid` + `rssi` + `scan_id`
     (matches `convert_ipin2024.py` schema; AP vocabulary built across
     all retained traces in one pass, written to `ap_vocab.json`).
   - Emit `imu.csv` from acce + gyro + (ahrs if present) merged on
     IMU timestamps.
   - Emit `odometry.csv` as a header-only stub (phone dataset has none,
     same as IPIN).
   Output root: `data/msiln_site1_b1/path_NN/`.
   - **Acceptance:** all retained traces (target ≥ 150 of the 160)
     converted with no NaN/Inf; converter wall-clock < 10 min; output
     layout `diff` against `data/ipin2024_floor-2/path_00/` shows
     identical file names + column headers (only data differs).

3. **Write `configs/data/msiln_site1_b1.yaml`.** Mirror
   `configs/data/ipin2024_floor-2.yaml` verbatim, only changing:
   - `name`, `collection_dir`, `source: msiln`
   - `split.{train,val,test}_paths` → read from `split.json` written
     in step 1 (use the same `path_NN` integers).
   Keep `modalities: [wifi, imu]`, `wifi_norm: raw`,
   `wifi_max_stale_s: null`, `imu_frame: world`,
   `windows: {imu: 32, wifi: 1}` — same proven settings as IPIN.
   - **Acceptance:** `from src.pipeline.fusion.builder import
     available_datasets, load_config; 'msiln_site1_b1' in
     available_datasets()` returns `True`; `load_config('msiln_site1_b1')`
     succeeds without error.

4. **Sanity-inspect with `inspect_01_rawdata.py`.** Run the existing
   probe on the converted dataset; it should reproduce (within ±10%)
   the per-floor numbers from RESULT_01's `inspect_msiln.txt` B1 row
   (`0.51 Hz WiFi`, `~50 Hz IMU`, `229.8 × 146.2 m extent`).
   - **Acceptance:** report saved under `runs/overnight/iter_02/`; any
     metric off by >10% is flagged as a converter bug and step 5 is
     skipped pending fix. If converter is correct, train and val IMU
     rate, WiFi rate, NaN counts, GT extent all match within tolerance.

5. **Run `scripts/baselines.py --dataset msiln_site1_b1`.** Same script
   that produces the IPIN baseline table — no modification needed. It
   emits the trivial floors per split:
   - `mean_train_pos` (centroid floor)
   - `wifi_knn_k5` (best-possible WiFi-only with carry-forward)
   - `imu_kalman` (or whatever IMU floor the script computes)
   Report **per-path distribution** (median, p25, p75, p90, max), not
   only the mean.
   - **Acceptance:** `runs/baselines/msiln_site1_b1/<timestamp>/eval.json`
     produced; train/val/test rows present; pasted into RESULT_02.

6. **Compare per-sample vs per-waypoint metric (sanity).** Take the
   centroid baseline predictions and evaluate them two ways: (a) at
   every IMU sample (per-sample MAE), (b) at waypoint timestamps only
   (per-waypoint MAE — the Kaggle leaderboard convention). Single
   notebook-cell or script addition, throwaway. The two numbers should
   differ by < 20 % if linear interpolation isn't biasing things.
   - **Acceptance:** both numbers reported in RESULT_02 with the gap
     quoted as a percentage; if the gap is > 20 %, flag as an open
     question for the scientist (interpolation may need revisiting).

## Sources

- Vendored Microsoft ILN 2.0 starter (Demand #3, do not modify):
  `C:\Users\FabLab\AppData\Local\Temp\msiln20\io_f.py`
- Github upstream of the starter:
  https://github.com/location-competition/indoor-location-competition-20
- Kaggle competition page (metric + leaderboard reference):
  https://www.kaggle.com/competitions/indoor-location-navigation/
- Existing converters to mirror:
  `scripts/convert_ipin2024.py` (linear waypoint interpolation, AP
  vocabulary, `split.json` pattern), `scripts/convert_ronin.py`.
- Existing config to mirror: `configs/data/ipin2024_floor-2.yaml`.

## What to report back

In `handoff/results/RESULT_02_msiln-convert-and-baselines.md`:

1. Per-step pass/fail with the measured number against each acceptance.
2. The `split.json` content (or its day-mapping summary): train/val/test
   counts + day(s) per split.
3. Converter wall-clock; any traces skipped + reason (e.g., < 3
   waypoints).
4. Diff of `inspect_01_rawdata.py` (msiln_site1_b1) vs RESULT_01's
   `inspect_msiln.txt` B1 row — should be flat.
5. **Baselines table** with per-split per-path distribution:

   | split | metric | mean | median | p25 | p75 | p90 | max |
   |---|---|---|---|---|---|---|---|
   | val | centroid | … | … | … | … | … | … |
   | val | wifi_knn | … | … | … | … | … | … |
   | val | imu_drift | … | … | … | … | … | … |
   | test | centroid | … | … | … | … | … | … |
   | test | wifi_knn | … | … | … | … | … | … |
   | test | imu_drift | … | … | … | … | … | … |

6. Per-sample vs per-waypoint metric gap (centroid baseline, val split).
7. **One open question for scientist** — should PLAN_03 train the
   FusionTransformer first (apples-to-apples with IPIN, no changes)
   or upgrade to the per-AP set-transformer encoder
   ([arXiv 2506.00656](https://arxiv.org/abs/2506.00656)) first?
   Engineer's gut call, with one-line justification.

## Reversibility

- Step 1 (`data/msiln_site1_b1/split.json`): **permanent** — committed.
- Step 2 (`scripts/convert_msiln.py`): **permanent** — joins the
  `scripts/convert_*.py` family, committed.
- Step 2 (`data/msiln_site1_b1/path_NN/*.csv`): **leave UNTRACKED for
  this iteration** (gitignored under `data/msiln_*` until we know the
  dataset is useful enough to DVC-add — defer to PLAN_03/04). The
  user can `dvc add` later in one shot.
- Step 3 (`configs/data/msiln_site1_b1.yaml`): **permanent** — committed.
- Steps 4–6 (`runs/overnight/iter_02/*`, `runs/baselines/msiln_*`):
  gitignored under `runs/`. Numbers go in RESULT_02.

**Total touched files in `src/`:** zero. **No model training in this plan.**

**Demand #3:** `io_f.py` is imported via `importlib`, not modified or
re-vendored into `src/`. Any compatibility shims (e.g., numpy / pandas
version quirks) go in `scripts/convert_msiln.py`, never in the
vendored source.
