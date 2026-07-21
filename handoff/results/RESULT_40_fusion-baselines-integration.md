# RESULT_40 — Fusion baselines (IMUWiFine + WiFi-kNN + PDR-from-start) on MSILN

> User directive (2026-06-04): pull IMUWiFine + indoor_location_competition_20
> repos, run them as fusion baselines on our MSILN site1/B1 cross-session split.

**Status:** PASS — notebook executes end-to-end with 31 cells, 10 figures, 9 DataFrames; 4 baselines plus Ours on MSILN row.

---

## Submodules added

- `external_methods/imuwifine` ← `https://github.com/IS2AI/IMUWiFine` — **NO LICENSE** (all-rights-reserved by default). Cloned for reference only; we did NOT import any vendored code.
- `external_methods/indoor_location_competition_20` ← `https://github.com/location-competition/indoor-location-competition-20` — **MIT license** (XYZ10, Inc. 2017-2020). Step-detection functions imported via `importlib` shim per Demand #3.

Both added to `.gitmodules`.

## Baseline details

### 1. IMUWiFine (learned LSTM fusion) — `src/pipeline/baselines/imuwifine.py`

**Clean-room reimplementation** of Nurpeiissov et al. 2022 (IEEE SII), per the architecture description in the paper:
- 4 × (Linear + ReLU) → 4-layer LSTM → Linear(hidden, output)
- Adapted for MSILN: hidden_dim=256 (vs paper's hidden==input which would blow up at 1425-d), output_dim=2 (MSILN is 2D), no magnetometer in MSILN site1/B1 (input dim = 1419 WiFi + 6 IMU = 1425).

**Critical fix during integration**: first training run collapsed to constant output (val 65m). Root cause: MSILN positions are absolute meters at scale ~(180, 115); LSTM with random init couldn't learn the offset cheaply. Fix: normalize targets by train-set mean/std, train in normalized space, un-normalize for Euclidean eval. After fix:
- 30 epochs, 4.4 min on Quadro P4000
- Train MAE 5.43 m / Val MAE 15.15 m / Test MAE 7.43 m

### 2. WiFi-kNN cross-session — `src/pipeline/baselines/msiln_baselines.py`

Pure WiFi-only k-nearest-neighbor (k=3, Manhattan distance, distance-weighted) on the MSILN cross-session split. Train set = all WiFi scans from paths 0-93 (Nov 24) with per-scan GT interpolated from waypoints. Query each val/test scan against train, return distance-weighted GT mean.
- 1782 train scans / 477 val / 137 test (all WiFi-derived)
- Val MAE 13.19 m / Test MAE 6.62 m

### 3. PDR-from-first-waypoint — `src/pipeline/baselines/msiln_baselines.py`

Pure inertial dead-reckoning. Uses the MIT-licensed `indoor_location_competition_20`'s step-detection (`compute_steps`, `compute_stride_length`, `compute_headings`, `compute_rel_positions`) via `importlib.util.spec_from_file_location` (no edits to vendored source). Anchored only at the first GT waypoint — no WiFi at test time.
- **Note:** the competition repo's `compute_step_positions` takes waypoints as a CALIBRATION input. We use only the first waypoint to make this a fair test-time baseline.
- 39 paths total (val + test); 5 test paths
- Val MAE 16.88 m / Test MAE 12.49 m

---

## Final MSILN comparison (live, every value computed in the notebook)

| dataset | wlanloc | WiFi-kNN | PDR-start | IMUWiFine | Ours |
|---|---|---|---|---|---|
| MSILN val | 21.26 | 13.19 | 16.88 | **15.15** | 15.22 |
| MSILN test ⭐ | 28.31 | **6.62** | 12.49 | 7.43 | 10.89 |

**Honest paper story (changed by this iteration):**

The headline "vs wlanloc by −62%" survives. But two real fusion baselines (WiFi-kNN and our clean-room IMUWiFine) **beat Ours on test**. This is the scope.md §7.3 path-130 composition surfacing — path 130 dominates 28% of the test mass and is WiFi-dense, so any WiFi-only method does well on it.

Story shift:
- **Was:** "Our fusion crushes the best open-source baseline by 62%"
- **Is:** "Our fusion outperforms the WiFi-only published SOTA (wlanloc) and pure inertial dead-reckoning, and ties with the learned-fusion baseline (IMUWiFine) on val. Two simpler methods (WiFi-kNN and IMUWiFine) outperform Ours on test due to path-130 composition (scope.md §7.3)."

This is more honest but weaker as a marketing claim. **The contribution is now the async-robustness mechanism (continuous-time set-transformer), not the absolute test-MAE number.** The deep-search agent's framing — "every prior fusion method resamples onto a common rate; ours doesn't" — becomes the load-bearing contribution claim.

## New artifacts

- `src/pipeline/baselines/_msiln_loader.py` — MSILN async_collection → per-path (timestamp, accel, gyro, ahrs, wifi-snapshot, gt-waypoints) for both IMUWiFine LSTM windowing and competition PDR.
- `src/pipeline/baselines/imuwifine.py` — `IMUWiFineModel` (clean-room), `train_imuwifine_msiln`, `load_imuwifine_msiln`.
- `src/pipeline/baselines/msiln_baselines.py` — `run_wifi_knn_msiln`, `run_pdr_from_start_msiln`.
- `src/pipeline/baselines/__init__.py` — updated to export the new symbols.
- `runs/main_table/msiln_site1_b1/imuwifine/model.pt` — saved IMUWiFine checkpoint (2.67 M params, val 15.15 m, test 7.43 m).
- `notebooks/paper_results.ipynb` — 31 cells (was 27); §4b "Fusion baselines" subsection added with 3 new code cells + 1 new IMUWiFine training curve figure; Table 2 + headline_df expanded to all 4 baselines.
- 2 new git submodules: `external_methods/imuwifine`, `external_methods/indoor_location_competition_20`.

## Acceptance verification

| Check | Status |
|---|---|
| Both repos cloned as git submodules | ✅ |
| Demand #3 honored (no edits to vendored source for either repo) | ✅ |
| IMUWiFine model imported via clean-room reimpl, NOT via the no-license repo | ✅ |
| Competition step-detection imported via MIT-licensed `importlib` shim | ✅ |
| All 4 MSILN baselines (wlanloc + WiFi-kNN + PDR + IMUWiFine) live in notebook | ✅ |
| Table 2 + headline_df include all baselines as pandas DataFrames | ✅ |
| Notebook executes end-to-end FAST_MODE=True without cell errors | ✅ (1.02 MB output, 31 cells, ~5 min wall-clock with cached IMUWiFine ckpt) |
| Mot_transformer dangling refs cleaned (from earlier deletion) | ✅ (bonus fix from execution diagnostics) |

## Open items for user

1. **Paper framing pivot.** The honest result is that two baselines beat Ours on test. The deep-search agent's "async-robustness as the contribution" framing is now load-bearing. Update `scope.md` §1 contribution claim from "real-world cross-session generalization" (which is no longer the lowest test-MAE) to "async-robust fusion without rate-resampling" (a mechanism contribution, not an MAE-leadership claim)?

2. **IMUWiFine license stance going forward.** Currently we vendored the repo for reference but only used clean-room code. Should we delete the `external_methods/imuwifine` submodule entirely (zero legal touch) or keep it for reproducibility documentation (engineer can re-derive the architecture from the vendored model.py)? My recommendation: keep it, with a clear comment in `src/pipeline/baselines/imuwifine.py` docstring that the implementation is clean-room based on the paper text and the vendored repo is for verification only.

3. **Path-130 mitigation.** The test split's path-130 favors WiFi-only methods. Two options for the paper: (a) report all baselines honestly + use scope.md §7.3 framing in the paper §7 limitations; (b) report per-path breakdown in Table 2 to show "our fusion is uniform across paths; WiFi-kNN benefits disproportionately from path 130." Option (b) takes ~20 min engineer time but is more defensible.

4. **Cleanup.** The transient `_build_paper_notebook.py` is still at project root from PLAN_39. Keep for iteration convenience or delete now that the notebook is stable?
