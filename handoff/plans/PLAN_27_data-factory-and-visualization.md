# Plan 27 — `src/pipeline/data/` dataset factory + `src/pipeline/visualization/` plotters

> Second consolidation iter. Builds the dataset side of the
> notebook's §0 pre-section: a uniform `load_dataset(name)` API
> across all 7 datasets, `dataset_stats(name)` for the stats table,
> `preprocessing_demo(name, modality)` for the side-by-side
> figures, and a `visualization/` package with the multi-panel
> plotters the notebook + scripts both call.

## Hypothesis

Today each dataset has its own ad-hoc loader scattered across
`src/pipeline/data/` (existing), iteration-scoped `_*.py`
scripts, and inline code in eval runners. The 7 datasets are:

| name | path | modalities | shape | source |
|---|---|---|---|---|
| `webots` | `data/async_collection/` | WiFi+IMU+Camera+Odom | 18 paths, async sensor rates | RESULT_06+ |
| `msiln_site1_b1` | `data/msiln_site1_b1/` | WiFi+IMU | 133 traces, cross-session Nov/Dec | RESULT_15 |
| `imuwifine_floor4` | `data/imuwifine_floor4/` | WiFi+IMU† | 80 paths, "two raw formats" | RESULT_19 |
| `ipin2024_floor0` | `data/ipin2024_floor0/` | WiFi+IMU | 16 paths | RESULT_22 |
| `ronin_canonical` | `data/ronin_frdr/{train,unseen,Pretrained_Models}/` | IMU only | 32 unseen test seqs (FRDR) | RESULT_07 |
| `tartanair_hospital` | `data/tartanair_hospital/hospital/hospital/Easy/P000/` | Camera only | 563 RGB frames | RESULT_08 |
| `uji_indoorloc` | `data/uji_indoorloc/{trainingData,validationData}.csv` | WiFi only | per-scan, 19937/1111 | RESULT_01 |

(† IMUWiFine test paths lack IMU — RESULT_20 audit. The factory
documents this in `dataset_stats`.)

After this iter:
- `from src.pipeline.data import load_dataset, dataset_stats,
  preprocessing_demo` works for all 7 names.
- `from src.pipeline.visualization import plot_dataset_overview,
  plot_per_trajectory, plot_staleness_curve, plot_preprocessing_demo`
  works for all 7 datasets and all 4 modalities.
- Every notebook §0 cell becomes ~2 lines: `stats =
  dataset_stats(name); plot_dataset_overview(name)`.
- Per-trajectory and staleness plots that lived in
  iteration-scoped scripts get promoted to the library.

## Steps

### Step 0 — Inventory current data loaders (10 min)

Engineer audits `src/pipeline/data/` (restored RESULT_06) and the
iteration scripts to map: which dataset's loading lives where?
The DataModule for Webots already exists; the others may be
partial / inline in eval runners.

```powershell
ls src/pipeline/data/
grep -r "data/async_collection" src/ scripts/ --include="*.py" -l
grep -r "data/msiln_site1_b1" src/ scripts/ --include="*.py" -l
# … etc for each dataset
```

**Acceptance**: 1-row-per-dataset table mapping current loader
location → planned new home in `src/pipeline/data/<name>.py`.

### Step 1 — Build per-dataset loader modules (25 min)

Engineer creates one module per dataset. Each exposes the SAME
3 functions (uniform API):

```
src/pipeline/data/
  __init__.py          # re-exports load_dataset, dataset_stats, preprocessing_demo, list_datasets
  factory.py           # the dispatcher (load_dataset(name) -> module's load())
  webots.py            # def load(...); def stats(); def preprocessing_demo(modality)
  msiln.py             # same shape
  imuwifine.py         # same shape — handles "two raw formats" honestly
  ipin2024.py          # same shape (floor 0 default, floor-2/-1 optional)
  ronin_canonical.py   # same shape — IMU only
  tartanair.py         # same shape — Camera only; hospital subset
  uji.py               # same shape — WiFi only, per-scan
  _common.py           # shared helpers: path resolution, normalization, etc.
```

Per-module API:

```python
# src/pipeline/data/webots.py (template)
"""Webots Tiago async_collection — 18-path 4-modality fusion dataset.

Sensor rates per CLAUDE.md:
  IMU ~31 Hz, Odometry ~15 Hz, Ground Truth ~10 Hz,
  WiFi ~1 Hz, Camera ~5 Hz.

Canonical CLAUDE.md split:
  train = paths [1, 3-12]  (11 paths)
  val   = paths [2, 13, 14] (3 paths)
  test  = paths [15, 16, 17] (3 paths)
"""
from __future__ import annotations

def load(split=None, modalities=None, K=4, batch_size=128, **kwargs):
    """Return a DataModule with train/val/test DataLoaders. If split
    is named ('train'/'val'/'test'), return that loader only."""
    # ... wraps the existing FusionTrainer-compatible DataModule

def stats() -> dict:
    """Return per-split sample counts + modality availability +
    sensor rates + per-path summary. Notebook renders this as a
    DataFrame."""
    return {
        "name": "webots",
        "modalities_available": ["wifi", "imu", "camera", "odom"],
        "splits": {"train": {...}, "val": {...}, "test": {...}},
        "sensor_rates_hz": {"imu": 31, "odom": 15, "gt": 10, "wifi": 1, "camera": 5},
        "per_path_summary": [...],   # list of dicts (path_id, length, n_samples, modalities)
        "known_caveats": [],          # e.g. ["Test paths use the canonical [15,16,17] split"]
        "source_result": "RESULT_06+",
    }

def preprocessing_demo(modality: str, n_samples: int = 1) -> dict:
    """Return raw and preprocessed paired samples for the given
    modality. Notebook renders side-by-side. Keys:
      'raw': numpy array or dict of arrays
      'preprocessed': same shape
      'description_raw': short str
      'description_preprocessed': short str
      'preprocessing_pipeline': list of steps applied
    """
    if modality == "wifi":
        # raw RSSI -100..0 → (rssi+100)/100 [0,1] → Anchor2Vec embedding
        ...
    elif modality == "imu":
        # raw 6ch device-frame → world-frame (RoNIN-style rotation)
        ...
    elif modality == "camera":
        # raw RGB → ImageNet-norm → DPVO-norm (2x-0.5)
        ...
    elif modality == "odom":
        # raw 7-column → P-B Δ-features (RESULT_04 winner)
        ...
```

**Same shape for every dataset.** UJI's `preprocessing_demo`
only handles WiFi (the others raise NotImplementedError with a
useful message); RoNIN canonical only handles IMU; TartanAir only
handles camera.

**Honest dataset-specific notes recorded in `stats()`'s
`known_caveats`**:
- IMUWiFine test: "test paths lack IMU by dataset design
  (RESULT_20 audit)"
- IPIN floor 0: "small-train regime (174 WiFi scans + 6924 IMU
  windows train) — fusion can overfit"
- MSILN: "cross-session Nov train / Dec test; test set has
  per-path composition where path 130 dominates"
- RoNIN canonical: "evaluated with RoNIN's own
  `compute_ate_rte` metric (raw RMSE, GT-start-anchored)"
- TartanAir hospital: "no IMU; image-only TartanAir v1; P000
  sample only"
- UJI: "per-scan, no temporal axis"

**Acceptance**: each module exports the 3 functions; smoke
imports work; `dataset_stats(name)` returns a non-empty dict
for every name.

### Step 2 — Build the dispatcher `factory.py` (5 min)

```python
# src/pipeline/data/factory.py
from importlib import import_module
from typing import Any

_REGISTRY = {
    "webots":            "src.pipeline.data.webots",
    "msiln_site1_b1":    "src.pipeline.data.msiln",
    "imuwifine_floor4":  "src.pipeline.data.imuwifine",
    "ipin2024_floor0":   "src.pipeline.data.ipin2024",
    "ronin_canonical":   "src.pipeline.data.ronin_canonical",
    "tartanair_hospital":"src.pipeline.data.tartanair",
    "uji_indoorloc":     "src.pipeline.data.uji",
}

def list_datasets() -> list[str]:
    return list(_REGISTRY)

def _resolve(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list_datasets()}")
    return import_module(_REGISTRY[name])

def load_dataset(name: str, **kwargs) -> Any:
    return _resolve(name).load(**kwargs)

def dataset_stats(name: str) -> dict:
    return _resolve(name).stats()

def preprocessing_demo(name: str, modality: str, **kwargs) -> dict:
    return _resolve(name).preprocessing_demo(modality, **kwargs)
```

`__init__.py` re-exports the 4 functions + `list_datasets`.

**Acceptance**: `from src.pipeline.data import load_dataset,
dataset_stats, preprocessing_demo, list_datasets;
print(list_datasets())` returns the 7 names.

### Step 3 — Build `src/pipeline/visualization/` plotters (25 min)

```
src/pipeline/visualization/
  __init__.py
  dataset_overview.py    # plot_dataset_overview(name)
  trajectory.py          # plot_per_trajectory, plot_path_trajectory
  evaluation.py          # plot_staleness_curve, plot_main_results_heatmap, plot_subset_eval_bar
  preprocessing.py       # plot_preprocessing_demo(raw, preprocessed, modality)
  _style.py              # shared matplotlib style (font sizes, color cycle)
```

Each function:
- Accepts a `dataset_stats` dict OR a clean structured input
  (engineer's choice — pick the simpler one).
- Returns a matplotlib `Figure` (so the notebook can both display
  inline AND save to disk).
- Has a sensible default size + DPI for paper figures.

Per-plotter:

**`plot_dataset_overview(name)` — multi-panel figure** (1 per dataset):
- Panel A: trajectory map (overlaid paths colored by split)
- Panel B: RSSI distribution (if WiFi available; histogram of
  detected APs per scan + RSSI value distribution)
- Panel C: IMU channel histograms (if IMU available; 6 subplots
  for gyro/accel xyz)
- Panel D: per-path length distribution (bar chart)

Datasets where some panels don't apply (UJI no trajectory map;
TartanAir Camera only) skip those panels and document.

**`plot_per_trajectory(pred_xy, gt_xy, path_id, title=None)`**:
- 2D path plot: predicted vs ground-truth.
- Used by §1-2 of notebook + by `scripts/eval_*.py` --save-plots.

**`plot_staleness_curve(lags, mae_values, label=None)`**:
- The robustness paper-figure (RESULT_14 style).
- Linear fit overlay + slope label.

**`plot_subset_eval_bar(subset_dict, title=None)`**:
- 6-bar comparison: `only:wifi` / `only:imu` / ... / `full`.
- Used in notebook §3 to surface the dead-reckoning regime.

**`plot_main_results_heatmap(table_df)`**:
- 6-row × N-arch heatmap (rows = datasets, cols = architectures,
  cell color = MAE normalized by row max).
- Bonus visualization for the cross-dataset comparison.

**`plot_preprocessing_demo(demo_dict, modality)`**:
- 2-panel side-by-side raw vs preprocessed.
- For WiFi: RSSI histogram (raw) → embedding heatmap
  (preprocessed).
- For IMU: time-series 6-channel (raw device-frame) → 6-channel
  (world-frame).
- For Camera: RGB image (raw) → DPVO-norm visualization.
- For Odom: time-series raw columns → Δ-features.

**Acceptance**: every plotter accepts well-defined inputs and
returns a matplotlib Figure; smoke test renders one of each.

### Step 4 — Verification: run the dataset pre-section "manually" (10 min)

Engineer writes a smoke script `scripts/_smoke_data_visualization.py`
that for each of the 7 datasets:
1. Calls `stats = dataset_stats(name)` → prints summary.
2. Calls `fig = plot_dataset_overview(name); fig.savefig(...)` →
   confirms a non-empty figure is produced.
3. For each available modality in the dataset, calls
   `preprocessing_demo(name, modality)` → confirms shape and
   keys.

This is the smoke pass for the notebook §0 cells. If it works,
the notebook §0 just inlines these calls.

**Acceptance**: smoke script runs cleanly for all 7 datasets;
7 dataset-overview PNGs saved under
`runs/overnight/run2_iter_27/dataset_overviews/`; ≤ 30 s
total runtime.

### Step 5 — Documentation (5 min)

Update:
- `docs/PIPELINE.md` (or similar canonical pipeline doc):
  add a section on `src/pipeline/data/` + `visualization/`
  with usage examples.
- `CLAUDE.md`: add the new modules to the "Pipeline
  Architecture" table.

## Sources

- All existing RESULT_NN files for the dataset-specific caveats
  surfaced during run-2.
- `src/pipeline/data/` (restored RESULT_06).
- Iteration scripts under `scripts/` for the per-dataset loaders
  to inline.
- `runs/overnight/run2_iter_*/test_paths/*.png` for the
  per-trajectory plot patterns already battle-tested.
- RESULT_26 `src/pipeline/baselines/` — the visualization package
  may want to call baselines (e.g. `load_basic_encoder4` to
  render DPVO patch features in the preprocessing demo).

## What to report back

In `handoff/results/RESULT_27_data-factory-and-visualization.md`:

1. **Step 0** — current-loader inventory table.
2. **Step 1** — `src/pipeline/data/<name>.py` files created; each
   exports `load`, `stats`, `preprocessing_demo`; smoke imports
   pass.
3. **Step 2** — `factory.py` + `__init__.py` work; `list_datasets()`
   returns 7 names.
4. **Step 3** — `src/pipeline/visualization/` plotter modules; one
   sample figure per plotter type included as PNG.
5. **Step 4** — smoke pass through all 7 datasets; dataset overview
   PNGs filed.
6. **Step 5** — documentation files updated.
7. **One open question** for scientist.

## Reversibility

- Step 1 (loader modules): permanent; engineer commits.
- Step 2 (factory): permanent.
- Step 3 (visualization package): permanent.
- Step 4 (smoke script): permanent; `_smoke_*.py` underscore
  convention.

Files committed: `src/pipeline/data/{factory.py, _common.py,
webots.py, msiln.py, imuwifine.py, ipin2024.py, ronin_canonical.py,
tartanair.py, uji.py}` + updated `__init__.py`;
`src/pipeline/visualization/*`; updated docs.

**Compute budget**: ≤ 80 min.
- Step 0: 10 min (inventory).
- Step 1: 25 min (7 modules × ~5 min each — most logic is reused
  from existing scripts).
- Step 2: 5 min (dispatcher).
- Step 3: 25 min (5 plotter modules — share style helpers).
- Step 4: 10 min (smoke + save 7 PNGs).
- Step 5: 5 min (docs).

If overrun: cut Step 3's `plot_main_results_heatmap` (it's a
bonus visualization; the table itself is the load-bearing
deliverable in PLAN_29's `MainResultsTable`).

If a dataset's `preprocessing_demo` is tricky to implement
cleanly (e.g. camera DPVO trunk activation visualization needs
the trunk loaded), provide a minimal placeholder that returns
the raw modality alone with a `note: "preprocessing visualization
TBD"` key. Notebook §0 still renders, just with thinner content
for that modality.

If `src/pipeline/data/` already has substantial loader code from
prior iters, ENGINEER'S CALL whether to (a) wrap existing
implementations in the new per-dataset modules or (b) refactor
inline. Pick whichever is smaller-surface.
