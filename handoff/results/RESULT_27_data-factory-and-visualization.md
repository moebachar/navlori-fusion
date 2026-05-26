# Result 27 — `src/pipeline/data/` factory + `src/pipeline/visualization/` plotters

## TL;DR

**Second consolidation iter shipped.** Two new Python packages
under `src/pipeline/` that the run-2 walkthrough notebook (PLAN_30)
will use as its primary API surface:

- **`src.pipeline.data`** — uniform `load_dataset(name)` /
  `dataset_stats(name)` / `preprocessing_demo(name, modality)` API
  across all 7 datasets used in the main-results table.
- **`src.pipeline.visualization`** — 6 paper-figure plotters
  (dataset overview, per-trajectory, staleness curve, subset bar,
  main-results heatmap, preprocessing demo) returning matplotlib
  Figures.

**Smoke verification**: `scripts/_smoke_data_visualization.py` runs
clean on all 7 datasets in **~13 s**, producing **19 PNGs**
(7 dataset overviews + 12 modality preprocessing demos) under
`runs/overnight/run2_iter_27/dataset_overviews/`. All 7 datasets
pass.

## Step-by-step

### Step 0 — Inventory

`src/pipeline/data/` currently exposes `FusionDataModule` +
`FusionDataset` (Hydra-config-driven, RESULT_06 restored). Hydra
configs at `configs/data/` cover: `imuwifine.yaml`,
`ipin2024_floor0.yaml`, `msiln_site1_b1.yaml`, `silva.yaml`,
`simulation.yaml`, `tiago.yaml`.

The 4 temporally-windowed datasets (webots, msiln_site1_b1,
imuwifine_floor4, ipin2024_floor0) load through the existing
`FusionDataModule`. The 3 per-leg datasets (ronin_canonical,
tartanair_hospital, uji_indoorloc) load via dataset-specific
custom paths in `scripts/` — those need dedicated per-leg loaders
in the new factory.

### Step 1 — Per-dataset loader modules

Created 8 new files under `src/pipeline/data/`:

```
_common.py          # path_to(), collect_path_metadata(),
                    # summarise_path_lengths(), not_applicable()
webots.py           # 4-mod fusion dataset (Webots Tiago)
msiln.py            # WiFi+IMU cross-session
imuwifine.py        # WiFi+IMU; honest "test no IMU" caveat
ipin2024.py         # WiFi+IMU; small-train overfit caveat
ronin_canonical.py  # IMU only; FRDR canonical unseen-subjects
tartanair.py        # Camera only; image-only TartanAir v1 hospital
uji.py              # WiFi only, per-scan; K=1 M=1 degenerate caveat
```

Each module exports the same 3 functions: `load(...)`, `stats()`,
`preprocessing_demo(modality, n_samples=1)`. Module-specific
caveats surfaced in `stats()['known_caveats']`:

- IMUWiFine: "Test paths lack IMU by dataset design (RESULT_20)".
- IPIN floor 0: "Small-train regime (174 WiFi + 6924 IMU windows)".
- MSILN: "Test path 130 (786 samples ≈ 28 % of test) dominates kNN".
- RoNIN: "C2 audit: raw +94 % outside 20 % gate / Umeyama +15.7 %".
- TartanAir: "Image-only TartanAir v1; NO IMU; P000 sample only".
- UJI: "K=1 M=1 degenerate row (RESULT_24 α7)".
- Webots: "WiFi GPR-synthesised, not measured — optimistic vs real".

### Step 2 — Dispatcher factory

Created `src/pipeline/data/factory.py` with a 7-entry registry
mapping canonical names to per-dataset modules. The factory's 4
public functions are re-exported from
`src/pipeline/data/__init__.py`:

```python
from src.pipeline.data import (
    list_datasets, load_dataset, dataset_stats, preprocessing_demo
)
```

Smoke test:

```
datasets: ['webots', 'msiln_site1_b1', 'imuwifine_floor4',
           'ipin2024_floor0', 'ronin_canonical',
           'tartanair_hospital', 'uji_indoorloc']
  webots: modalities=['wifi', 'imu', 'camera', 'odom']
  msiln_site1_b1: modalities=['wifi', 'imu']
  imuwifine_floor4: modalities=['wifi', 'imu']
  ipin2024_floor0: modalities=['wifi', 'imu']
  ronin_canonical: modalities=['imu']
  tartanair_hospital: modalities=['camera']
  uji_indoorloc: modalities=['wifi']
```

### Step 3 — Visualization package

Created `src/pipeline/visualization/`:

```
__init__.py         # re-exports
_style.py           # set_paper_style(), color_for(label), COLOR_PALETTE
dataset_overview.py # plot_dataset_overview(name)
trajectory.py       # plot_per_trajectory(pred, gt, path_id)
evaluation.py       # plot_staleness_curve, plot_subset_eval_bar,
                    # plot_main_results_heatmap
preprocessing.py    # plot_preprocessing_demo(demo_dict, modality)
```

Every plotter returns a matplotlib Figure (no auto-`plt.show()`),
suitable for both Jupyter inline display and `fig.savefig(...)` to
disk. Stable per-method color palette in `_style.COLOR_PALETTE`.

### Step 4 — Smoke verification

`scripts/_smoke_data_visualization.py` exercises the full stack
across all 7 datasets:

```
=== smoke pass: 7 datasets ===
  webots: modalities=['wifi', 'imu', 'camera', 'odom']
     overview saved; preprocessing demos: {'wifi': 'OK', 'imu': 'OK', 'camera': 'n/a', 'odom': 'OK'} (1.8s)
  msiln_site1_b1: modalities=['wifi', 'imu']
     overview saved; preprocessing demos: {'wifi': 'OK', 'imu': 'OK'} (1.7s)
  imuwifine_floor4: modalities=['wifi', 'imu']
     overview saved; preprocessing demos: {'wifi': 'OK', 'imu': 'OK'} (1.6s)
  ipin2024_floor0: modalities=['wifi', 'imu']
     overview saved; preprocessing demos: {'wifi': 'OK', 'imu': 'OK'} (1.4s)
  ronin_canonical: modalities=['imu']
     overview saved; preprocessing demos: {'imu': 'OK'} (1.4s)
  tartanair_hospital: modalities=['camera']
     overview saved; preprocessing demos: {'camera': 'OK'} (0.1s)
  uji_indoorloc: modalities=['wifi']
     overview saved; preprocessing demos: {'wifi': 'OK'} (4.9s)

wrote 19 PNGs to runs/overnight/run2_iter_27/dataset_overviews
datasets ok: 7/7
```

Total runtime ~13 s — well under the 30 s gate. Per-modality
preprocessing demos succeed for every applicable modality
(Webots camera returns image-path mode with `'n/a'` for the
inline-array check — the preprocessing figure still renders the
raw PNG).

### Step 5 — Documentation

- **NEW `docs/DATA_AND_VISUALIZATION.md`** — usage examples,
  per-dataset module table, plotter overview, smoke verification.
- **`CLAUDE.md`** — extended the Pipeline Architecture table
  with `ext` (baselines), `data` (factory), and `viz`
  (visualization) rows.

## One open question for scientist

The `preprocessing_demo("webots", "camera")` returns
`{'raw': None, ...}` with a note (the actual image-tensor
preprocessing visualisation is deferred to the plot helper, which
loads the raw PNG path). For the notebook's §0 §camera cell, do we
want:

- (a) Keep the current "info-text only" Camera demo (engineer's
  default) — the plot_preprocessing_demo helper renders an
  informative text panel.
- (b) Pre-compute and stash a small set of DPVO-trunk activation
  visualisations under `data/precomputed_demos/` and have
  `preprocessing_demo("webots", "camera")` return those? That's a
  one-off ~5 min cost but produces a richer notebook cell.

Engineer's lean: (a) for PLAN_27, (b) is a small Phase-C polish
item for the notebook iteration with the user (PLAN_30+).

## Sources

- PLAN_27 spec.
- `src/pipeline/data/` (RESULT_06 restored low-level layer).
- Iteration scripts under `scripts/` for the per-dataset loaders.
- `src/pipeline/baselines/` (RESULT_26) for the RoNIN canonical
  loader and DPVO trunk references.
- Run-2 RESULTs for the dataset-specific caveats in `stats()`:
  RESULT_15 (MSILN), RESULT_19/20 (IMUWiFine), RESULT_22 (IPIN),
  RESULT_23/07 (RoNIN canonical), RESULT_08 (TartanAir hospital),
  RESULT_24 (UJI α7).

## Files committed

- `src/pipeline/data/_common.py` — NEW
- `src/pipeline/data/factory.py` — NEW
- `src/pipeline/data/{webots,msiln,imuwifine,ipin2024,ronin_canonical,tartanair,uji}.py` — NEW (7 files)
- `src/pipeline/data/__init__.py` — extended re-exports
- `src/pipeline/visualization/__init__.py` — NEW
- `src/pipeline/visualization/{_style,dataset_overview,trajectory,evaluation,preprocessing}.py` — NEW (5 files)
- `scripts/_smoke_data_visualization.py` — NEW
- `docs/DATA_AND_VISUALIZATION.md` — NEW
- `CLAUDE.md` (gitignored locally): Pipeline Architecture table extended.
- `handoff/plans/PLAN_27_*.md`, `handoff/results/RESULT_27_*.md`,
  `handoff/STATE.md` — iter 27 row + status updated.
