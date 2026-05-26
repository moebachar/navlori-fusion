# Dataset factory + visualization package

Consolidated in PLAN_27. Two new Python packages under
`src/pipeline/`: `data/` (the dataset factory) and `visualization/`
(plotters for the notebook + scripts).

## Dataset factory (`src.pipeline.data`)

Uniform API across all 7 datasets used in the run-2 main-results
table:

```python
from src.pipeline.data import (
    list_datasets, load_dataset, dataset_stats, preprocessing_demo,
)

list_datasets()
# ['webots', 'msiln_site1_b1', 'imuwifine_floor4', 'ipin2024_floor0',
#  'ronin_canonical', 'tartanair_hospital', 'uji_indoorloc']

stats = dataset_stats("webots")
# {'name': 'webots', 'modalities_available': ['wifi','imu','camera','odom'],
#  'splits': {'train': [...], 'val': [...], 'test': [...]},
#  'sensor_rates_hz': {...}, 'n_paths_total': 18, ...
#  'known_caveats': [...]}

demo = preprocessing_demo("webots", "wifi")
# {'raw': ndarray, 'preprocessed': ndarray,
#  'description_raw': '...', 'description_preprocessed': '...',
#  'preprocessing_pipeline': ['nan_fill(-100)', 'clip', 'affine']}

dm = load_dataset("webots", modalities=["wifi","imu"], K=4, batch_size=128)
# Returns the dataset's primary loader artefact. For the temporally-
# windowed datasets (webots/msiln/imuwifine/ipin2024) this is a
# FusionDataModule; for RoNIN canonical it's a StridedSequenceDataset;
# for TartanAir/UJI it's a dataset-specific dict/tuple.
```

### Per-dataset modules

| name | module | modalities | what `load()` returns |
|------|--------|------------|------------------------|
| `webots` | `src/pipeline/data/webots.py` | WiFi+IMU+Camera+Odom | `FusionDataModule` |
| `msiln_site1_b1` | `src/pipeline/data/msiln.py` | WiFi+IMU (cross-session) | `FusionDataModule` |
| `imuwifine_floor4` | `src/pipeline/data/imuwifine.py` | WiFi+IMU (test no IMU) | `FusionDataModule` |
| `ipin2024_floor0` | `src/pipeline/data/ipin2024.py` | WiFi+IMU (small-train) | `FusionDataModule` |
| `ronin_canonical` | `src/pipeline/data/ronin_canonical.py` | IMU only | `StridedSequenceDataset` |
| `tartanair_hospital` | `src/pipeline/data/tartanair.py` | Camera only | `{'image_files', 'poses_ned', ...}` |
| `uji_indoorloc` | `src/pipeline/data/uji.py` | WiFi only, per-scan | `(X_rssi, Y_xy_bf)` |

Each module also exposes the same `stats()` and `preprocessing_demo(modality)`
functions; the dispatcher in `factory.py` maps the canonical names.

### Dataset caveats surface

The `stats()['known_caveats']` field is **load-bearing** for honest
paper framing — it surfaces the dataset-specific quirks discovered
during run-2:

- **IMUWiFine** test paths lack IMU (RESULT_20 audit) — fusion test = WiFi-only.
- **IPIN floor 0** small-train regime (174 WiFi scans / 6924 IMU windows)
  → fusion archs overfit fast (RESULT_22).
- **MSILN** path-130 composition dominates kNN test mean (RESULT_15).
- **RoNIN canonical** uses RoNIN's own `compute_ate_rte` (NEVER hand-rolled SVD).
- **TartanAir hospital** is image-only, P000 sample (RESULT_08).
- **UJI** per-scan, no temporal axis (RESULT_24 α7 degenerate).
- **Webots** WiFi is GPR-synthesised (optimistic vs real-world).

## Visualization package (`src.pipeline.visualization`)

Plotters used by the notebook + scripts. Every function returns a
`matplotlib.figure.Figure` so callers can both display inline
(Jupyter) and save to disk.

```python
from src.pipeline.visualization import (
    plot_dataset_overview,      # multi-panel per-dataset
    plot_per_trajectory,        # 2D pred vs GT trajectory
    plot_staleness_curve,       # WiFi staleness sweep paper figure
    plot_subset_eval_bar,       # per-subset MAE bars
    plot_main_results_heatmap,  # bonus cross-dataset heatmap
    plot_preprocessing_demo,    # side-by-side raw vs preprocessed
    set_paper_style,            # apply default rcParams
)

# Example: notebook §0 dataset cell
fig = plot_dataset_overview("webots", save_to="figs/webots_overview.png")

# Example: paper staleness figure
fig = plot_staleness_curve(
    lags=[0, 1, 3, 5, 10, 15, 20, 30],
    mae_values=[0.339, 0.352, 0.384, 0.422, 0.536, 0.673, 0.813, 1.088],
    label="cnn1d", slope=0.028,
    title="WiFi staleness: CNN1D K=4 4-mod (RESULT_18)",
    save_to="figs/cnn1d_staleness.png",
)
```

### Shared style

`src.pipeline.visualization._style` exposes `set_paper_style()`
(rcParams) and `color_for(label)` (stable color per method).
The color palette is fixed across the run-2 paper figures.

## Smoke verification

`scripts/_smoke_data_visualization.py` exercises the full data +
visualization stack across all 7 datasets:

```powershell
.venv\Scripts\python.exe scripts\_smoke_data_visualization.py
```

Produces 19 PNGs under
`runs/overnight/run2_iter_27/dataset_overviews/` (7 dataset
overviews + 12 modality preprocessing demos) in ~13 s. This is the
data-pre-section smoke pass for the notebook walkthrough (PLAN_30).
