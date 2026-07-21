# External dependencies — SOTA baseline repositories

All open-source SOTA baselines used in this project live under
`external_methods/` as **Git submodules**. They are vendored
unmodified per **Demand #3** of the project rubric (no edits to
upstream sources; all compat shims live in OUR wrapper code at
`src/pipeline/baselines/_shims.py`).

## Setup after `git clone`

```powershell
git clone https://github.com/<owner>/navlori-fusion.git
cd navlori-fusion
git submodule update --init --recursive
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

After submodule init, the six vendored repos are in
`external_methods/{wlan_localization,ronin,tartanvo,dpvo,imuwifine,indoor_location_competition_20}/`.
(`imuwifine` and `indoor_location_competition_20` were added 2026-06-04,
RESULT_40, for the MSILN fusion baselines — see §5/§6 below.)

## Submodule inventory

### 1. `external_methods/wlan_localization` — WiFi SOTA

| field | value |
|-------|-------|
| upstream | https://github.com/sharan-naribole/wlan_localization |
| branch | `master` |
| pinned commit | `5e1949dac00b779268eca26b13081e9ee901c47e` |
| license | MIT |
| what we use | `src/wlan_localization/models/position_regressor.py::PositionRegressor`; `src/wlan_localization/data/preprocessor.py::DataPreprocessor` |
| our loader | `src.pipeline.baselines.load_position_regressor()`, `load_preprocessor()` |
| used by RESULT | 01 (UJI), 15 (MSILN), 19 (IMUWiFine), 22 (IPIN floor 0) |

Loading approach: their package `__init__` drags in
`imbalanced-learn` ↔ `scikit-learn` version conflicts in our venv,
so the loader uses `importlib.util.spec_from_file_location` to
import their two class files directly. No edits to vendored
sources.

### 2. `external_methods/ronin` — IMU SOTA

| field | value |
|-------|-------|
| upstream | https://github.com/Sachini/ronin |
| branch | `master` |
| pinned commit | `805b7f0f28bb164ce89ada9ac05a9470dbe3d715` |
| license | MIT |
| what we use | `source/model_resnet1d.py::{BasicBlock1D, FCOutputModule, ResNet1D}`, `source/data_glob_speed.py::{GlobSpeedSequence, StridedSequenceDataset}`, `source/metric.py::compute_ate_rte`; `lists/list_{train,test_unseen,val}.txt` |
| our loader | `src.pipeline.baselines.{BasicBlock1D, ..., load_test_list, load_train_list}` |
| used by RESULT | 02 (a000 proxy), 07 (canonical unseen-subjects), 23 (CNN1D/LSTM-attn aggregator over IMUCNN sub-windows) |

Compat shim needed: `np.int = int` (removed in numpy 1.20+).
Applied in `src.pipeline.baselines._shims.apply_np_int_shim()`,
called automatically by `src.pipeline.baselines.ronin` on import.

Pretrained ResNet1D checkpoint (reproduces paper's 5.14 m on
canonical unseen-subjects): not in the submodule; placed by the
user under `data/ronin_frdr/pretrained_resnet/ronin_resnet/checkpoint_gsn_latest.pt`
(extracted from FRDR dataset 538, RESULT_07).

### 3. `external_methods/tartanvo` — Camera SOTA (visual odometry)

| field | value |
|-------|-------|
| upstream | https://github.com/castacks/tartanvo |
| branch | `python3` (not `master` — needed for Python 3.10+ compat) |
| pinned commit | `ec2ecc38b590ff27b76cc7818cf198d6420dce4b` |
| license | MIT |
| what we use | full inference pipeline (`TartanVO.py` + `Network/` + `Datasets/`) |
| our loader | `src.pipeline.baselines.load_vo_module()`, `apply_tartanvo_shims()` |
| used by RESULT | 08 (TartanAir hospital P000 last-20% slice ATE 0.012 m) |

Compat shims needed (3):
- `scipy.spatial.transform.Rotation.as_dcm` → `as_matrix` (renamed in
  scipy 1.4).
- `numpy.linalg.linalg` (deprecated nested submodule, referenced by
  the PWC correlation kernel).
- `cupy.cuda.compile_with_cache` (removed in cupy 12+; mapped to
  `RawModule`).

All applied by `apply_tartanvo_shims()`. Demand #3 honoured: shims
monkey-patch our process's loaded modules; vendored sources are
untouched.

**TartanVO weights**: not in the submodule by upstream design (their
README points to a download). User places `tartanvo_1914.pkl`
under `external_methods/tartanvo/` (the runner code expects it
relative to the repo root).

### 4. `external_methods/dpvo` — DPVO (deep patch visual odometry)

| field | value |
|-------|-------|
| upstream | https://github.com/princeton-vl/DPVO |
| branch | `main` |
| pinned commit | (current `main` HEAD as of submodule add; not pinned to a specific commit because we use only the `BasicEncoder4` extractor which is stable) |
| license | BSD (3-clause) |
| what we use | `dpvo/extractor.py::BasicEncoder4` (the dense patch feature extractor) |
| our loader | `src.pipeline.baselines.BasicEncoder4`, `get_basic_encoder4_class()`, `load_basic_encoder4(weights_path)` |
| used by | `src/pipeline/encoders/dpvo_motion.py` (the trainable Camera fusion encoder) |
| used by RESULT | 03 (Camera encoder audit), 08 (Camera ext-SOTA), 09-18 (4-modality fusion with DPVOMotionEncoder) |

**Windows limitation**: DPVO's full SLAM pipeline needs `lietorch`
and `altcorr` custom CUDA ops which do NOT build on Windows. Our
project only uses the encoder trunk (extractor), which runs on
CPU/GPU without those custom ops. RESULT_08's full SLAM evaluation
was done via TartanVO instead, on the same TartanAir hospital
sequence.

If Linux/WSL2 build is needed for a future Phase C extension:
```bash
cd external_methods/dpvo
pip install -e .
python setup.py build_ext --inplace
```
(Requires CUDA toolkit + GCC + an Eigen install. See upstream
README.)

## 5. `external_methods/indoor_location_competition_20` — MSILN dataset tooling

*(Promoted from data-only reference to submodule on 2026-06-04, RESULT_40 —
the earlier "not a submodule, 2.1 GB" stance was reversed once the MSILN
baselines needed its `compute_f`/`io_f` utilities at import time.)*

| field | value |
|-------|-------|
| upstream | https://github.com/location-competition/indoor-location-competition-20 |
| pinned commit | see `git submodule status` |
| license | (refer to upstream) |
| what we use | `compute_f`/`io_f` utilities + starter structure for the Microsoft Indoor Localization 2.0 (MSILN) dataset |
| our loader | `src.pipeline.baselines._msiln_loader`, `src.pipeline.data.msiln` |
| referenced by | `scripts/convert_msiln.py`, RESULT_15/40, `configs/data/msiln_site1_b1.yaml` |

## 6. `external_methods/imuwifine` — WiFi+IMU learned-fusion baseline

*(Added 2026-06-04, RESULT_40.)*

| field | value |
|-------|-------|
| upstream | https://github.com/IS2AI/IMUWiFine |
| pinned commit | see `git submodule status` |
| license | **NONE published (all rights reserved)** — only clean-room re-implementations are executed (`src/pipeline/baselines/imuwifine.py`); the submodule is reference-only. Keep-vs-drop decision pending (RESULT_40 item 2). |
| what we use | architecture/protocol reference for the clean-room LSTM baseline on MSILN; IMUWiFine floor-4 dataset provenance |
| our loader | `src.pipeline.baselines.imuwifine` (clean-room, no upstream imports) |
| referenced by | RESULT_40, paper §6.3 (IMUWiFine fl.4) |

## Compat shims summary

All shims live in `src/pipeline/baselines/_shims.py`. Each is
idempotent. None of them edits vendored source files (Demand #3).

| shim | what it patches | which baseline needs it |
|------|-----------------|--------------------------|
| `apply_np_int_shim()` | `numpy.int = int` | RoNIN's `data_glob_speed.py` |
| `apply_scipy_as_dcm_shim()` | `Rotation.as_dcm = as_matrix` (& `from_dcm`) | TartanVO `Datasets/transformation.py` |
| `apply_numpy_linalg_submodule_shim()` | `numpy.linalg.linalg = numpy.linalg` | TartanVO PWC correlation kernel |
| `apply_cupy_compat_shim()` | `cupy.cuda.compile_with_cache` via `RawModule` | TartanVO PWC correlation kernel |

The `tartanvo` loader calls all 3 of its needed shims automatically
via `apply_tartanvo_shims()`. The `ronin` loader applies
`apply_np_int_shim()` automatically on import. `wlanloc` and
`dpvo_trunk` need no shims.

## Reproducibility check

After cloning and submodule-init, the simplest smoke test is:

```powershell
.venv\Scripts\python.exe -c "from src.pipeline.baselines import *; print(EXTERNAL_METHODS)"
.venv\Scripts\python.exe scripts\eval_wlanloc_uji.py
```

The latter should print `global mean Euclidean = 15.171 m`
(reproduces RESULT_01's headline number).
