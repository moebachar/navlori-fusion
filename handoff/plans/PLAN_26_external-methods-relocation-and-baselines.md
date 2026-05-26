# Plan 26 — Relocate external SOTA repos to `external_methods/`, centralise loaders in `src/pipeline/baselines/`

> First iteration of the post-run-2 consolidation. The user's
> directive: pull all SOTA dependencies into one folder
> `external_methods/` so they can ship as Git submodules on
> GitHub; adapt every wrapper to import from the new path.
>
> This plan does that PLUS centralises the per-baseline
> importlib + shim machinery into `src/pipeline/baselines/` —
> so wrappers go from ~30-50 lines of boilerplate per baseline
> down to ~3 lines of import. The notebook will use the same
> `src.pipeline.baselines.*` API as the scripts.
>
> Foundational iteration: subsequent consolidation plans
> (data, fusion, training, evaluation, visualization, notebook)
> all depend on `external_methods/` being in place.

## Hypothesis

Today the SOTA dependencies are scattered:

| repo | current location | how we use it | license |
|---|---|---|---|
| `sharan-naribole/wlan_localization` | `C:\Users\FabLab\AppData\Local\Temp\wlan_localization\` | WiFi SOTA: `PositionRegressor`, `DataPreprocessor` | MIT |
| `Sachini/ronin` | `C:\Users\FabLab\AppData\Local\Temp\ronin\` | IMU SOTA: `model_resnet1d.{BasicBlock1D, FCOutputModule, ResNet1D}`, `data_glob_speed.{GlobSpeedSequence}`, `metric.compute_ate_rte` | MIT |
| `castacks/tartanvo` (python3 branch) | `C:\Users\FabLab\AppData\Local\Temp\tartanvo\` | Camera SOTA: VO pipeline | MIT |
| `princeton-vl/DPVO` | `external/dpvo/` (partial — only extractor.py + dpvo.pth) | encoder trunk for `DPVOMotionEncoder` | BSD |
| (MSILN starter) | `C:\Users\FabLab\AppData\Local\Temp\msiln20\` | (not used as SOTA — data only) | n/a |

After this plan:
- All five repos live under `<project_root>/external_methods/<repo_name>/`
  as Git submodules.
- A single `src/pipeline/baselines/` Python package exposes a clean
  API for each baseline (load + shim + run).
- Every existing wrapper (`scripts/eval_*.py` + `_eval_*.py`)
  imports from `src.pipeline.baselines.*` instead of carrying
  duplicated `importlib`/`sys.path` machinery.
- Running `eval_wlanloc_uji.py` after the relocation produces
  the same UJI number as RESULT_01 (15.17 m global).

## Steps

### Step 0 — Inventory + decision: full submodule clones vs partial vendoring (10 min)

For each repo, decide:
- **Full Git submodule clone**: clean GitHub presentation; user
  runs `git submodule update --init --recursive` after cloning.
  Default for all 4 SOTA repos.
- **Already-partial vendoring**: `external/dpvo/` currently has
  only `extractor.py` + `__init__.py` (not the full DPVO repo —
  `lietorch`/`altcorr` CUDA ops never built on Windows). Decision:
  either (a) full DPVO submodule + document Windows build
  limitation; (b) keep the partial vendoring and document it as
  an "extractor-only" use.

**Acceptance**: documented decision per repo; for each, the
upstream URL + a pinned commit hash from the upstream history.

Engineer probes each `Temp/<repo>/.git` to get the current
commit hash:

```powershell
foreach ($r in 'wlan_localization','ronin','tartanvo','msiln20') {
    pushd "C:\Users\FabLab\AppData\Local\Temp\$r"
    git rev-parse HEAD
    git remote get-url origin
    popd
}
```

(Note: tartanvo is on python3 branch, not master — engineer
records the branch name too.)

### Step 1 — Create `external_methods/` and add submodules (15 min)

```powershell
git submodule add -b master https://github.com/sharan-naribole/wlan_localization external_methods/wlan_localization
git submodule add -b master https://github.com/Sachini/ronin external_methods/ronin
git submodule add -b python3 https://github.com/castacks/tartanvo external_methods/tartanvo
git submodule add -b master https://github.com/princeton-vl/DPVO external_methods/dpvo
```

(Branch flags exact-pinned per the Step 0 inventory.)

For each submodule, pin to the commit hash from Step 0 to lock
the exact version we tested with:

```powershell
foreach ($r in 'wlan_localization','ronin','tartanvo','dpvo') {
    pushd "external_methods/$r"
    git checkout <commit-hash-from-step-0>
    popd
}
git add external_methods .gitmodules
```

MSILN starter (`msiln20`): NOT a SOTA baseline, just a data
starter. Document in `docs/EXTERNAL_DEPENDENCIES.md` but
**don't add as a submodule** — its 2.1 GB of bundled data
shouldn't live in our `external_methods/`. Document the URL +
how to extract data only.

**Acceptance**: `.gitmodules` is written; `git submodule status`
shows all 4 submodules at the pinned commits.

### Step 2 — Verify each submodule's content matches what we use (10 min)

For each submodule, confirm the files our wrappers import exist
at the new path:

| baseline | files imported | new path |
|---|---|---|
| wlan_localization | `src/wlan_localization/models/position_regressor.py`, `src/wlan_localization/data/preprocessor.py` | `external_methods/wlan_localization/src/wlan_localization/...` |
| ronin | `source/model_resnet1d.py`, `source/data_glob_speed.py`, `source/metric.py`, `source/ronin_resnet.py`, `lists/*.txt` | `external_methods/ronin/source/...`, `external_methods/ronin/lists/...` |
| tartanvo | `Network/`, `Datasets/`, vendored runner files | `external_methods/tartanvo/...` |
| dpvo | `dpvo/extractor.py` (and full repo for future Linux/WSL2 lietorch build) | `external_methods/dpvo/dpvo/extractor.py` |

If a file is missing in the submodule (e.g. wrong branch / different
folder layout), document and either: switch branch in Step 1, OR
adapt the loader path in Step 3.

**Acceptance**: every file referenced by an existing wrapper has
a confirmed new-path location.

### Step 3 — Create `src/pipeline/baselines/` package (25 min)

Engineer creates:

```
src/pipeline/baselines/
  __init__.py         # exports load_wlanloc, load_ronin_resnet1d, load_tartanvo, load_dpvo_trunk
  _paths.py           # EXTERNAL_METHODS_ROOT, path constants per submodule
  _shims.py           # apply_scipy_as_dcm_shim, apply_cupy_compat_shim, apply_numpy_linalg_shim, apply_np_int_shim
  wlanloc.py          # load_position_regressor(), load_preprocessor()
  ronin.py            # load_resnet1d(win, out_dim, **kwargs), load_glob_speed_sequence(), compute_ate_rte (re-export)
  tartanvo.py         # apply_tartanvo_shims(), run_vo_pipeline(image_dir, pose_file, ...)
  dpvo_trunk.py       # load_basic_encoder4(weights_path)
```

#### `_paths.py`

```python
"""Canonical paths to external_methods submodules. Edit only if
the project root layout changes."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_METHODS = PROJECT_ROOT / "external_methods"

WLANLOC_SRC = EXTERNAL_METHODS / "wlan_localization" / "src"
RONIN_SRC = EXTERNAL_METHODS / "ronin" / "source"
RONIN_LISTS = EXTERNAL_METHODS / "ronin" / "lists"
TARTANVO_ROOT = EXTERNAL_METHODS / "tartanvo"
DPVO_ROOT = EXTERNAL_METHODS / "dpvo"
```

#### `_shims.py`

```python
"""All runtime compat shims for vendored baselines. Each function
is idempotent (safe to call multiple times). NO vendored sources
edited; shims monkey-patch our process's loaded modules only —
Demand #3 honoured."""

def apply_np_int_shim():
    """RoNIN's data_glob_speed uses np.int (removed in numpy 1.20+)."""
    import numpy as np
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]

def apply_scipy_as_dcm_shim():
    """TartanVO's vendored transformation.py uses Rotation.as_dcm,
    renamed to as_matrix in scipy 1.4."""
    from scipy.spatial.transform import Rotation
    if not hasattr(Rotation, "as_dcm"):
        Rotation.as_dcm = Rotation.as_matrix
        Rotation.from_dcm = Rotation.from_matrix

def apply_numpy_linalg_submodule_shim():
    """numpy.linalg.linalg was a deprecated nested submodule;
    TartanVO's PWC correlation references it."""
    import numpy.linalg as nplinalg
    if not hasattr(nplinalg, "linalg"):
        nplinalg.linalg = nplinalg  # type: ignore[attr-defined]

def apply_cupy_compat_shim():
    """cupy.cuda.compile_with_cache was removed in cupy 12+;
    map to a RawModule-wrapped compat class. Used by TartanVO's
    Network/PWC/correlation.py CUDA kernels."""
    import cupy
    if hasattr(cupy.cuda, "compile_with_cache"):
        return
    class _CompatModule:
        def __init__(self, source):
            self._mod = cupy.RawModule(code=source)
        def get_function(self, name):
            return self._mod.get_function(name)
    cupy.cuda.compile_with_cache = lambda src: _CompatModule(src)
```

#### `wlanloc.py`

```python
"""WiFi SOTA baseline — wlan_localization (Naribole, MIT).

Bypass-import their `PositionRegressor` + `DataPreprocessor` from
the vendored source: their package `__init__` drags in
`imbalanced-learn` ↔ `scikit-learn` version conflicts that don't
resolve cleanly in our venv.

Demand #3: no source edits in the submodule; this loader uses
`importlib.util.spec_from_file_location` to load the two class
files directly."""
from __future__ import annotations
import importlib.util
import sys
import types
import logging
from ._paths import WLANLOC_SRC

def _stub_wlanloc_logger():
    """Pre-create stub modules wlan_localization.utils.logger so
    relative imports inside their source resolve to a no-op
    logger getter."""
    pkg = types.ModuleType("wlan_localization")
    utils = types.ModuleType("wlan_localization.utils")
    logmod = types.ModuleType("wlan_localization.utils.logger")
    logmod.get_logger = lambda name: logging.getLogger(name)
    sys.modules.setdefault("wlan_localization", pkg)
    sys.modules.setdefault("wlan_localization.utils", utils)
    sys.modules.setdefault("wlan_localization.utils.logger", logmod)

def _load_pure(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, WLANLOC_SRC / "wlan_localization" / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_position_regressor():
    _stub_wlanloc_logger()
    return _load_pure(
        "models/position_regressor.py", "wlan_pos_reg").PositionRegressor

def load_preprocessor():
    _stub_wlanloc_logger()
    return _load_pure(
        "data/preprocessor.py", "wlan_preproc").DataPreprocessor
```

#### `ronin.py`

```python
"""IMU SOTA baseline — Sachini/ronin (MIT).

Exports the canonical class definitions + the metric function.
Adds the RoNIN `source/` to sys.path so their internal absolute
imports work."""
from __future__ import annotations
import sys
from ._paths import RONIN_SRC, RONIN_LISTS
from ._shims import apply_np_int_shim

if str(RONIN_SRC) not in sys.path:
    sys.path.insert(0, str(RONIN_SRC))

apply_np_int_shim()  # required before any of their data files load

from model_resnet1d import BasicBlock1D, FCOutputModule, ResNet1D  # noqa: E402
from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset  # noqa: E402
from metric import compute_ate_rte, compute_absolute_trajectory_error  # noqa: E402

__all__ = [
    "BasicBlock1D", "FCOutputModule", "ResNet1D",
    "GlobSpeedSequence", "StridedSequenceDataset",
    "compute_ate_rte", "compute_absolute_trajectory_error",
    "load_test_list", "load_train_list",
]

def load_test_list(name="list_test_unseen.txt"):
    return [ln.strip() for ln in (RONIN_LISTS / name).read_text().splitlines() if ln.strip()]

def load_train_list():
    return load_test_list("list_train.txt")
```

#### `tartanvo.py`

```python
"""Camera SOTA baseline — castacks/tartanvo, python3 branch (MIT).

Apply 4 shims, expose a run_on_folder() that mirrors their
`vo_trajectory_from_folder.py` runner but in-process so we don't
have to `chdir` into the vendored repo."""
from __future__ import annotations
import sys
from ._paths import TARTANVO_ROOT
from ._shims import (
    apply_scipy_as_dcm_shim,
    apply_numpy_linalg_submodule_shim,
    apply_cupy_compat_shim,
)

def apply_tartanvo_shims():
    """Apply all 3 shims TartanVO needs (np.int not required for
    inference path). Idempotent."""
    apply_scipy_as_dcm_shim()
    apply_numpy_linalg_submodule_shim()
    apply_cupy_compat_shim()

def _ensure_path():
    if str(TARTANVO_ROOT) not in sys.path:
        sys.path.insert(0, str(TARTANVO_ROOT))

def load_vo_module():
    """Import TartanVO's TartanVO class. Caller can instantiate
    with their checkpoint path and run inference."""
    apply_tartanvo_shims()
    _ensure_path()
    from TartanVO import TartanVO  # noqa: E402  type: ignore
    return TartanVO
```

#### `dpvo_trunk.py`

```python
"""DPVO patch trunk (princeton-vl/DPVO, BSD).

We use ONLY `BasicEncoder4` from extractor.py — the dense patch
feature extractor. DPVO's full SLAM pipeline (lietorch + altcorr
custom CUDA ops) is not buildable on Windows; that's documented
and not addressed here."""
from __future__ import annotations
import sys
import torch
from ._paths import DPVO_ROOT

def _ensure_path():
    if str(DPVO_ROOT) not in sys.path:
        sys.path.insert(0, str(DPVO_ROOT))

def load_basic_encoder4(weights_path=None):
    """Load the DPVO patch encoder. If weights_path is provided,
    load the matching state_dict subset; otherwise return random-
    initialised."""
    _ensure_path()
    from dpvo.extractor import BasicEncoder4  # noqa: E402  type: ignore
    enc = BasicEncoder4(output_dim=128, norm_fn="instance")
    if weights_path is not None:
        sd = torch.load(weights_path, map_location="cpu")
        # DPVO's state_dict keys are prefixed; subset to encoder
        encoder_sd = {
            k[len("patchify."):]: v for k, v in sd.items()
            if k.startswith("patchify.")
        }
        enc.load_state_dict(encoder_sd, strict=False)
    return enc
```

#### `__init__.py`

```python
"""Centralised SOTA baseline loaders. All wrappers + the notebook
import from here so the path/shim machinery lives in one place."""
from .wlanloc import load_position_regressor, load_preprocessor
from .ronin import (
    BasicBlock1D, FCOutputModule, ResNet1D,
    GlobSpeedSequence, StridedSequenceDataset,
    compute_ate_rte, compute_absolute_trajectory_error,
    load_test_list, load_train_list,
)
from .tartanvo import apply_tartanvo_shims, load_vo_module
from .dpvo_trunk import load_basic_encoder4
from ._paths import (
    EXTERNAL_METHODS, WLANLOC_SRC, RONIN_SRC, RONIN_LISTS,
    TARTANVO_ROOT, DPVO_ROOT,
)
from ._shims import (
    apply_np_int_shim, apply_scipy_as_dcm_shim,
    apply_numpy_linalg_submodule_shim, apply_cupy_compat_shim,
)

__all__ = [
    "load_position_regressor", "load_preprocessor",
    "BasicBlock1D", "FCOutputModule", "ResNet1D",
    "GlobSpeedSequence", "StridedSequenceDataset",
    "compute_ate_rte", "compute_absolute_trajectory_error",
    "load_test_list", "load_train_list",
    "apply_tartanvo_shims", "load_vo_module",
    "load_basic_encoder4",
    "EXTERNAL_METHODS", "WLANLOC_SRC", "RONIN_SRC", "RONIN_LISTS",
    "TARTANVO_ROOT", "DPVO_ROOT",
    "apply_np_int_shim", "apply_scipy_as_dcm_shim",
    "apply_numpy_linalg_submodule_shim", "apply_cupy_compat_shim",
]
```

**Acceptance**: `python -c "from src.pipeline.baselines import *;
print(EXTERNAL_METHODS)"` returns the absolute project path
ending in `external_methods/`.

### Step 4 — Migrate every wrapper to use the new package (20 min)

Identified wrappers to update (the engineer greps `WLANLOC_SRC`,
`RONIN_SRC`, `r"C:\Users\FabLab\AppData\Local\Temp"` to find them):

- `scripts/eval_wlanloc_uji.py` (committed, canonical)
- `scripts/eval_wlanloc_ipin.py` (committed, canonical)
- `scripts/_eval_wlanloc_msiln.py` (iter-15)
- `scripts/_eval_wlanloc_imuwifine.py` (iter-19)
- `scripts/_eval_wlanloc_ipin_floor0.py` (iter-22) — if exists
- `scripts/eval_ronin_ate_fixed.py` (canonical)
- `scripts/eval_ronin_ipin.py` (canonical)
- `scripts/_eval_ronin_imuwifine.py` (iter-19)
- `scripts/_eval_ronin_a000_branchY.py` (iter-02)
- `scripts/_train_ronin_canonical_arch.py` (iter-23)
- `scripts/_eval_tartanvo_hospital.py` (iter-08)
- `src/pipeline/encoders/dpvo_motion.py` (DPVO trunk loader)

Per-wrapper migration pattern (example for `eval_wlanloc_uji.py`):

```python
# BEFORE — boilerplate at the top:
WLANLOC_SRC = Path(r"C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src")
def _stub_logger(): ...
def _load_pure(...): ...
def _load_position_regressor():
    _stub_logger()
    return _load_pure("models/position_regressor.py", "wlan_pos_reg").PositionRegressor
# ... ~40 lines

# AFTER — single import:
from src.pipeline.baselines import load_position_regressor, load_preprocessor
PositionRegressor = load_position_regressor()
DataPreprocessor = load_preprocessor()
```

Same shape per wrapper. Engineer's job is mechanical: delete the
boilerplate, replace with `from src.pipeline.baselines import ...`.

**Acceptance**: each wrapper imports from `src.pipeline.baselines`
ONLY (no remaining `r"C:\Users\FabLab\AppData\Local\Temp"` paths
in `scripts/` or `src/` after this step).

### Step 5 — Verification: re-run one canonical eval (10 min)

Run `eval_wlanloc_uji.py` and confirm it produces the same number
as RESULT_01:
- Expected: global mode val mean Euclid **15.17 m** (within
  ±0.5 m of the RESULT_01 number; small variance from
  preprocessor randomisation is acceptable).

If the number drifts more than that, something in the migration
broke. Diagnose:
- Did the submodule pin a different commit than the Temp/ clone?
- Did the shim machinery not apply in the right order?

**Acceptance**: at least one canonical wrapper reproduces its
RESULT number through the new loaders.

### Step 6 — Documentation + dependency manifest (10 min)

Create / update:

- **`docs/EXTERNAL_DEPENDENCIES.md`** — one section per submodule:
  - Repo URL
  - Branch + pinned commit hash
  - License
  - What we use it for (which RESULT_NN; which `src.pipeline.baselines.*`
    function exposes it)
  - Setup notes: `git submodule update --init --recursive` after
    `git clone`; Windows lietorch/altcorr unavailability for DPVO
    SLAM; cupy installation note for TartanVO.
- **`README.md`** — top-level: clone-and-setup section pointing
  at the submodule init command.
- **`.gitmodules`** — auto-generated by `git submodule add`.
- **`CLAUDE.md`** — update path references from
  `C:\Users\FabLab\AppData\Local\Temp\...` to
  `external_methods/...`.

**Acceptance**: documentation files updated; clean checkout +
submodule init + venv install reproduces the SOTA-eval pipeline
on a fresh machine (engineer's call whether to actually test this
or just document the expected sequence).

## Sources

- User directive 2026-05-26 ~12:30 local: pull SOTA dependencies
  into one `external_methods/` folder; adapt scripts to import
  from the new source; GitHub presentation as dependency modules.
- RESULT_01 (UJI canonical numbers for the verification target).
- All existing wrappers under `scripts/` and the DPVO loader in
  `src/pipeline/encoders/dpvo_motion.py`.

## What to report back

In `handoff/results/RESULT_26_external-methods-relocation-and-baselines.md`:

1. **Step 0** — inventory of upstream URLs + pinned commit hashes
   for the 4 submodules; MSILN starter handling note.
2. **Step 1** — `.gitmodules` content; `git submodule status`
   output.
3. **Step 2** — file presence check table (every file we import
   confirmed at the new path).
4. **Step 3** — `src/pipeline/baselines/` package contents +
   smoke import verification.
5. **Step 4** — list of wrappers migrated + line-count savings
   table (boilerplate removed per file).
6. **Step 5** — verification number (eval_wlanloc_uji should
   reproduce 15.17 m); any drift explained.
7. **Step 6** — documentation files updated.
8. **One open question** for scientist.

## Reversibility

- Step 1 (submodule adds): permanent; reversible by `git submodule
  deinit + rm .gitmodules`.
- Step 3 (baselines package): permanent; new files under
  `src/pipeline/baselines/`.
- Step 4 (wrapper migrations): permanent; each wrapper edit is a
  small targeted change reversible via `git revert` per file.
- Steps 5-6: throwaway eval + documentation.

Files committed: `.gitmodules`, the 4 submodule entries (Git
submodule references, not the submodule contents themselves),
`src/pipeline/baselines/*.py`, migrated wrapper edits, updated
docs.

**Compute budget**: ≤ 75 min.
- Step 0: 10 min (inventory).
- Step 1: 15 min (submodule add × 4 + pin + commit).
- Step 2: 10 min (file presence checks).
- Step 3: 25 min (write the baselines package; engineer can
  copy the templates above).
- Step 4: 20 min (migrate ~10-12 wrappers, mechanical).
- Step 5: 10 min (verification run).
- Step 6: 10 min (documentation).

If overrun: drop Step 5's full verification to a smoke import
only (confirm imports work; defer number-reproduction to PLAN_27
when running through the full pipeline anyway). Don't skip
Step 4 — incomplete migration leaves the codebase half-old-half-new.

If a submodule URL has moved or the upstream commit is unavailable
(repo deleted, forks etc.): document the obstacle and pin a
fork-or-mirror that has the needed files. The vendored `Temp/`
copies are our source of truth for the exact code that produced
RESULT_NN numbers.

## Iteration scope after this plan

| iter | scope | depends on |
|---|---|---|
| 26 | `external_methods/` + `src/pipeline/baselines/` (THIS PLAN) | — |
| 27 | `src/pipeline/data/` factory + `src/pipeline/visualization/` for the notebook's dataset pre-section | 26 |
| 28 | `src/pipeline/fusion/` + `encoders/` + `training/` consolidation (build_arch factory, public FusionTrainer methods, load_trained helper) | 26 |
| 29 | `src/pipeline/evaluation/MainResultsTable` + scripts/eval_*.py triage + configs + docs sweep | 26, 27, 28 |
| 30 | `notebooks/run2_walkthrough.ipynb` scaffold (§0 datasets pre-section + §1-3 phase sections + §4 gaps + §5 framing + §6 reproducibility) | 26, 27, 28, 29 |

After PLAN_30 the user iterates directly with engineer on the
notebook polish; the consolidation handoff is complete.
