# Plan 29 — `src/pipeline/evaluation/MainResultsTable` + canonical `scripts/eval_*.py` triage + configs/docs sweep

> Fourth consolidation iter. After PLAN_26 (baselines), PLAN_27
> (data + viz), PLAN_28 (fusion + encoders + training): all the
> APIs are in place. This iter builds the paper-table assembler,
> promotes ~4-5 canonical eval scripts to thin wrappers, and
> sweeps configs + docs to reflect the consolidated state.
>
> **Exclusions baked in per `handoff/SCIENTIST_NOTE_notebook-
> exclusions.md`**: IPIN row dropped from MainResultsTable;
> MoTTransformer column dropped from paper-facing columns.
> Both stay in the codebase for reproducibility but are not
> featured in paper-facing deliverables.

## Hypothesis

After this iter:
- `from src.pipeline.evaluation import MainResultsTable;
  table = MainResultsTable.from_archive(); table.to_dataframe()`
  returns the paper-ready 6-row × N-column comparison
  (Webots / IMUWiFine / MSILN / RoNIN canonical / TartanAir
  hospital / UJI; columns = per-leg SOTAs + incumbent + cnn1d +
  lstm_attn).
- 5 canonical `scripts/eval_*.py` thin wrappers exist; each is
  ~30 lines built entirely on consolidated APIs (no boilerplate).
- `configs/stage_c/fusion.yaml` default = CNN1D winner config;
  alternative archs named variants.
- `docs/SOTA_BASELINES.md` rewritten with the run-2 6-row main
  table + 3-architecture comparison + cross-cutting findings.

This is one focused experiment: assemble the paper-facing
artifacts.

## Steps

### Step 0 — `MainResultsTable` class (25 min)

`src/pipeline/evaluation/main_results_table.py`:

```python
"""Paper-ready main results table assembler.

Reads from:
  - `handoff/results/RESULT_NN_*.md` (parsed for headline numbers)
  - `runs/overnight/run2_iter_*/` JSONs (machine-readable
    per-run metrics)
  - `runs/baselines/` (legacy / pre-run-2 baselines)

Schema (paper-facing; excludes IPIN per SCIENTIST_NOTE):

| dataset            | modalities         | wlan_localization | RoNIN ResNet1D | TartanVO | Anchor2Vec | DPVOMotion | IMUCNN | incumbent | cnn1d (winner) | lstm_attn |
|---|---|---|---|---|---|---|---|---|---|---|
| Webots sim         | WiFi+IMU+Cam+Odom  | …                | …             | …        | …          | …          | …      | 0.417 t   | **0.339 t**    | 0.340 t   |
| IMUWiFine fl.4 (1) | WiFi+IMU           | 4.17 v / 8.50 t  | 26.84 v       | …        | …          | …          | …      | …         | 1.40 v / 7.09  | 1.26 v    |
| MSILN site1/B1     | WiFi+IMU           | 21.26 v          | …             | …        | …          | …          | …      | …         | (PLAN_15)      | (PLAN_15) |
| RoNIN canonical (2)| IMU only           | …                | **5.14 raw**  | …        | …          | …          | 9.96   | …         | 7.59 raw       | 7.50 raw  |
| TartanAir hosp.    | Camera only        | …                | …             | 0.518 f  | …          | 0.293      | …      | …         | n/a (3)        | n/a (3)   |
| UJI IndoorLoc      | WiFi only val      | 15.17            | …             | …        | **8.69**   | …          | …      | …         | 8.72           | 8.43      |

Notes:
  (1) IMUWiFine test split lacks IMU per dataset design
      (RESULT_20 audit); fusion test column = WiFi-only.
  (2) RoNIN raw / Umeyama-aligned ATE; reuses RESULT_07's
      pretrained ResNet1D number.
  (3) Camera external-SOTA validation queued as Phase C extension
      (paper-soft per RESULT_08).
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "handoff" / "results"
RUNS_DIR = ROOT / "runs" / "overnight"


# Datasets included in paper-facing main table (drops IPIN per
# SCIENTIST_NOTE_notebook-exclusions.md)
PAPER_DATASETS = [
    "webots",
    "imuwifine_floor4",
    "msiln_site1_b1",
    "ronin_canonical",
    "tartanair_hospital",
    "uji_indoorloc",
]

# Architectures included in paper-facing columns (drops
# mot_transformer per SCIENTIST_NOTE_notebook-exclusions.md;
# MoTTransformer code stays in src/pipeline/fusion/ for repro)
PAPER_ARCHS = ["incumbent", "cnn1d", "lstm_attn"]

# Per-leg SOTAs that appear in the table
SOTA_COLS = [
    "wlan_localization",
    "RoNIN_ResNet1D",
    "TartanVO",
    "Anchor2Vec",
    "DPVOMotion",
    "IMUCNN",
]


@dataclass
class TableCell:
    """One number in the main table. Numbers can be val-only,
    test-only, val/test paired, or n/a."""
    val: float | None = None
    test: float | None = None
    metric: str = "MAE"  # or "ATE", "ATE_aligned"
    source: str = ""     # e.g. "RESULT_17"
    note: str = ""       # short footnote ref


class MainResultsTable:
    """Reads run-2 archive + JSONs, returns the paper main table
    as a pandas DataFrame.

    Paper-facing presentation excludes IPIN dataset + MoTTransformer
    architecture per SCIENTIST_NOTE_notebook-exclusions.md (2026-05-26).
    Both remain in the codebase for reproducibility.
    """

    def __init__(self, cells: dict[tuple[str, str], TableCell]):
        self.cells = cells  # (dataset, column) -> TableCell

    @classmethod
    def from_archive(cls,
                     datasets: list[str] | None = None,
                     archs: list[str] | None = None) -> "MainResultsTable":
        """Build the table from RESULT_NN parsing + JSONs."""
        datasets = datasets or PAPER_DATASETS
        archs = archs or PAPER_ARCHS
        cells = _harvest_all(datasets, archs)
        return cls(cells)

    def to_dataframe(self,
                     value_format: str = "val_test_paired") -> pd.DataFrame:
        """Render as DataFrame. value_format options:
          - 'val_test_paired': "1.40 v / 7.09 t"
          - 'val_only', 'test_only', 'best' (whichever is available)"""
        rows = []
        cols = SOTA_COLS + PAPER_ARCHS
        for dataset in PAPER_DATASETS:
            row = {"dataset": dataset}
            for col in cols:
                cell = self.cells.get((dataset, col))
                row[col] = _format_cell(cell, value_format) if cell else "n/a"
            rows.append(row)
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """For SUMMARY.md / docs / paper-supp"""
        return self.to_dataframe().to_markdown(index=False)

    def cell(self, dataset: str, column: str) -> TableCell | None:
        return self.cells.get((dataset, column))


# Module-level helpers (private)
def _harvest_all(datasets, archs) -> dict[tuple[str, str], TableCell]:
    """Walk RESULT_NN files + JSONs, harvest the numbers."""
    cells = {}
    # ... per-dataset, per-column harvesting from RESULTs
    # Engineer implements by reading the saved JSONs under
    # runs/overnight/run2_iter_NN/<arch>/metrics.jsonl AND parsing
    # RESULT_NN headline-numbers tables (regex on the markdown
    # "|method|val MAE|test MAE|" patterns is acceptable; or
    # better, harvest only from the JSONs and treat RESULT_NN as
    # documentation).
    return cells


def _format_cell(cell: TableCell, value_format: str) -> str:
    """Format a cell for display."""
    if cell is None:
        return "n/a"
    if value_format == "val_test_paired":
        parts = []
        if cell.val is not None:
            parts.append(f"{cell.val:.2f} v")
        if cell.test is not None:
            parts.append(f"{cell.test:.2f} t")
        return " / ".join(parts) if parts else "n/a"
    # ... other formats
```

`src/pipeline/evaluation/__init__.py` exports `MainResultsTable`.

**Acceptance**: `from src.pipeline.evaluation import MainResultsTable;
t = MainResultsTable.from_archive(); df = t.to_dataframe(); print(df)`
prints the 6-row × 9-column DataFrame, every cell populated or
explicitly "n/a", matching the paper-facing schema.

### Step 1 — Canonical `scripts/eval_*.py` thin wrappers (20 min)

Promote 5 canonical eval entry points. Each is ≤ 30 lines and
imports entirely from the consolidated APIs.

#### `scripts/eval_uji.py` (NEW canonical)

```python
"""Reproduce RESULT_01 numbers on UJIIndoorLoc:
  - wlan_localization SOTA  : 15.17 m val mean Euclid (global mode)
  - Anchor2Vec (ours)       :  8.69 m val mean Euclid

Replaces the iteration-scoped `scripts/eval_uji_wifi.py` +
`scripts/eval_wlanloc_uji.py`. All logic now in the pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.pipeline.baselines import load_position_regressor, load_preprocessor
from src.pipeline.data import load_dataset
from src.pipeline.encoders import Anchor2Vec
# ... ~25 lines: argparse → load_dataset("uji_indoorloc") →
# run wlan_localization global mode → train Anchor2Vec at
# the canonical config → write JSON
```

Similarly:
- `scripts/eval_ronin_canonical.py` — reuses RESULT_07's pretrained
  ResNet1D checkpoint via `src.pipeline.baselines.load_resnet1d_pretrained`
  (engineer adds this helper if not already present).
- `scripts/eval_msiln.py` — reuses RESULT_15 pattern.
- `scripts/eval_imuwifine.py` — reuses RESULT_19 pattern.
- `scripts/eval_tartanair_hospital.py` — reuses RESULT_08 pattern.

**NO `scripts/eval_ipin_floor0.py`** per the exclusion directive.
Existing iter-scoped `scripts/_eval_*.py` from PLAN_22 stay as
historical artifacts; engineer's call whether to delete or keep.

Each new canonical wrapper documents at the top:
- Which RESULT_NN it reproduces.
- Expected output numbers (the paper-facing-canonical values).
- Run command + flags.

**Acceptance**: each canonical wrapper produces its
RESULT_NN headline number within ±0.5 % when run on the
current archive.

### Step 2 — `configs/stage_c/fusion.yaml` cleanup (5 min)

The winner config (CNN1D + K=4 + B=128 + 4-mod Webots) becomes
the default. Other variants named for clarity:

```yaml
# configs/stage_c/fusion.yaml
default_arch: cnn1d
archs:
  cnn1d:
    K: 4
    batch_size: 128
    epochs: 90
    lr: 1.3e-3
    optimizer: AdamW
    scheduler: OneCycleLR
    loss: Huber(delta=0.5)
    modality_dropout: 0.4
    instant_dropout: 0.45
  incumbent: { ... K=4 B=128 ... }  # run-1 baseline
  lstm_attn: { ... }
  tcn:      { ... }
  # NOTE: mot_transformer config exists in codebase but is NOT
  # promoted as a named variant — see SCIENTIST_NOTE_notebook-
  # exclusions.md (kept for reproducibility, not paper-facing).
```

(Engineer adapts to existing config structure — Hydra
conventions, etc.)

**Acceptance**: a fresh `build_arch("cnn1d")` from the default
config reproduces the RESULT_17 winner number.

### Step 3 — Documentation sweep (15 min)

#### `docs/SOTA_BASELINES.md` — rewrite top-level

Replace with run-2 6-row main table (using
`MainResultsTable.to_markdown()`) + cross-cutting findings
section (already in SUMMARY.md but mirrored here for the
paper-supp). Exclude IPIN row + MoTTransformer column per the
directive.

#### `docs/fusion_pipeline.md` — update to 3-arch story

CNN1D winner declaration + LSTM-attn runner-up (dead-reckoning
3-dataset structural finding) + incumbent baseline reference.
**Don't mention MoTTransformer in the body**; one footnote at
the end can say "`src/pipeline/fusion/mot_transformer.py` is a
documented honest-negative experiment retained for repository
reproducibility — see `handoff/results/RESULT_21_*` if curious."

#### `CLAUDE.md` — paper-facing status

Update "Stage A + B/C Complete" section to reflect run-2
findings + the paper-facing exclusions. The "What's Next"
section can list the optional follow-ups from SUMMARY.md.

#### `README.md` — entry-point pointer

Point new readers at: `external_methods/` setup → `notebooks/run2_walkthrough.ipynb` → `scripts/eval_*.py` for individual benchmark reproduction.

**Acceptance**: all 4 docs files updated; no remaining IPIN /
MoTTransformer mentions in paper-facing locations except as
reproducibility footnotes.

### Step 4 — Smoke verification (10 min)

`scripts/_smoke_evaluation.py`:

```python
from src.pipeline.evaluation import MainResultsTable

# 1. Build the table from archive
t = MainResultsTable.from_archive()
df = t.to_dataframe()

# 2. Sanity checks
assert len(df) == 6                              # 6 datasets (IPIN excluded)
assert "mot_transformer" not in df.columns        # MoTTransformer excluded
assert df.loc[df.dataset == "webots", "cnn1d"].iloc[0] == "0.28 v / 0.34 t"  # CNN1D winner

# 3. Render to markdown (paper-ready)
print(t.to_markdown())
```

Plus a thin smoke run of one canonical eval (e.g. `eval_uji.py`
should reproduce 15.17 m wlan_localization and 8.69 m
Anchor2Vec).

**Acceptance**: smoke runs end-to-end; assertions pass;
canonical eval reproduces within ±0.5 %.

### Step 5 — Open Q logging (5 min)

If during MainResultsTable harvest the engineer surfaces
discrepancies between RESULT_NN headline numbers and saved
JSONs (e.g. paper-facing numbers were "best epoch" but JSON
has all epochs), document the harvest rule in the class
docstring. The class's default should be "best val MAE epoch"
per the run-2 convention.

## Sources

- `handoff/SUMMARY.md` (run-2 final).
- `handoff/SCIENTIST_NOTE_notebook-exclusions.md` (this directive).
- All `handoff/results/RESULT_01-25_*.md` (the source-of-truth
  numbers).
- `runs/overnight/run2_iter_*/` JSONs.
- `runs/baselines/msiln_site1_b1/baselines.json` (legacy).
- `src/pipeline/{baselines,data,fusion,training,visualization}/`
  (consolidated APIs from PLAN_26-28).

## What to report back

In `handoff/results/RESULT_29_main-results-table-and-scripts-triage.md`:

1. **Step 0** — `MainResultsTable` class shipped; sample
   DataFrame output included as table.
2. **Step 1** — 5 canonical `scripts/eval_*.py` (paths +
   line counts); verification numbers per wrapper.
3. **Step 2** — `configs/stage_c/fusion.yaml` cleaned;
   default = CNN1D winner.
4. **Step 3** — 4 docs files updated; diff summary.
5. **Step 4** — smoke verification output.
6. **Step 5** — any harvest-rule decisions documented in the
   class.
7. **One open question** for scientist.

## Reversibility

- Step 0 (`MainResultsTable`): NEW file under
  `src/pipeline/evaluation/`. Permanent.
- Step 1 (canonical eval scripts): NEW thin wrappers under
  `scripts/`. Engineer commits.
- Step 2 (config cleanup): permanent.
- Step 3 (docs sweep): permanent.
- Step 4-5: smoke + documentation.

Files committed: `src/pipeline/evaluation/main_results_table.py`,
5 NEW `scripts/eval_*.py`, updated `configs/stage_c/fusion.yaml`,
`docs/{SOTA_BASELINES,fusion_pipeline}.md`, `CLAUDE.md`,
`README.md`.

**Compute budget**: ≤ 80 min.
- Step 0: 25 min (MainResultsTable + harvest logic).
- Step 1: 20 min (5 wrappers; each ≤ 30 lines, mechanical).
- Step 2: 5 min.
- Step 3: 15 min (4 doc files).
- Step 4: 10 min (smoke + 1 canonical eval).
- Step 5: 5 min.

If overrun: cut Step 1 to 3 canonical wrappers
(`eval_uji.py`, `eval_ronin_canonical.py`,
`eval_tartanair_hospital.py` — these are the per-leg-SOTA
reproductions that the paper most needs). The 2-modality
fusion reproductions (msiln + imuwifine) can be deferred or
added in PLAN_30.

## Iteration scope after this plan

- **30 (FINAL)**: `notebooks/run2_walkthrough.ipynb` scaffold
  using every consolidated API. 6 datasets × 3 architectures
  paper-facing presentation. After this, user iterates with
  engineer directly. Estimated 60-90 min.
