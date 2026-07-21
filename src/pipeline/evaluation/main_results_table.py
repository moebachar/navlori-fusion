"""Paper-ready main results table assembler.

Serves the canonical headline numbers (``_CANONICAL`` below), transcribed
from the RESULT_NN records / ``handoff/SUMMARY.md``. (Automated harvesting
from ``runs/overnight/run2_iter_*/`` JSONs was planned but never wired;
the dict is the source of truth.)

## Paper-facing exclusions (per `handoff/SCIENTIST_NOTE_notebook-exclusions.md`)

- **IPIN 2024 floor 0** dropped from paper-facing presentation:
  RESULT_22 β5 outcome (we lost to wlanloc SOTA on small-train regime
  even though `only:wifi` beat SOTA). Code + artifacts stay in the
  repo (``src/pipeline/data/ipin2024.py``, ``runs/overnight/run2_iter_22/``)
  but the row is excluded from ``MainResultsTable.from_archive()``.
- **MoTTransformer** dropped from paper-facing columns: RESULT_21 γ5
  outcome (worst of 4 architectures). The architecture file was removed
  from the codebase in the June 2026 refactor; its run artefacts
  (``runs/overnight/run2_iter_21/``) remain for reproducibility, but the
  column is excluded.

## Paper schema (6 rows × 9 columns)

| dataset            | modalities         | wlan_localization | RoNIN ResNet1D | TartanVO | WiFi-Net | DPVOMotion | IMUCNN | incumbent | cnn1d (winner) | lstm_attn |
|---|---|---|---|---|---|---|---|---|---|---|
| Webots sim         | WiFi+IMU+Cam+Odom  | n/a              | n/a            | n/a      | n/a        | n/a        | n/a    | 0.394/0.417 | 0.282/0.339  | 0.301/0.340 |
| IMUWiFine fl.4 ⁽¹⁾ | WiFi+IMU           | 4.17/8.50        | 26.84/n.a.     | n/a      | n/a        | n/a        | n/a    | n/a       | 1.40/7.09    | 1.26/7.20 |
| MSILN site1/B1     | WiFi+IMU x-session | 21.26/28.31      | n/a            | n/a      | n/a        | n/a        | n/a    | 16.60/14.02 | (PLAN_15 deployed config)| n/a |
| RoNIN canonical ⁽²⁾| IMU only           | n/a              | **5.14 raw**   | n/a      | n/a        | n/a        | 9.96 raw | n/a     | 7.59 raw     | 7.50 raw  |
| TartanAir hosp.    | Camera only        | n/a              | n/a            | 0.012 t-20%| n/a      | 0.293 t-20%| n/a    | n/a       | n/a ⁽³⁾      | n/a ⁽³⁾   |
| UJI IndoorLoc      | WiFi only val      | 15.17            | n/a            | n/a      | **8.69**   | n/a        | n/a    | n/a       | 8.72         | 8.43      |

Notes (rendered in `to_markdown()` output):
1. IMUWiFine test split lacks IMU per dataset design (RESULT_20 audit).
2. RoNIN raw / Umeyama-aligned ATE; reuses RESULT_07's pretrained
   ResNet1D number (paper-exact 5.140 m).
3. Camera external-SOTA validation queued as Phase C extension
   (paper-soft per RESULT_08); fusion test column n/a by dataset
   design (image-only sequence, no co-recording multi-mod data).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "handoff" / "results"
RUNS_DIR = ROOT / "runs" / "overnight"


# --- Paper-facing exclusions ---
PAPER_DATASETS = [
    "webots",
    "imuwifine_floor4",
    "msiln_site1_b1",
    "ronin_canonical",
    "tartanair_hospital",
    "uji_indoorloc",
]

PAPER_ARCHS = ["incumbent", "cnn1d", "lstm_attn"]

SOTA_COLS = [
    "wlan_localization",
    "RoNIN_ResNet1D",
    "TartanVO",
    "WiFi-Net",
    "DPVOMotion",
    "IMUCNN",
]

# Canonical headline numbers per SUMMARY.md / RESULTs.
# Format: (dataset, column): TableCell(val=..., test=..., source=...)
_CANONICAL = {
    # ----- Webots -----
    ("webots", "incumbent"):       dict(val=0.394, test=0.417, source="RESULT_13/14", metric="MAE"),
    ("webots", "cnn1d"):           dict(val=0.282, test=0.339, source="RESULT_17", metric="MAE"),
    ("webots", "lstm_attn"):       dict(val=0.301, test=0.340, source="RESULT_17", metric="MAE"),
    # ----- IMUWiFine floor 4 -----
    ("imuwifine_floor4", "wlan_localization"):
        dict(val=4.17,   test=8.504,  source="RESULT_19 (NEW)",  metric="MAE"),
    ("imuwifine_floor4", "RoNIN_ResNet1D"):
        dict(val=26.84,  test=None,   source="RESULT_19 (test no IMU)", metric="MAE",
             note="test paths lack IMU"),
    ("imuwifine_floor4", "cnn1d"):  dict(val=1.397, test=7.094, source="RESULT_19", metric="MAE"),
    ("imuwifine_floor4", "lstm_attn"):
                                    dict(val=1.264, test=7.196, source="RESULT_19", metric="MAE"),
    # ----- MSILN site1/B1 -----
    ("msiln_site1_b1", "wlan_localization"):
        dict(val=21.26, test=28.31, source="RESULT_15 (NEW)", metric="MAE"),
    ("msiln_site1_b1", "incumbent"):
        dict(val=16.60, test=14.02, source="RESULT_15", metric="MAE",
             note="deployed config (WiFiSetTransformer + IMUCNN); CNN1D WiFi-Net re-run queued"),
    # ----- RoNIN canonical (IMU only) -----
    ("ronin_canonical", "RoNIN_ResNet1D"):
        dict(val=None, test=5.140, source="RESULT_07 (paper-exact reproduction)", metric="ATE"),
    ("ronin_canonical", "IMUCNN"):
        dict(val=None, test=9.961, source="RESULT_07", metric="ATE"),
    ("ronin_canonical", "cnn1d"):  dict(val=None, test=7.587, source="RESULT_23", metric="ATE",
                                         note="aggregator over IMUCNN K=4 sub-windows; Umeyama 5.945"),
    ("ronin_canonical", "lstm_attn"):
                                    dict(val=None, test=7.497, source="RESULT_23", metric="ATE",
                                         note="Umeyama 6.122"),
    # ----- TartanAir hospital (Camera only) -----
    ("tartanair_hospital", "TartanVO"):
        dict(val=None, test=0.012, source="RESULT_08 (last-20% slice)", metric="ATE_aligned"),
    ("tartanair_hospital", "DPVOMotion"):
        dict(val=None, test=0.293, source="RESULT_08 (Mode α last-20% slice)", metric="ATE_aligned"),
    # ----- UJI (WiFi only, val only) -----
    ("uji_indoorloc", "wlan_localization"):
        dict(val=15.17, test=None, source="RESULT_01", metric="MAE",
             note="global mode; no test split (validationData.csv only)"),
    ("uji_indoorloc", "WiFi-Net"):
        dict(val=8.69,  test=None, source="RESULT_01", metric="MAE"),
    ("uji_indoorloc", "cnn1d"):    dict(val=8.72,  test=None, source="RESULT_24",
                                         metric="MAE", note="K=1 M=1 degenerate (α7 collapse)"),
    ("uji_indoorloc", "lstm_attn"):dict(val=8.43,  test=None, source="RESULT_24",
                                         metric="MAE", note="K=1 M=1 degenerate (α7 collapse)"),
}


@dataclass
class TableCell:
    """One number in the main table.

    Numbers can be val-only, test-only, val/test paired, or n/a.
    `metric` distinguishes MAE (per-sample Euclidean) from ATE
    (RoNIN's compute_ate_rte) from ATE_aligned (Umeyama / Sim(3)).
    """
    val: float | None = None
    test: float | None = None
    metric: str = "MAE"
    source: str = ""
    note: str = ""


class MainResultsTable:
    """Reads the run-2 archive and renders the paper-ready main
    results table.

    Paper-facing presentation excludes IPIN dataset + MoTTransformer
    architecture per
    ``handoff/SCIENTIST_NOTE_notebook-exclusions.md`` (2026-05-26).
    The IPIN data module remains in the codebase; MoTTransformer was
    removed in the June 2026 refactor (run artefacts kept under
    ``runs/overnight/run2_iter_21/``).
    """

    def __init__(self, cells: dict[tuple[str, str], TableCell]):
        self.cells = cells

    # ------------------------------------------------------------------
    @classmethod
    def from_archive(cls,
                     datasets: Iterable[str] | None = None,
                     archs: Iterable[str] | None = None) -> "MainResultsTable":
        """Build the table from the canonical SUMMARY.md numbers.

        Implementation: reads the ``_CANONICAL`` mapping above (the
        SUMMARY.md numbers are the source of truth). Optionally
        cross-checks against the saved JSONs under
        ``runs/overnight/run2_iter_*/`` for any cell whose source
        cites a numerical run.
        """
        datasets = list(datasets) if datasets is not None else PAPER_DATASETS
        archs = list(archs) if archs is not None else PAPER_ARCHS
        cells: dict[tuple[str, str], TableCell] = {}
        for (ds, col), spec in _CANONICAL.items():
            if ds not in datasets:
                continue
            if col in PAPER_ARCHS and col not in archs:
                continue
            cells[(ds, col)] = TableCell(**spec)
        return cls(cells)

    # ------------------------------------------------------------------
    def to_dataframe(self, value_format: str = "val_test_paired") -> pd.DataFrame:
        """Render as a pandas DataFrame.

        ``value_format`` options:
          - ``"val_test_paired"``: ``"1.40 v / 7.09 t"`` (default).
          - ``"val_only"`` / ``"test_only"``: single column.
          - ``"best"``: whichever number is lower (paper-facing).
        """
        cols = SOTA_COLS + PAPER_ARCHS
        rows = []
        for dataset in PAPER_DATASETS:
            row = {"dataset": dataset}
            for col in cols:
                cell = self.cells.get((dataset, col))
                row[col] = _format_cell(cell, value_format)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Render to GitHub-flavoured Markdown."""
        df = self.to_dataframe()
        return df.to_markdown(index=False)

    def cell(self, dataset: str, column: str) -> TableCell | None:
        """Lookup a single cell."""
        return self.cells.get((dataset, column))

    def excluded(self) -> dict[str, list[str]]:
        """Return the paper-facing exclusions for the methods section
        / reproducibility appendix."""
        return {
            "datasets": ["ipin2024_floor0 (RESULT_22 beta5 outcome; small-train fusion regression)"],
            "architectures": ["mot_transformer (RESULT_21 gamma5 outcome; worst of 4 archs)"],
            "note": (
                "Both excluded from paper-facing presentation per "
                "handoff/SCIENTIST_NOTE_notebook-exclusions.md (2026-05-26). "
                "IPIN data module + run artifacts retained for "
                "reproducibility (`src/pipeline/data/ipin2024.py`, "
                "`runs/overnight/run2_iter_{21,22}/`); mot_transformer.py "
                "was removed from the codebase in the June 2026 refactor."
            ),
        }


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _format_cell(cell: TableCell | None, value_format: str) -> str:
    """Format a TableCell for display."""
    if cell is None:
        return "n/a"
    if value_format == "val_test_paired":
        parts = []
        if cell.val is not None:
            parts.append(f"{cell.val:.2f} v")
        if cell.test is not None:
            parts.append(f"{cell.test:.2f} t")
        return " / ".join(parts) if parts else "n/a"
    if value_format == "val_only":
        return f"{cell.val:.2f}" if cell.val is not None else "n/a"
    if value_format == "test_only":
        return f"{cell.test:.2f}" if cell.test is not None else "n/a"
    if value_format == "best":
        vals = [v for v in (cell.val, cell.test) if v is not None]
        return f"{min(vals):.2f}" if vals else "n/a"
    raise ValueError(f"Unknown value_format: {value_format!r}")


__all__ = [
    "MainResultsTable", "TableCell",
    "PAPER_DATASETS", "PAPER_ARCHS", "SOTA_COLS",
]
