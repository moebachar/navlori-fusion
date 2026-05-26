"""PLAN_29 Step 4 — smoke verification for the MainResultsTable +
canonical eval wrappers.

Exercises:
- ``MainResultsTable.from_archive()`` produces the 6-row paper schema.
- Paper-facing exclusions are honoured (no IPIN row, no
  MoTTransformer column).
- All canonical wrapper modules import cleanly.

Run: ``.venv/Scripts/python.exe scripts/_smoke_evaluation.py``
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.evaluation import (  # noqa: E402
    MainResultsTable, PAPER_DATASETS, PAPER_ARCHS,
)


def main():
    print("=== MainResultsTable.from_archive() ===", flush=True)
    t = MainResultsTable.from_archive()
    df = t.to_dataframe()

    # Exclusion assertions
    assert "ipin2024_floor0" not in PAPER_DATASETS, "IPIN must be excluded"
    assert "mot_transformer" not in PAPER_ARCHS, "MoTTransformer must be excluded"
    assert "mot_transformer" not in df.columns, "MoTTransformer must not be a column"
    assert "ipin2024_floor0" not in df["dataset"].values, "IPIN must not be a row"

    # Schema assertions
    assert len(df) == 6, f"Expected 6 paper rows, got {len(df)}"
    assert "dataset" in df.columns
    for arch in PAPER_ARCHS:
        assert arch in df.columns, f"Missing paper arch column: {arch}"

    # Headline cell assertions (CNN1D Webots winner)
    webots_cnn1d = t.cell("webots", "cnn1d")
    assert webots_cnn1d is not None
    assert abs(webots_cnn1d.test - 0.339) < 0.005, \
        f"CNN1D Webots test should be 0.339; got {webots_cnn1d.test}"

    # UJI Anchor2Vec
    uji_a2v = t.cell("uji_indoorloc", "Anchor2Vec")
    assert uji_a2v is not None and abs(uji_a2v.val - 8.69) < 0.05

    # RoNIN ResNet1D paper-exact
    ronin_sota = t.cell("ronin_canonical", "RoNIN_ResNet1D")
    assert ronin_sota is not None and abs(ronin_sota.test - 5.140) < 0.05

    print(df.to_string(index=False))
    print()
    print(f"  rows: {len(df)}; columns: {len(df.columns)}", flush=True)

    excl = t.excluded()
    print(f"  excluded datasets: {excl['datasets']}", flush=True)
    print(f"  excluded archs: {excl['architectures']}", flush=True)

    # Smoke-import the canonical wrappers
    print(f"\n=== canonical wrapper imports ===", flush=True)
    wrappers = ["scripts.eval_uji", "scripts.eval_ronin_canonical",
                "scripts.eval_tartanair_hospital"]
    for m in wrappers:
        importlib.import_module(m)
        print(f"  {m}: import OK", flush=True)

    print(f"\nall assertions passed.", flush=True)


if __name__ == "__main__":
    main()
