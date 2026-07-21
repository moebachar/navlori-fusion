"""M5 real-robustness eval on IMUWiFine floor-4 (test split).

Loads the trained transformer fusion checkpoint and reports:
  - evaluate_subsets("test")  -> all / only-WiFi / only-IMU
  - evaluate_staleness("wifi", "test")  -> MAE vs stale instants

Artifacts:
  revision/real_robustness_m5/imuwifine_subsets_test.json
  revision/real_robustness_m5/imuwifine_staleness_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.training import load_trained  # noqa: E402


def main() -> None:
    out_dir = REPO_ROOT / "revision" / "real_robustness_m5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[M5/imuwifine] loading trained transformer (K=4) ...", flush=True)
    tr = load_trained(
        "runs/main_table/imuwifine/transformer",
        arch="transformer",
        dataset="imuwifine",
        K=4,
    )
    print(f"[M5/imuwifine] modalities={tr.modalities} n={tr.n}", flush=True)

    print("[M5/imuwifine] evaluate_subsets(test) ...", flush=True)
    subsets = tr.evaluate_subsets("test")
    (out_dir / "imuwifine_subsets_test.json").write_text(
        json.dumps(subsets, indent=2)
    )
    print(json.dumps(subsets, indent=2), flush=True)

    print("[M5/imuwifine] evaluate_staleness(wifi, test) ...", flush=True)
    staleness = tr.evaluate_staleness("wifi", "test")
    (out_dir / "imuwifine_staleness_test.json").write_text(
        json.dumps(staleness, indent=2)
    )
    print(json.dumps(staleness, indent=2), flush=True)

    print("[M5/imuwifine] done.", flush=True)


if __name__ == "__main__":
    main()
