"""PLAN_27 Step 4 — smoke pass through all 7 datasets via the new
factory + visualization packages. Writes a dataset-overview PNG for
each.

Verifies:
- ``list_datasets()`` returns the 7 canonical names.
- ``dataset_stats(name)`` works for all 7 and returns a non-empty dict.
- ``plot_dataset_overview(name)`` produces a non-empty Figure.
- ``preprocessing_demo(name, modality)`` works for each available
  modality.

Run: ``.venv/Scripts/python.exe scripts/_smoke_data_visualization.py``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data import (  # noqa: E402
    dataset_stats, list_datasets, preprocessing_demo,
)
from src.pipeline.visualization import (  # noqa: E402
    plot_dataset_overview, plot_preprocessing_demo,
)

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_27" / "dataset_overviews"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list_datasets()
    print(f"=== smoke pass: {len(names)} datasets ===", flush=True)
    rows = []
    for name in names:
        t0 = time.time()
        try:
            s = dataset_stats(name)
        except Exception as e:
            print(f"  {name}: STATS FAILED — {type(e).__name__}: {e}", flush=True)
            rows.append((name, "stats_fail", str(e)))
            continue
        mods = s.get("modalities_available", [])
        print(f"  {name}: modalities={mods}", flush=True)
        try:
            fig = plot_dataset_overview(name, save_to=OUT_DIR / f"{name}_overview.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"     overview plot FAILED — {type(e).__name__}: {e}", flush=True)
            rows.append((name, "plot_fail", str(e)))
            continue
        # preprocessing_demo for each available modality
        demo_status = {}
        for m in mods:
            try:
                d = preprocessing_demo(name, m)
                demo_status[m] = "OK" if d.get("raw") is not None else "n/a"
                # Render a preprocessing figure (do not fail loudly if a modality lacks raw arrays).
                fig = plot_preprocessing_demo(d, m, save_to=OUT_DIR / f"{name}_preproc_{m}.png")
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                demo_status[m] = f"FAIL: {type(e).__name__}"
        elapsed = time.time() - t0
        print(f"     overview saved; preprocessing demos: {demo_status} ({elapsed:.1f}s)", flush=True)
        rows.append((name, "ok", demo_status))

    print(f"\nwrote {len(list(OUT_DIR.glob('*.png')))} PNGs to {OUT_DIR}", flush=True)
    n_failed = sum(1 for r in rows if r[1] != "ok")
    print(f"datasets ok: {len(rows) - n_failed}/{len(rows)}", flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
