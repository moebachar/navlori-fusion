"""Extract each top-lead's code_sketch from the workflow output and write
runnable files under experiments/leads_night2/. Also write
experiments/state/night2_leads.json so the night2 chain can advance.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

OUT_DIR = REPO / "experiments" / "leads_night2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LEADS_JSON = REPO / "experiments" / "state" / "night2_leads.json"

WF_OUTPUT = Path(r"C:\Users\FabLab\AppData\Local\Temp\claude"
                  r"\x--navlori-fusion\d8eb7630-f005-4cc0-859b-12f4f3e28313"
                  r"\tasks\w07sod666.output")


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:60]


def main() -> None:
    d = json.loads(WF_OUTPUT.read_text())
    top = d["result"]["top_leads"]
    print(f"Extracting {len(top)} leads")

    out_index: list[dict] = []
    for lead in top:
        rank = lead["rank"]
        name = lead["name"]
        family = lead["family"]
        spec = lead["small_test_spec"]
        sketch = spec.get("code_sketch", "")
        # Determine filename — use synth's suggested runner_filename
        runner_path = spec.get("runner_filename")
        if runner_path and runner_path.startswith("experiments/leads/"):
            # Keep the synth's intended layout but move into leads_night2/
            fname = Path(runner_path).name
        else:
            fname = f"lead_{rank:02d}_{slugify(name)}.py"
        out_p = OUT_DIR / fname
        out_p.write_text(sketch, encoding="utf-8")
        print(f"  #{rank} -> {out_p.relative_to(REPO)} ({len(sketch)} chars)")
        out_index.append({
            "rank": rank, "name": slugify(name), "family": family,
            "runner": str(out_p.relative_to(REPO)).replace("\\", "/"),
            "epochs": int(spec.get("epochs", 12)),
            "dataset": "msiln_site1_b1",
            "seed": 42,
            "expected_lift_m": float(lead.get("expected_lift_m", 0)),
            "success_criterion": spec.get("success_criterion", ""),
            "rationale": lead.get("rationale", ""),
        })

    LEADS_JSON.write_text(json.dumps(out_index, indent=2), encoding="utf-8")
    print(f"\nWrote {LEADS_JSON}: {len(out_index)} leads")


if __name__ == "__main__":
    main()
