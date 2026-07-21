"""Post-M1+M2-batch finalisation.

Runs after batch_m1_m2.py completes. Does three things:

1. Re-runs the aggregator to refresh revision/artifacts/m1_m2_table.{md,tex,csv}.
2. Appends the final mean +/- std numbers to revision/PAPER_INSERTS.md.
3. Prints a short headline summary.

Safe to call any time after at least one run has completed — but the
appended PAPER_INSERTS section is overwritten on every call (idempotent
between sentinel markers), so call it once after the batch is done.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "revision" / "ablation_m1_timeenc" / "manifest.json"
PAPER_INSERTS = REPO / "revision" / "PAPER_INSERTS.md"
SENTINEL_START = "<!-- M1_M2_TABLE_START -->"
SENTINEL_END = "<!-- M1_M2_TABLE_END -->"

sys.path.insert(0, str(REPO / "revision" / "runners"))
from aggregate_m1_m2 import main as aggregate_main  # noqa: E402


def render_paper_section() -> str:
    if not MANIFEST.exists():
        return "_(M1+M2 manifest not found.)_"
    data = json.loads(MANIFEST.read_text())
    ok = [r for r in data if r.get("status") == "ok"]
    if not ok:
        return "_(M1+M2 manifest is empty.)_"

    from collections import defaultdict
    import math

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in ok:
        grouped[(r["dataset"], r["time_enc_mode"])].append(r)

    def ms(xs: list[float]) -> tuple[float, float]:
        n = len(xs)
        if n == 0:
            return float("nan"), float("nan")
        mu = sum(xs) / n
        if n < 2:
            return mu, 0.0
        var = sum((x - mu) ** 2 for x in xs) / (n - 1)
        return mu, math.sqrt(var)

    DISP = {"learned_continuous": "Learned continuous (ours)",
            "none": "No time encoding",
            "binned": "Binned (log-quantized)",
            "posindex": "Positional index (rank)"}
    DSDISP = {"simulation_2mod": "Webots", "msiln_site1_b1": "MSILN site1/B1"}

    lines: list[str] = []
    lines.append(SENTINEL_START)
    lines.append("")
    lines.append("## M1 + M2 — Time-encoding ablation × seed variance")
    lines.append("")
    lines.append(
        "All cells report `mean ± std` over the seeds listed in the n column. "
        "Seeds={42, 7, 123}. Headline = test MAE in metres."
    )
    lines.append("")
    for dataset in ["simulation_2mod", "msiln_site1_b1"]:
        lines.append(f"### {DSDISP[dataset]}")
        lines.append("")
        lines.append("| Time encoding | n | val MAE (m) | test MAE (m) |")
        lines.append("|---|---:|---:|---:|")
        for mode in ["learned_continuous", "none", "binned", "posindex"]:
            entries = grouped.get((dataset, mode), [])
            vals = [e["val_mae_m"] for e in entries]
            tests = [e["test_mae_m"] for e in entries]
            v_mu, v_sd = ms(vals)
            t_mu, t_sd = ms(tests)
            if not entries:
                cv = "pending"; ct = "pending"
            else:
                cv = f"{v_mu:.3f} ± {v_sd:.3f}"
                ct = f"{t_mu:.3f} ± {t_sd:.3f}"
            lines.append(f"| {DISP[mode]} | {len(entries)} | {cv} | {ct} |")
        lines.append("")

    # Quick interpretation
    lines.append("### Take-aways (auto)")
    lines.append("")
    for dataset in ["simulation_2mod", "msiln_site1_b1"]:
        baseline = grouped.get((dataset, "learned_continuous"), [])
        if not baseline:
            continue
        bmu, _ = ms([e["test_mae_m"] for e in baseline])
        for mode in ["none", "binned", "posindex"]:
            es = grouped.get((dataset, mode), [])
            if not es:
                continue
            mu, _ = ms([e["test_mae_m"] for e in es])
            delta = mu - bmu
            pct = 100.0 * delta / bmu if bmu != 0 else float("nan")
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"- **{DSDISP[dataset]} / {DISP[mode]}**: test MAE {mu:.3f} m "
                f"(Δ vs learned-continuous: {sign}{delta:.3f} m, {sign}{pct:.1f} %)."
            )
        lines.append("")
    lines.append(SENTINEL_END)
    return "\n".join(lines)


def splice_into_paper_inserts(section: str) -> None:
    text = (PAPER_INSERTS.read_text(encoding="utf-8")
            if PAPER_INSERTS.exists() else "")
    if SENTINEL_START in text and SENTINEL_END in text:
        new = re.sub(
            f"{re.escape(SENTINEL_START)}.*?{re.escape(SENTINEL_END)}",
            section,
            text,
            count=1,
            flags=re.S,
        )
    else:
        new = text.rstrip() + "\n\n" + section + "\n"
    PAPER_INSERTS.write_text(new, encoding="utf-8")
    print(f"[post] PAPER_INSERTS.md updated with M1+M2 section "
          f"({len(section)} chars).")


def main() -> None:
    aggregate_main()
    section = render_paper_section()
    splice_into_paper_inserts(section)
    print("\n[post] DONE.")


if __name__ == "__main__":
    main()
