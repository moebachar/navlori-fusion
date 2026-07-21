"""Aggregate D3 period-range sub-ablation into markdown + LaTeX, splice into
PAPER_INSERTS.md between sentinel markers.

Compares each period range to the M1+M2 ``learned_continuous`` baseline
(the (0.05, 120) seed-42 value, NOT the 3-seed mean, since D3 is single-seed).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D3_MANIFEST = REPO / "revision" / "k_period_d3" / "period_manifest.json"
M1_MANIFEST = REPO / "revision" / "ablation_m1_timeenc" / "manifest.json"
PAPER_INSERTS = REPO / "revision" / "PAPER_INSERTS.md"
ARTIFACTS = REPO / "revision" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SENTINEL_START = "<!-- D3_PERIOD_TABLE_START -->"
SENTINEL_END = "<!-- D3_PERIOD_TABLE_END -->"

DEFAULT_PERIOD = (0.05, 120.0)
DSDISP = {"simulation_2mod": "Webots", "msiln_site1_b1": "MSILN site1/B1"}
VARDISP = {
    "narrow": "narrow (0.5, 10) s",
    "wide": "wide (0.01, 600) s",
    "shifted": "shifted (0.1, 30) s",
}


def main() -> None:
    if not D3_MANIFEST.exists():
        print(f"[d3-agg] no manifest yet at {D3_MANIFEST}")
        return
    d3 = json.loads(D3_MANIFEST.read_text())
    m1 = json.loads(M1_MANIFEST.read_text()) if M1_MANIFEST.exists() else []

    # baseline: seed=42, learned_continuous (the same single-seed point
    # the D3 runs compare against).
    base = {}
    for r in m1:
        if (r.get("status") == "ok"
                and r.get("time_enc_mode") == "learned_continuous"
                and r.get("seed") == 42):
            base[r["dataset"]] = r["test_mae_m"]

    # Render markdown
    lines: list[str] = []
    lines.append(SENTINEL_START)
    lines.append("")
    lines.append("## D3 — Period-range sensitivity")
    lines.append("")
    lines.append(
        f"Single-seed (42) sweep of the continuous-time encoding's period "
        f"range. Default is `({DEFAULT_PERIOD[0]:g}, {DEFAULT_PERIOD[1]:g}) s`; "
        "deltas are vs the M1+M2 baseline at the same seed."
    )
    lines.append("")
    for dataset in ["simulation_2mod", "msiln_site1_b1"]:
        lines.append(f"### {DSDISP[dataset]}")
        lines.append("")
        lines.append("| Period range | val MAE (m) | test MAE (m) | Δ vs default |")
        lines.append("|---|---:|---:|---:|")
        # baseline row first
        base_val = base.get(dataset, float("nan"))
        lines.append(f"| **default** (0.05, 120) s | — | **{base_val:.3f}** | — |")
        for variant in ["narrow", "wide", "shifted"]:
            rec = next((r for r in d3
                        if r.get("dataset") == dataset
                        and r.get("variant_label") == variant
                        and r.get("status") == "ok"), None)
            if rec is None:
                lines.append(f"| {VARDISP[variant]} | pending | pending | — |")
                continue
            v = rec["val_mae_m"]; t = rec["test_mae_m"]
            d = t - base_val
            sign = "+" if d >= 0 else ""
            lines.append(
                f"| {VARDISP[variant]} | {v:.3f} | {t:.3f} | {sign}{d:.3f} m |"
            )
        lines.append("")

    lines.append("### Take-aways (auto)")
    lines.append("")
    # Find min/max per dataset (including default)
    for dataset in ["simulation_2mod", "msiln_site1_b1"]:
        all_tests: list[tuple[str, float]] = []
        if dataset in base:
            all_tests.append(("default", base[dataset]))
        for variant in ["narrow", "wide", "shifted"]:
            rec = next((r for r in d3
                        if r.get("dataset") == dataset
                        and r.get("variant_label") == variant
                        and r.get("status") == "ok"), None)
            if rec is not None:
                all_tests.append((variant, rec["test_mae_m"]))
        if not all_tests:
            continue
        all_tests.sort(key=lambda x: x[1])
        best_n, best_v = all_tests[0]
        worst_n, worst_v = all_tests[-1]
        lines.append(
            f"- **{DSDISP[dataset]}**: best = `{best_n}` ({best_v:.3f} m), "
            f"worst = `{worst_n}` ({worst_v:.3f} m), spread = {worst_v - best_v:.3f} m."
        )
    lines.append("")
    lines.append(SENTINEL_END)
    section = "\n".join(lines)

    # Splice into PAPER_INSERTS
    text = (PAPER_INSERTS.read_text(encoding="utf-8")
            if PAPER_INSERTS.exists() else "")
    if SENTINEL_START in text and SENTINEL_END in text:
        new = re.sub(
            f"{re.escape(SENTINEL_START)}.*?{re.escape(SENTINEL_END)}",
            section, text, count=1, flags=re.S,
        )
    else:
        new = text.rstrip() + "\n\n" + section + "\n"
    PAPER_INSERTS.write_text(new, encoding="utf-8")
    print(f"[d3-agg] PAPER_INSERTS.md updated with D3 section ({len(section)} chars).")

    # Also dump CSV
    csv_p = ARTIFACTS / "d3_period_table.csv"
    with csv_p.open("w") as f:
        f.write("dataset,variant,min_period,max_period,val_mae_m,test_mae_m\n")
        for r in d3:
            if r.get("status") != "ok":
                continue
            f.write(f"{r['dataset']},{r['variant_label']},"
                    f"{r['min_period']},{r['max_period']},"
                    f"{r['val_mae_m']:.4f},{r['test_mae_m']:.4f}\n")
    print(f"[d3-agg] wrote {csv_p}")


if __name__ == "__main__":
    main()
