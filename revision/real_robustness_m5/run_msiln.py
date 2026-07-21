"""M5: Real-data robustness reproduction on MSILN.

Reproduces the modality-dropout + WiFi-staleness curves on MSILN
(site1/B1 cross-session test split) using the already-trained
transformer checkpoint at runs/main_table/msiln_site1_b1/transformer.

Writes:
  revision/real_robustness_m5/msiln_subsets_test.json
  revision/real_robustness_m5/msiln_staleness_test.json
  revision/real_robustness_m5/msiln_summary.md
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

# Ensure project root on path when run as a plain script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "revision" / "real_robustness_m5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS_OUT = OUT_DIR / "msiln_subsets_test.json"
STALE_OUT = OUT_DIR / "msiln_staleness_test.json"
SUMMARY_OUT = OUT_DIR / "msiln_summary.md"

CKPT_DIR = "runs/main_table/msiln_site1_b1/transformer"


def _fmt_table_subsets(subsets: dict) -> str:
    lines = ["| Subset | MAE (m) | RMSE (m) |", "|---|---|---|"]
    for label, vals in subsets.items():
        lines.append(f"| {label} | {vals['mae']:.3f} | {vals['rmse']:.3f} |")
    return "\n".join(lines)


def _fmt_table_stale(stale: dict) -> str:
    lines = ["| Staleness (K steps) | MAE (m) | RMSE (m) |", "|---|---|---|"]
    for label, vals in stale.items():
        lines.append(f"| {label} | {vals['mae']:.3f} | {vals['rmse']:.3f} |")
    return "\n".join(lines)


def _is_monotone(stale: dict) -> tuple[bool, list[tuple[int, float]]]:
    """Return (monotone_non_decreasing, ordered (stale_int, mae) list)."""
    ordered = []
    for label, vals in stale.items():
        # label of form 'stale=N'
        try:
            n = int(label.split("=")[1])
        except Exception:
            continue
        ordered.append((n, float(vals["mae"])))
    ordered.sort(key=lambda t: t[0])
    monotone = all(b >= a - 1e-6 for (_, a), (_, b) in zip(ordered, ordered[1:]))
    return monotone, ordered


def main() -> int:
    print(f"[M5] repo_root = {REPO_ROOT}", flush=True)
    print(f"[M5] ckpt_dir  = {CKPT_DIR}", flush=True)
    print(f"[M5] out_dir   = {OUT_DIR}", flush=True)

    from src.pipeline.training import load_trained

    print("[M5] Loading trained MSILN transformer (K=4) ...", flush=True)
    tr = load_trained(CKPT_DIR, arch="transformer",
                      dataset="msiln_site1_b1", K=4)
    print(f"[M5] modalities = {tr.modalities}", flush=True)
    print(f"[M5] n[test] = {tr.n.get('test')}", flush=True)
    print(f"[M5] run_dir = {tr.run_path}", flush=True)

    print("[M5] evaluate_subsets('test') ...", flush=True)
    subsets = tr.evaluate_subsets("test")
    print(f"[M5] subsets done: {list(subsets.keys())}", flush=True)
    for k, v in subsets.items():
        print(f"    {k:18s}  MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}",
              flush=True)

    # Copy side-effect file too (subsets.json -> msiln_subsets_test.json)
    side = tr.run_path / "subsets.json"
    if side.is_file():
        shutil.copy2(side, SUBSETS_OUT)
        print(f"[M5] copied side-effect {side} -> {SUBSETS_OUT}", flush=True)
    else:
        SUBSETS_OUT.write_text(json.dumps(subsets, indent=2))
        print(f"[M5] wrote {SUBSETS_OUT}", flush=True)

    print("[M5] evaluate_staleness(modality='wifi', split='test') ...",
          flush=True)
    stale = tr.evaluate_staleness(modality="wifi", split="test")
    print(f"[M5] staleness done: {list(stale.keys())}", flush=True)
    for k, v in stale.items():
        print(f"    {k:12s}  MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}",
              flush=True)

    side_s = tr.run_path / "staleness_wifi.json"
    if side_s.is_file():
        shutil.copy2(side_s, STALE_OUT)
        print(f"[M5] copied side-effect {side_s} -> {STALE_OUT}", flush=True)
    else:
        STALE_OUT.write_text(json.dumps(stale, indent=2))
        print(f"[M5] wrote {STALE_OUT}", flush=True)

    # --- Markdown summary --------------------------------------------------
    all_mae = subsets.get("all", {}).get("mae", float("nan"))
    wifi_mae = subsets.get("only:wifi", {}).get("mae", float("nan"))
    imu_mae = subsets.get("only:imu", {}).get("mae", float("nan"))
    monotone, ordered = _is_monotone(stale)
    K_max = max(n for n, _ in ordered) if ordered else 0
    mae_at_K = next((m for n, m in ordered if n == K_max), float("nan"))
    mae_fresh = next((m for n, m in ordered if n == 0), float("nan"))

    if monotone:
        degradation = "graceful (monotone non-decreasing)"
    else:
        # Look for a cliff: a single big jump dominating the curve.
        jumps = [(ordered[i + 1][0], ordered[i + 1][1] - ordered[i][1])
                 for i in range(len(ordered) - 1)]
        biggest = max(jumps, key=lambda t: t[1]) if jumps else (None, 0.0)
        degradation = (f"non-monotone / cliff at stale={biggest[0]} "
                       f"(jump +{biggest[1]:.2f} m)")

    headline = (f"MSILN cross-session test — all={all_mae:.2f} m, "
                f"WiFi-only={wifi_mae:.2f} m, IMU-only={imu_mae:.2f} m, "
                f"stale=K({K_max})={mae_at_K:.2f} m "
                f"(fresh {mae_fresh:.2f} m); degradation: {degradation}.")
    print(f"[M5] HEADLINE: {headline}", flush=True)

    md = [
        "# M5 - Real-data robustness on MSILN (site1/B1, cross-session test)",
        "",
        "Reproduces the modality-dropout and WiFi-staleness curves on the",
        "real MSILN test split (5 traces, +11-12 days from training session)",
        "using the existing transformer checkpoint",
        f"`{CKPT_DIR}` (no retraining).",
        "",
        f"**Headline.** {headline}",
        "",
        "## Modality-dropout (test split)",
        "",
        _fmt_table_subsets(subsets),
        "",
        "## WiFi staleness (test split, K=4)",
        "",
        _fmt_table_stale(stale),
        "",
        f"Monotone non-decreasing across staleness: **{monotone}**.",
        "A monotone curve = graceful degradation (temporal fusion propagating",
        "motion from the last good WiFi fix); a non-monotone jump at one step",
        "= cliff behaviour (single-instant-style failure).",
        "",
    ]
    SUMMARY_OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"[M5] wrote {SUMMARY_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
