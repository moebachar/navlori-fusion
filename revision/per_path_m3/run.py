"""Per-path MAE on MSILN site1/B1 test split.

Reports MAE/RMSE for every test path + macro (unweighted) average and the
sample-weighted aggregate (the existing headline number) for sanity. The
macro average prevents any single large path from dominating the metric.

Outputs:
  revision/per_path_m3/msiln_test_per_path.json
  revision/per_path_m3/msiln_test_per_path.md
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.pipeline.training import load_trained  # noqa: E402


def main() -> None:
    out_dir = REPO / "revision" / "per_path_m3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[per_path_m3] loading trained MSILN fusion model...", flush=True)
    tr = load_trained(
        "runs/main_table/msiln_site1_b1/transformer",
        arch="transformer",
        dataset="msiln_site1_b1",
        K=4,
    )

    print(f"[per_path_m3] modalities = {tr.modalities}", flush=True)
    print(f"[per_path_m3] n test samples = {tr.n.get('test', 'n/a')}", flush=True)

    preds, tgts = tr.predict("test")
    preds_np = preds.numpy()
    tgts_np = tgts.numpy()
    err = np.linalg.norm(preds_np - tgts_np, axis=1)  # (N,) Euclidean per sample

    # Pull per-sample path_ids from the test dataset's GT rows. predict()
    # iterates in split order (s=0..n step batch_size), so _gt_rows[i] aligns
    # one-to-one with preds[i]. Use len(preds) as the safety guard.
    ds = tr.dm.test_ds
    assert ds is not None, "MSILN test dataset is None — config split missing?"
    gt_rows = ds._gt_rows
    pids = np.array([r["path_id"] for r in gt_rows], dtype=np.int64)
    if len(pids) != len(err):
        raise RuntimeError(
            f"Length mismatch: gt_rows={len(pids)} vs predictions={len(err)}"
        )
    print(f"[per_path_m3] unique test path_ids = {sorted(set(int(p) for p in pids))}",
          flush=True)

    total_n = int(len(err))
    per_path: dict[str, dict[str, float | int]] = OrderedDict()
    path_fractions: dict[str, float] = OrderedDict()
    for pid in sorted(set(int(p) for p in pids)):
        mask = pids == pid
        n_p = int(mask.sum())
        e_p = err[mask]
        mae_p = float(e_p.mean())
        rmse_p = float(np.sqrt((e_p ** 2).mean()))
        per_path[str(pid)] = {"n": n_p, "mae_m": mae_p, "rmse_m": rmse_p}
        path_fractions[str(pid)] = n_p / total_n

    macro_mae = float(np.mean([v["mae_m"] for v in per_path.values()]))
    sample_weighted_mae = float(err.mean())

    # Dominant path
    dom_pid = max(path_fractions.items(), key=lambda kv: kv[1])

    record = {
        "dataset": "msiln_site1_b1",
        "split": "test",
        "checkpoint": "runs/main_table/msiln_site1_b1/transformer",
        "arch": "transformer",
        "K": 4,
        "modalities": list(tr.modalities),
        "total_samples": total_n,
        "per_path": per_path,
        "path_fractions": path_fractions,
        "macro_mae_m": macro_mae,
        "sample_weighted_mae_m": sample_weighted_mae,
        "dominant_path": {
            "path_id": int(dom_pid[0]),
            "fraction": dom_pid[1],
            "n": per_path[dom_pid[0]]["n"],
        },
    }

    json_path = out_dir / "msiln_test_per_path.json"
    json_path.write_text(json.dumps(record, indent=2))
    print(f"[per_path_m3] wrote {json_path}", flush=True)

    # Markdown table
    lines = [
        "# MSILN site1/B1 — Per-path test MAE",
        "",
        f"Checkpoint: `runs/main_table/msiln_site1_b1/transformer` "
        f"(arch=transformer, K=4)",
        f"Modalities: {', '.join(tr.modalities)}",
        f"Total test samples: {total_n}",
        "",
        "| path_id | n | fraction | MAE (m) | RMSE (m) |",
        "|--------:|--:|---------:|--------:|---------:|",
    ]
    for pid_str, stats in per_path.items():
        frac = path_fractions[pid_str]
        lines.append(
            f"| {pid_str} | {stats['n']} | {frac:.3f} | "
            f"{stats['mae_m']:.3f} | {stats['rmse_m']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"| **macro avg** | — | — | **{macro_mae:.3f}** | — |"
    )
    lines.append(
        f"| sample-weighted (aggregate sanity) | {total_n} | 1.000 | "
        f"{sample_weighted_mae:.3f} | — |"
    )
    lines.append("")
    lines.append(
        f"Dominant path: **{dom_pid[0]}** "
        f"({dom_pid[1]*100:.1f}% of test samples, "
        f"n={per_path[dom_pid[0]]['n']}).  "
        f"Macro MAE = {macro_mae:.3f} m vs sample-weighted "
        f"{sample_weighted_mae:.3f} m."
    )
    md_path = out_dir / "msiln_test_per_path.md"
    md_path.write_text("\n".join(lines))
    print(f"[per_path_m3] wrote {md_path}", flush=True)

    print(f"[per_path_m3] MACRO MAE  = {macro_mae:.4f} m", flush=True)
    print(f"[per_path_m3] SAMPLE-WEIGHTED MAE = {sample_weighted_mae:.4f} m", flush=True)
    print(f"[per_path_m3] DOMINANT path = {dom_pid[0]} "
          f"({dom_pid[1]*100:.2f}% of {total_n} samples)", flush=True)


if __name__ == "__main__":
    main()
