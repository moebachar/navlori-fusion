"""Load run artifacts, dataset stats, and live training metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

RUNS_DIR = Path("runs")
DATA_DIR = Path("data/async_collection")

MODALITY_META = {
    "wifi":   {"color": "#3B82F6", "icon": "📡", "label": "WiFi",     "encoder": "Anchor2Vec"},
    "imu":    {"color": "#10B981", "icon": "🔄", "label": "IMU",      "encoder": "CNN-1D"},
    "odom":   {"color": "#F59E0B", "icon": "🛞", "label": "Odometry", "encoder": "CNN-1D"},
    "camera": {"color": "#EF4444", "icon": "📷", "label": "Vision",   "encoder": "ViT-B/16"},
}

STAGE_META = [
    {"id": "A", "name": "Encoders",          "status": "done",    "color": "#10B981"},
    {"id": "B", "name": "Temporal Align",    "status": "pending", "color": "#6B7280"},
    {"id": "C", "name": "Cross-Modal Fusion","status": "pending", "color": "#6B7280"},
    {"id": "D", "name": "KalmanNet",         "status": "pending", "color": "#6B7280"},
    {"id": "E", "name": "Uncertainty",       "status": "pending", "color": "#6B7280"},
]


def list_runs() -> list[dict]:
    """Return all completed runs sorted by finish time (newest first)."""
    runs = []
    for run_dir in sorted(RUNS_DIR.glob("*/"), reverse=True):
        hist_path = run_dir / "history.json"
        meta_path = run_dir / "meta.json"
        eval_path = run_dir / "eval.json"
        if not hist_path.exists():
            continue
        hist = json.loads(hist_path.read_text())
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        evl  = json.loads(eval_path.read_text()) if eval_path.exists() else {}
        runs.append({**meta, **hist, "eval": evl, "run_dir": str(run_dir)})
    return runs


def best_run_per_modality() -> dict[str, dict]:
    """Return the run with lowest best_val_mae for each modality."""
    best: dict[str, dict] = {}
    for run in list_runs():
        mod = run.get("modality", "")
        if mod not in best or run["best_val_mae"] < best[mod]["best_val_mae"]:
            best[mod] = run
    return best


def load_metrics_live(run_dir: str) -> list[dict]:
    """Read JSONL metrics file — safe for concurrent writes."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def active_run() -> dict | None:
    """Return the most recently started run that has no history.json yet (still training)."""
    for run_dir in sorted(RUNS_DIR.glob("*/"), reverse=True):
        if (run_dir / "meta.json").exists() and not (run_dir / "history.json").exists():
            meta = json.loads((run_dir / "meta.json").read_text())
            metrics = load_metrics_live(str(run_dir))
            return {"meta": meta, "metrics": metrics, "run_dir": str(run_dir)}
    return None


def dataset_stats() -> dict:
    """Compute dataset-level statistics from the data directory."""
    stats: dict = {"paths": [], "total_samples": 0, "modalities": {}}
    for path_dir in sorted(DATA_DIR.glob("path_*/")):
        gt = path_dir / "ground_truth.csv"
        if not gt.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(gt)
            if len(df) == 0:
                continue
            stats["paths"].append({
                "id": path_dir.name,
                "n_gt": len(df),
                "x_range": [float(df["gt_x"].min()), float(df["gt_x"].max())],
                "y_range": [float(df["gt_y"].min()), float(df["gt_y"].max())],
                "duration": float(df["sim_time"].max() - df["sim_time"].min()),
            })
            stats["total_samples"] += len(df)
        except Exception:
            pass
    return stats


@torch.no_grad()
def extract_embeddings_from_run(run_dir: str, dm, modality: str, n_max: int = 2000):
    """Load a saved encoder and extract embeddings from the val set."""
    import sys
    sys.path.insert(0, ".")
    from src.pipeline.encoders import Anchor2Vec, IMUCNN, OdomCNN, VisionViT

    run_path = Path(run_dir)
    meta_path = run_path / "meta.json"
    if not meta_path.exists():
        return None, None

    # Reconstruct encoder based on modality
    enc_map = {
        "wifi":   lambda: Anchor2Vec(n_aps=32, embed_dim=128, n_anchors=64),
        "imu":    lambda: IMUCNN(in_features=9, embed_dim=128),
        "odom":   lambda: OdomCNN(in_features=7, embed_dim=128),
        "camera": lambda: VisionViT(embed_dim=128, freeze_backbone=True),
    }
    if modality not in enc_map:
        return None, None

    encoder = enc_map[modality]()
    weights = run_path / "encoder.pt"
    if weights.exists():
        encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    encoder.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)

    X, y = dm.val_ds.get_tensors(modality) if modality != "camera" else (None, None)
    if X is None:
        return None, None

    # Subsample
    if len(X) > n_max:
        idx = torch.randperm(len(X))[:n_max]
        X, y = X[idx], y[idx]

    all_z = []
    bs = 256
    for i in range(0, len(X), bs):
        xb = X[i:i+bs].to(device)
        z = encoder(xb)
        if z.ndim == 3:
            z = z.mean(1)
        all_z.append(z.cpu())

    return torch.cat(all_z).numpy(), y.numpy()
