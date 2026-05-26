"""Shared helpers for per-dataset loader modules.

Each ``src/pipeline/data/<dataset>.py`` module exposes the SAME 3
functions: ``load``, ``stats``, ``preprocessing_demo``. The
helpers here cover the common operations.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def path_to(rel: str) -> Path:
    """Resolve a path relative to the repo root."""
    return ROOT / rel


def collect_path_metadata(collection_dir: Path) -> list[dict]:
    """Walk ``collection_dir/path_*/metadata.json`` and return the list."""
    out = []
    if not collection_dir.is_dir():
        return out
    for sub in sorted(collection_dir.glob("path_*")):
        meta_p = sub / "metadata.json"
        if meta_p.exists():
            try:
                out.append(json.loads(meta_p.read_text()))
            except Exception:
                pass
    return out


def summarise_path_lengths(per_path_meta: list[dict]) -> dict:
    """Aggregate path durations / sample counts across paths."""
    if not per_path_meta:
        return {"n_paths": 0, "duration_total_s": 0}
    durations = [m.get("duration_s", 0) for m in per_path_meta]
    return {
        "n_paths": len(per_path_meta),
        "duration_total_s": float(sum(durations)),
        "duration_per_path_mean": float(sum(durations) / len(durations)),
        "duration_per_path_min": float(min(durations)) if durations else 0,
        "duration_per_path_max": float(max(durations)) if durations else 0,
    }


def not_applicable(modality: str, dataset: str) -> dict:
    """Standard `preprocessing_demo` return value for not-supported modalities."""
    return {
        "raw": None,
        "preprocessed": None,
        "description_raw": f"{modality!r} not present in {dataset!r}",
        "description_preprocessed": "n/a",
        "preprocessing_pipeline": [],
        "note": (
            f"Dataset {dataset!r} does not include the {modality!r} modality. "
            f"Call preprocessing_demo() with one of the modalities listed in "
            f"stats()['modalities_available'] instead."
        ),
    }


__all__ = ["ROOT", "path_to", "collect_path_metadata",
           "summarise_path_lengths", "not_applicable"]
