"""Dataset factory — dispatch ``load_dataset(name)`` / ``stats(name)``
/ ``preprocessing_demo(name, modality)`` to the per-dataset modules.

The notebook §0 pre-section uses these three functions to render
every dataset's overview cell uniformly.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_REGISTRY: dict[str, str] = {
    "webots":             "src.pipeline.data.webots",
    "msiln_site1_b1":     "src.pipeline.data.msiln",
    "imuwifine_floor4":   "src.pipeline.data.imuwifine",
    "ipin2024_floor0":    "src.pipeline.data.ipin2024",
    "ronin_canonical":    "src.pipeline.data.ronin_canonical",
    "tartanair_hospital": "src.pipeline.data.tartanair",
    "uji_indoorloc":      "src.pipeline.data.uji",
}


def list_datasets() -> list[str]:
    """Return the canonical dataset names accepted by the factory."""
    return list(_REGISTRY)


def _resolve(name: str):
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {list_datasets()}"
        )
    return import_module(_REGISTRY[name])


def load_dataset(name: str, **kwargs) -> Any:
    """Build / return the dataset's primary loader artefact.

    For temporally-windowed datasets (webots, msiln_site1_b1,
    imuwifine_floor4, ipin2024_floor0) this returns a
    ``FusionDataModule``. For RoNIN canonical it returns a
    ``StridedSequenceDataset``. For TartanAir / UJI it returns a
    dataset-specific dict / tuple — see each module's docstring.
    """
    return _resolve(name).load(**kwargs)


def dataset_stats(name: str) -> dict:
    """Return a dict of structured statistics for the dataset
    (used by ``plot_dataset_overview`` + the notebook §0 cells)."""
    return _resolve(name).stats()


def preprocessing_demo(name: str, modality: str, **kwargs) -> dict:
    """Return a raw / preprocessed pair of samples for the given
    (dataset, modality) — for the notebook's side-by-side
    preprocessing figures."""
    return _resolve(name).preprocessing_demo(modality, **kwargs)


__all__ = ["list_datasets", "load_dataset", "dataset_stats", "preprocessing_demo"]
