"""DPVO patch encoder trunk — princeton-vl/DPVO (BSD).

We use ONLY ``BasicEncoder4`` from ``dpvo/extractor.py`` — the dense
patch feature extractor. DPVO's full SLAM pipeline (``lietorch`` +
``altcorr`` custom CUDA ops) does NOT build on Windows; that's
documented in ``docs/EXTERNAL_DEPENDENCIES.md`` and not addressed in
this codebase. The extractor alone runs on CPU/GPU and provides the
frozen vision trunk for ``DPVOMotionEncoder``.
"""
from __future__ import annotations

import sys

from ._paths import DPVO_ROOT


def _ensure_path() -> None:
    if str(DPVO_ROOT) not in sys.path:
        sys.path.insert(0, str(DPVO_ROOT))


def get_basic_encoder4_class():
    """Return the ``BasicEncoder4`` class (for callers that want to
    instantiate it themselves, e.g. ``DPVOMotionEncoder`` wraps it
    with an ImageNet un-normalisation + motion head)."""
    _ensure_path()
    from dpvo.extractor import BasicEncoder4  # noqa: E402  type: ignore
    return BasicEncoder4


# Eagerly expose the class for callers that import it directly
# (matches the legacy ``from external.dpvo import BasicEncoder4`` pattern).
BasicEncoder4 = get_basic_encoder4_class()


def load_basic_encoder4(weights_path=None):
    """Load the DPVO patch encoder.

    If ``weights_path`` is provided, load the matching state-dict
    subset (DPVO's saved checkpoints prefix encoder weights with
    ``patchify.``); otherwise return a random-initialised encoder.
    """
    import torch
    _ensure_path()
    from dpvo.extractor import BasicEncoder4  # noqa: E402  type: ignore
    enc = BasicEncoder4(output_dim=128, norm_fn="instance")
    if weights_path is not None:
        sd = torch.load(weights_path, map_location="cpu")
        encoder_sd = {
            k[len("patchify."):]: v for k, v in sd.items()
            if k.startswith("patchify.")
        }
        enc.load_state_dict(encoder_sd, strict=False)
    return enc


__all__ = ["BasicEncoder4", "get_basic_encoder4_class", "load_basic_encoder4"]
