"""Stage A — Modality-specific encoders.

Vision encoders kept here are motion-only (DPVO). The place-recognition
vision encoders (ACEVision, VisionViT) were removed 2026-05-20 — they were
trained with absolute-position targets, which the audit identified as a
memorization trap on small / single-floor datasets. Re-introduce only when
there is a dataset with enough environment diversity to justify them.
"""

from .base import BaseEncoder
from .dpvo_full import DPVOFullEncoder
from .dpvo_motion import DPVOMotionEncoder
from .imu import IMUCNN
from .odom import OdomCNN
from .wifi import Anchor2Vec
from .wifi_set import WiFiSetTransformer

__all__ = [
    "BaseEncoder",
    "Anchor2Vec",
    "DPVOFullEncoder",
    "DPVOMotionEncoder",
    "IMUCNN",
    "OdomCNN",
    "WiFiSetTransformer",
]
