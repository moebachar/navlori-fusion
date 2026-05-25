"""Stage A — Modality-specific encoders."""

from .base import BaseEncoder
from .imu import IMUCNN
from .odom import OdomCNN
from .vision import VisionViT
from .wifi import Anchor2Vec
from .wifi_set import WiFiSetTransformer

__all__ = [
    "BaseEncoder",
    "Anchor2Vec",
    "IMUCNN",
    "OdomCNN",
    "VisionViT",
    "WiFiSetTransformer",
]
