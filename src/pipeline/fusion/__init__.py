"""Stage C — Cross-modal fusion modules."""

from .base import BaseFusion
from .transformer import ContinuousTimeEncoding, FusionTransformer

__all__ = ["BaseFusion", "ContinuousTimeEncoding", "FusionTransformer"]
