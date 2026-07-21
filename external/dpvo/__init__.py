"""Vendored subset of Princeton-VL/DPVO.

Only the modules we actually use as a frozen feature extractor inside
`pipeline.encoders.dpvo_motion`. Upstream license preserved alongside.

Source: https://github.com/princeton-vl/DPVO
"""
from .extractor import BasicEncoder, BasicEncoder4, ResidualBlock

__all__ = ["BasicEncoder", "BasicEncoder4", "ResidualBlock"]
