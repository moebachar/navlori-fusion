"""Base encoder interface for all modality-specific encoders (Stage A)."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """All encoders map raw sensor input to a fixed-dim token in shared embedding space.

    Subclasses must implement:
        - forward(x) -> Tensor of shape (batch, embed_dim) or (batch, n_tokens, embed_dim)
        - input_spec: property describing expected input format
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw sensor data into embedding space."""
        ...

    @property
    @abstractmethod
    def input_spec(self) -> dict:
        """Describe expected input: shape, dtype, modality name."""
        ...
