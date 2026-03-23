"""Base cross-modal fusion interface (Stage C)."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFusion(ABC, nn.Module):
    """Combines temporally-aligned tokens from all modalities into a fused representation.

    Subclasses must implement:
        - forward(tokens, source_ids, mask) -> fused representation
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        source_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse multi-modal tokens.

        Args:
            tokens: (batch, n_tokens, embed_dim) temporally-aligned from Stage B
            source_ids: (batch, n_tokens) modality identifiers
            mask: (batch, n_tokens) boolean mask for absent modalities
        """
        ...
