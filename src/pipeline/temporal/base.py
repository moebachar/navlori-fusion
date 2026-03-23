"""Base temporal alignment interface (Stage B)."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTemporalAligner(ABC, nn.Module):
    """Assigns temporal context to tokens so fusion can reason about observation timing.

    Subclasses must implement:
        - forward(tokens, timestamps, source_ids) -> temporally-contextualized tokens
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        timestamps: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add temporal context to encoded tokens.

        Args:
            tokens: (batch, n_tokens, embed_dim) from Stage A encoders
            timestamps: (batch, n_tokens) continuous timestamps
            source_ids: (batch, n_tokens) integer modality identifiers
        """
        ...
