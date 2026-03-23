"""Base differentiable filter interface (Stage D)."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFilter(ABC, nn.Module):
    """Converts fused representation into position estimate with temporal consistency.

    Implements predict-update cycle with physics priors.

    Subclasses must implement:
        - predict(state) -> predicted next state
        - update(predicted_state, observation) -> corrected state
        - forward(fused_seq) -> trajectory of position estimates
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Physics-based state propagation."""
        ...

    @abstractmethod
    def update(self, predicted: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Learned correction step."""
        ...

    @abstractmethod
    def forward(self, fused_seq: torch.Tensor) -> torch.Tensor:
        """Full predict-update cycle over a sequence.

        Args:
            fused_seq: (batch, seq_len, fused_dim) from Stage C

        Returns:
            positions: (batch, seq_len, 2) predicted (x, y) trajectory
        """
        ...
