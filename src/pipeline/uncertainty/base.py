"""Base uncertainty quantification interface (Stage E)."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseUncertainty(ABC, nn.Module):
    """Wraps position estimates with calibrated uncertainty bounds.

    Subclasses must implement:
        - calibrate(predictions, targets) -> calibration state
        - forward(predictions) -> (predictions, lower_bound, upper_bound)
    """

    @abstractmethod
    def calibrate(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Calibrate uncertainty on held-out data."""
        ...

    @abstractmethod
    def forward(self, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predictions with uncertainty bounds.

        Returns:
            (predictions, lower_bound, upper_bound) each of shape (batch, 2)
        """
        ...
