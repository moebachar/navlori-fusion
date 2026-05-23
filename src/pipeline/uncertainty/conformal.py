"""Conformal position uncertainty (Stage E).

Split-conformal prediction wraps a *trained, frozen* FusionTransformer with
a calibrated uncertainty radius. It is distribution-free: given any model
and a calibration set held out from training, it guarantees (in
expectation, under exchangeability) that the true position falls within
the predicted region with probability ≥ 1 − α.

For 2-D position the natural conformity score is the Euclidean residual
‖pred − target‖. The calibrated radius is the rank-based (1 − α) quantile
of calibration residuals; the prediction region is the disc of that radius
around each prediction. Per-axis ``lower``/``upper`` bounds (the
``BaseUncertainty`` contract) are reported as the axis-aligned square that
encloses the disc.

Reference: Vovk et al., *Algorithmic Learning in a Random World* (2005);
Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (2021).
"""

from __future__ import annotations

import math

import torch

from .base import BaseUncertainty


class ConformalPosition(BaseUncertainty):
    """Split-conformal uncertainty for (x, y) predictions.

    Parameters
    ----------
    alpha : float
        Miscoverage rate. ``alpha=0.1`` → 90% target coverage.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("radius", torch.tensor(float("nan")))

    @torch.no_grad()
    def calibrate(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Set the radius from calibration-set residuals.

        Uses the finite-sample-valid quantile level
        ``ceil((n+1)(1−α)) / n`` so coverage holds for the given n, not
        just asymptotically.
        """
        res = torch.linalg.norm(predictions - targets, dim=1)  # (n,)
        n = res.numel()
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.radius = torch.quantile(res, level).to(self.radius)

    @torch.no_grad()
    def forward(
        self, predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(predictions, lower, upper)`` — the enclosing square."""
        if torch.isnan(self.radius):
            raise RuntimeError("ConformalPosition.calibrate() must run first")
        r = self.radius
        return predictions, predictions - r, predictions + r

    @torch.no_grad()
    def coverage(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Empirical coverage: fraction of targets inside the radius."""
        res = torch.linalg.norm(predictions - targets, dim=1)
        return float((res <= self.radius).float().mean())
