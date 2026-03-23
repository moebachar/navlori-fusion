"""Tests for pipeline composition."""

import torch

from src.pipeline import FusionPipeline


class SimpleEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).unsqueeze(1)


def test_pipeline_encoder_only():
    """Pipeline with only Stage A should pass through encoded tokens."""
    pipe = FusionPipeline(encoders={"wifi": SimpleEncoder(10, 64), "imu": SimpleEncoder(6, 64)})
    inputs = {"wifi": torch.randn(2, 10), "imu": torch.randn(2, 6)}
    out = pipe(inputs)
    assert "position" in out
    assert out["position"].shape == (2, 2, 64)  # 2 modalities, 1 token each


def test_pipeline_missing_modality():
    """Pipeline should handle missing modalities gracefully."""
    pipe = FusionPipeline(encoders={"wifi": SimpleEncoder(10, 64), "imu": SimpleEncoder(6, 64)})
    inputs = {"wifi": torch.randn(2, 10)}  # imu missing
    out = pipe(inputs)
    assert out["position"].shape == (2, 1, 64)  # only wifi token
