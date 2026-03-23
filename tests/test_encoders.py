"""Tests for Stage A encoders — verify interface contract."""

import torch

from src.encoders.base import BaseEncoder


class DummyEncoder(BaseEncoder):
    """Minimal encoder for testing the base interface."""

    def __init__(self, in_features: int, embed_dim: int):
        super().__init__(embed_dim)
        self.proj = torch.nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).unsqueeze(1)  # (batch, 1, embed_dim)

    @property
    def input_spec(self) -> dict:
        return {"shape": (-1,), "dtype": torch.float32, "modality": "test"}


def test_base_encoder_output_shape():
    enc = DummyEncoder(in_features=10, embed_dim=128)
    x = torch.randn(4, 10)
    out = enc(x)
    assert out.shape == (4, 1, 128)


def test_base_encoder_embed_dim():
    enc = DummyEncoder(in_features=10, embed_dim=64)
    assert enc.embed_dim == 64


def test_base_encoder_input_spec():
    enc = DummyEncoder(in_features=10, embed_dim=128)
    spec = enc.input_spec
    assert "modality" in spec
    assert "shape" in spec
