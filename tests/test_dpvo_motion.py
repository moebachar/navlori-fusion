"""Smoke tests for DPVOMotionEncoder.

Verify:
1. Constructs and loads `dpvo.pth` weights into the frozen trunk.
2. Trunk parameters are frozen; head parameters are trainable.
3. forward(B, 2, 3, H, W) -> (B, embed_dim) with finite values.
4. Single-frame fallback path (B, 3, H, W) — duplicates the frame.
5. Dataset yields (2, 3, 480, 640) when windows['camera'] = 2.
6. extract_backbone_features path produces consistent shapes.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.pipeline.encoders import DPVOMotionEncoder


WEIGHTS = Path("runs/_weights/dpvo.pth")
EMBED_DIM = 128
BATCH = 2
H, W = 480, 640

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(),
    reason=f"DPVO weights not found at {WEIGHTS}; "
           "run `python scripts/fetch_dpvo_weights.py`.",
)


@pytest.fixture(scope="module")
def encoder():
    return DPVOMotionEncoder(embed_dim=EMBED_DIM, n_patches=64)


def test_construct_and_freeze(encoder):
    # Trunk frozen.
    trunk_params = list(encoder.trunk.parameters())
    assert len(trunk_params) > 0
    assert all(not p.requires_grad for p in trunk_params)
    # Head trainable.
    head_params = list(encoder.head.parameters())
    assert len(head_params) > 0
    assert all(p.requires_grad for p in head_params)


def test_forward_pair_shape(encoder):
    x = torch.randn(BATCH, 2, 3, H, W)
    out = encoder(x)
    assert out.shape == (BATCH, EMBED_DIM)
    assert torch.isfinite(out).all()


def test_forward_single_frame_fallback(encoder):
    """Single (B, 3, H, W) input: duplicate frame, expect well-defined output."""
    x = torch.randn(BATCH, 3, H, W)
    out = encoder(x)
    assert out.shape == (BATCH, EMBED_DIM)
    assert torch.isfinite(out).all()


def test_input_spec(encoder):
    spec = encoder.input_spec
    assert spec["modality"] == "camera"
    assert spec["shape"] == (2, 3, 480, 640)


def test_dataset_camera_pair():
    """Dataset returns (2, 3, 480, 640) when windows['camera'] = 2."""
    from src.pipeline.data.dataset import FusionDataset

    ds = FusionDataset(
        data_dir="data/async_collection",
        path_ids=[1, 2],
        modalities=["camera"],
        windows={"camera": 2},
        normalize=False,
    )
    sample = ds[0]
    assert sample["camera"].shape == (2, 3, 480, 640)


def test_extract_backbone_features(encoder, tmp_path):
    """Cached features round-trip and the head path reproduces forward()."""
    from torch.utils.data import DataLoader, TensorDataset

    # Tiny synthetic dataset of frame pairs.
    n = 4
    x = torch.randn(n, 2, 3, H, W)
    y = torch.randn(n, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    cache = tmp_path / "feat.pt"
    feats, tgts = encoder.extract_backbone_features(
        loader, device="cpu", cache_path=cache,
    )
    assert feats.shape == (n, encoder.n_patches, encoder._patch_token_dim)
    assert tgts.shape == (n, 2)
    assert cache.exists()

    # Cache hit returns same data.
    feats2, tgts2 = encoder.extract_backbone_features(
        loader, device="cpu", cache_path=cache,
    )
    assert torch.allclose(feats, feats2)
    assert torch.allclose(tgts, tgts2)

    # Head over cached tokens reproduces forward() (modulo dropout — we eval).
    encoder.eval()
    full = encoder(x)
    head_only = encoder.head(feats)
    assert head_only.shape == full.shape
    assert torch.allclose(full, head_only, atol=1e-5)
