"""Tests for modality-specific encoders (Stage A).

Verify output shapes, gradient flow, and BaseEncoder contract.
"""

import pytest
import torch

from src.pipeline.encoders import Anchor2Vec, IMUCNN, OdomCNN, VisionViT, BaseEncoder


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

BATCH = 4
EMBED_DIM = 64


# ------------------------------------------------------------------
# Base encoder contract
# ------------------------------------------------------------------

class DummyEncoder(BaseEncoder):
    def __init__(self):
        super().__init__(embed_dim=32)
        self.proj = torch.nn.Linear(10, 32)

    def forward(self, x):
        return self.proj(x)

    @property
    def input_spec(self):
        return {"modality": "test", "shape": (10,), "dtype": "float32"}


def test_base_encoder_contract():
    enc = DummyEncoder()
    assert enc.embed_dim == 32
    assert enc.input_spec["modality"] == "test"
    out = enc(torch.randn(2, 10))
    assert out.shape == (2, 32)


# ------------------------------------------------------------------
# WiFi — Anchor2Vec
# ------------------------------------------------------------------

class TestAnchor2Vec:
    def test_output_shape(self):
        enc = Anchor2Vec(n_aps=117, embed_dim=EMBED_DIM, n_anchors=32)
        x = torch.randn(BATCH, 1, 117)
        out = enc(x)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_output_shape_squeezed(self):
        """Should also accept (batch, n_aps) without window dim."""
        enc = Anchor2Vec(n_aps=117, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH, 117)
        out = enc(x)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_gradients_flow(self):
        enc = Anchor2Vec(n_aps=32, embed_dim=EMBED_DIM, n_anchors=8)
        x = torch.randn(BATCH, 1, 32, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert enc.anchors.grad is not None

    def test_is_base_encoder(self):
        enc = Anchor2Vec(n_aps=10, embed_dim=EMBED_DIM)
        assert isinstance(enc, BaseEncoder)
        assert enc.embed_dim == EMBED_DIM
        assert enc.input_spec["modality"] == "wifi"

    def test_pca_reduced_input(self):
        """Works with PCA-reduced WiFi (e.g. 32 dims instead of 117)."""
        enc = Anchor2Vec(n_aps=32, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH, 1, 32)
        out = enc(x)
        assert out.shape == (BATCH, EMBED_DIM)


# ------------------------------------------------------------------
# IMU — CNN-1D
# ------------------------------------------------------------------

class TestIMUCNN:
    def test_output_shape(self):
        enc = IMUCNN(in_features=9, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH, 32, 9)
        out = enc(x)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_custom_channels(self):
        enc = IMUCNN(in_features=9, embed_dim=EMBED_DIM, channels=(16, 32))
        x = torch.randn(BATCH, 32, 9)
        out = enc(x)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_different_window_size(self):
        enc = IMUCNN(in_features=9, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH, 64, 9)  # larger window
        out = enc(x)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_gradients_flow(self):
        enc = IMUCNN(in_features=9, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH, 32, 9, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_is_base_encoder(self):
        enc = IMUCNN(in_features=9, embed_dim=EMBED_DIM)
        assert isinstance(enc, BaseEncoder)
        assert enc.input_spec["modality"] == "imu"


# ------------------------------------------------------------------
# Odometry — CNN-1D
# ------------------------------------------------------------------

class TestOdomCNN:
    def test_output_shape(self):
        enc = OdomCNN(in_features=7, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH, 16, 7)
        out = enc(x)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_gradients_flow(self):
        enc = OdomCNN(in_features=7, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH, 16, 7, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_is_base_encoder(self):
        enc = OdomCNN(in_features=7, embed_dim=EMBED_DIM)
        assert isinstance(enc, BaseEncoder)
        assert enc.input_spec["modality"] == "odom"

    def test_fewer_params_than_imu(self):
        """Odom defaults to lighter channels than IMU."""
        imu = IMUCNN(in_features=9, embed_dim=EMBED_DIM)
        odom = OdomCNN(in_features=7, embed_dim=EMBED_DIM)
        imu_params = sum(p.numel() for p in imu.parameters())
        odom_params = sum(p.numel() for p in odom.parameters())
        assert odom_params < imu_params


# ------------------------------------------------------------------
# Vision — ViT
# ------------------------------------------------------------------

class TestVisionViT:
    @pytest.fixture(scope="class")
    def encoder(self):
        return VisionViT(embed_dim=EMBED_DIM, backbone="vit_b_16", freeze_backbone=True)

    def test_output_shape(self, encoder):
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, EMBED_DIM)

    def test_backbone_frozen(self, encoder):
        for name, param in encoder.vit.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_head_trainable(self, encoder):
        for param in encoder.head.parameters():
            assert param.requires_grad

    def test_is_base_encoder(self, encoder):
        assert isinstance(encoder, BaseEncoder)
        assert encoder.input_spec["modality"] == "camera"

    def test_gradient_flows_to_head(self, encoder):
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        head_param = list(encoder.head.parameters())[0]
        assert head_param.grad is not None


# ------------------------------------------------------------------
# Parameter counts
# ------------------------------------------------------------------

class TestParameterCounts:
    def test_wifi_lightweight(self):
        enc = Anchor2Vec(n_aps=117, embed_dim=128, n_anchors=64)
        n = sum(p.numel() for p in enc.parameters())
        print(f"WiFi Anchor2Vec: {n:,} params")
        assert n < 100_000

    def test_imu_lightweight(self):
        enc = IMUCNN(in_features=9, embed_dim=128)
        n = sum(p.numel() for p in enc.parameters())
        print(f"IMU CNN: {n:,} params")
        assert n < 100_000

    def test_odom_lightweight(self):
        enc = OdomCNN(in_features=7, embed_dim=128)
        n = sum(p.numel() for p in enc.parameters())
        print(f"Odom CNN: {n:,} params")
        assert n < 50_000
