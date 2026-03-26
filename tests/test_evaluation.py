"""Tests for encoder evaluation metrics.

Uses synthetic embeddings (no real encoder needed) to verify
that each metric returns correct types, shapes, and sensible values.
"""

import numpy as np
import pytest

from src.pipeline.evaluation import (
    alignment_uniformity,
    effective_dimensionality,
    knn_probe,
    linear_probe,
    temporal_smoothness,
    trustworthiness,
)


# ------------------------------------------------------------------
# Fixtures — synthetic data
# ------------------------------------------------------------------

@pytest.fixture
def good_embeddings():
    """Embeddings that correlate with position (encoder doing well)."""
    rng = np.random.RandomState(42)
    n = 200
    y = rng.uniform(-10, 10, (n, 2)).astype(np.float32)
    # Embedding = position + small noise (good encoder)
    z = np.hstack([y, rng.randn(n, 30).astype(np.float32) * 0.1])
    return z, y


@pytest.fixture
def random_embeddings():
    """Random embeddings unrelated to position (bad encoder)."""
    rng = np.random.RandomState(42)
    n = 200
    y = rng.uniform(-10, 10, (n, 2)).astype(np.float32)
    z = rng.randn(n, 32).astype(np.float32)
    return z, y


@pytest.fixture
def collapsed_embeddings():
    """All embeddings map to the same point (degenerate encoder)."""
    n = 200
    rng = np.random.RandomState(42)
    y = rng.uniform(-10, 10, (n, 2)).astype(np.float32)
    z = np.ones((n, 32), dtype=np.float32) * 0.5
    return z, y


# ------------------------------------------------------------------
# Linear Probe
# ------------------------------------------------------------------

class TestLinearProbe:
    def test_returns_correct_keys(self, good_embeddings):
        z, y = good_embeddings
        result = linear_probe(z[:150], y[:150], z[150:], y[150:], epochs=50)
        assert "mae" in result
        assert "rmse" in result
        assert "mean_euclidean" in result

    def test_good_beats_random(self, good_embeddings, random_embeddings):
        z_good, y = good_embeddings
        z_bad, _ = random_embeddings
        good_result = linear_probe(z_good[:150], y[:150], z_good[150:], y[150:], epochs=200)
        bad_result = linear_probe(z_bad[:150], y[:150], z_bad[150:], y[150:], epochs=200)
        assert good_result["mae"] < bad_result["mae"]

    def test_values_positive(self, good_embeddings):
        z, y = good_embeddings
        result = linear_probe(z[:150], y[:150], z[150:], y[150:], epochs=50)
        assert result["mae"] > 0
        assert result["rmse"] > 0


# ------------------------------------------------------------------
# Alignment & Uniformity
# ------------------------------------------------------------------

class TestAlignmentUniformity:
    def test_returns_correct_keys(self, good_embeddings):
        z, y = good_embeddings
        result = alignment_uniformity(z, y)
        assert "alignment" in result
        assert "uniformity" in result

    def test_good_alignment_lower(self, good_embeddings, random_embeddings):
        z_good, y = good_embeddings
        z_bad, _ = random_embeddings
        good_au = alignment_uniformity(z_good, y, distance_threshold=2.0)
        bad_au = alignment_uniformity(z_bad, y, distance_threshold=2.0)
        # Good embeddings: nearby positions → nearby embeddings → lower alignment
        assert good_au["alignment"] < bad_au["alignment"]

    def test_collapsed_bad_uniformity(self, collapsed_embeddings, good_embeddings):
        z_col, y_col = collapsed_embeddings
        z_good, y_good = good_embeddings
        col_au = alignment_uniformity(z_col, y_col)
        good_au = alignment_uniformity(z_good, y_good)
        # Collapsed: all at same point → distances=0 → exp(0)=1 → log(1)=0
        # Good: spread out → distances large → exp(-t*d²)≈0 → log(small) < 0
        # Higher uniformity = worse (less spread). Collapsed = 0, good < 0.
        assert col_au["uniformity"] > good_au["uniformity"]


# ------------------------------------------------------------------
# Effective Dimensionality
# ------------------------------------------------------------------

class TestEffectiveDimensionality:
    def test_returns_correct_keys(self, good_embeddings):
        z, _ = good_embeddings
        result = effective_dimensionality(z)
        assert "participation_ratio" in result
        assert "dims_95" in result
        assert "embed_dim" in result

    def test_collapsed_low_dimensionality(self, collapsed_embeddings):
        z, _ = collapsed_embeddings
        result = effective_dimensionality(z)
        assert result["participation_ratio"] == 0.0  # fully collapsed

    def test_spread_higher_dimensionality(self, good_embeddings, collapsed_embeddings):
        z_good, _ = good_embeddings
        z_col, _ = collapsed_embeddings
        pr_good = effective_dimensionality(z_good)["participation_ratio"]
        pr_col = effective_dimensionality(z_col)["participation_ratio"]
        assert pr_good > pr_col  # 0 for collapsed, >0 for spread

    def test_dims_95_within_range(self, good_embeddings):
        z, _ = good_embeddings
        result = effective_dimensionality(z)
        assert 1 <= result["dims_95"] <= result["embed_dim"]


# ------------------------------------------------------------------
# Temporal Smoothness
# ------------------------------------------------------------------

class TestTemporalSmoothness:
    def test_returns_correct_keys(self, good_embeddings):
        z, y = good_embeddings
        result = temporal_smoothness(z, y)
        assert "correlation" in result
        assert "mean_dz" in result
        assert "std_dz" in result

    def test_correlated_high(self):
        """If embedding change is proportional to displacement, corr ≈ 1."""
        n = 300
        # Trajectory with varying speed (acceleration/deceleration)
        t = np.linspace(0, 1, n)
        x = np.cumsum(np.abs(np.sin(t * 10)))  # varying step sizes
        y_pos = np.stack([x, x * 0.5], axis=1).astype(np.float32)
        # Embedding = scaled position — perfect proportionality
        z = (y_pos * 3).astype(np.float32)
        result = temporal_smoothness(z, y_pos)
        assert result["correlation"] > 0.99

    def test_random_low_correlation(self, random_embeddings):
        z, y = random_embeddings
        result = temporal_smoothness(z, y)
        assert abs(result["correlation"]) < 0.5


# ------------------------------------------------------------------
# KNN Probe
# ------------------------------------------------------------------

class TestKNNProbe:
    def test_returns_correct_keys(self, good_embeddings):
        z, y = good_embeddings
        result = knn_probe(z[:150], y[:150], z[150:], y[150:])
        assert "mae" in result
        assert "mean_euclidean" in result

    def test_good_beats_random(self, good_embeddings, random_embeddings):
        z_good, y = good_embeddings
        z_bad, _ = random_embeddings
        good_result = knn_probe(z_good[:150], y[:150], z_good[150:], y[150:])
        bad_result = knn_probe(z_bad[:150], y[:150], z_bad[150:], y[150:])
        assert good_result["mae"] < bad_result["mae"]


# ------------------------------------------------------------------
# Trustworthiness
# ------------------------------------------------------------------

class TestTrustworthiness:
    def test_returns_correct_keys(self, good_embeddings):
        z, y = good_embeddings
        result = trustworthiness(y, z, k=5)
        assert "trustworthiness" in result

    def test_score_in_range(self, good_embeddings):
        z, y = good_embeddings
        result = trustworthiness(y, z, k=5)
        assert 0 <= result["trustworthiness"] <= 1

    def test_identity_perfect(self):
        """If embeddings = input, trustworthiness should be 1."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        result = trustworthiness(X, X, k=5)
        assert result["trustworthiness"] > 0.99
