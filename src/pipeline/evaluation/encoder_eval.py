"""Encoder evaluation metrics — all functions are encoder-agnostic.

Each metric takes pre-computed embeddings (and optionally positions/timestamps)
and returns a scalar or small dict of scores.

Usage::

    z_train, y_train = extract_embeddings(encoder, train_loader)
    z_val, y_val     = extract_embeddings(encoder, val_loader)

    lp = linear_probe(z_train, y_train, z_val, y_val)
    au = alignment_uniformity(z_val, y_val)
    ed = effective_dimensionality(z_val)
    ts = temporal_smoothness(z_val, y_val)
    kp = knn_probe(z_train, y_train, z_val, y_val)
    tw = trustworthiness(X_val_raw, z_val)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# 1. Linear Probe
# ======================================================================

def linear_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, float]:
    """Train a linear layer on frozen embeddings to predict (x, y).

    Args:
        z_train: (N_train, embed_dim) training embeddings
        y_train: (N_train, 2) training positions
        z_val:   (N_val, embed_dim) validation embeddings
        y_val:   (N_val, 2) validation positions
        epochs:  training epochs for the linear head
        lr:      learning rate
        device:  torch device

    Returns:
        dict with 'mae' (meters) and 'rmse' (meters)
    """
    embed_dim = z_train.shape[1]

    # Build a simple linear head
    head = nn.Linear(embed_dim, 2).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    z_t = torch.tensor(z_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    # Train
    head.train()
    for _ in range(epochs):
        pred = head(z_t)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    head.eval()
    z_v = torch.tensor(z_val, dtype=torch.float32, device=device)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = head(z_v)

    diff = (pred - y_v).cpu().numpy()
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    euclidean = np.sqrt((diff ** 2).sum(axis=1)).mean()

    return {"mae": float(mae), "rmse": float(rmse), "mean_euclidean": float(euclidean)}


# ======================================================================
# 2. Alignment & Uniformity (Wang & Isola, ICML 2020)
# ======================================================================

def alignment_uniformity(
    z: np.ndarray,
    y: np.ndarray,
    distance_threshold: float = 1.0,
    t: float = 2.0,
    max_samples: int = 1000,
) -> dict[str, float]:
    """Compute alignment and uniformity of embeddings.

    Args:
        z: (N, embed_dim) embeddings
        y: (N, 2) positions — used to define "positive pairs"
        distance_threshold: max physical distance (meters) for a pair
                            to be considered "similar"
        t: temperature for uniformity (default 2, per paper)
        max_samples: subsample to this many points to keep O(N²) tractable

    Returns:
        dict with 'alignment' (lower=better) and 'uniformity' (lower=better)
    """
    # Subsample to avoid O(N²·D) memory blowup
    if len(z) > max_samples:
        idx = np.random.choice(len(z), max_samples, replace=False)
        z, y = z[idx], y[idx]

    z = _l2_normalize(z)
    n = len(z)

    # --- Pairwise squared distances in embedding space (memory-efficient) ---
    # ||a-b||² = ||a||² + ||b||² - 2·aᵀb  →  never materialises (N,N,D)
    z_sq = (z ** 2).sum(axis=1)                    # (N,)
    dot   = z @ z.T                                 # (N, N)
    z_dists_sq = z_sq[:, None] + z_sq[None, :] - 2 * dot  # (N, N)
    z_dists_sq = np.maximum(z_dists_sq, 0.0)       # numerical guard

    # --- Pairwise physical distances ---
    y_sq = (y ** 2).sum(axis=1)
    y_dot = y @ y.T
    pos_dists = np.sqrt(np.maximum(y_sq[:, None] + y_sq[None, :] - 2 * y_dot, 0.0))

    # --- Alignment ---
    pos_mask = pos_dists < distance_threshold
    np.fill_diagonal(pos_mask, False)
    alignment = float(z_dists_sq[pos_mask].mean()) if pos_mask.any() else 0.0

    # --- Uniformity ---
    off_diag = ~np.eye(n, dtype=bool)
    uniformity = float(np.log(np.exp(-t * z_dists_sq[off_diag]).mean()))

    return {"alignment": alignment, "uniformity": uniformity}


# ======================================================================
# 3. Effective Dimensionality
# ======================================================================

def effective_dimensionality(z: np.ndarray) -> dict[str, float]:
    """Compute effective dimensionality via participation ratio.

    Args:
        z: (N, embed_dim) embeddings

    Returns:
        dict with 'participation_ratio', 'dims_95' (dims for 95% variance),
        and 'embed_dim' (total available dimensions)
    """
    # Center the embeddings
    z_centered = z - z.mean(axis=0)

    # SVD
    _, s, _ = np.linalg.svd(z_centered, full_matrices=False)
    variance = s ** 2

    # Participation ratio: (sum σ²)² / sum(σ⁴)
    total_var = variance.sum()
    if total_var < 1e-12:
        # Collapsed: all embeddings identical
        return {"participation_ratio": 0.0, "dims_95": 0, "embed_dim": z.shape[1]}
    pr = float(total_var ** 2 / (variance ** 2).sum())

    # Dims needed for 95% variance
    cumvar = np.cumsum(variance) / total_var
    dims_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    return {
        "participation_ratio": pr,
        "dims_95": dims_95,
        "embed_dim": z.shape[1],
    }


# ======================================================================
# 4. Temporal Smoothness
# ======================================================================

def temporal_smoothness(
    z: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Correlation between embedding changes and physical displacement.

    Expects z and y to be in temporal order (consecutive timestamps).

    Args:
        z: (N, embed_dim) embeddings in time order
        y: (N, 2) positions in time order

    Returns:
        dict with 'correlation' (Pearson r between ||Δz|| and ||Δy||),
        'mean_dz' and 'std_dz'
    """
    # Compute consecutive differences
    dz = np.linalg.norm(np.diff(z, axis=0), axis=1)  # (N-1,)
    dy = np.linalg.norm(np.diff(y, axis=0), axis=1)  # (N-1,)

    # Pearson correlation
    if dz.std() < 1e-10 or dy.std() < 1e-10:
        corr = 0.0
    else:
        corr = float(np.corrcoef(dz, dy)[0, 1])

    return {
        "correlation": corr,
        "mean_dz": float(dz.mean()),
        "std_dz": float(dz.std()),
    }


# ======================================================================
# 5. KNN Probe
# ======================================================================

def knn_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    """Non-parametric position prediction using k-nearest neighbors.

    Args:
        z_train: (N_train, embed_dim)
        y_train: (N_train, 2)
        z_val:   (N_val, embed_dim)
        y_val:   (N_val, 2)
        k:       number of neighbors

    Returns:
        dict with 'mae' and 'mean_euclidean' in meters
    """
    from sklearn.neighbors import KNeighborsRegressor

    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn.fit(z_train, y_train)
    y_pred = knn.predict(z_val)

    diff = y_val - y_pred
    mae = float(np.abs(diff).mean())
    euclidean = float(np.sqrt((diff ** 2).sum(axis=1)).mean())

    return {"mae": mae, "mean_euclidean": euclidean}


# ======================================================================
# 6. Trustworthiness (Venna & Kaski, 2006)
# ======================================================================

def trustworthiness(
    X: np.ndarray,
    z: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Neighborhood preservation: how well local structure is maintained.

    Args:
        X: (N, input_dim) raw input features
        z: (N, embed_dim) embeddings
        k: neighborhood size

    Returns:
        dict with 'trustworthiness' score in [0, 1] (higher=better)
    """
    from sklearn.manifold import trustworthiness as sk_trust

    score = float(sk_trust(X, z, n_neighbors=k))
    return {"trustworthiness": score}


# ======================================================================
# Helper: extract embeddings from encoder + dataloader
# ======================================================================

@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataloader,
    modality: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run encoder on all batches, collect embeddings and targets.

    Args:
        encoder:    nn.Module that maps (batch, window, features) → (batch, embed_dim)
        dataloader: yields dicts with modality key and 'target'
        modality:   which key to feed to the encoder
        device:     torch device

    Returns:
        (embeddings, positions) as numpy arrays
    """
    encoder = encoder.to(device).eval()
    all_z = []
    all_y = []

    for batch in dataloader:
        # Handle both TensorDataset (tuple) and FusionDataset (dict)
        if isinstance(batch, (list, tuple)):
            x, y = batch[0].to(device), batch[1]
        else:
            x, y = batch[modality].to(device), batch["target"]

        z = encoder(x)
        # If encoder outputs (batch, n_tokens, embed_dim), mean-pool tokens
        if z.ndim == 3:
            z = z.mean(dim=1)

        all_z.append(z.cpu().numpy())
        all_y.append(y.numpy())

    return np.concatenate(all_z), np.concatenate(all_y)


# ======================================================================
# Full evaluation report
# ======================================================================

def evaluate_encoder(
    encoder: nn.Module,
    train_loader,
    val_loader,
    modality: str,
    raw_val_inputs: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, dict]:
    """Run all 6 evaluation metrics on an encoder.

    Args:
        encoder:         the encoder module
        train_loader:    training DataLoader
        val_loader:      validation DataLoader
        modality:        modality key (e.g. "imu", "wifi")
        raw_val_inputs:  (N_val, input_dim) raw features for trustworthiness
                         (optional — skipped if None)
        device:          torch device

    Returns:
        dict of metric_name → results dict
    """
    z_train, y_train = extract_embeddings(encoder, train_loader, modality, device)
    z_val, y_val = extract_embeddings(encoder, val_loader, modality, device)

    results = {
        "linear_probe": linear_probe(z_train, y_train, z_val, y_val, device=device),
        "alignment_uniformity": alignment_uniformity(z_val, y_val),
        "effective_dimensionality": effective_dimensionality(z_val),
        "temporal_smoothness": temporal_smoothness(z_val, y_val),
        "knn_probe": knn_probe(z_train, y_train, z_val, y_val),
    }

    if raw_val_inputs is not None:
        results["trustworthiness"] = trustworthiness(raw_val_inputs, z_val)

    return results


def print_report(results: dict[str, dict], modality: str = "") -> None:
    """Pretty-print evaluation results."""
    header = f"Encoder Evaluation Report — {modality}" if modality else "Encoder Evaluation Report"
    print(f"\n{'=' * 55}")
    print(f"  {header}")
    print(f"{'=' * 55}")

    if "linear_probe" in results:
        lp = results["linear_probe"]
        print(f"  Linear Probe:     MAE={lp['mae']:.3f}m  RMSE={lp['rmse']:.3f}m  Euclid={lp['mean_euclidean']:.3f}m")

    if "knn_probe" in results:
        kp = results["knn_probe"]
        print(f"  KNN Probe (k=5):  MAE={kp['mae']:.3f}m  Euclid={kp['mean_euclidean']:.3f}m")

    if "alignment_uniformity" in results:
        au = results["alignment_uniformity"]
        print(f"  Alignment:        {au['alignment']:.4f}  (lower=better)")
        print(f"  Uniformity:       {au['uniformity']:.4f}  (lower=better)")

    if "effective_dimensionality" in results:
        ed = results["effective_dimensionality"]
        print(f"  Effective Dim:    PR={ed['participation_ratio']:.1f}/{ed['embed_dim']}  95%var={ed['dims_95']} dims")

    if "temporal_smoothness" in results:
        ts = results["temporal_smoothness"]
        print(f"  Temporal Smooth:  corr={ts['correlation']:.3f}  mean_dz={ts['mean_dz']:.4f}")

    if "trustworthiness" in results:
        tw = results["trustworthiness"]
        print(f"  Trustworthiness:  {tw['trustworthiness']:.4f}  (higher=better)")

    print(f"{'=' * 55}\n")


# ======================================================================
# Internal helpers
# ======================================================================

def _l2_normalize(z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return z / norms


