"""DPVO-based motion encoder.

Wraps DPVO's pretrained patch feature extractor (``Patchifier.fnet``) — a
small ResNet-style CNN trained on TartanAir as part of Princeton-VL's
DPVO (Deep Patch Visual Odometry, NeurIPS 2023) — and uses it as a
**frozen, scene-agnostic** feature backbone for two consecutive frames.
On top of the frozen trunk we:

    1. Sample N patches on a coarse grid in frame ``t-1``.
    2. Cross-correlate each patch's descriptor against frame ``t``'s
       feature map.
    3. Soft-argmax to recover sub-pixel ``(dx, dy)`` per patch.
    4. Concat ``[patch_feat | dx | dy | ‖flow‖ | corr_peak]`` into a
       per-patch motion token.
    5. Trainable head: per-patch projection → attentive pooling → MLP
       → 128-D shared-embedding token.

The result is a 128-D motion token in the project's shared encoder
embedding space, suitable for Stage C cross-attention fusion.

Why this differs from Niantic-ACE
---------------------------------
ACE encodes "where the camera is" (absolute scene coordinates → PnP for
absolute pose); its trunk is scene-agnostic but its head is per-scene.
DPVO encodes "how the camera is moving" between frames; both trunk and
weights are *fully* scene-agnostic — the same ``dpvo.pth`` works on KITTI,
EuRoC, our Webots Tiago, etc., because it's just learning to track image
patches, not scene geometry.

Input convention
----------------
The dataset (with ``windows['camera'] >= 2`` and a temporal
``camera_stride`` >= ~5) yields ImageNet-normalised RGB pairs
``(B, 2, 3, H, W)``. We un-normalise to ``[0, 1]`` and apply DPVO's own
normalisation (``2x - 0.5``, matching ``VONet.forward``). At
``camera_stride=1`` (default in the dataset) the inter-frame motion on
the stride-4 feature grid is sub-pixel and the encoder is degenerate; the
DPVO config sets stride=5 (~1 s gap, ~15 cm motion) so soft-argmax sees
a real correlation peak shift.

References
----------
Teed, Lipson, Deng. "Deep Patch Visual Odometry." NeurIPS 2023.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.baselines import BasicEncoder4

from .base import BaseEncoder

# ImageNet un-normalisation constants (the dataset applies these forward).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Trainable head
# ---------------------------------------------------------------------------

class _AttentivePool(nn.Module):
    """Single-query MHA pooling: ``(B, N, D) -> (B, D)``."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        q = self.q.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class _MotionHead(nn.Module):
    """Trainable suffix: ``(B, N, in_dim) -> (B, embed_dim)``.

    Wrapping the per-patch projection + attentive pool + MLP into a single
    module lets us expose this as ``encoder.head`` for ``EncoderTrainer``'s
    vision-cache path: the trainer caches the per-patch motion tokens once
    (everything before this head is frozen / parameter-free) and trains
    only the parameters in here.
    """

    def __init__(self, in_dim: int, embed_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.token_proj = nn.Linear(in_dim, embed_dim)
        self.pool = _AttentivePool(embed_dim, num_heads=4)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.token_proj(tokens)   # (B, N, embed_dim)
        pooled = self.pool(z)         # (B, embed_dim)
        return self.mlp(pooled)       # (B, embed_dim)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class DPVOMotionEncoder(BaseEncoder):
    """Frozen DPVO ``fnet`` + patch tracking + trainable head.

    Parameters
    ----------
    embed_dim
        Output dim of the projection (default 128, to match other project
        encoders).
    weights_path
        Path to ``dpvo.pth`` from the Princeton-VL release. Run
        ``python scripts/fetch_dpvo_weights.py`` to populate.
    n_patches
        Number of patches sampled per frame, on a sqrt(n) x sqrt(n) coarse
        grid in frame ``t-1`` (default 64 → 8x8 grid).
    head_hidden
        Width of the MLP hidden layer (default 256).
    dropout
        Dropout applied before the projection output.
    input_is_imagenet_normalised
        If True (default), assume input is ImageNet-normalised RGB and
        reverse that before DPVO's own preprocessing.
    """

    #: Stride of DPVO's BasicEncoder4 over the input grid.
    OUTPUT_STRIDE = 4

    #: Feature-map channels produced by DPVO's fnet.
    TRUNK_FEATURE_DIM = 128

    def __init__(
        self,
        embed_dim: int = 128,
        weights_path: str | Path = "runs/_weights/dpvo.pth",
        n_patches: int = 64,
        head_hidden: int = 256,
        dropout: float = 0.1,
        input_is_imagenet_normalised: bool = True,
        search_radius: int = 32,
    ):
        super().__init__(embed_dim)
        # Half-size of the local correlation search window, in feature-map
        # cells. The correlation is masked outside ±search_radius of each
        # patch's frame-1 location: a global cosine search is ill-posed in
        # low-texture indoor scenes (a uniform patch matches everywhere),
        # so we bound it the way DPVO bounds its search via geometry. 32
        # cells ≈ 128 image px ≈ generous coverage for ~1 s of robot motion.
        self.search_radius = int(search_radius)

        # ----- Frozen scene-agnostic trunk (DPVO's fnet) -----
        self.trunk = BasicEncoder4(output_dim=self.TRUNK_FEATURE_DIM, norm_fn="instance")
        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(
                f"DPVO weights not found at {wp}. Run "
                "`python scripts/fetch_dpvo_weights.py` to download."
            )
        full_state = torch.load(wp, weights_only=True, map_location="cpu")
        # The release stores keys as `module.patchify.fnet.<name>` because
        # DPVO trains under DataParallel. Strip the prefix to match BasicEncoder4.
        prefix = "module.patchify.fnet."
        fnet_state = {
            k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)
        }
        if not fnet_state:
            raise RuntimeError(
                f"No fnet keys found in {wp} (looked for prefix '{prefix}'). "
                f"Got keys like: {list(full_state.keys())[:3]}..."
            )
        self.trunk.load_state_dict(fnet_state, strict=True)
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.trunk.eval()

        # ----- Preprocessing buffers (no params, move with module) -----
        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("_imagenet_std", _IMAGENET_STD.clone())
        self.input_is_imagenet_normalised = input_is_imagenet_normalised

        # ----- Patch grid (fixed, sqrt(n_patches) x sqrt(n_patches)) -----
        self.n_patches = n_patches
        gs = int(round(n_patches ** 0.5))
        if gs * gs != n_patches:
            raise ValueError(
                f"n_patches must be a perfect square; got {n_patches}"
            )
        self._grid_size = gs

        # ----- Trainable head (per-patch proj + pool + MLP) -----
        # per-patch token = [trunk_feat (128) | dx | dy | ‖flow‖ | corr_peak] = 132
        self._patch_token_dim = self.TRUNK_FEATURE_DIM + 4
        self.head = _MotionHead(
            in_dim=self._patch_token_dim,
            embed_dim=embed_dim,
            hidden=head_hidden,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _to_dpvo_range(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ImageNet-normalised RGB to DPVO's expected range.

        DPVO's ``VONet.forward`` does: ``images = 2 * (images / 255.0) - 0.5``.
        Our pipeline already loads RGB as float in [0, 1] then ImageNet-normalises.
        Reversing ImageNet gives [0, 1]; DPVO's transform on [0, 1] is
        equivalently ``2 * x - 0.5``.
        """
        if self.input_is_imagenet_normalised:
            x = x * self._imagenet_std + self._imagenet_mean  # -> [0, 1]
        return 2.0 * x - 0.5

    # ------------------------------------------------------------------
    # Patch sampling + correlation
    # ------------------------------------------------------------------

    def _patch_coords(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Coarse grid of patch centres in feature-map coordinates.

        Returns
        -------
        Tensor of shape ``(N, 2)`` with (x, y) in pixel coordinates.
        """
        gs = self._grid_size
        # Avoid the very edge of the feature map for stability.
        ys = torch.linspace(0.5, h - 1.5, gs, device=device, dtype=dtype)
        xs = torch.linspace(0.5, w - 1.5, gs, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.flatten(), gy.flatten()], dim=-1)

    @staticmethod
    def _bilinear_sample(fmap: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Bilinear sample ``(B, C, H, W)`` at sub-pixel ``(B, N, 2)`` coords.

        Returns
        -------
        Tensor of shape ``(B, N, C)``.
        """
        B, C, H, W = fmap.shape
        x = coords[..., 0] / max(W - 1, 1) * 2 - 1
        y = coords[..., 1] / max(H - 1, 1) * 2 - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(1)  # (B, 1, N, 2)
        sampled = F.grid_sample(fmap, grid, mode="bilinear",
                                align_corners=True, padding_mode="border")
        return sampled.squeeze(2).transpose(1, 2).contiguous()

    @staticmethod
    def _windowed_soft_argmax(
        scores: torch.Tensor, radius: int = 3, temperature: float = 20.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Windowed soft-argmax over a 2D correlation map.

        A plain soft-argmax over the whole ``H x W`` map (here 120x160 ≈ 19k
        cells) is degenerate: ``softmax`` spreads probability mass over
        thousands of cells, so the expectation collapses toward the map
        centroid regardless of where the true peak is — every patch reports
        ~zero flow. This is exactly how the encoder was broken.

        Instead we do what DPVO's own correlation does: take a **hard
        argmax** for the integer peak (valid anywhere in the map, so large
        motions are fine), then a **soft-argmax inside a small
        ``(2r+1)x(2r+1)`` window** around it for sub-pixel refinement. The
        peak can no longer collapse — the window is local by construction.

        Parameters
        ----------
        scores
            ``(B, N, H, W)`` correlation scores per patch.
        radius
            Half-size of the sub-pixel window (default 3 → 7x7, matching
            DPVO's ``altcorr`` window).
        temperature
            Softmax temperature for the window. ``scores`` are cosine
            similarities in [-1, 1]; a temperature ~20 gives the softmax
            enough contrast to localise the sub-pixel peak instead of
            averaging the window flat.

        Returns
        -------
        coords : ``(B, N, 2)`` matched (x, y) in feature-map pixels.
        sharp  : ``(B, N)`` peak sharpness — max softmax prob *within* the
                 window, in [0, 1]. ~1 = crisp unambiguous match,
                 ~1/(2r+1)^2 = flat/ambiguous.
        """
        B, N, _H, W = scores.shape
        r = radius
        device, dtype = scores.device, scores.dtype

        # --- global hard-argmax → integer peak (handles arbitrary motion) ---
        peak_idx = scores.view(B, N, -1).argmax(dim=-1)        # (B, N)
        py = torch.div(peak_idx, W, rounding_mode="floor")     # (B, N) int
        px = peak_idx - py * W

        # --- gather a (2r+1)x(2r+1) window around each peak ---
        # Pad with a very negative value so out-of-bounds cells vanish in
        # the softmax (peaks near the map edge stay well-defined).
        pad = F.pad(scores, (r, r, r, r), value=-1e4)          # (B,N,H+2r,W+2r)
        off = torch.arange(-r, r + 1, device=device)
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
        n_idx = torch.arange(N, device=device).view(1, N, 1, 1)
        y_idx = (py + r).view(B, N, 1, 1) + off.view(1, 1, -1, 1)
        x_idx = (px + r).view(B, N, 1, 1) + off.view(1, 1, 1, -1)
        window = pad[b_idx, n_idx, y_idx, x_idx]               # (B,N,2r+1,2r+1)

        # --- soft-argmax inside the window (temperature-scaled) ---
        probs = F.softmax(temperature * window.reshape(B, N, -1), dim=-1)
        sharp = probs.max(dim=-1).values                       # (B,N)
        oy, ox = torch.meshgrid(off.to(dtype), off.to(dtype), indexing="ij")
        ex = (probs * ox.reshape(-1)).sum(dim=-1)              # sub-pixel dx
        ey = (probs * oy.reshape(-1)).sum(dim=-1)              # sub-pixel dy

        coords = torch.stack([px.to(dtype) + ex, py.to(dtype) + ey], dim=-1)
        return coords, sharp

    @staticmethod
    def _mask_to_local_window(
        corr: torch.Tensor, centres: torch.Tensor, radius: int,
    ) -> torch.Tensor:
        """Set correlation outside ±radius of each patch's centre to -1e4.

        A global argmax over the whole correlation map is ill-posed in
        low-texture scenes — a patch on a uniform floor/wall matches
        thousands of cells equally and the argmax lands on an arbitrary far
        cell. Restricting the search to a local window makes a uniform
        patch correctly report ~zero flow (and a low ``sharp``), and keeps
        textured patches locked onto their true, nearby correspondence.

        Parameters
        ----------
        corr     : ``(B, N, H, W)`` correlation map.
        centres  : ``(B, N, 2)`` per-patch (x, y) search-window centres
                   (the patch locations in frame ``t-1``).
        radius   : half-size of the window in feature-map cells.
        """
        B, N, H, W = corr.shape
        dev = corr.device
        hh = torch.arange(H, device=dev).view(1, 1, H, 1)
        ww = torch.arange(W, device=dev).view(1, 1, 1, W)
        cx = centres[..., 0].view(B, N, 1, 1)
        cy = centres[..., 1].view(B, N, 1, 1)
        inside = ((hh - cy).abs() <= radius) & ((ww - cx).abs() <= radius)
        return corr.masked_fill(~inside, -1e4)

    # ------------------------------------------------------------------
    # Frozen forward up to per-patch tokens (cacheable)
    # ------------------------------------------------------------------

    def _trunk_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Run the frozen trunk on a batch of frames.

        Parameters
        ----------
        frames
            ``(B, 3, H, W)`` already in DPVO range.

        Returns
        -------
        Tensor of shape ``(B, 128, H/4, W/4)`` — DPVO's `fnet` output, scaled
        by 1/4 to match upstream convention.
        """
        with torch.no_grad():
            out = self.trunk(frames.unsqueeze(1))   # (B, 1, 128, H/4, W/4)
        return out.squeeze(1) / 4.0

    def _patch_motion_tokens(
        self, fmap_prev: torch.Tensor, fmap_curr: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-patch motion descriptors from two feature maps.

        Returns ``(B, N, 132)`` patch tokens.
        """
        B, _C, H, W = fmap_prev.shape
        coords_prev = self._patch_coords(H, W, fmap_prev.device, fmap_prev.dtype)
        coords_prev_b = coords_prev.unsqueeze(0).expand(B, -1, -1)        # (B, N, 2)

        # Patch descriptors from frame t-1.
        patches = self._bilinear_sample(fmap_prev, coords_prev_b)         # (B, N, C)

        # Cross-correlate against frame t. We L2-normalise both the patch
        # descriptors and the target feature map so the correlation is a
        # COSINE similarity in [-1, 1]. Without this the raw dot product is
        # energy-biased — it peaks at high-norm image regions, not at the
        # true correspondence — which made every match degenerate. DPVO
        # itself never hits this because it only correlates in a tiny
        # geometry-predicted window; our global search needs the cosine.
        patches_n = F.normalize(patches, dim=-1)                          # (B, N, C)
        fmap_curr_n = F.normalize(fmap_curr, dim=1)                       # (B, C, H, W)
        corr = torch.einsum("bnc,bchw->bnhw", patches_n, fmap_curr_n)     # (B, N, H, W)
        # Bound the search to a local window — a global argmax is degenerate
        # in low-texture scenes (see _mask_to_local_window).
        corr = self._mask_to_local_window(corr, coords_prev_b, self.search_radius)
        coords_curr, sharp = self._windowed_soft_argmax(corr, radius=3)   # (B, N, 2), (B, N)

        flow = coords_curr - coords_prev_b                                # (B, N, 2)
        flow_mag = flow.norm(dim=-1, keepdim=True)
        sharp_col = sharp.unsqueeze(-1)

        return torch.cat([patches, flow, flow_mag, sharp_col], dim=-1)    # (B, N, 132)

    def _frozen_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen path: ``(B, 2, 3, H, W) -> (B, n_patches, 132)``.

        Used by both ``forward`` and ``extract_backbone_features``.
        """
        if x.ndim == 4:
            x = x.unsqueeze(1).expand(-1, 2, -1, -1, -1)
        x = self._to_dpvo_range(x)
        fmap_prev = self._trunk_features(x[:, 0])
        fmap_curr = self._trunk_features(x[:, 1])
        return self._patch_motion_tokens(fmap_prev, fmap_curr)

    # ------------------------------------------------------------------
    # BaseEncoder API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a frame pair into a 128-D motion token.

        Parameters
        ----------
        x
            ``(B, 2, 3, H, W)`` ImageNet-normalised RGB frame pair, ordered
            ``[t-1, t]``. As a fallback for callers passing a single frame
            ``(B, 3, H, W)`` we duplicate it (zero motion).

        Returns
        -------
        Tensor of shape ``(B, embed_dim)``.
        """
        tokens = self._frozen_tokens(x)        # (B, N, 132)
        return self.head(tokens)               # (B, embed_dim)

    @torch.no_grad()
    def demo_forward(self, raw_input):
        """Per-encoder introspection (notebook §0). Runs the frozen
        trunk on the input image pair, returns the per-patch motion
        tokens (B, n_patches, 132) as ``intermediate``.

        ``raw_input`` should be ``(2, 3, H, W)`` or ``(B, 2, 3, H, W)``
        ImageNet-normalised float RGB frame pair.
        """
        import numpy as np
        x = raw_input
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 4:  # (2, 3, H, W) -> (1, 2, 3, H, W)
            x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        self.eval()
        tokens = self._frozen_tokens(x)
        encoded = self.head(tokens)
        return {
            "raw": x.detach().cpu().numpy(),
            "preprocessed": x.detach().cpu().numpy(),
            "intermediate": tokens.detach().cpu().numpy(),  # (B, n_patches, 132)
            "encoded": encoded.detach().cpu().numpy(),
            "description": (
                f"DPVOMotionEncoder: frozen DPVO BasicEncoder4 trunk on "
                f"frame pair (t-1, t) -> {self.n_patches} patches × "
                f"{self._patch_token_dim} channels (trunk feat + dx, dy, "
                f"‖flow‖, corr_peak) -> {self.embed_dim}-d token via "
                f"trainable head."
            ),
        }

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "camera",
            "shape": (2, 3, 480, 640),
            "dtype": "float32",
            "normalisation": ("imagenet" if self.input_is_imagenet_normalised
                              else "unit"),
        }

    # ------------------------------------------------------------------
    # Trainer-cache integration
    # ------------------------------------------------------------------

    def extract_backbone_features(
        self,
        dataloader,
        device: str = "cuda",
        cache_path: str | Path | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute per-patch motion tokens once and cache to disk.

        Everything before the head is either frozen (the DPVO trunk) or
        parameter-free (correlation, soft-argmax), so caching the
        ``(N_samples, n_patches, 132)`` tokens is safe — it doesn't bypass
        any trainable layer. The trainer then trains only ``self.head`` on
        the cached tensors, mirroring the ACE / DINOv2 pattern.

        Returns
        -------
        features : ``(N_samples, n_patches, 132)`` tensor
        targets  : ``(N_samples, 2)`` GT (x, y) tensor
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                saved = torch.load(cache_path, weights_only=True)
                return saved["features"], saved["targets"]

        self.to(device).eval()
        feats, tgts = [], []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(device), batch[1]
                else:
                    x, y = batch["camera"].to(device), batch["target"]
                tokens = self._frozen_tokens(x.to(device))   # (B, N, 132)
                feats.append(tokens.cpu())
                tgts.append(y)

        features = torch.cat(feats)
        targets = torch.cat(tgts)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "targets": targets}, cache_path)
        return features, targets
