"""Vision encoder — pretrained ViT with projection head.

Reference: Shao et al., "EffLoc" — lightweight ViT backbone for 6-DOF pose
regression, reducing FLOPs by 86.8% vs AtLoc. We use a similar approach:
pretrained ViT backbone (frozen or fine-tunable) + learned projection head.

Architecture:
    1. ViT backbone (pretrained on ImageNet) → CLS token
    2. Linear projection → embed_dim
    3. LayerNorm

The backbone can be frozen (feature extraction) or unfrozen (fine-tuning).
Default: frozen backbone, only the projection head is trainable. This is
appropriate for Stage A evaluation; unfreeze for end-to-end training.

Input:  (batch, 3, 224, 224) — RGB image, ImageNet-normalized
Output: (batch, embed_dim)
"""

import torch
import torch.nn as nn
from torchvision import models

from .base import BaseEncoder


class VisionViT(BaseEncoder):
    """Pretrained ViT encoder for camera images.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension (default 128).
    backbone : str
        Which ViT variant to use: "vit_b_16" or "vit_b_32" (default "vit_b_16").
    freeze_backbone : bool
        If True, freeze backbone weights (only train projection head).
        Default True.
    dropout : float
        Dropout rate for projection head. Default 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        backbone: str = "vit_b_16",
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim)

        # Load pretrained ViT
        weights_map = {
            "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
            "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.DEFAULT),
        }
        if backbone not in weights_map:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(weights_map)}")

        model_fn, weights = weights_map[backbone]
        self.vit = model_fn(weights=weights)

        # ViT hidden dim (768 for vit_b_*)
        vit_hidden = self.vit.hidden_dim

        # Remove the classification head — we only need the feature extractor
        self.vit.heads = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Projection head
        self.head = nn.Sequential(
            nn.Linear(vit_hidden, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    @torch.no_grad()
    def extract_backbone_features(
        self,
        dataloader,
        device: str = "cuda",
        cache_path: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen backbone once over all images, return cached features.

        Since the backbone is frozen, its output never changes — cache it once
        and train only the projection head on the 768-dim vectors. Eliminates
        all disk I/O from the training loop.

        Args:
            dataloader: DataLoader yielding camera batches
            device:     device to run backbone on
            cache_path: if given, save/load features from this .pt file so the
                        extraction only ever runs once across sessions

        Returns:
            features: (N, vit_hidden) float32 on CPU
            targets:  (N, 2)          float32 on CPU
        """
        from pathlib import Path

        if cache_path is not None and Path(cache_path).exists():
            saved = torch.load(cache_path, weights_only=True)
            print(f"  Loaded backbone features from {cache_path}", flush=True)
            return saved["features"], saved["targets"]

        self.vit.to(device).eval()
        all_feat, all_y = [], []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1]
            else:
                x, y = batch["camera"].to(device), batch["target"]
            feat = self.vit(x).cpu()
            all_feat.append(feat)
            all_y.append(y)

        features = torch.cat(all_feat)
        targets = torch.cat(all_y)

        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "targets": targets}, cache_path)
            print(f"  Saved backbone features to {cache_path}", flush=True)

        return features, targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224) — ImageNet-normalized RGB image

        Returns:
            (batch, embed_dim)
        """
        # ViT backbone → CLS token features
        features = self.vit(x)  # (batch, vit_hidden)

        # Project to shared embedding space
        return self.head(features)  # (batch, embed_dim)

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "camera",
            "shape": (3, 224, 224),
            "dtype": "float32",
        }
