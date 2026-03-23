"""Pipeline composer — wires Stages A through E into a full model."""

import torch
import torch.nn as nn


class FusionPipeline(nn.Module):
    """Composes modality encoders, temporal alignment, fusion, filter, and uncertainty.

    Each stage is optional and controlled by config. This allows testing
    individual stages in isolation before composing the full system.
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module] | None = None,
        temporal_aligner: nn.Module | None = None,
        fusion_module: nn.Module | None = None,
        state_filter: nn.Module | None = None,
        uncertainty_module: nn.Module | None = None,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders or {})
        self.temporal_aligner = temporal_aligner
        self.fusion_module = fusion_module
        self.state_filter = state_filter
        self.uncertainty_module = uncertainty_module

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        timestamps: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the pipeline on a batch of multi-modal inputs.

        Args:
            inputs: dict mapping modality name -> raw sensor tensor
            timestamps: (batch, n_tokens) continuous timestamps
            source_ids: (batch, n_tokens) modality identifiers

        Returns:
            dict with 'position' and optionally 'uncertainty' keys
        """
        # Stage A: Encode each modality
        tokens = []
        for name, encoder in self.encoders.items():
            if name in inputs:
                tokens.append(encoder(inputs[name]))
        tokens = torch.cat(tokens, dim=1)  # (batch, total_tokens, embed_dim)

        # Stage B: Temporal alignment
        if self.temporal_aligner is not None and timestamps is not None:
            tokens = self.temporal_aligner(tokens, timestamps, source_ids)

        # Stage C: Cross-modal fusion
        if self.fusion_module is not None:
            fused = self.fusion_module(tokens, source_ids)
        else:
            fused = tokens

        # Stage D: State estimation
        if self.state_filter is not None:
            positions = self.state_filter(fused)
        else:
            positions = fused

        result = {"position": positions}

        # Stage E: Uncertainty
        if self.uncertainty_module is not None:
            positions, lower, upper = self.uncertainty_module(positions)
            result["uncertainty"] = {"lower": lower, "upper": upper}

        return result
