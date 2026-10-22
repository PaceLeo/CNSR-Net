"""Learned multi-scale spatial attention aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional

from wildfire_detection.spatial_hierarchical import PyramidScale

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    weighted_evidence: torch.Tensor
    attention_weights: dict[str, torch.Tensor]
    flat_attention_weights: torch.Tensor


class SpatialAttentionAggregator(nn.Module):
    """Aggregate all pyramid locations with globally normalized attention."""

    def __init__(
        self,
        pyramid_scales: Sequence[PyramidScale],
        config: AttentionConfig,
    ):
        super().__init__()
        self.pyramid_scales = tuple(pyramid_scales)
        input_dim = 4 + len(self.pyramid_scales)
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.raw_score_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(config.score_scale_init)))
        )
        self.score_bias = nn.Parameter(torch.tensor(config.score_bias_init))

    @property
    def score_scale(self) -> torch.Tensor:
        return functional.softplus(self.raw_score_scale)

    @staticmethod
    def _position_features(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        columns = torch.linspace(
            0.0, 1.0, width, device=device, dtype=dtype
        )
        row_grid, column_grid = torch.meshgrid(
            rows,
            columns,
            indexing="ij",
        )
        return torch.stack((row_grid, column_grid), dim=-1)

    def forward(
        self,
        evidence_maps: dict[str, torch.Tensor],
        persistence_maps: dict[str, torch.Tensor],
    ) -> AttentionOutput:
        features = []
        evidence_values = []
        scale_lengths: list[tuple[str, int, int]] = []

        for scale_index, scale in enumerate(self.pyramid_scales):
            evidence = evidence_maps[scale.name]
            persistence = persistence_maps[scale.name]
            batch_size, height, width = evidence.shape
            positions = self._position_features(
                height,
                width,
                evidence.device,
                evidence.dtype,
            ).expand(batch_size, -1, -1, -1)
            scale_one_hot = torch.zeros(
                (batch_size, height, width, len(self.pyramid_scales)),
                device=evidence.device,
                dtype=evidence.dtype,
            )
            scale_one_hot[..., scale_index] = 1.0
            features.append(
                torch.cat(
                    (
                        evidence.unsqueeze(-1),
                        persistence.unsqueeze(-1),
                        positions,
                        scale_one_hot,
                    ),
                    dim=-1,
                ).reshape(batch_size, height * width, -1)
            )
            evidence_values.append(evidence.reshape(batch_size, -1))
            scale_lengths.append((scale.name, height, width))

        flat_features = torch.cat(features, dim=1)
        flat_evidence = torch.cat(evidence_values, dim=1)
        attention_logits = self.attention_mlp(flat_features).squeeze(-1)
        flat_weights = torch.softmax(attention_logits, dim=1)
        weighted_evidence = (flat_weights * flat_evidence).sum(dim=1)
        logits = self.score_scale * weighted_evidence + self.score_bias

        attention_weights = {}
        start = 0
        for scale_name, height, width in scale_lengths:
            end = start + height * width
            attention_weights[scale_name] = flat_weights[:, start:end].reshape(
                flat_weights.shape[0], height, width
            )
            start = end

        return AttentionOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            weighted_evidence=weighted_evidence,
            attention_weights=attention_weights,
            flat_attention_weights=flat_weights,
        )

    @staticmethod
    def attention_entropy(flat_weights: torch.Tensor) -> torch.Tensor:
        """Return positive entropy, whose minimization favors sparsity."""
        return -(
            flat_weights * flat_weights.clamp_min(1e-12).log()
        ).sum(dim=1).mean()
