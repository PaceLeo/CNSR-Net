"""Residual adapters for distilling exact crop CLS features from shared ROIs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as functional


class MultiScaleROIAdapter(torch.nn.Module):
    """Apply one lightweight residual projection per approximated scale."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int,
        scale_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.scale_names = tuple(str(name) for name in scale_names)
        self.adapters = torch.nn.ModuleDict()
        for name in self.scale_names:
            projection = torch.nn.Sequential(
                torch.nn.LayerNorm(self.feature_dim),
                torch.nn.Linear(self.feature_dim, self.bottleneck_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.bottleneck_dim, self.feature_dim),
            )
            torch.nn.init.zeros_(projection[-1].weight)
            torch.nn.init.zeros_(projection[-1].bias)
            self.adapters[name] = projection

    def forward(self, scale_name: str, features: torch.Tensor) -> torch.Tensor:
        if scale_name not in self.adapters:
            return features
        residual = self.adapters[scale_name](features.float())
        return functional.normalize(features.float() + residual, dim=-1)


def load_roi_adapter(
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> MultiScaleROIAdapter:
    """Restore a trained ROI adapter and freeze it for feature extraction."""
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
    )
    adapter = MultiScaleROIAdapter(
        feature_dim=int(checkpoint["feature_dim"]),
        bottleneck_dim=int(checkpoint["bottleneck_dim"]),
        scale_names=tuple(checkpoint["scale_names"]),
    )
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.to(device).eval().requires_grad_(False)
    return adapter
