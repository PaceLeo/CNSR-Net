"""Direct CLIP-distance prediction from whole-frame student features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .pyramid import PyramidScale
from .spatial_matcher import PatchEncoder, SpatialHierarchicalMatcher, SpatialNoveltyMap
from .whole_frame_student import (
    FrozenResNet50ROIBackbone,
    WholeFrameROIHead,
    WholeFrameStudentSpatialHierarchicalMatcher,
)


class WholeFrameDistanceHead(torch.nn.Module):
    """Correct student pair distances toward exact CLIP pair distances."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        scale_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.scale_names = tuple(str(name) for name in scale_names)
        self.heads = torch.nn.ModuleDict()
        for name in self.scale_names:
            head = torch.nn.Sequential(
                torch.nn.LayerNorm(2 * self.feature_dim),
                torch.nn.Linear(2 * self.feature_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_dim, 1),
            )
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
            self.heads[name] = head

    def forward(
        self,
        scale_name: str,
        current: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        delta = current.float() - reference.float()
        base_distance = torch.linalg.vector_norm(delta, dim=-1)
        pair_features = torch.cat(
            (delta.abs(), current.float() * reference.float()), dim=-1
        )
        logit = self.heads[scale_name](pair_features).squeeze(-1)
        correction = functional.softplus(logit) / math.log(2.0)
        return base_distance * correction


def load_whole_frame_distance_head(
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> WholeFrameDistanceHead:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
    )
    head = WholeFrameDistanceHead(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        scale_names=tuple(checkpoint["scale_names"]),
    )
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    return head.to(device).eval().requires_grad_(False)


class WholeFrameDistanceSpatialHierarchicalMatcher(
    WholeFrameStudentSpatialHierarchicalMatcher
):
    """Use learned pair distances for student scales and exact coarse CLIP."""

    def __init__(
        self,
        encoder: PatchEncoder,
        pyramid_scales: Sequence[PyramidScale],
        backbone: FrozenResNet50ROIBackbone,
        student_head: WholeFrameROIHead,
        distance_head: WholeFrameDistanceHead,
        num_prototypes: int = 8,
        patch_batch_size: int = 36,
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        normalization_epsilon: float = 1e-6,
        source_scale_name: str = "coarse",
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(
            encoder=encoder,
            pyramid_scales=pyramid_scales,
            backbone=backbone,
            student_head=student_head,
            num_prototypes=num_prototypes,
            patch_batch_size=patch_batch_size,
            device=device,
            use_amp=use_amp,
            normalization_epsilon=normalization_epsilon,
            source_scale_name=source_scale_name,
            amp_dtype=amp_dtype,
        )
        self.distance_head = distance_head

    def _direct_distance_map(
        self,
        scale_name: str,
        features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            distances = self.distance_head(
                scale_name,
                features.unsqueeze(-2),
                prototypes.to(self.device),
            )
        return distances.amin(dim=-1)

    def compute_spatial_novelty(
        self,
        camera_id: str,
        frame: np.ndarray | torch.Tensor | Image.Image,
        timestamp: float,
        normalize: bool = True,
    ) -> SpatialNoveltyMap:
        bank = self.camera_banks[camera_id]
        features = self.encode_frame(frame)
        raw_distances = {}
        novelties = {}
        for scale in self.pyramid_scales:
            prototypes = bank.scales[scale.name].prototypes.to(self.device)
            if scale.name == self.source_scale.name:
                raw = SpatialHierarchicalMatcher._raw_distance_map(
                    features[scale.name], prototypes
                )
            else:
                raw = self._direct_distance_map(
                    scale.name, features[scale.name], prototypes
                )
            raw_distances[scale.name] = raw
            novelties[scale.name] = (
                self.normalize_distances(camera_id, {scale.name: raw})[
                    scale.name
                ]
                if normalize
                else raw
            )
        return SpatialNoveltyMap(
            timestamp=timestamp,
            scale_novelties=novelties,
            raw_distances=raw_distances,
            raw_features=features,
        )

    def fit_camera_normalization(
        self,
        camera_id: str,
        threshold_frames: Sequence[np.ndarray | torch.Tensor | Image.Image],
    ) -> None:
        distance_maps = [
            self.compute_spatial_novelty(
                camera_id, frame, timestamp=index, normalize=False
            ).raw_distances
            for index, frame in enumerate(threshold_frames)
        ]
        self.fit_camera_normalization_from_distances(camera_id, distance_maps)
