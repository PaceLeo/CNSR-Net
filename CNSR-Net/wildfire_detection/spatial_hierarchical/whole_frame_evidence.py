"""Direct semantic-evidence prediction from whole-frame student features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as functional

from .pyramid import PyramidScale
from .spatial_matcher import PatchEncoder, SpatialNoveltyMap
from .whole_frame_student import (
    FrozenResNet50ROIBackbone,
    WholeFrameROIHead,
    WholeFrameStudentSpatialHierarchicalMatcher,
)


class WholeFrameSemanticEvidenceHead(torch.nn.Module):
    """Predict final semantic evidence from a student residual vector."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        scale_names: Sequence[str],
        initial_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.scale_names = tuple(str(name) for name in scale_names)
        self.initial_factor = float(initial_factor)
        self.heads = torch.nn.ModuleDict()
        input_dim = 2 * self.feature_dim + 1
        initial_bias = math.log(math.expm1(self.initial_factor))
        for name in self.scale_names:
            head = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.hidden_dim, 1),
            )
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.constant_(head[-1].bias, initial_bias)
            self.heads[name] = head

    def forward(
        self,
        scale_name: str,
        residual: torch.Tensor,
        raw_distance: torch.Tensor,
    ) -> torch.Tensor:
        residual = residual.float()
        residual_unit = functional.normalize(residual, dim=-1, eps=1e-8)
        inputs = torch.cat(
            (
                residual_unit,
                residual_unit.abs(),
                torch.log1p(raw_distance.float()).unsqueeze(-1),
            ),
            dim=-1,
        )
        factor = functional.softplus(
            self.heads[scale_name](inputs).squeeze(-1)
        )
        return raw_distance.float() * factor


def load_whole_frame_semantic_evidence_head(
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> WholeFrameSemanticEvidenceHead:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
    )
    head = WholeFrameSemanticEvidenceHead(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        scale_names=tuple(checkpoint["scale_names"]),
        initial_factor=float(checkpoint["initial_factor"]),
    )
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    return head.to(device).eval().requires_grad_(False)


class WholeFrameSemanticEvidenceSpatialHierarchicalMatcher(
    WholeFrameStudentSpatialHierarchicalMatcher
):
    """Use exact coarse CLIP and distilled evidence on dense scales."""

    def __init__(
        self,
        encoder: PatchEncoder,
        pyramid_scales: Sequence[PyramidScale],
        backbone: FrozenResNet50ROIBackbone,
        student_head: WholeFrameROIHead,
        evidence_head: WholeFrameSemanticEvidenceHead,
        evidence_scale_names: Sequence[str] | None = None,
        evidence_blend_weight: float = 1.0,
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
        self.evidence_head = evidence_head
        self.evidence_scale_names = frozenset(
            evidence_scale_names
            if evidence_scale_names is not None
            else (scale.name for scale in self.target_scales)
        )
        self.evidence_blend_weight = float(evidence_blend_weight)

    def semantic_evidence_overrides(
        self,
        camera_id: str,
        spatial_map: SpatialNoveltyMap,
    ) -> dict[str, tuple[torch.Tensor, float]]:
        prototype_means = self.prototype_means(camera_id)
        evidence = {}
        with torch.inference_mode():
            for scale in self.target_scales:
                name = scale.name
                if name not in self.evidence_scale_names:
                    continue
                residual = (
                    spatial_map.raw_features[name] - prototype_means[name]
                )
                evidence[name] = (
                    self.evidence_head(
                        name,
                        residual,
                        spatial_map.raw_distances[name],
                    ),
                    self.evidence_blend_weight,
                )
        return evidence
