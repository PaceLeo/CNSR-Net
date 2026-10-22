"""Vectorized normal-relative semantic residual scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from wildfire_detection.spatial_hierarchical import (
    SpatialHierarchicalMatcher,
    SpatialNoveltyMap,
)

from .config import SemanticWeights
from .semantic_prototypes import SemanticPrototypes


@dataclass(frozen=True)
class SemanticResidualMap:
    timestamp: float
    spatial_novelty: dict[str, torch.Tensor]
    raw_distances: dict[str, torch.Tensor]
    semantic_alignment: dict[str, torch.Tensor]
    semantic_components: dict[str, dict[str, torch.Tensor]]
    semantic_evidence: dict[str, torch.Tensor]
    residual_vectors: dict[str, torch.Tensor] | None = None


class SemanticResidualComputer:
    """Project local visual changes onto wildfire semantic directions."""

    def __init__(
        self,
        spatial_matcher: SpatialHierarchicalMatcher,
        semantic_prototypes: SemanticPrototypes,
        mode: str,
        weights: SemanticWeights,
        use_contrast: bool = True,
        residual_epsilon: float = 1e-8,
        residual_reference: str = "mean",
    ):
        self.spatial_matcher = spatial_matcher
        self.semantic_prototypes = semantic_prototypes
        self.mode = mode
        self.weights = weights
        self.use_contrast = use_contrast
        self.residual_epsilon = residual_epsilon
        if residual_reference not in ("mean", "nearest"):
            raise ValueError(f"Unknown residual reference: {residual_reference}")
        self.residual_reference = residual_reference

    @staticmethod
    def _positive_projection(
        residual_unit: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("...d,d->...", residual_unit, direction).relu()

    def _single_alignment(
        self,
        residual_unit: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = (
            self.semantic_prototypes.contrast_embedding
            if self.use_contrast
            else self.semantic_prototypes.fire_embedding
        ).to(residual_unit.device)
        fire = self._positive_projection(residual_unit, target)
        return fire, {"fire": fire}

    def _multidim_alignment(
        self,
        residual_unit: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prototypes = self.semantic_prototypes
        smoke = self._positive_projection(
            residual_unit,
            prototypes.smoke_direction.to(residual_unit.device),
        )
        flame = self._positive_projection(
            residual_unit,
            prototypes.flame_direction.to(residual_unit.device),
        )
        cloud = self._positive_projection(
            residual_unit,
            prototypes.cloud_direction.to(residual_unit.device),
        )
        combined = (
            self.weights.smoke * smoke
            + self.weights.flame * flame
            - self.weights.cloud * cloud
        ).clamp(0.0, 1.0)
        return combined, {"smoke": smoke, "flame": flame, "cloud": cloud}

    def compute_from_spatial(
        self,
        camera_id: str,
        spatial_map: SpatialNoveltyMap,
        return_debug_info: bool = False,
    ) -> SemanticResidualMap:
        """Add semantic direction constraints to an existing spatial map."""
        references = (
            self.spatial_matcher.nearest_prototypes(
                camera_id, spatial_map.raw_features
            )
            if self.residual_reference == "nearest"
            else self.spatial_matcher.prototype_means(camera_id)
        )
        alignments: dict[str, torch.Tensor] = {}
        components: dict[str, dict[str, torch.Tensor]] = {}
        evidence: dict[str, torch.Tensor] = {}
        residual_vectors: dict[str, torch.Tensor] = {}

        for scale_name, features in spatial_map.raw_features.items():
            residual = features - references[scale_name].to(features.device)
            residual_unit = residual / torch.linalg.vector_norm(
                residual,
                dim=-1,
                keepdim=True,
            ).clamp_min(self.residual_epsilon)

            if self.mode == "multidim":
                alignment, scale_components = self._multidim_alignment(
                    residual_unit
                )
            else:
                alignment, scale_components = self._single_alignment(
                    residual_unit
                )

            alignments[scale_name] = alignment
            components[scale_name] = scale_components
            evidence[scale_name] = (
                spatial_map.raw_distances[scale_name] * alignment
            )
            if return_debug_info:
                residual_vectors[scale_name] = residual

        for scale_name, (distilled, blend_weight) in (
            self.spatial_matcher.semantic_evidence_overrides(
                camera_id, spatial_map
            ).items()
        ):
            raw_distance = spatial_map.raw_distances[scale_name]
            distilled = torch.lerp(
                evidence[scale_name], distilled, blend_weight
            )
            distilled_alignment = torch.where(
                raw_distance > self.residual_epsilon,
                distilled / raw_distance.clamp_min(self.residual_epsilon),
                torch.zeros_like(distilled),
            )
            evidence[scale_name] = distilled
            alignments[scale_name] = distilled_alignment
            components[scale_name] = {"distilled": distilled_alignment}

        return SemanticResidualMap(
            timestamp=spatial_map.timestamp,
            spatial_novelty=spatial_map.scale_novelties,
            raw_distances=spatial_map.raw_distances,
            semantic_alignment=alignments,
            semantic_components=components,
            semantic_evidence=evidence,
            residual_vectors=residual_vectors if return_debug_info else None,
        )

    def compute_semantic_evidence(
        self,
        camera_id: str,
        frame: np.ndarray | torch.Tensor | Image.Image,
        timestamp: float,
        return_debug_info: bool = False,
    ) -> SemanticResidualMap:
        spatial_map = self.spatial_matcher.compute_spatial_novelty(
            camera_id,
            frame,
            timestamp,
            normalize=True,
        )
        return self.compute_from_spatial(
            camera_id,
            spatial_map,
            return_debug_info=return_debug_info,
        )
