"""Calibrated temporal persistence and spatial-neighborhood gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional

from .config import TemporalConfig
from .temporal_buffer import TemporalEvidenceBuffer, TemporalEvidenceFrame


@dataclass(frozen=True)
class TemporalFilteredMap:
    timestamp: float
    semantic_evidence: dict[str, torch.Tensor]
    temporal_persistence: dict[str, torch.Tensor]
    persistence_scores: dict[str, torch.Tensor]
    filtered_evidence: dict[str, torch.Tensor]
    gating_mask: dict[str, torch.Tensor]


class TemporalPersistenceFilter:
    """Suppress evidence that does not persist for the configured duration."""

    def __init__(self, config: TemporalConfig):
        self.config = config
        self.buffer = TemporalEvidenceBuffer(config.window_seconds)
        self.camera_low_thresholds: dict[
            str, dict[str, torch.Tensor]
        ] = {}

    def fit_low_thresholds(
        self,
        camera_id: str,
        threshold_fit_evidence: Sequence[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Fit position-local thresholds on camera normal evidence."""
        scale_names = threshold_fit_evidence[0].keys()
        thresholds = {}
        for scale_name in scale_names:
            samples = torch.stack(
                [
                    evidence[scale_name].detach().float().cpu()
                    for evidence in threshold_fit_evidence
                ]
            )
            position_threshold = torch.quantile(
                samples,
                self.config.low_threshold_quantile,
                dim=0,
            )
            scale_floor = torch.quantile(
                samples.flatten(),
                self.config.low_threshold_quantile,
            )
            thresholds[scale_name] = position_threshold.clamp_min(scale_floor)
        self.camera_low_thresholds[camera_id] = thresholds
        return thresholds

    def reset(self, camera_id: str) -> None:
        self.buffer.reset(camera_id)

    def _temporal_persistence(
        self,
        history: tuple[TemporalEvidenceFrame, ...],
        scale_name: str,
        low_threshold: torch.Tensor,
    ) -> torch.Tensor:
        device = history[-1].evidence[scale_name].device
        threshold = low_threshold.to(device)
        active_observations = torch.stack(
            [
                (frame.evidence[scale_name] > threshold).float()
                for frame in history
            ]
        ).sum(dim=0)
        contribution = (
            self.config.nominal_sampling_seconds / self.config.window_seconds
        )
        return (active_observations * contribution).clamp_max(1.0)

    @staticmethod
    def _spatial_persistence(persistence: torch.Tensor) -> torch.Tensor:
        return functional.max_pool2d(
            persistence[None, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0]

    def filter_evidence(
        self,
        camera_id: str,
        timestamp: float,
        semantic_evidence: dict[str, torch.Tensor],
    ) -> TemporalFilteredMap:
        """Update one camera and gate its current semantic evidence."""
        low_thresholds = self.camera_low_thresholds[camera_id]
        history = self.buffer.update(camera_id, timestamp, semantic_evidence)
        temporal_scores: dict[str, torch.Tensor] = {}
        effective_scores: dict[str, torch.Tensor] = {}
        filtered_evidence: dict[str, torch.Tensor] = {}
        gating_masks: dict[str, torch.Tensor] = {}

        for scale_name, evidence in semantic_evidence.items():
            temporal = self._temporal_persistence(
                history,
                scale_name,
                low_thresholds[scale_name],
            )
            effective = (
                self._spatial_persistence(temporal)
                if self.config.use_spatial_neighborhood
                else temporal
            )
            gate = (
                effective
                > self.config.persistence_thresholds[scale_name]
            )
            temporal_scores[scale_name] = temporal
            effective_scores[scale_name] = effective
            gating_masks[scale_name] = gate
            filtered_evidence[scale_name] = evidence * gate

        return TemporalFilteredMap(
            timestamp=float(timestamp),
            semantic_evidence=semantic_evidence,
            temporal_persistence=temporal_scores,
            persistence_scores=effective_scores,
            filtered_evidence=filtered_evidence,
            gating_mask=gating_masks,
        )
