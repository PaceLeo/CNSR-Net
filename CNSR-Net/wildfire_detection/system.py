"""Default A3 Camera-conditioned Normal-relative Semantic Residual Network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from wildfire_detection.semantic_residual import (
    SemanticPrototypeBuilder,
    SemanticResidualComputer,
    SemanticResidualMap,
    load_semantic_config,
)
from wildfire_detection.spatial_hierarchical import (
    SpatialHierarchicalMatcher,
    SpatialNoveltyMap,
    build_spatial_matcher,
    load_frozen_patch_encoder,
)
from wildfire_detection.temporal_filtering import (
    TemporalFilteredMap,
    TemporalPersistenceFilter,
    load_temporal_config,
)
from wildfire_detection.variants import (
    MAIN_METHOD,
    MAIN_VARIANT,
    calibration_payload,
)


@dataclass(frozen=True)
class DetectionResult:
    timestamp: float
    camera_id: str
    spatial_novelty_map: SpatialNoveltyMap
    semantic_residual_map: SemanticResidualMap
    temporal_filtered_map: TemporalFilteredMap
    fire_score: float
    is_fire: bool
    confidence: float
    model_variant: str
    evidence_heatmap: dict[str, torch.Tensor]
    attention_heatmap: dict[str, torch.Tensor]
    top_evidence_locations: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class _AggregationResult:
    score: float
    attention_heatmap: dict[str, torch.Tensor]


def _cpu_maps(maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: values.detach().float().cpu()
        for name, values in maps.items()
    }


class _BaseCNSRNet:
    """Shared spatial, semantic, temporal, and calibration pipeline."""

    def __init__(
        self,
        config_path: str | Path,
        artifact_path: str | Path,
        calibration_method: str,
        model_variant: str,
        device: str | torch.device = "cuda",
    ):
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.device = torch.device(device)
        self.model_variant = model_variant

        self.encoder = load_frozen_patch_encoder(
            self.config["models"]["clip_checkpoint"],
            device=self.device,
        )
        self.spatial_matcher = build_spatial_matcher(
            encoder=self.encoder,
            config_path=self.config["components"]["spatial"],
            device=self.device,
        )
        semantic_config = load_semantic_config(
            self.config["components"]["semantic"]
        )
        semantic_prototypes = SemanticPrototypeBuilder(self.encoder).build(
            semantic_config.prompts
        )
        self.semantic_computer = SemanticResidualComputer(
            spatial_matcher=self.spatial_matcher,
            semantic_prototypes=semantic_prototypes,
            mode=semantic_config.mode,
            weights=semantic_config.weights,
            use_contrast=semantic_config.use_contrast,
            residual_reference=semantic_config.residual_reference,
        )
        self.temporal_config = load_temporal_config(
            self.config["components"]["temporal"]
        )
        self.temporal_filter = TemporalPersistenceFilter(self.temporal_config)

        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(artifact, Mapping):
            raise ValueError("Model artifact must be a mapping")
        self._artifact = artifact
        self.global_margin = float(
            calibration_payload(artifact, calibration_method)["global_margin"]
        )
        self.camera_thresholds: dict[str, float] = {}

    def _aggregate(
        self,
        evidence: dict[str, torch.Tensor],
        persistence: dict[str, torch.Tensor],
    ) -> _AggregationResult:
        raise NotImplementedError

    def _semantic(
        self,
        camera_id: str,
        frame: Any,
        timestamp: float,
        normalize: bool,
    ) -> tuple[SpatialNoveltyMap, SemanticResidualMap]:
        spatial = self.spatial_matcher.compute_spatial_novelty(
            camera_id,
            frame,
            timestamp,
            normalize=normalize,
        )
        semantic = self.semantic_computer.compute_from_spatial(
            camera_id,
            spatial,
        )
        return spatial, semantic

    def _normal_history(
        self,
        camera_id: str,
        frames: Sequence[Any],
        timestamps: Sequence[float],
    ) -> list[dict[str, Any]]:
        observations = []
        for frame, timestamp in zip(frames, timestamps):
            spatial, semantic = self._semantic(
                camera_id, frame, float(timestamp), normalize=False
            )
            observations.append({
                "timestamp": float(timestamp),
                "raw_distances": _cpu_maps(spatial.raw_distances),
                "semantic_evidence": _cpu_maps(semantic.semantic_evidence),
            })
        return observations

    def calibrate_camera(
        self,
        camera_id: str,
        prototype_fit_frames: Sequence[Any],
        threshold_fit_frames: Sequence[Any],
        temporal_warmup_frames: Sequence[Any] = (),
        threshold_timestamps: Sequence[float] | None = None,
        warmup_timestamps: Sequence[float] | None = None,
        *,
        activation_fit_frames: Sequence[Any] | None = None,
        activation_timestamps: Sequence[float] | None = None,
    ) -> float:
        """Fit normality and thresholds, with optional earlier activation history."""
        separate_activation = (
            self._artifact.get("normal_history_protocol", {}).get(
                "activation_fit", "shared"
            ) == "separate"
        )
        if separate_activation != (activation_fit_frames is not None):
            raise ValueError(
                "activation_fit_frames must match the shared/separate normal "
                "history used to fit this calibration artifact"
            )
        nominal = self.temporal_config.nominal_sampling_seconds
        if threshold_timestamps is None:
            threshold_timestamps = tuple(
                index * nominal for index in range(len(threshold_fit_frames))
            )
        if activation_fit_frames is not None and activation_timestamps is None:
            activation_timestamps = tuple(
                float(threshold_timestamps[0])
                + (index - len(activation_fit_frames)) * nominal
                for index in range(len(activation_fit_frames))
            )
        if warmup_timestamps is None:
            warmup_origin = (
                float(activation_timestamps[0])
                if activation_fit_frames is not None
                else float(threshold_timestamps[0])
            )
            warmup_timestamps = tuple(
                warmup_origin + (index - len(temporal_warmup_frames)) * nominal
                for index in range(len(temporal_warmup_frames))
            )

        self.spatial_matcher.fit_camera_prototypes(
            camera_id,
            prototype_fit_frames,
        )
        threshold_intermediate = self._normal_history(
            camera_id, threshold_fit_frames, threshold_timestamps
        )
        activation_intermediate = (
            self._normal_history(camera_id, activation_fit_frames, activation_timestamps)
            if activation_fit_frames is not None else []
        )

        self.spatial_matcher.fit_camera_normalization_from_distances(
            camera_id,
            [
                item["raw_distances"]
                for item in activation_intermediate + threshold_intermediate
            ],
        )
        self.temporal_filter.fit_low_thresholds(
            camera_id,
            [
                item["semantic_evidence"]
                for item in (activation_intermediate or threshold_intermediate)
            ],
        )
        self.temporal_filter.reset(camera_id)

        for frame, timestamp in zip(
            temporal_warmup_frames,
            warmup_timestamps,
        ):
            _, semantic = self._semantic(
                camera_id,
                frame,
                float(timestamp),
                normalize=True,
            )
            self.temporal_filter.filter_evidence(
                camera_id,
                float(timestamp),
                _cpu_maps(semantic.semantic_evidence),
            )

        for item in activation_intermediate:
            self.temporal_filter.filter_evidence(
                camera_id, item["timestamp"], item["semantic_evidence"]
            )

        threshold_scores = []
        for item in threshold_intermediate:
            temporal = self.temporal_filter.filter_evidence(
                camera_id,
                item["timestamp"],
                item["semantic_evidence"],
            )
            aggregation = self._aggregate(
                temporal.filtered_evidence,
                temporal.persistence_scores,
            )
            threshold_scores.append(aggregation.score)
        threshold = max(threshold_scores) + self.global_margin
        self.camera_thresholds[camera_id] = threshold
        return threshold

    @staticmethod
    def _top_locations(
        evidence_map: dict[str, torch.Tensor],
        top_k: int = 5,
    ) -> tuple[tuple[str, int, int], ...]:
        locations = []
        for scale_name, values in evidence_map.items():
            for flat_index, value in enumerate(values.flatten()):
                row, column = divmod(flat_index, values.shape[1])
                locations.append(
                    (float(value), scale_name, int(row), int(column))
                )
        locations.sort(reverse=True)
        return tuple(
            (scale_name, row, column)
            for _, scale_name, row, column in locations[:top_k]
        )

    def process_frame(
        self,
        camera_id: str,
        frame: Any,
        timestamp: float,
    ) -> DetectionResult:
        """Process one chronological frame for a calibrated camera."""
        spatial, semantic = self._semantic(
            camera_id,
            frame,
            timestamp,
            normalize=True,
        )
        temporal = self.temporal_filter.filter_evidence(
            camera_id,
            timestamp,
            _cpu_maps(semantic.semantic_evidence),
        )
        aggregation = self._aggregate(
            temporal.filtered_evidence,
            temporal.persistence_scores,
        )
        score = aggregation.score
        threshold = self.camera_thresholds[camera_id]
        average_persistence = float(
            torch.cat(
                [
                    values.float().flatten()
                    for values in temporal.persistence_scores.values()
                ]
            ).mean()
        )
        confidence = score * (0.5 + 0.5 * average_persistence)
        evidence_heatmap = _cpu_maps(temporal.filtered_evidence)
        location_map = aggregation.attention_heatmap or evidence_heatmap
        return DetectionResult(
            timestamp=float(timestamp),
            camera_id=camera_id,
            spatial_novelty_map=spatial,
            semantic_residual_map=semantic,
            temporal_filtered_map=temporal,
            fire_score=score,
            is_fire=score > threshold,
            confidence=confidence,
            model_variant=self.model_variant,
            evidence_heatmap=evidence_heatmap,
            attention_heatmap=aggregation.attention_heatmap,
            top_evidence_locations=self._top_locations(location_map),
        )


class CNSRNet(_BaseCNSRNet):
    """Run the default A3 model with maximum temporal-evidence aggregation."""

    def __init__(
        self,
        config_path: str | Path,
        calibration_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        *,
        trained_model_path: str | Path | None = None,
    ):
        if calibration_path is None:
            calibration_path = trained_model_path
        elif trained_model_path is not None:
            raise ValueError(
                "Pass calibration_path or trained_model_path, not both"
            )
        if calibration_path is None:
            raise ValueError("calibration_path is required")
        super().__init__(
            config_path=config_path,
            artifact_path=calibration_path,
            calibration_method=MAIN_METHOD,
            model_variant=MAIN_VARIANT,
            device=device,
        )

    def _aggregate(
        self,
        evidence: dict[str, torch.Tensor],
        persistence: dict[str, torch.Tensor],
    ) -> _AggregationResult:
        del persistence
        values = torch.cat(
            [scale_values.float().flatten() for scale_values in evidence.values()]
        )
        return _AggregationResult(
            score=float(values.max()),
            attention_heatmap={},
        )
