"""Multi-scale matching by reusing a prefix of the CLIP visual transformer."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.ops import roi_align

from .pyramid import PyramidScale
from .roi_adapter import MultiScaleROIAdapter
from .spatial_matcher import (
    PatchEncoder,
    SpatialHierarchicalMatcher,
    SpatialNoveltyMap,
)


class SplitTokenEncoder(PatchEncoder, Protocol):
    hidden_dim: int
    token_grid_size: int
    transformer_layers: int

    def encode_patches_to_layer(
        self,
        patches: torch.Tensor,
        split_layer: int,
    ) -> torch.Tensor:
        ...

    def encode_tokens_from_layer(
        self,
        tokens: torch.Tensor,
        split_layer: int,
    ) -> torch.Tensor:
        ...


class LateBlockSpatialHierarchicalMatcher(SpatialHierarchicalMatcher):
    """Stitch coarse intermediate tokens and rerun CLIP suffix blocks per ROI."""

    def __init__(
        self,
        encoder: SplitTokenEncoder,
        pyramid_scales: Sequence[PyramidScale],
        num_prototypes: int = 8,
        patch_batch_size: int = 36,
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        normalization_epsilon: float = 1e-6,
        source_scale_name: str = "coarse",
        split_layer: int = 18,
        late_batch_size: int = 32,
        source_cls_weight: float = 0.5,
        amp_dtype: torch.dtype = torch.float16,
        roi_adapter: MultiScaleROIAdapter | None = None,
        exact_refine_topk: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            pyramid_scales=pyramid_scales,
            num_prototypes=num_prototypes,
            patch_batch_size=patch_batch_size,
            device=device,
            use_amp=use_amp,
            normalization_epsilon=normalization_epsilon,
        )
        self.source_scale = next(
            scale
            for scale in self.pyramid_scales
            if scale.name == source_scale_name
        )
        if not 0 <= split_layer < encoder.transformer_layers:
            raise ValueError(
                "split_layer must leave at least one CLIP block to rerun"
            )
        self.split_layer = int(split_layer)
        self.late_batch_size = int(late_batch_size)
        self.source_cls_weight = float(source_cls_weight)
        self.amp_dtype = amp_dtype
        self.roi_adapter = roi_adapter
        self.exact_refine_topk = {
            str(name): int(count)
            for name, count in (exact_refine_topk or {}).items()
            if int(count) > 0
        }
        self._geometry_cache: dict[
            tuple[int, int, str, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _encode_prototype_frame(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
    ) -> dict[str, torch.Tensor]:
        if self.exact_refine_topk:
            return SpatialHierarchicalMatcher.encode_frame(self, frame)
        return self.encode_frame(frame)

    @staticmethod
    def _intersection_area(first: torch.Tensor, second: torch.Tensor) -> float:
        start = torch.maximum(first[:2], second[:2])
        end = torch.minimum(first[2:], second[2:])
        size = (end - start).clamp_min(0)
        return float(size.prod())

    def _source_core_rois(
        self,
        height: int,
        width: int,
        source_boxes: torch.Tensor,
        token_grid_size: int,
    ) -> torch.Tensor:
        _, core_boxes = self._patch_bounds(
            height,
            width,
            PyramidScale(
                name=self.source_scale.name,
                grid_size=self.source_scale.grid_size,
                overlap=0.0,
            ),
        )
        source_boxes = source_boxes.detach().cpu()
        source_sizes = source_boxes[:, 2:] - source_boxes[:, :2]
        starts = (core_boxes[:, :2] - source_boxes[:, :2]) / source_sizes
        ends = (core_boxes[:, 2:] - source_boxes[:, :2]) / source_sizes
        local_boxes = torch.cat((starts, ends), dim=-1) * token_grid_size
        indices = torch.arange(len(local_boxes), dtype=local_boxes.dtype)
        return torch.cat((indices.unsqueeze(-1), local_boxes), dim=-1).to(
            self.device
        )

    def _target_geometry(
        self,
        target_scale: PyramidScale,
        height: int,
        width: int,
        canvas_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, target_scale.name, canvas_size)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        _, target_boxes = self._patch_bounds(height, width, target_scale)
        _, source_cores = self._patch_bounds(
            height,
            width,
            PyramidScale(
                name=self.source_scale.name,
                grid_size=self.source_scale.grid_size,
                overlap=0.0,
            ),
        )
        canvas_height = self.source_scale.grid_size[0] * canvas_size
        canvas_width = self.source_scale.grid_size[1] * canvas_size
        canvas_boxes = target_boxes.clone()
        canvas_boxes[:, [0, 2]] *= canvas_width / width
        canvas_boxes[:, [1, 3]] *= canvas_height / height
        roi_indices = torch.zeros(len(canvas_boxes), 1)
        target_rois = torch.cat((roi_indices, canvas_boxes), dim=-1).to(
            self.device
        )

        weights = torch.zeros(
            len(target_boxes), len(source_cores), dtype=torch.float32
        )
        for target_index, target_box in enumerate(target_boxes):
            for source_index, source_core in enumerate(source_cores):
                weights[target_index, source_index] = self._intersection_area(
                    target_box, source_core
                )
        weights /= weights.sum(dim=-1, keepdim=True)
        geometry = target_rois, weights.to(self.device)
        self._geometry_cache[key] = geometry
        return geometry

    def _stitch_canvas(
        self,
        source_hidden: torch.Tensor,
        height: int,
        width: int,
        source_boxes: torch.Tensor,
    ) -> torch.Tensor:
        grid_size = self.encoder.token_grid_size
        token_maps = source_hidden[:, 1:].reshape(
            len(source_hidden), grid_size, grid_size, self.encoder.hidden_dim
        )
        token_maps = token_maps.permute(0, 3, 1, 2)
        core_tokens = roi_align(
            token_maps,
            self._source_core_rois(
                height, width, source_boxes, grid_size
            ),
            output_size=(grid_size, grid_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        source_rows, source_columns = self.source_scale.grid_size
        return (
            core_tokens.reshape(
                source_rows,
                source_columns,
                self.encoder.hidden_dim,
                grid_size,
                grid_size,
            )
            .permute(2, 0, 3, 1, 4)
            .reshape(
                self.encoder.hidden_dim,
                source_rows * grid_size,
                source_columns * grid_size,
            )
        )

    def _encode_target_scale(
        self,
        target_scale: PyramidScale,
        canvas: torch.Tensor,
        source_hidden: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        grid_size = self.encoder.token_grid_size
        target_rois, source_weights = self._target_geometry(
            target_scale, height, width, grid_size
        )
        patch_tokens = roi_align(
            canvas.unsqueeze(0),
            target_rois,
            output_size=(grid_size, grid_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).flatten(2).transpose(1, 2)
        source_cls = source_weights.to(source_hidden.dtype) @ source_hidden[:, 0]
        patch_mean = patch_tokens.mean(dim=1)
        target_cls = (
            self.source_cls_weight * source_cls
            + (1.0 - self.source_cls_weight) * patch_mean
        )
        target_tokens = torch.cat((target_cls.unsqueeze(1), patch_tokens), dim=1)
        encoded = []
        for start in range(0, len(target_tokens), self.late_batch_size):
            encoded.append(
                self.encoder.encode_tokens_from_layer(
                    target_tokens[start : start + self.late_batch_size],
                    self.split_layer,
                )
            )
        result = functional.normalize(torch.cat(encoded).float(), dim=-1).reshape(
            *target_scale.grid_size, self.encoder.feature_dim
        )
        if self.roi_adapter is not None:
            result = self.roi_adapter(target_scale.name, result)
        return result

    def encode_frame(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
    ) -> dict[str, torch.Tensor]:
        frame_tensor = self._frame_tensor(frame).to(self.device)
        _, height, width = frame_tensor.shape
        _, source_boxes = self._patch_bounds(
            height, width, self.source_scale
        )
        source_boxes = source_boxes.to(self.device)
        patches = roi_align(
            frame_tensor.unsqueeze(0),
            [source_boxes],
            output_size=(
                self.encoder.input_resolution,
                self.encoder.input_resolution,
            ),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        patches = (patches - self._mean) / self._std

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp and self.device.type == "cuda",
        ):
            source_hidden = self.encoder.encode_patches_to_layer(
                patches, self.split_layer
            )
            source_features = self.encoder.encode_tokens_from_layer(
                source_hidden, self.split_layer
            )
            canvas = self._stitch_canvas(
                source_hidden, height, width, source_boxes
            )
            features = {
                self.source_scale.name: source_features.reshape(
                    *self.source_scale.grid_size, self.encoder.feature_dim
                )
            }
            for scale in self.pyramid_scales:
                if scale.name == self.source_scale.name:
                    continue
                features[scale.name] = self._encode_target_scale(
                    scale,
                    canvas,
                    source_hidden,
                    height,
                    width,
                )
        return features

    def _refine_features(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
        features: dict[str, torch.Tensor],
        candidate_scores: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        refined = {name: values.clone() for name, values in features.items()}
        frame_tensor = self._frame_tensor(frame).to(self.device)
        _, height, width = frame_tensor.shape
        boxes = []
        positions: list[tuple[str, int]] = []
        for scale in self.pyramid_scales:
            count = min(
                self.exact_refine_topk.get(scale.name, 0),
                scale.grid_size[0] * scale.grid_size[1],
            )
            if count == 0:
                continue
            indices = torch.topk(
                candidate_scores[scale.name].flatten(), count
            ).indices.detach().cpu()
            _, scale_boxes = self._patch_bounds(height, width, scale)
            boxes.append(scale_boxes[indices])
            positions.extend((scale.name, int(index)) for index in indices)

        if not positions:
            return refined
        all_boxes = torch.cat(boxes).to(self.device)
        encoded_batches = []
        with torch.inference_mode():
            for start in range(0, len(positions), self.patch_batch_size):
                patches = roi_align(
                    frame_tensor.unsqueeze(0),
                    [all_boxes[start : start + self.patch_batch_size]],
                    output_size=(
                        self.encoder.input_resolution,
                        self.encoder.input_resolution,
                    ),
                    spatial_scale=1.0,
                    sampling_ratio=2,
                    aligned=True,
                )
                patches = (patches - self._mean) / self._std
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp and self.device.type == "cuda",
                ):
                    encoded_batches.append(self.encoder.encode_patches(patches))
        encoded = functional.normalize(
            torch.cat(encoded_batches).float(), dim=-1
        )
        for feature, (scale_name, position) in zip(encoded, positions):
            refined[scale_name].view(-1, self.encoder.feature_dim)[position] = (
                feature
            )
        return refined

    def refine_spatial_novelty(
        self,
        camera_id: str,
        frame: np.ndarray | torch.Tensor | Image.Image,
        spatial_map: SpatialNoveltyMap,
        candidate_scores: dict[str, torch.Tensor],
        normalize: bool = True,
    ) -> SpatialNoveltyMap:
        if not self.exact_refine_topk:
            return spatial_map
        features = self._refine_features(
            frame, spatial_map.raw_features, candidate_scores
        )
        bank = self.camera_banks[camera_id]
        raw_distances = {
            scale.name: self._raw_distance_map(
                features[scale.name],
                bank.scales[scale.name].prototypes.to(self.device),
            )
            for scale in self.pyramid_scales
        }
        novelties = (
            self.normalize_distances(camera_id, raw_distances)
            if normalize
            else raw_distances
        )
        return SpatialNoveltyMap(
            timestamp=spatial_map.timestamp,
            scale_novelties=novelties,
            raw_distances=raw_distances,
            raw_features=features,
        )
