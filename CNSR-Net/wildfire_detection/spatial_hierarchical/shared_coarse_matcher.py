"""Multi-scale matching from shared CLIP tokens of the coarse crops."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.ops import roi_align

from .pyramid import PyramidScale
from .roi_adapter import MultiScaleROIAdapter
from .spatial_matcher import PatchEncoder, SpatialHierarchicalMatcher


class SharedTokenEncoder(PatchEncoder, Protocol):
    def encode_patches_with_tokens(
        self,
        patches: torch.Tensor,
        feature_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CLS features and token grids shaped [B, H, W, D]."""
        ...


class SharedCoarseSpatialHierarchicalMatcher(SpatialHierarchicalMatcher):
    """Encode coarse crops once and pool their tokens for finer scales."""

    def __init__(
        self,
        encoder: SharedTokenEncoder,
        pyramid_scales: Sequence[PyramidScale],
        num_prototypes: int = 8,
        patch_batch_size: int = 256,
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        normalization_epsilon: float = 1e-6,
        source_scale_name: str = "coarse",
        token_layer: int = 24,
        roi_output_size: int = 3,
        context_weight: float = 0.5,
        amp_dtype: torch.dtype = torch.float16,
        roi_adapter: MultiScaleROIAdapter | None = None,
    ):
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
        self.token_layer = token_layer
        self.roi_output_size = roi_output_size
        self.context_weight = context_weight
        self.amp_dtype = amp_dtype
        self.roi_adapter = roi_adapter
        self._geometry_cache: dict[
            tuple[int, int, str, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
        ] = {}

    @staticmethod
    def _intersection(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        start = torch.maximum(first[:2], second[:2])
        end = torch.minimum(first[2:], second[2:])
        size = end - start
        if bool((size <= 0).any()):
            return None
        return torch.cat((start, end)), size.prod()

    def _pooling_geometry(
        self,
        target_scale: PyramidScale,
        height: int,
        width: int,
        source_boxes: torch.Tensor,
        token_height: int,
        token_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cache_key = (
            height,
            width,
            target_scale.name,
            token_height,
            token_width,
        )
        cached = self._geometry_cache.get(cache_key)
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
        source_boxes = source_boxes.detach().cpu()

        roi_rows = []
        target_indices = []
        source_indices = []
        piece_areas = []
        for target_index, target_box in enumerate(target_boxes):
            for source_index, source_core in enumerate(source_cores):
                intersection = self._intersection(target_box, source_core)
                if intersection is None:
                    continue
                intersection_box, area = intersection
                source_box = source_boxes[source_index]
                source_size = source_box[2:] - source_box[:2]
                local_start = (
                    (intersection_box[:2] - source_box[:2]) / source_size
                )
                local_end = (
                    (intersection_box[2:] - source_box[:2]) / source_size
                )
                local_box = torch.cat((local_start, local_end))
                local_box[[0, 2]] *= token_width
                local_box[[1, 3]] *= token_height
                roi_rows.append(
                    torch.cat(
                        (
                            torch.tensor(
                                [source_index],
                                dtype=local_box.dtype,
                            ),
                            local_box,
                        )
                    )
                )
                target_indices.append(target_index)
                source_indices.append(source_index)
                piece_areas.append(area)

        target_index_tensor = torch.tensor(target_indices, device=self.device)
        source_index_tensor = torch.tensor(source_indices, device=self.device)
        area_tensor = torch.stack(piece_areas).to(self.device)
        target_count = len(target_boxes)
        area_sums = torch.zeros(target_count, device=self.device)
        area_sums.index_add_(0, target_index_tensor, area_tensor)
        weights = area_tensor / area_sums[target_index_tensor]
        geometry = (
            torch.stack(roi_rows).to(self.device),
            target_index_tensor,
            source_index_tensor,
            weights,
            target_count,
        )
        self._geometry_cache[cache_key] = geometry
        return geometry

    def _pool_scale(
        self,
        target_scale: PyramidScale,
        height: int,
        width: int,
        source_boxes: torch.Tensor,
        source_cls: torch.Tensor,
        token_maps: torch.Tensor,
    ) -> torch.Tensor:
        token_height, token_width = token_maps.shape[-2:]
        (
            roi_rows_tensor,
            target_index_tensor,
            source_index_tensor,
            weights,
            target_count,
        ) = self._pooling_geometry(
            target_scale,
            height,
            width,
            source_boxes,
            token_height,
            token_width,
        )
        pooled_pieces = roi_align(
            token_maps,
            roi_rows_tensor,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).mean(dim=(-2, -1))
        piece_features = pooled_pieces + (
            self.context_weight * source_cls[source_index_tensor]
        )
        pooled = torch.zeros(
            target_count,
            self.encoder.feature_dim,
            device=self.device,
            dtype=pooled_pieces.dtype,
        )
        pooled.index_add_(
            0,
            target_index_tensor,
            piece_features * weights.to(pooled_pieces.dtype).unsqueeze(-1),
        )
        pooled = functional.normalize(pooled, dim=-1).reshape(
            *target_scale.grid_size,
            self.encoder.feature_dim,
        )
        if self.roi_adapter is not None:
            pooled = self.roi_adapter(target_scale.name, pooled)
        return pooled

    def encode_frame(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
    ) -> dict[str, torch.Tensor]:
        frame_tensor = self._frame_tensor(frame).to(self.device)
        _, height, width = frame_tensor.shape
        source_positions, source_boxes = self._patch_bounds(
            height,
            width,
            self.source_scale,
        )
        source_boxes = source_boxes.to(self.device)
        cls_batches = []
        token_batches = []

        with torch.inference_mode():
            for start in range(0, len(source_positions), self.patch_batch_size):
                end = min(start + self.patch_batch_size, len(source_positions))
                patches = roi_align(
                    frame_tensor.unsqueeze(0),
                    [source_boxes[start:end]],
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
                    cls_features, token_grid = (
                        self.encoder.encode_patches_with_tokens(
                            patches,
                            self.token_layer,
                        )
                    )
                cls_batches.append(cls_features.float())
                token_batches.append(token_grid.float())

        source_cls = torch.cat(cls_batches)
        token_maps = torch.cat(token_batches).permute(0, 3, 1, 2)
        features = {
            self.source_scale.name: source_cls.reshape(
                *self.source_scale.grid_size,
                self.encoder.feature_dim,
            )
        }
        for scale in self.pyramid_scales:
            if scale.name == self.source_scale.name:
                continue
            features[scale.name] = self._pool_scale(
                scale,
                height,
                width,
                source_boxes,
                source_cls,
                token_maps,
            )
        for scale_name, values in features.items():
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError(
                    f"Shared-coarse encoding produced non-finite {scale_name} "
                    "features"
                )
        return features
