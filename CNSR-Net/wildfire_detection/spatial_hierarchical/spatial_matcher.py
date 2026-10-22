"""Dense multi-scale spatial prototype matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.ops import roi_align

from .kmedoids import pam_kmedoids
from .prototype_bank import CameraPrototypeBank, ScalePrototypeBank
from .pyramid import PyramidScale


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class PatchEncoder(Protocol):
    input_resolution: int
    feature_dim: int

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class SpatialNoveltyMap:
    timestamp: float
    scale_novelties: dict[str, torch.Tensor]
    raw_distances: dict[str, torch.Tensor]
    raw_features: dict[str, torch.Tensor]


class SpatialHierarchicalMatcher:
    """Build camera-local medoids and compute dense novelty maps."""

    def __init__(
        self,
        encoder: PatchEncoder,
        pyramid_scales: Sequence[PyramidScale],
        num_prototypes: int = 8,
        patch_batch_size: int = 256,
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        normalization_epsilon: float = 1e-6,
    ):
        self.encoder = encoder
        self.pyramid_scales = tuple(pyramid_scales)
        self.num_prototypes = num_prototypes
        self.patch_batch_size = patch_batch_size
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.normalization_epsilon = normalization_epsilon
        self.camera_banks: dict[str, CameraPrototypeBank] = {}

        self._mean = torch.tensor(_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_CLIP_STD, device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def _frame_tensor(frame: np.ndarray | torch.Tensor | Image.Image) -> torch.Tensor:
        if isinstance(frame, Image.Image):
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        elif isinstance(frame, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(frame))
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
        elif isinstance(frame, torch.Tensor):
            tensor = frame
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)!r}")

        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("Frames must have shape [H, W, 3] or [3, H, W]")
        if tensor.dtype == torch.uint8:
            return tensor.float().div_(255.0)
        return tensor.float()

    @staticmethod
    def _patch_bounds(
        height: int,
        width: int,
        scale: PyramidScale,
    ) -> tuple[list[tuple[int, int]], torch.Tensor]:
        grid_height, grid_width = scale.grid_size
        cell_height = height / grid_height
        cell_width = width / grid_width
        overlap_height = int(cell_height * scale.overlap / 2)
        overlap_width = int(cell_width * scale.overlap / 2)

        positions: list[tuple[int, int]] = []
        boxes: list[list[float]] = []
        for row in range(grid_height):
            for column in range(grid_width):
                y_start = max(0, int(row * cell_height) - overlap_height)
                y_end = min(
                    height,
                    int((row + 1) * cell_height) + overlap_height,
                )
                x_start = max(0, int(column * cell_width) - overlap_width)
                x_end = min(
                    width,
                    int((column + 1) * cell_width) + overlap_width,
                )
                positions.append((row, column))
                boxes.append([x_start, y_start, x_end, y_end])
        return positions, torch.tensor(boxes, dtype=torch.float32)

    def encode_frame(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
    ) -> dict[str, torch.Tensor]:
        frame_tensor = self._frame_tensor(frame).to(self.device)
        _, height, width = frame_tensor.shape
        features = {
            scale.name: torch.empty(
                (*scale.grid_size, self.encoder.feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
            for scale in self.pyramid_scales
        }

        positions: list[tuple[str, int, int]] = []
        all_boxes: list[torch.Tensor] = []
        for scale in self.pyramid_scales:
            scale_positions, scale_boxes = self._patch_bounds(
                height,
                width,
                scale,
            )
            positions.extend(
                (scale.name, row, column)
                for row, column in scale_positions
            )
            all_boxes.append(scale_boxes)
        boxes = torch.cat(all_boxes, dim=0).to(self.device)

        with torch.inference_mode():
            for start in range(0, len(positions), self.patch_batch_size):
                end = min(start + self.patch_batch_size, len(positions))
                patches = roi_align(
                    frame_tensor.unsqueeze(0),
                    [boxes[start:end]],
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
                    dtype=torch.float16,
                    enabled=self.use_amp and self.device.type == "cuda",
                ):
                    encoded = self.encoder.encode_patches(patches)
                encoded = torch.nn.functional.normalize(
                    encoded.float(),
                    dim=-1,
                )
                for feature, (scale_name, row, column) in zip(
                    encoded,
                    positions[start:end],
                ):
                    features[scale_name][row, column] = feature
        return features

    def _encode_prototype_frame(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
    ) -> dict[str, torch.Tensor]:
        return self.encode_frame(frame)

    def fit_camera_prototypes(
        self,
        camera_id: str,
        normal_frames: Sequence[np.ndarray | torch.Tensor | Image.Image],
    ) -> CameraPrototypeBank:
        if len(normal_frames) < self.num_prototypes:
            raise ValueError(
                f"Camera {camera_id} has {len(normal_frames)} prototype frames, "
                f"but k={self.num_prototypes}"
            )

        encoded_frames = [
            self._encode_prototype_frame(frame) for frame in normal_frames
        ]
        scale_banks: dict[str, ScalePrototypeBank] = {}
        for scale in self.pyramid_scales:
            frame_features = torch.stack(
                [features[scale.name].cpu() for features in encoded_frames]
            )
            sample_count, grid_height, grid_width, feature_dim = (
                frame_features.shape
            )
            flattened = frame_features.reshape(sample_count, -1, feature_dim)
            prototype_positions = []
            for position in range(flattened.shape[1]):
                result = pam_kmedoids(
                    flattened[:, position, :].numpy(),
                    self.num_prototypes,
                )
                prototype_positions.append(torch.from_numpy(result.medoids))
            prototypes = torch.stack(prototype_positions).reshape(
                grid_height,
                grid_width,
                self.num_prototypes,
                feature_dim,
            )
            scale_banks[scale.name] = ScalePrototypeBank(
                scale_name=scale.name,
                grid_size=scale.grid_size,
                prototypes=prototypes.float(),
            )

        camera_bank = CameraPrototypeBank(
            camera_id=camera_id,
            scales=scale_banks,
        )
        self.camera_banks[camera_id] = camera_bank
        return camera_bank

    @staticmethod
    def _raw_distance_map(
        features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        return torch.linalg.vector_norm(
            features.unsqueeze(-2) - prototypes,
            dim=-1,
        ).amin(dim=-1)

    def fit_camera_normalization(
        self,
        camera_id: str,
        threshold_frames: Sequence[np.ndarray | torch.Tensor | Image.Image],
    ) -> None:
        bank = self.camera_banks[camera_id]
        distance_samples = {scale.name: [] for scale in self.pyramid_scales}
        for frame in threshold_frames:
            features = self.encode_frame(frame)
            for scale in self.pyramid_scales:
                scale_bank = bank.scales[scale.name]
                distances = self._raw_distance_map(
                    features[scale.name],
                    scale_bank.prototypes.to(self.device),
                )
                distance_samples[scale.name].append(distances.cpu())

        self.fit_camera_normalization_from_distances(
            camera_id,
            [
                {
                    scale.name: distance_samples[scale.name][frame_index]
                    for scale in self.pyramid_scales
                }
                for frame_index in range(len(threshold_frames))
            ],
        )

    def fit_camera_normalization_from_distances(
        self,
        camera_id: str,
        distance_maps: Sequence[dict[str, torch.Tensor]],
    ) -> None:
        """Fit position-wise distance statistics from encoded normal maps."""
        bank = self.camera_banks[camera_id]
        for scale in self.pyramid_scales:
            samples = torch.stack(
                [
                    distances[scale.name].detach().float().cpu()
                    for distances in distance_maps
                ]
            )
            scale_bank = bank.scales[scale.name]
            scale_bank.distance_mean = samples.mean(dim=0)
            scale_bank.distance_std = samples.std(dim=0, unbiased=False)

    def normalize_distances(
        self,
        camera_id: str,
        raw_distances: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply fitted camera-local z normalization to distance maps."""
        bank = self.camera_banks[camera_id]
        normalized = {}
        for scale in self.pyramid_scales:
            if scale.name not in raw_distances:
                continue
            scale_bank = bank.scales[scale.name]
            if (
                scale_bank.distance_mean is None
                or scale_bank.distance_std is None
            ):
                raise ValueError(
                    f"Camera {camera_id} normalization is not fitted"
                )
            raw = raw_distances[scale.name]
            mean = scale_bank.distance_mean.to(raw.device)
            std = scale_bank.distance_std.to(raw.device).clamp_min(
                self.normalization_epsilon
            )
            normalized[scale.name] = (raw - mean) / std
        return normalized

    def compute_spatial_novelty(
        self,
        camera_id: str,
        frame: np.ndarray | torch.Tensor | Image.Image,
        timestamp: float,
        normalize: bool = True,
    ) -> SpatialNoveltyMap:
        bank = self.camera_banks[camera_id]
        features = self.encode_frame(frame)
        raw_distances: dict[str, torch.Tensor] = {}
        novelties: dict[str, torch.Tensor] = {}

        for scale in self.pyramid_scales:
            scale_bank = bank.scales[scale.name]
            raw = self._raw_distance_map(
                features[scale.name],
                scale_bank.prototypes.to(self.device),
            )
            raw_distances[scale.name] = raw
            if normalize:
                novelties[scale.name] = self.normalize_distances(
                    camera_id,
                    {scale.name: raw},
                )[scale.name]
            else:
                novelties[scale.name] = raw

        return SpatialNoveltyMap(
            timestamp=timestamp,
            scale_novelties=novelties,
            raw_distances=raw_distances,
            raw_features=features,
        )

    def refine_spatial_novelty(
        self,
        camera_id: str,
        frame: np.ndarray | torch.Tensor | Image.Image,
        spatial_map: SpatialNoveltyMap,
        candidate_scores: dict[str, torch.Tensor],
        normalize: bool = True,
    ) -> SpatialNoveltyMap:
        """Optionally refine candidates after semantic scoring."""
        return spatial_map

    def prototype_means(
        self,
        camera_id: str,
    ) -> dict[str, torch.Tensor]:
        bank = self.camera_banks[camera_id]
        return {
            name: scale_bank.prototypes.mean(dim=-2).to(self.device)
            for name, scale_bank in bank.scales.items()
        }

    def nearest_prototypes(
        self,
        camera_id: str,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Retrieve the same position-local medoid used by novelty distance."""
        bank = self.camera_banks[camera_id]
        matched = {}
        for name, values in features.items():
            prototypes = bank.scales[name].prototypes.to(values.device)
            distances = torch.linalg.vector_norm(
                values.unsqueeze(-2) - prototypes, dim=-1
            )
            nearest = distances.argmin(dim=-1, keepdim=True)
            indices = nearest.unsqueeze(-1).expand(
                *nearest.shape, values.shape[-1]
            )
            matched[name] = prototypes.gather(-2, indices).squeeze(-2)
        return matched

    def semantic_evidence_overrides(
        self,
        camera_id: str,
        spatial_map: SpatialNoveltyMap,
    ) -> dict[str, tuple[torch.Tensor, float]]:
        """Return learned evidence maps for scales that bypass CLIP semantics."""
        return {}

    def save_camera_bank(self, camera_id: str, path: str | Path) -> None:
        self.camera_banks[camera_id].save(path)

    def load_camera_bank(self, path: str | Path) -> CameraPrototypeBank:
        bank = CameraPrototypeBank.load(path)
        self.camera_banks[bank.camera_id] = bank
        return bank
