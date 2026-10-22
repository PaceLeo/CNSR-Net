"""Whole-frame CNN student for dense CLIP crop embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.models import resnet50
from torchvision.ops import roi_align

from .pyramid import PyramidScale
from .spatial_matcher import PatchEncoder, SpatialHierarchicalMatcher


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenResNet50ROIBackbone(torch.nn.Module):
    """Pool multi-level ImageNet features for every pyramid region."""

    channels = (256, 512, 1024, 2048)

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device,
        pool_size: int = 1,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.pool_size = int(pool_size)
        self.amp_dtype = amp_dtype
        self.descriptor_dim = sum(self.channels) * self.pool_size**2
        self.model = resnet50(weights=None)
        state = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _frame_tensor(
        frame: np.ndarray | torch.Tensor | Image.Image,
    ) -> torch.Tensor:
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

    def _feature_maps(self, frame_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = (frame_tensor.unsqueeze(0) - self.mean) / self.std
        model = self.model
        values = model.relu(model.bn1(model.conv1(values)))
        values = model.maxpool(values)
        c2 = model.layer1(values)
        c3 = model.layer2(c2)
        c4 = model.layer3(c3)
        c5 = model.layer4(c4)
        return c2, c3, c4, c5

    def encode_tensor(
        self,
        frame_tensor: torch.Tensor,
        pyramid_scales: Sequence[PyramidScale],
    ) -> dict[str, torch.Tensor]:
        frame_tensor = frame_tensor.to(self.device)
        _, height, width = frame_tensor.shape
        scale_boxes = []
        counts = []
        for scale in pyramid_scales:
            _, boxes = SpatialHierarchicalMatcher._patch_bounds(
                height, width, scale
            )
            scale_boxes.append(boxes)
            counts.append(len(boxes))
        all_boxes = torch.cat(scale_boxes).to(self.device)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            maps = self._feature_maps(frame_tensor)
            pooled_levels = []
            for values in maps:
                boxes = all_boxes.clone()
                boxes[:, [0, 2]] *= values.shape[-1] / width
                boxes[:, [1, 3]] *= values.shape[-2] / height
                pooled = roi_align(
                    values,
                    [boxes],
                    output_size=(self.pool_size, self.pool_size),
                    spatial_scale=1.0,
                    sampling_ratio=2,
                    aligned=True,
                ).flatten(1)
                pooled_levels.append(functional.normalize(pooled.float(), dim=-1))
            descriptors = torch.cat(pooled_levels, dim=-1)

        result = {}
        offset = 0
        for scale, count in zip(pyramid_scales, counts, strict=True):
            result[scale.name] = descriptors[offset : offset + count].reshape(
                *scale.grid_size, self.descriptor_dim
            )
            offset += count
        return result

    def encode_frame(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
        pyramid_scales: Sequence[PyramidScale],
    ) -> dict[str, torch.Tensor]:
        return self.encode_tensor(self._frame_tensor(frame), pyramid_scales)


class WholeFrameROIHead(torch.nn.Module):
    """Map pooled whole-frame CNN descriptors into the CLIP image space."""

    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        feature_dim: int,
        scale_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.scale_names = tuple(str(name) for name in scale_names)
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(self.descriptor_dim),
            torch.nn.Linear(self.descriptor_dim, self.hidden_dim),
            torch.nn.GELU(),
        )
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(self.hidden_dim, self.feature_dim)
                for name in self.scale_names
            }
        )

    def forward(self, scale_name: str, descriptors: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(descriptors.float())
        return functional.normalize(self.heads[scale_name](hidden).float(), dim=-1)


def load_whole_frame_roi_head(
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> WholeFrameROIHead:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=True,
    )
    head = WholeFrameROIHead(
        descriptor_dim=int(checkpoint["descriptor_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        feature_dim=int(checkpoint["feature_dim"]),
        scale_names=tuple(checkpoint["scale_names"]),
    )
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    return head.to(device).eval().requires_grad_(False)


class WholeFrameStudentSpatialHierarchicalMatcher(SpatialHierarchicalMatcher):
    """Use exact coarse CLIP crops and one CNN pass for denser scales."""

    def __init__(
        self,
        encoder: PatchEncoder,
        pyramid_scales: Sequence[PyramidScale],
        backbone: FrozenResNet50ROIBackbone,
        student_head: WholeFrameROIHead,
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
            num_prototypes=num_prototypes,
            patch_batch_size=patch_batch_size,
            device=device,
            use_amp=use_amp,
            normalization_epsilon=normalization_epsilon,
        )
        self.source_scale = next(
            scale for scale in self.pyramid_scales if scale.name == source_scale_name
        )
        self.target_scales = tuple(
            scale for scale in self.pyramid_scales if scale.name != source_scale_name
        )
        self.backbone = backbone
        self.student_head = student_head
        self.amp_dtype = amp_dtype

    def _encode_source(
        self,
        frame_tensor: torch.Tensor,
    ) -> torch.Tensor:
        _, height, width = frame_tensor.shape
        _, boxes = self._patch_bounds(height, width, self.source_scale)
        patches = roi_align(
            frame_tensor.unsqueeze(0),
            [boxes.to(self.device)],
            output_size=(self.encoder.input_resolution, self.encoder.input_resolution),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        patches = (patches - self._mean) / self._std
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(patches), self.patch_batch_size):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp and self.device.type == "cuda",
                ):
                    encoded.append(
                        self.encoder.encode_patches(
                            patches[start : start + self.patch_batch_size]
                        )
                    )
        return functional.normalize(torch.cat(encoded).float(), dim=-1).reshape(
            *self.source_scale.grid_size, self.encoder.feature_dim
        )

    def encode_frame(
        self,
        frame: np.ndarray | torch.Tensor | Image.Image,
    ) -> dict[str, torch.Tensor]:
        frame_tensor = self._frame_tensor(frame).to(self.device)
        descriptors = self.backbone.encode_tensor(frame_tensor, self.target_scales)
        features = {self.source_scale.name: self._encode_source(frame_tensor)}
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp and self.device.type == "cuda",
        ):
            for scale in self.target_scales:
                features[scale.name] = self.student_head(
                    scale.name, descriptors[scale.name]
                )
        return features
