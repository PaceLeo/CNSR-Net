"""Construct spatial matchers from the spatial component configuration."""

from pathlib import Path

import torch
import yaml

from .frozen_encoder import FrozenPatchEncoder
from .late_block_matcher import LateBlockSpatialHierarchicalMatcher
from .pyramid import load_pyramid_scales
from .roi_adapter import load_roi_adapter
from .shared_coarse_matcher import SharedCoarseSpatialHierarchicalMatcher
from .spatial_matcher import SpatialHierarchicalMatcher
from .whole_frame_distance import (
    WholeFrameDistanceSpatialHierarchicalMatcher,
    load_whole_frame_distance_head,
)
from .whole_frame_evidence import (
    WholeFrameSemanticEvidenceSpatialHierarchicalMatcher,
    load_whole_frame_semantic_evidence_head,
)
from .whole_frame_student import (
    FrozenResNet50ROIBackbone,
    WholeFrameStudentSpatialHierarchicalMatcher,
    load_whole_frame_roi_head,
)


def build_spatial_matcher(
    encoder: FrozenPatchEncoder,
    config_path: str | Path,
    device: str | torch.device,
) -> SpatialHierarchicalMatcher:
    """Build the independent-crop or shared-coarse matcher."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    spatial = config["spatial_hierarchical"]
    inference = config["inference"]
    device = torch.device(device)
    common = {
        "encoder": encoder,
        "pyramid_scales": load_pyramid_scales(config_path),
        "num_prototypes": int(spatial["num_prototypes"]),
        "patch_batch_size": int(inference["patch_batch_size"]),
        "device": device,
        "use_amp": bool(inference["use_amp"]) and device.type == "cuda",
    }
    encoding = inference.get("encoding", {"mode": "independent_patches"})
    mode = encoding["mode"]
    if mode == "independent_patches":
        return SpatialHierarchicalMatcher(**common)
    if mode == "shared_coarse_tokens":
        amp_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        adapter_checkpoint = encoding.get("roi_adapter_checkpoint")
        roi_adapter = (
            load_roi_adapter(adapter_checkpoint, device)
            if adapter_checkpoint
            else None
        )
        return SharedCoarseSpatialHierarchicalMatcher(
            **common,
            source_scale_name=str(encoding["source_scale"]),
            token_layer=int(encoding["token_layer"]),
            roi_output_size=int(encoding["roi_output_size"]),
            context_weight=float(encoding["context_weight"]),
            amp_dtype=amp_dtypes[str(encoding["amp_dtype"])],
            roi_adapter=roi_adapter,
        )
    if mode == "shared_prefix_late_blocks":
        amp_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        adapter_checkpoint = encoding.get("roi_adapter_checkpoint")
        roi_adapter = (
            load_roi_adapter(adapter_checkpoint, device)
            if adapter_checkpoint
            else None
        )
        return LateBlockSpatialHierarchicalMatcher(
            **common,
            source_scale_name=str(encoding["source_scale"]),
            split_layer=int(encoding["split_layer"]),
            late_batch_size=int(encoding["late_batch_size"]),
            source_cls_weight=float(encoding["source_cls_weight"]),
            amp_dtype=amp_dtypes[str(encoding["amp_dtype"])],
            roi_adapter=roi_adapter,
            exact_refine_topk={
                str(name): int(count)
                for name, count in encoding.get(
                    "exact_refine_topk", {}
                ).items()
            },
        )
    if mode == "whole_frame_student":
        amp_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        amp_dtype = amp_dtypes[str(encoding["amp_dtype"])]
        backbone = FrozenResNet50ROIBackbone(
            checkpoint_path=encoding["backbone_checkpoint"],
            device=device,
            pool_size=int(encoding["pool_size"]),
            amp_dtype=amp_dtype,
        )
        student_head = load_whole_frame_roi_head(
            encoding["student_checkpoint"], device
        )
        return WholeFrameStudentSpatialHierarchicalMatcher(
            **common,
            backbone=backbone,
            student_head=student_head,
            source_scale_name=str(encoding["source_scale"]),
            amp_dtype=amp_dtype,
        )
    if mode == "whole_frame_distance_student":
        amp_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        amp_dtype = amp_dtypes[str(encoding["amp_dtype"])]
        backbone = FrozenResNet50ROIBackbone(
            checkpoint_path=encoding["backbone_checkpoint"],
            device=device,
            pool_size=int(encoding["pool_size"]),
            amp_dtype=amp_dtype,
        )
        student_head = load_whole_frame_roi_head(
            encoding["student_checkpoint"], device
        )
        distance_head = load_whole_frame_distance_head(
            encoding["distance_checkpoint"], device
        )
        return WholeFrameDistanceSpatialHierarchicalMatcher(
            **common,
            backbone=backbone,
            student_head=student_head,
            distance_head=distance_head,
            source_scale_name=str(encoding["source_scale"]),
            amp_dtype=amp_dtype,
        )
    if mode == "whole_frame_semantic_evidence_student":
        amp_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        amp_dtype = amp_dtypes[str(encoding["amp_dtype"])]
        backbone = FrozenResNet50ROIBackbone(
            checkpoint_path=encoding["backbone_checkpoint"],
            device=device,
            pool_size=int(encoding["pool_size"]),
            amp_dtype=amp_dtype,
        )
        student_head = load_whole_frame_roi_head(
            encoding["student_checkpoint"], device
        )
        evidence_head = load_whole_frame_semantic_evidence_head(
            encoding["evidence_checkpoint"], device
        )
        return WholeFrameSemanticEvidenceSpatialHierarchicalMatcher(
            **common,
            backbone=backbone,
            student_head=student_head,
            evidence_head=evidence_head,
            evidence_scale_names=tuple(
                str(name)
                for name in encoding.get(
                    "evidence_scales", student_head.scale_names
                )
            ),
            evidence_blend_weight=float(
                encoding.get("evidence_blend_weight", 1.0)
            ),
            source_scale_name=str(encoding["source_scale"]),
            amp_dtype=amp_dtype,
        )
    raise ValueError(f"Unsupported spatial encoding mode: {mode}")
