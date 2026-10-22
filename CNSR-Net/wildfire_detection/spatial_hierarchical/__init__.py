"""Spatial hierarchical prototype matching."""

from .frozen_encoder import FrozenPatchEncoder, load_frozen_patch_encoder
from .factory import build_spatial_matcher
from .kmedoids import KMedoidsResult, pam_kmedoids
from .late_block_matcher import LateBlockSpatialHierarchicalMatcher
from .pyramid import PyramidScale, load_pyramid_scales
from .prototype_bank import CameraPrototypeBank, ScalePrototypeBank
from .roi_adapter import MultiScaleROIAdapter, load_roi_adapter
from .shared_coarse_matcher import SharedCoarseSpatialHierarchicalMatcher
from .spatial_matcher import SpatialHierarchicalMatcher, SpatialNoveltyMap
from .whole_frame_distance import (
    WholeFrameDistanceHead,
    WholeFrameDistanceSpatialHierarchicalMatcher,
    load_whole_frame_distance_head,
)
from .whole_frame_evidence import (
    WholeFrameSemanticEvidenceHead,
    WholeFrameSemanticEvidenceSpatialHierarchicalMatcher,
    load_whole_frame_semantic_evidence_head,
)
from .whole_frame_student import (
    FrozenResNet50ROIBackbone,
    WholeFrameROIHead,
    WholeFrameStudentSpatialHierarchicalMatcher,
    load_whole_frame_roi_head,
)

__all__ = [
    "CameraPrototypeBank",
    "MultiScaleROIAdapter",
    "FrozenPatchEncoder",
    "FrozenResNet50ROIBackbone",
    "KMedoidsResult",
    "LateBlockSpatialHierarchicalMatcher",
    "PyramidScale",
    "ScalePrototypeBank",
    "SharedCoarseSpatialHierarchicalMatcher",
    "SpatialHierarchicalMatcher",
    "SpatialNoveltyMap",
    "WholeFrameROIHead",
    "WholeFrameDistanceHead",
    "WholeFrameDistanceSpatialHierarchicalMatcher",
    "WholeFrameSemanticEvidenceHead",
    "WholeFrameSemanticEvidenceSpatialHierarchicalMatcher",
    "WholeFrameStudentSpatialHierarchicalMatcher",
    "build_spatial_matcher",
    "load_frozen_patch_encoder",
    "load_pyramid_scales",
    "load_roi_adapter",
    "load_whole_frame_roi_head",
    "load_whole_frame_distance_head",
    "load_whole_frame_semantic_evidence_head",
    "pam_kmedoids",
]
