"""Attention aggregation and camera-window conformal calibration."""

from .attention import AttentionOutput, SpatialAttentionAggregator
from .config import (
    AttentionConfig,
    ConformalConfig,
    DevelopmentRoleConfig,
    Phase4Config,
    load_phase4_config,
)
from .conformal import CameraWindowConformalCalibrator
from .roles import DevelopmentRoles, split_development_cameras
from .trainer import AttentionSample, AttentionTrainer, TrainingHistory

__all__ = [
    "AttentionConfig",
    "AttentionOutput",
    "AttentionSample",
    "AttentionTrainer",
    "CameraWindowConformalCalibrator",
    "ConformalConfig",
    "DevelopmentRoleConfig",
    "DevelopmentRoles",
    "Phase4Config",
    "SpatialAttentionAggregator",
    "TrainingHistory",
    "load_phase4_config",
    "split_development_cameras",
]
