"""Fixed camera-disjoint roles inside the development split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import DevelopmentRoleConfig


@dataclass(frozen=True)
class DevelopmentRoles:
    attention_train: tuple[str, ...]
    attention_validation: tuple[str, ...]
    conformal_calibration: tuple[str, ...]


def split_development_cameras(
    camera_ids: Sequence[str],
    config: DevelopmentRoleConfig,
) -> DevelopmentRoles:
    """Validate and return the configured camera-disjoint roles."""
    configured_roles = (
        config.attention_train,
        config.attention_validation,
        config.conformal_calibration,
    )
    configured_cameras = [
        camera_id for role in configured_roles for camera_id in role
    ]
    if len(configured_cameras) != len(set(configured_cameras)):
        raise ValueError("Development camera roles must be disjoint")
    if set(configured_cameras) != set(camera_ids):
        raise ValueError(
            "Configured development cameras do not match the experiment "
            "development split"
        )
    return DevelopmentRoles(
        attention_train=config.attention_train,
        attention_validation=config.attention_validation,
        conformal_calibration=config.conformal_calibration,
    )
