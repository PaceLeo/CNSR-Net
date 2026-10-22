"""Stable model-variant names and artifact helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Keep the experiment IDs stable so released metric files remain traceable.
MAIN_METHOD = "A3_spatial_semantic_temporal"
PLUS_METHOD = "Full_system"

MAIN_MODEL_NAME = "CNSR-Net"
PLUS_MODEL_NAME = "CNSR-Net Plus"
MAIN_VARIANT = "main"
PLUS_VARIANT = "plus"

MAIN_ARTIFACT_FILENAME = "cnsr_net_calibration.pt"
PLUS_ARTIFACT_FILENAME = "cnsr_net_plus_model.pt"


def calibration_payload(
    artifact: Mapping[str, Any],
    method: str,
) -> Mapping[str, Any]:
    """Read one method's calibration from a variant or legacy joint artifact."""
    raw = artifact.get("calibration")
    if not isinstance(raw, Mapping):
        raise ValueError("Model artifact does not contain calibration metadata")
    if method in raw and isinstance(raw[method], Mapping):
        return raw[method]
    if artifact.get("method") == method and "global_margin" in raw:
        return raw
    raise ValueError(f"Model artifact has no calibration for {method}")


def build_main_calibration_artifact(
    joint_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the training-free CNSR-Net calibration from a joint artifact."""
    artifact = {
        "schema_version": int(joint_artifact.get("schema_version", 1)),
        "model_name": MAIN_MODEL_NAME,
        "variant": MAIN_VARIANT,
        "method": MAIN_METHOD,
        "calibration": dict(calibration_payload(joint_artifact, MAIN_METHOD)),
    }
    for field in ("experiment", "role_set", "confirmation_used"):
        if field in joint_artifact:
            artifact[field] = joint_artifact[field]
    return artifact


def build_plus_model_artifact(
    joint_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Mark the learned Full-system artifact as the optional Plus variant."""
    if "state_dict" not in joint_artifact:
        raise ValueError("CNSR-Net Plus artifact requires attention weights")
    artifact = dict(joint_artifact)
    artifact.update(
        {
            "model_name": PLUS_MODEL_NAME,
            "variant": PLUS_VARIANT,
            "method": PLUS_METHOD,
        }
    )
    return artifact
