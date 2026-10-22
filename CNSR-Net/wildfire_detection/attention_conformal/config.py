"""Phase 4 attention and conformal configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AttentionConfig:
    seed: int
    hidden_dim: int
    dropout: float
    entropy_weight: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_epochs: int
    early_stopping_patience: int
    score_scale_init: float
    score_bias_init: float
    horizontal_flip_probability: float


@dataclass(frozen=True)
class DevelopmentRoleConfig:
    attention_train: tuple[str, ...]
    attention_validation: tuple[str, ...]
    conformal_calibration: tuple[str, ...]


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float
    calibration_statistic: str


@dataclass(frozen=True)
class Phase4Config:
    attention: AttentionConfig
    development_roles: DevelopmentRoleConfig
    conformal: ConformalConfig


def load_phase4_config(path: str | Path) -> Phase4Config:
    """Load attention training and camera-level calibration settings."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    attention = raw["attention"]
    roles_path = Path(raw["development_roles"]["config"]).expanduser()
    if not roles_path.is_absolute():
        roles_path = config_path.parent / roles_path
    with roles_path.open("r", encoding="utf-8") as file:
        roles = yaml.safe_load(file)["roles"]
    conformal = raw["conformal"]
    return Phase4Config(
        attention=AttentionConfig(
            seed=int(attention["seed"]),
            hidden_dim=int(attention["hidden_dim"]),
            dropout=float(attention["dropout"]),
            entropy_weight=float(attention["entropy_weight"]),
            learning_rate=float(attention["learning_rate"]),
            weight_decay=float(attention["weight_decay"]),
            batch_size=int(attention["batch_size"]),
            num_epochs=int(attention["num_epochs"]),
            early_stopping_patience=int(
                attention["early_stopping_patience"]
            ),
            score_scale_init=float(attention["score_scale_init"]),
            score_bias_init=float(attention["score_bias_init"]),
            horizontal_flip_probability=float(
                attention["horizontal_flip_probability"]
            ),
        ),
        development_roles=DevelopmentRoleConfig(
            attention_train=tuple(str(value) for value in roles["training"]),
            attention_validation=tuple(
                str(value) for value in roles["validation"]
            ),
            conformal_calibration=tuple(
                str(value) for value in roles["calibration"]
            ),
        ),
        conformal=ConformalConfig(
            alpha=float(conformal["alpha"]),
            calibration_statistic=str(
                conformal["calibration_statistic"]
            ),
        ),
    )
