"""Configuration for semantic residual scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SemanticWeights:
    smoke: float
    flame: float
    cloud: float


@dataclass(frozen=True)
class SemanticConfig:
    mode: str
    use_contrast: bool
    weights: SemanticWeights
    prompts: dict[str, tuple[str, ...]]
    residual_reference: str = "mean"


def load_semantic_config(path: str | Path) -> SemanticConfig:
    """Load semantic prompts and multidimensional weights from YAML."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)["semantic_residual"]

    return SemanticConfig(
        mode=str(raw["mode"]),
        use_contrast=bool(raw["use_contrast"]),
        weights=SemanticWeights(
            smoke=float(raw["weights"]["smoke"]),
            flame=float(raw["weights"]["flame"]),
            cloud=float(raw["weights"]["cloud"]),
        ),
        prompts={
            name: tuple(str(prompt) for prompt in prompts)
            for name, prompts in raw["prompts"].items()
        },
        residual_reference=str(raw.get("residual_reference", "mean")),
    )
