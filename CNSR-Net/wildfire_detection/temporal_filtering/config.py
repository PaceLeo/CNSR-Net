"""Configuration for time-aware persistence filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TemporalConfig:
    window_seconds: float
    nominal_sampling_seconds: float
    low_threshold_quantile: float
    persistence_thresholds: dict[str, float]
    use_spatial_neighborhood: bool


def load_temporal_config(path: str | Path) -> TemporalConfig:
    """Load the physical-time persistence configuration from YAML."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)["temporal_filtering"]

    return TemporalConfig(
        window_seconds=float(raw["window_seconds"]),
        nominal_sampling_seconds=float(raw["nominal_sampling_seconds"]),
        low_threshold_quantile=float(raw["low_threshold_quantile"]),
        persistence_thresholds={
            name: float(threshold)
            for name, threshold in raw["persistence_thresholds"].items()
        },
        use_spatial_neighborhood=bool(raw["use_spatial_neighborhood"]),
    )
