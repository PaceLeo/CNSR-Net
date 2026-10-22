"""Spatial pyramid configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PyramidScale:
    name: str
    grid_size: tuple[int, int]
    overlap: float


def load_pyramid_scales(path: str | Path) -> tuple[PyramidScale, ...]:
    """Load spatial pyramid scales from a phase configuration file."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return tuple(
        PyramidScale(
            name=item["name"],
            grid_size=(int(item["grid_size"][0]), int(item["grid_size"][1])),
            overlap=float(item["overlap"]),
        )
        for item in raw["spatial_hierarchical"]["pyramid_scales"]
    )
