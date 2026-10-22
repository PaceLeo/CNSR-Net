"""Build reusable text prototypes in CNSR-Net's frozen embedding space."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol, Sequence

import torch
import torch.nn.functional as functional


class TextEncoder(Protocol):
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        ...


@dataclass
class SemanticPrototypes:
    fire_embedding: torch.Tensor
    normal_embedding: torch.Tensor
    contrast_embedding: torch.Tensor
    smoke_embedding: torch.Tensor
    flame_embedding: torch.Tensor
    cloud_embedding: torch.Tensor
    smoke_direction: torch.Tensor
    flame_direction: torch.Tensor
    cloud_direction: torch.Tensor

    def similarities_to_normal(self) -> dict[str, float]:
        return {
            name: float(embedding @ self.normal_embedding)
            for name, embedding in {
                "fire": self.fire_embedding,
                "smoke": self.smoke_embedding,
                "flame": self.flame_embedding,
                "cloud": self.cloud_embedding,
            }.items()
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            field.name: getattr(self, field.name).detach().float().cpu()
            for field in fields(self)
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticPrototypes":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**{name: value.float() for name, value in state.items()})


class SemanticPrototypeBuilder:
    """Aggregate prompt ensembles and derive normal-relative directions."""

    def __init__(self, text_encoder: TextEncoder):
        self.text_encoder = text_encoder

    def _build_single(self, prompts: Sequence[str]) -> torch.Tensor:
        embeddings = self.text_encoder.encode_texts(prompts).float()
        return functional.normalize(embeddings.mean(dim=0), dim=0)

    @staticmethod
    def _direction(
        target: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        return functional.normalize(target - normal, dim=0)

    def build(self, prompts: dict[str, Sequence[str]]) -> SemanticPrototypes:
        """Build fire, normal, smoke, flame, and cloud prototypes."""
        embeddings = {
            name: self._build_single(prompts[name])
            for name in ("fire", "normal", "smoke", "flame", "cloud")
        }
        normal = embeddings["normal"]
        return SemanticPrototypes(
            fire_embedding=embeddings["fire"],
            normal_embedding=normal,
            contrast_embedding=self._direction(embeddings["fire"], normal),
            smoke_embedding=embeddings["smoke"],
            flame_embedding=embeddings["flame"],
            cloud_embedding=embeddings["cloud"],
            smoke_direction=self._direction(embeddings["smoke"], normal),
            flame_direction=self._direction(embeddings["flame"], normal),
            cloud_direction=self._direction(embeddings["cloud"], normal),
        )
