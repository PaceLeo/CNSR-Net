"""Tensor-backed camera prototype banks."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ScalePrototypeBank:
    scale_name: str
    grid_size: tuple[int, int]
    prototypes: torch.Tensor
    distance_mean: torch.Tensor | None = None
    distance_std: torch.Tensor | None = None


@dataclass
class CameraPrototypeBank:
    camera_id: str
    scales: dict[str, ScalePrototypeBank]

    def state_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "scales": {
                name: {
                    "scale_name": bank.scale_name,
                    "grid_size": list(bank.grid_size),
                    "prototypes": bank.prototypes.cpu(),
                    "distance_mean": (
                        None
                        if bank.distance_mean is None
                        else bank.distance_mean.cpu()
                    ),
                    "distance_std": (
                        None
                        if bank.distance_std is None
                        else bank.distance_std.cpu()
                    ),
                }
                for name, bank in self.scales.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "CameraPrototypeBank":
        scales = {
            name: ScalePrototypeBank(
                scale_name=item["scale_name"],
                grid_size=(int(item["grid_size"][0]), int(item["grid_size"][1])),
                prototypes=item["prototypes"].float().cpu(),
                distance_mean=(
                    None
                    if item["distance_mean"] is None
                    else item["distance_mean"].float().cpu()
                ),
                distance_std=(
                    None
                    if item["distance_std"] is None
                    else item["distance_std"].float().cpu()
                ),
            )
            for name, item in state["scales"].items()
        }
        return cls(camera_id=state["camera_id"], scales=scales)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "CameraPrototypeBank":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_state_dict(state)
