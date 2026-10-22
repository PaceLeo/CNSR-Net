"""Timestamped per-camera evidence buffers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalEvidenceFrame:
    timestamp: float
    evidence: dict[str, torch.Tensor]


class TemporalEvidenceBuffer:
    """Retain evidence observations inside a physical time window."""

    def __init__(self, window_seconds: float):
        self.window_seconds = window_seconds
        self.camera_buffers: dict[str, deque[TemporalEvidenceFrame]] = {}

    def reset(self, camera_id: str) -> None:
        self.camera_buffers[camera_id] = deque()

    def update(
        self,
        camera_id: str,
        timestamp: float,
        evidence: dict[str, torch.Tensor],
    ) -> tuple[TemporalEvidenceFrame, ...]:
        buffer = self.camera_buffers.setdefault(camera_id, deque())
        if buffer and timestamp <= buffer[-1].timestamp:
            raise ValueError(f"Timestamps for {camera_id} must increase")

        buffer.append(
            TemporalEvidenceFrame(
                timestamp=float(timestamp),
                evidence={name: values.detach() for name, values in evidence.items()},
            )
        )
        cutoff = timestamp - self.window_seconds
        while buffer and buffer[0].timestamp <= cutoff:
            buffer.popleft()
        return tuple(buffer)

    def history(self, camera_id: str) -> tuple[TemporalEvidenceFrame, ...]:
        return tuple(self.camera_buffers.get(camera_id, ()))
