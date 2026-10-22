"""Camera-window conformal calibration for dependent frame sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass
class CameraWindowConformalCalibrator:
    """Calibrate the maximum normal-window excess across cameras."""

    alpha: float
    global_margin: float | None = None
    camera_baselines: dict[str, float] = field(default_factory=dict)
    camera_thresholds: dict[str, float] = field(default_factory=dict)
    calibration_excesses: dict[str, float] = field(default_factory=dict)
    order_statistic: int | None = None

    @staticmethod
    def camera_baseline(threshold_fit_scores: Sequence[float]) -> float:
        return max(float(score) for score in threshold_fit_scores)

    def fit(
        self,
        threshold_fit_scores: Mapping[str, Sequence[float]],
        negative_window_scores: Mapping[str, Sequence[float]],
    ) -> float:
        """Fit one nonconformity statistic per calibration camera."""
        camera_ids = sorted(threshold_fit_scores)
        if camera_ids != sorted(negative_window_scores):
            raise ValueError("Calibration camera sets must match")
        excesses = {}
        for camera_id in camera_ids:
            baseline = self.camera_baseline(threshold_fit_scores[camera_id])
            excesses[camera_id] = (
                max(float(score) for score in negative_window_scores[camera_id])
                - baseline
            )

        return self.fit_excesses(excesses)

    def fit_excesses(self, excesses: Mapping[str, float]) -> float:
        """Fit from one pre-aggregated nonconformity value per camera."""
        excesses = {
            str(camera_id): float(value)
            for camera_id, value in excesses.items()
        }

        sample_count = len(excesses)
        order = math.ceil((sample_count + 1) * (1.0 - self.alpha))
        self.order_statistic = order
        self.calibration_excesses = excesses
        if order > sample_count:
            self.global_margin = math.inf
        else:
            self.global_margin = sorted(excesses.values())[order - 1]
        return self.global_margin

    def calibrate_camera(
        self,
        camera_id: str,
        threshold_fit_scores: Sequence[float],
    ) -> float:
        if self.global_margin is None:
            raise ValueError("Camera-window calibrator is not fitted")
        baseline = self.camera_baseline(threshold_fit_scores)
        threshold = baseline + self.global_margin
        self.camera_baselines[camera_id] = baseline
        self.camera_thresholds[camera_id] = threshold
        return threshold

    def predict(self, camera_id: str, score: float) -> bool:
        return float(score) > self.camera_thresholds[camera_id]

    def finite_sample_false_alarm_bound(self) -> float:
        if self.order_statistic is None:
            raise ValueError("Camera-window calibrator is not fitted")
        sample_count = len(self.calibration_excesses)
        return (sample_count + 1 - self.order_statistic) / (
            sample_count + 1
        )
