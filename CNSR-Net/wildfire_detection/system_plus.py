"""Optional CNSR-Net Plus model with learned attention aggregation."""

from __future__ import annotations

from pathlib import Path

import torch

from wildfire_detection.attention_conformal import (
    SpatialAttentionAggregator,
    load_phase4_config,
)
from wildfire_detection.system import _AggregationResult, _BaseCNSRNet
from wildfire_detection.variants import PLUS_METHOD, PLUS_VARIANT


class CNSRNetPlus(_BaseCNSRNet):
    """Run the former Full model as the optional learned-aggregation variant."""

    def __init__(
        self,
        config_path: str | Path,
        trained_model_path: str | Path,
        device: str | torch.device = "cuda",
    ):
        super().__init__(
            config_path=config_path,
            artifact_path=trained_model_path,
            calibration_method=PLUS_METHOD,
            model_variant=PLUS_VARIANT,
            device=device,
        )
        phase4_config = load_phase4_config(
            self.config["components"]["attention_conformal"]
        )
        scales = self.spatial_matcher.pyramid_scales
        self.attention_model = SpatialAttentionAggregator(
            scales,
            phase4_config.attention,
        ).to(self.device)
        self.attention_model.load_state_dict(
            self._artifact["state_dict"],
            strict=True,
        )
        self.attention_model.eval()

    def _aggregate(
        self,
        evidence: dict[str, torch.Tensor],
        persistence: dict[str, torch.Tensor],
    ) -> _AggregationResult:
        with torch.inference_mode():
            output = self.attention_model(
                {
                    name: values.unsqueeze(0).to(self.device)
                    for name, values in evidence.items()
                },
                {
                    name: values.unsqueeze(0).to(self.device)
                    for name, values in persistence.items()
                },
            )
        heatmap = {
            name: weights[0].detach().float().cpu()
            for name, weights in output.attention_weights.items()
        }
        return _AggregationResult(
            score=float(output.probabilities.item()),
            attention_heatmap=heatmap,
        )
