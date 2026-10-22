"""Frozen visual encoder used by CNSR-Net for local patch embeddings."""

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn.functional as functional


class FrozenPatchEncoder:
    """Encode normalized image patches with the frozen CLIP/Crane ViT-L."""

    def __init__(self, model: torch.nn.Module, input_resolution: int = 336):
        self.model = model
        self.input_resolution = input_resolution
        self.feature_dim = int(model.visual.output_dim)
        self.hidden_dim = int(model.visual.embed_dim)
        self.token_grid_size = int(
            (model.visual.positional_embedding.shape[0] - 1) ** 0.5
        )
        self.transformer_layers = len(model.visual.transformer.resblocks)

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        image_features, _ = self.model.encode_image(patches, feature_list=[])
        return functional.normalize(image_features.float(), dim=-1)

    def encode_patches_with_tokens(
        self,
        patches: torch.Tensor,
        feature_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized CLS features and one projected token grid."""
        image_features, layer_features = self.model.encode_image(
            patches,
            feature_list=[feature_layer],
        )
        tokens = layer_features[0]
        patch_count = tokens.shape[1] - 1
        grid_size = int(patch_count**0.5)
        if grid_size * grid_size != patch_count:
            raise ValueError(
                "Expected the selected CLIP layer to contain CLS plus a "
                "square patch-token grid"
            )
        token_grid = tokens[:, 1:].reshape(
            tokens.shape[0],
            grid_size,
            grid_size,
            tokens.shape[-1],
        )
        return (
            functional.normalize(image_features.float(), dim=-1),
            token_grid.float(),
        )

    def encode_patches_to_layer(
        self,
        patches: torch.Tensor,
        split_layer: int,
    ) -> torch.Tensor:
        """Run CLIP patch embedding and the first ``split_layer`` blocks."""
        if not 0 <= split_layer <= self.transformer_layers:
            raise ValueError(
                f"split_layer must be in [0, {self.transformer_layers}]"
            )
        visual = self.model.visual
        tokens = visual.conv1(patches.type(self.model.dtype))
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
        tokens = tokens.permute(0, 2, 1)
        cls = visual.class_embedding.to(tokens.dtype).expand(
            tokens.shape[0], 1, -1
        )
        tokens = torch.cat((cls, tokens), dim=1)
        grid_size = int((tokens.shape[1] - 1) ** 0.5)
        visual._check_interpolate_token_emebeddings(grid_size)
        tokens = visual.ln_pre(
            tokens + visual.positional_embedding.to(tokens.dtype)
        )
        tokens = tokens.permute(1, 0, 2)
        for block in visual.transformer.resblocks[:split_layer]:
            tokens = block(tokens)
            if isinstance(tokens, list):
                raise ValueError(
                    "Late-block encoding requires the standard CLIP visual path"
                )
        return tokens.permute(1, 0, 2)

    def encode_tokens_from_layer(
        self,
        tokens: torch.Tensor,
        split_layer: int,
    ) -> torch.Tensor:
        """Resume CLIP at ``split_layer`` and return normalized CLS features."""
        if not 0 <= split_layer <= self.transformer_layers:
            raise ValueError(
                f"split_layer must be in [0, {self.transformer_layers}]"
            )
        visual = self.model.visual
        hidden = tokens.permute(1, 0, 2)
        for block in visual.transformer.resblocks[split_layer:]:
            hidden = block(hidden)
            if isinstance(hidden, list):
                raise ValueError(
                    "Late-block encoding requires the standard CLIP visual path"
                )
        hidden = visual.ln_post(hidden.permute(1, 0, 2))
        projected = hidden @ visual.proj
        return functional.normalize(projected[:, 0].float(), dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        from models.prompt_ensemble import tokenize

        device = next(self.model.parameters()).device
        tokens = tokenize(list(texts)).to(device)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens)
        return functional.normalize(text_features.float(), dim=-1)


def load_frozen_patch_encoder(
    checkpoint_path: str,
    device: str | torch.device = "cuda",
    input_resolution: int = 336,
) -> FrozenPatchEncoder:
    """Load the cited Crane implementation as CNSR-Net's frozen encoder."""
    import models

    runtime_args = SimpleNamespace(
        dino_model="none",
        both_eattn_dattn=True,
        image_size=input_resolution,
    )
    design_details = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": 9,
        "learnabel_text_embedding_length": 4,
        "others": runtime_args,
    }
    model, _ = models.load(
        str(Path(checkpoint_path).expanduser()),
        device=device,
        design_details=design_details,
    )
    model.eval()
    model.requires_grad_(False)
    return FrozenPatchEncoder(model, input_resolution=input_resolution)
