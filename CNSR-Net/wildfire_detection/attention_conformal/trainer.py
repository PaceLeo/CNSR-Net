"""Balanced supervised training for the spatial attention aggregator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional

from .attention import SpatialAttentionAggregator
from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionSample:
    evidence_maps: dict[str, torch.Tensor]
    persistence_maps: dict[str, torch.Tensor]
    label: float


@dataclass(frozen=True)
class TrainingHistory:
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int


class AttentionTrainer:
    """Train with BCE logits, sparse-attention entropy, and early stopping."""

    def __init__(
        self,
        model: SpatialAttentionAggregator,
        config: AttentionConfig,
    ):
        self.model = model
        self.config = config

    @staticmethod
    def _batch(
        samples: Sequence[AttentionSample],
        indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        selected = [samples[int(index)] for index in indices]
        scale_names = selected[0].evidence_maps.keys()
        evidence = {
            name: torch.stack(
                [sample.evidence_maps[name].float() for sample in selected]
            ).to(device)
            for name in scale_names
        }
        persistence = {
            name: torch.stack(
                [sample.persistence_maps[name].float() for sample in selected]
            ).to(device)
            for name in scale_names
        }
        labels = torch.tensor(
            [sample.label for sample in selected],
            dtype=torch.float32,
            device=device,
        )
        return evidence, persistence, labels

    @staticmethod
    def _horizontal_flip(
        evidence: dict[str, torch.Tensor],
        persistence: dict[str, torch.Tensor],
        probability: float,
        generator: torch.Generator,
    ) -> None:
        batch_size = next(iter(evidence.values())).shape[0]
        flip_mask = torch.rand(batch_size, generator=generator) < probability
        for maps in (evidence, persistence):
            for values in maps.values():
                device_mask = flip_mask.to(values.device)
                values[device_mask] = values[device_mask].flip(dims=(-1,))

    @staticmethod
    def _positive_weight(
        samples: Sequence[AttentionSample],
        device: torch.device,
    ) -> torch.Tensor:
        positives = sum(sample.label == 1.0 for sample in samples)
        negatives = len(samples) - positives
        return torch.tensor(negatives / positives, device=device)

    def _validation_loss(
        self,
        samples: Sequence[AttentionSample],
        device: torch.device,
    ) -> float:
        self.model.eval()
        losses = []
        positive_weight = self._positive_weight(samples, device)
        with torch.inference_mode():
            for start in range(0, len(samples), self.config.batch_size):
                indices = torch.arange(
                    start,
                    min(start + self.config.batch_size, len(samples)),
                )
                evidence, persistence, labels = self._batch(
                    samples, indices, device
                )
                output = self.model(evidence, persistence)
                losses.append(
                    functional.binary_cross_entropy_with_logits(
                        output.logits,
                        labels,
                        pos_weight=positive_weight,
                    ).item()
                )
        return sum(losses) / len(losses)

    def fit(
        self,
        training_samples: Sequence[AttentionSample],
        validation_samples: Sequence[AttentionSample],
        device: str | torch.device,
    ) -> TrainingHistory:
        device = torch.device(device)
        self.model.to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        positive_weight = self._positive_weight(training_samples, device)
        generator = torch.Generator().manual_seed(self.config.seed)
        training_losses = []
        validation_losses = []
        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_losses = []
            permutation = torch.randperm(
                len(training_samples), generator=generator
            )
            for start in range(
                0, len(training_samples), self.config.batch_size
            ):
                indices = permutation[start : start + self.config.batch_size]
                evidence, persistence, labels = self._batch(
                    training_samples, indices, device
                )
                self._horizontal_flip(
                    evidence,
                    persistence,
                    self.config.horizontal_flip_probability,
                    generator,
                )
                output = self.model(evidence, persistence)
                classification_loss = (
                    functional.binary_cross_entropy_with_logits(
                        output.logits,
                        labels,
                        pos_weight=positive_weight,
                    )
                )
                entropy = self.model.attention_entropy(
                    output.flat_attention_weights
                )
                loss = (
                    classification_loss
                    + self.config.entropy_weight * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            training_losses.append(sum(epoch_losses) / len(epoch_losses))
            validation_loss = self._validation_loss(
                validation_samples, device
            )
            validation_losses.append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    epochs_without_improvement
                    >= self.config.early_stopping_patience
                ):
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return TrainingHistory(
            training_loss=tuple(training_losses),
            validation_loss=tuple(validation_losses),
            best_epoch=best_epoch,
        )
