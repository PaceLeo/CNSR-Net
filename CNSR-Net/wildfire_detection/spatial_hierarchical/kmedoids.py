"""A small, NumPy 2 compatible PAM K-Medoids implementation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMedoidsResult:
    medoid_indices: np.ndarray
    medoids: np.ndarray
    labels: np.ndarray
    inertia: float


def _pairwise_euclidean(features: np.ndarray) -> np.ndarray:
    differences = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(differences, axis=-1)


def _build_initial_medoids(distances: np.ndarray, k: int) -> list[int]:
    first_medoid = int(np.argmin(distances.sum(axis=1)))
    medoids = [first_medoid]
    nearest_distances = distances[:, first_medoid]

    while len(medoids) < k:
        best_candidate = -1
        best_cost = np.inf
        for candidate in range(distances.shape[0]):
            if candidate in medoids:
                continue
            candidate_cost = np.minimum(
                nearest_distances,
                distances[:, candidate],
            ).sum()
            if candidate_cost < best_cost:
                best_candidate = candidate
                best_cost = candidate_cost
        medoids.append(best_candidate)
        nearest_distances = np.minimum(
            nearest_distances,
            distances[:, best_candidate],
        )
    return medoids


def pam_kmedoids(
    features: np.ndarray,
    k: int,
    max_iterations: int = 100,
) -> KMedoidsResult:
    """Cluster features with deterministic BUILD plus PAM swap refinement."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("K-Medoids features must have shape [samples, features]")
    if not 1 <= k <= features.shape[0]:
        raise ValueError(
            f"K-Medoids requires 1 <= k <= samples, got k={k}, "
            f"samples={features.shape[0]}"
        )

    distances = _pairwise_euclidean(features)
    medoids = _build_initial_medoids(distances, k)
    current_cost = float(distances[:, medoids].min(axis=1).sum())

    for _ in range(max_iterations):
        best_cost = current_cost
        best_swap: tuple[int, int] | None = None
        medoid_set = set(medoids)
        for medoid_position in range(k):
            for candidate in range(features.shape[0]):
                if candidate in medoid_set:
                    continue
                trial_medoids = medoids.copy()
                trial_medoids[medoid_position] = candidate
                trial_cost = float(
                    distances[:, trial_medoids].min(axis=1).sum()
                )
                if trial_cost < best_cost - 1e-7:
                    best_cost = trial_cost
                    best_swap = medoid_position, candidate

        if best_swap is None:
            break
        medoids[best_swap[0]] = best_swap[1]
        current_cost = best_cost

    medoid_indices = np.asarray(medoids, dtype=np.int64)
    labels = np.argmin(distances[:, medoid_indices], axis=1)
    return KMedoidsResult(
        medoid_indices=medoid_indices,
        medoids=features[medoid_indices].copy(),
        labels=labels,
        inertia=current_cost,
    )
