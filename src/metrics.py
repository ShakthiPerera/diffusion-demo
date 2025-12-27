"""Precision, recall, density, and coverage metrics (PRDC)."""

from __future__ import annotations

import numpy as np
try:
    from sklearn.metrics import pairwise_distances  # type: ignore
except Exception:
    import numpy as _np

    def pairwise_distances(X: _np.ndarray, Y: _np.ndarray | None = None, metric: str = "euclidean", n_jobs: int = 1) -> _np.ndarray:  # noqa: N802
        if metric != "euclidean":
            raise ValueError("Only 'euclidean' metric is supported in fallback")
        if Y is None:
            Y = X
        diff = X[:, None, :] - Y[None, :, :]
        return _np.sqrt(_np.sum(diff * diff, axis=-1, dtype=_np.float64))

__all__ = ["compute_prdc"]


def _get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    indices = np.argpartition(unsorted, k, axis=axis)[..., : k + 1]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    return k_smallests.max(axis=axis)


def _compute_nearest_neighbour_distances(features: np.ndarray, nearest_k: int) -> np.ndarray:
    distances = pairwise_distances(features, metric="euclidean", n_jobs=1)
    return _get_kth_value(distances, k=nearest_k, axis=-1)


def compute_prdc(real_features: np.ndarray, fake_features: np.ndarray, nearest_k: int = 5) -> dict[str, float]:
    if nearest_k < 1:
        raise ValueError("nearest_k must be a positive integer")
    real_radii = _compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_radii = _compute_nearest_neighbour_distances(fake_features, nearest_k)
    dist_real_fake = pairwise_distances(real_features, fake_features, metric="euclidean", n_jobs=1)
    precision_mask = dist_real_fake < real_radii[:, np.newaxis]
    precision = precision_mask.any(axis=0).mean()
    recall_mask = dist_real_fake < fake_radii[np.newaxis, :]
    recall = recall_mask.any(axis=1).mean()
    density = (1.0 / float(nearest_k)) * precision_mask.sum(axis=0).mean()
    min_dist = dist_real_fake.min(axis=1)
    coverage = (min_dist < real_radii).mean()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }
