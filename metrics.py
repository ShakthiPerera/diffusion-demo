"""
Precision, recall, density, and coverage metrics for generative models.

This module provides a pure Python implementation of the PRDC metrics defined
in *Evaluating Precision and Recall for Generative Models* (Sajjadi et al.,
2018) and used in [1].  The implementation does not depend on any external
packages beyond NumPy and scikit‑learn for computing pairwise distances.  It
operates directly on raw data or feature embeddings.

The four metrics quantify different aspects of how well a set of generated
samples ``fake_features`` approximates a set of ground‑truth samples
``real_features``:

``precision``
    The fraction of generated samples that fall within the manifold of real
    samples.  Higher is better.

``recall``
    The fraction of real samples that are covered by the generated samples.
    Higher is better.

``density``
    Average number of real neighbours contained within the ``k``‑radius of
    generated points, scaled by ``1/k``.

``coverage``
    Fraction of real samples whose nearest generated sample lies within the
    real sample’s ``k``‑th nearest neighbour radius.

Example
-------

>>> from metrics import compute_prdc
>>> import numpy as np
>>> real = np.random.randn(1000, 2).astype(np.float32)
>>> fake = np.random.randn(1000, 2).astype(np.float32)
>>> metrics = compute_prdc(real, fake, nearest_k=5)
>>> print(metrics['precision'], metrics['recall'])

References
----------

[1] Kynkäänniemi, T., Karras, T., Lehtinen, J., & Aila, T. (2022).
    The Role of Precision, Recall, and Density in Generative Models.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

__all__ = ['compute_prdc']


def _get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Return the k‑th smallest values along the given axis.

    Uses ``np.argpartition`` to avoid fully sorting the array.

    Parameters
    ----------
    unsorted : ndarray
        Input array from which to extract k‑th smallest values.
    k : int
        The k index (0–based) indicating which smallest value to retrieve.
    axis : int, optional
        The axis along which to operate.  Default is the last axis.

    Returns
    -------
    ndarray
        Array of k‑th smallest values.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k + 1]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def _compute_nearest_neighbour_distances(features: np.ndarray, nearest_k: int) -> np.ndarray:
    """Compute distances to the k‑th nearest neighbour for each point.

    Parameters
    ----------
    features : ndarray of shape (n_samples, n_features)
        Feature vectors.
    nearest_k : int
        The k index used for the neighbourhood (i.e. the number of nearest
        neighbours to consider).

    Returns
    -------
    ndarray of shape (n_samples,)
        Distances from each point to its k‑th nearest neighbour in the same set.
    """
    # pairwise distances within the feature set
    distances = pairwise_distances(features, metric='euclidean', n_jobs=1)
    # k+1 because the nearest neighbour of a point is itself with distance 0
    radii = _get_kth_value(distances, k=nearest_k, axis=-1)
    return radii


def compute_prdc(real_features: np.ndarray, fake_features: np.ndarray, nearest_k: int = 5) -> dict[str, float]:
    """Compute precision, recall, density, and coverage metrics.

    Parameters
    ----------
    real_features : ndarray of shape (n_real, n_features)
        Feature representations of real samples.
    fake_features : ndarray of shape (n_fake, n_features)
        Feature representations of generated samples.
    nearest_k : int, optional
        Number of nearest neighbours to consider.  Default is 5.

    Returns
    -------
    dict
        Mapping with keys ``'precision'``, ``'recall'``, ``'density'``, and
        ``'coverage'``.
    """
    if nearest_k < 1:
        raise ValueError('nearest_k must be a positive integer')
    # compute radii for real and fake sets
    real_radii = _compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_radii = _compute_nearest_neighbour_distances(fake_features, nearest_k)
    # distances between real and fake samples
    dist_real_fake = pairwise_distances(real_features, fake_features, metric='euclidean', n_jobs=1)
    # precision: fraction of fake samples within some real radius
    precision_mask = dist_real_fake < real_radii[:, np.newaxis]
    precision = precision_mask.any(axis=0).mean()
    # recall: fraction of real samples within some fake radius
    recall_mask = dist_real_fake < fake_radii[np.newaxis, :]
    recall = recall_mask.any(axis=1).mean()
    # density: average number of real neighbours within the real radii around fake points
    density = (1.0 / float(nearest_k)) * precision_mask.sum(axis=0).mean()
    # coverage: fraction of real samples whose nearest fake neighbour lies within the real radius
    min_dist = dist_real_fake.min(axis=1)
    coverage = (min_dist < real_radii).mean()
    return {
        'precision': float(precision),
        'recall': float(recall),
        'density': float(density),
        'coverage': float(coverage),
    }
