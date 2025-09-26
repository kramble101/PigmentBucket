"""Clustering utilities using KMeans + silhouette selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .types import ClipFeatures, ClusterResult


@dataclass
class ClustererConfig:
    max_k: int = 8
    random_state: int = 42
    silhouette_sample_size: int = 1000


class Clusterer:
    """Abstract clusterer."""

    def cluster(self, features: Iterable[ClipFeatures]) -> ClusterResult:
        raise NotImplementedError


class KMeansClusterer(Clusterer):
    """KMeans clusterer with auto-K selection via silhouette score."""

    def __init__(self, config: ClustererConfig | None = None) -> None:
        self._config = config or ClustererConfig()

    def cluster(self, features: Iterable[ClipFeatures]) -> ClusterResult:  # noqa: D401
        feature_list = list(features)
        if not feature_list:
            return ClusterResult(labels={}, chosen_k=0, silhouette=None)

        ids = [cf.clip_id for cf in feature_list]
        matrix = np.stack([cf.vector for cf in feature_list])
        n_samples = matrix.shape[0]

        if n_samples <= 1:
            labels = {ids[0]: 0} if ids else {}
            return ClusterResult(labels=labels, chosen_k=1, silhouette=None)

        max_k = max(2, self._config.max_k)
        max_k = min(max_k, n_samples)
        best_k = 2
        best_score = -1.0
        best_labels = None

        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=self._config.random_state, n_init=10)
            assignments = model.fit_predict(matrix)
            score = self._compute_silhouette(matrix, assignments)
            if score is None:
                chosen = assignments
                best_k = k
                best_labels = assignments
                best_score = None
                break
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = assignments

        if best_labels is None:
            # fallback when silhouette could not be computed
            model = KMeans(n_clusters=best_k, random_state=self._config.random_state, n_init=10)
            best_labels = model.fit_predict(matrix)

        label_map = {clip_id: int(label) for clip_id, label in zip(ids, best_labels)}
        silhouette_value = best_score if best_score is not None and best_score >= 0 else None
        return ClusterResult(labels=label_map, chosen_k=best_k, silhouette=silhouette_value)

    def _compute_silhouette(self, matrix: np.ndarray, labels: np.ndarray) -> float | None:
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return None
        sample_size = min(self._config.silhouette_sample_size, matrix.shape[0])
        if sample_size < 2:
            return None
        try:
            kwargs = {"random_state": self._config.random_state}
            if sample_size < matrix.shape[0]:
                kwargs["sample_size"] = sample_size
            return float(silhouette_score(matrix, labels, **kwargs))
        except ValueError:
            return None
