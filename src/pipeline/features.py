"""Feature extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .types import ClipFeatures, FrameSample

np.random.seed(42)


class FeatureExtractionError(RuntimeError):
    """Raised when features cannot be computed for a clip."""


@dataclass
class FeatureExtractorConfig:
    histogram_bins: int = 8


class FeatureExtractor:
    """Transforms frame samples into deterministic feature vectors."""

    def __init__(self, config: FeatureExtractorConfig | None = None) -> None:
        self._config = config or FeatureExtractorConfig()
        self._hist_bins = max(2, self._config.histogram_bins)

    def extract(self, clip_id: str, samples: Iterable[FrameSample]) -> ClipFeatures:
        frames = [sample.data for sample in samples if sample.data is not None]
        if not frames:
            raise FeatureExtractionError("No sampled frames with pixel data")

        scalar_features: List[np.ndarray] = []
        hist_features: List[np.ndarray] = []
        for frame in frames:
            scalar, hist = self._frame_features(frame)
            scalar_features.append(scalar)
            hist_features.append(hist)

        scalar_matrix = np.stack(scalar_features, axis=0)
        hist_matrix = np.stack(hist_features, axis=0)

        scalar_mean = np.mean(scalar_matrix, axis=0)
        scalar_median = np.median(scalar_matrix, axis=0)
        hist_mean = np.mean(hist_matrix, axis=0)

        feature_vector = np.concatenate([scalar_mean, scalar_median, hist_mean]).astype(np.float32)
        metadata = {
            "feature_dim": int(feature_vector.shape[0]),
            "frame_count": len(frames),
        }
        return ClipFeatures(clip_id=clip_id, vector=feature_vector, metadata=metadata)

    # ------------------------------------------------------------------
    def _frame_features(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise FeatureExtractionError("Expected RGB frame with 3 channels")

        rgb = frame.astype(np.float32) / 255.0
        rgb_flat = rgb.reshape(-1, 3)
        mean_rgb = rgb_flat.mean(axis=0)
        std_rgb = rgb_flat.std(axis=0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 179.0  # Hue range in OpenCV
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0
        hsv_flat = hsv.reshape(-1, 3)
        mean_hsv = hsv_flat.mean(axis=0)

        hist_channels = []
        for channel in range(3):
            hist, _ = np.histogram(frame[:, :, channel], bins=self._hist_bins, range=(0, 256))
            hist = hist.astype(np.float32)
            total = float(hist.sum())
            if total > 0:
                hist /= total
            hist_channels.append(hist)
        hist_vector = np.concatenate(hist_channels)

        scalar_vector = np.concatenate([mean_rgb, std_rgb, mean_hsv])
        return scalar_vector.astype(np.float32), hist_vector
