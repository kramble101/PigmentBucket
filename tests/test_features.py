from __future__ import annotations

import numpy as np

from src.pipeline.features import FeatureExtractor, FeatureExtractorConfig
from src.pipeline.types import FrameSample


def test_feature_extractor_dimension_and_determinism() -> None:
    extractor = FeatureExtractor(FeatureExtractorConfig(histogram_bins=8))
    frame_a = np.full((12, 12, 3), 128, dtype=np.uint8)
    frame_b = np.full((12, 12, 3), 64, dtype=np.uint8)
    samples = [
        FrameSample(clip_id="clip", frame_index=0, timestamp_seconds=0.0, data=frame_a),
        FrameSample(clip_id="clip", frame_index=1, timestamp_seconds=0.5, data=frame_b),
    ]

    features_first = extractor.extract("clip", samples)
    features_second = extractor.extract("clip", samples)

    # 3(mean RGB) + 3(std RGB) + 3(mean HSV) + 3*8(hist bins) = 42
    assert features_first.vector.shape[0] == 42
    assert features_first.metadata["frame_count"] == len(samples)
    np.testing.assert_allclose(features_first.vector, features_second.vector)
