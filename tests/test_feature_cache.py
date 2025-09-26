from __future__ import annotations

from pathlib import Path

import numpy as np

from src.pipeline.analyzer import Analyzer, AnalyzerConfig
from src.pipeline.types import ClipContext, ClipFeatures, FrameSample


def _make_context(media_path: Path) -> ClipContext:
    return ClipContext(
        clip_id="clip-cache",
        clip_name="Clip Cache",
        duration_seconds=1.0,
        media_path=str(media_path),
        timeline_in="00:00:00:00",
        timeline_out="00:00:01:00",
    )


def test_feature_cache_hit_and_miss(tmp_path) -> None:
    media_file = tmp_path / "media.mov"
    media_file.write_bytes(b"stub")

    config = AnalyzerConfig(force_min_k=1, ignore_cache=False, cache_dir=tmp_path / "cache")
    analyzer = Analyzer(config)

    call_counts = {"sample": 0, "extract": 0}

    def fake_sample(_clip: ClipContext) -> list[FrameSample]:
        call_counts["sample"] += 1
        return [
            FrameSample(
                clip_id=_clip.clip_id,
                frame_index=0,
                timestamp_seconds=0.0,
                data=np.zeros((4, 4, 3), dtype=np.uint8),
            )
        ]

    def fake_extract(clip_id: str, samples) -> ClipFeatures:
        call_counts["extract"] += 1
        return ClipFeatures(
            clip_id=clip_id,
            vector=np.array([1.0, 2.0], dtype=np.float32),
            metadata={"feature_dim": 2, "frame_count": len(samples)},
        )

    analyzer._sampler.sample = fake_sample  # type: ignore[attr-defined]
    analyzer._feature_extractor.extract = fake_extract  # type: ignore[attr-defined]

    contexts = [_make_context(media_file)]

    result_first = analyzer.analyze(contexts)
    assert call_counts == {"sample": 1, "extract": 1}
    assert result_first.summary.get("cache_hits", 0) == 0

    call_counts.update({"sample": 0, "extract": 0})
    result_second = analyzer.analyze(contexts)
    assert call_counts == {"sample": 0, "extract": 0}
    assert result_second.summary.get("cache_hits") == 1
    assert result_second.clips[0].features.metadata.get("cache_hit") is True

    # When ignore_cache is requested, sampling/extraction should happen again
    config_ignore = AnalyzerConfig(force_min_k=1, ignore_cache=True, cache_dir=tmp_path / "cache")
    analyzer_ignore = Analyzer(config_ignore)
    analyzer_ignore._sampler.sample = fake_sample  # type: ignore[attr-defined]
    analyzer_ignore._feature_extractor.extract = fake_extract  # type: ignore[attr-defined]
    call_counts.update({"sample": 0, "extract": 0})
    analyzer_ignore.analyze(contexts)
    assert call_counts == {"sample": 1, "extract": 1}
