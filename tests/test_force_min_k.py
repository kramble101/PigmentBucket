from __future__ import annotations

import numpy as np

from src.pipeline.analyzer import Analyzer, AnalyzerConfig
from src.pipeline.types import AnalysisResult, ClipAnalysis, ClipContext, ClipFeatures, ClusterResult, FrameSample


def _build_contexts() -> list[ClipContext]:
    return [
        ClipContext(
            clip_id="clip-1",
            clip_name="Clip 1",
            duration_seconds=1.0,
            media_path=None,
            timeline_in="00:00:00:00",
            timeline_out="00:00:01:00",
        ),
        ClipContext(
            clip_id="clip-2",
            clip_name="Clip 2",
            duration_seconds=1.0,
            media_path=None,
            timeline_in="00:00:01:00",
            timeline_out="00:00:02:00",
        ),
    ]


def _wire_fake_pipeline(analyzer: Analyzer, vector: np.ndarray) -> None:
    def fake_sample(clip: ClipContext) -> list[FrameSample]:
        return [
            FrameSample(
                clip_id=clip.clip_id,
                frame_index=0,
                timestamp_seconds=0.0,
                data=np.zeros((4, 4, 3), dtype=np.uint8),
            )
        ]

    def fake_extract(clip_id: str, samples) -> ClipFeatures:
        return ClipFeatures(
            clip_id=clip_id,
            vector=vector.copy(),
            metadata={"feature_dim": vector.shape[0], "frame_count": len(samples)},
        )

    def fake_cluster(features) -> ClusterResult:
        ids = [feature.clip_id for feature in features]
        return ClusterResult(labels={clip_id: 0 for clip_id in ids}, chosen_k=1, silhouette=None)

    analyzer._sampler.sample = fake_sample  # type: ignore[attr-defined]
    analyzer._feature_extractor.extract = fake_extract  # type: ignore[attr-defined]
    analyzer._cluster_with_random_state = lambda feats, _: fake_cluster(feats)  # type: ignore[attr-defined]


def test_force_min_k_skips_coloring_when_disabled(tmp_path) -> None:
    config = AnalyzerConfig(force_min_k=1, ignore_cache=True, cache_dir=tmp_path / "cache")
    analyzer = Analyzer(config)
    _wire_fake_pipeline(analyzer, np.array([0.0, 0.0], dtype=np.float32))

    result = analyzer.analyze(_build_contexts())
    assert isinstance(result, AnalysisResult)
    assert result.summary["coloring"]["coloring"] == "skipped"
    assert result.summary["clusters"] == 1


def test_force_min_k_splits_when_enabled(tmp_path) -> None:
    config = AnalyzerConfig(force_min_k=2, ignore_cache=True, cache_dir=tmp_path / "cache")
    analyzer = Analyzer(config)
    _wire_fake_pipeline(analyzer, np.array([0.0, 0.0], dtype=np.float32))

    result = analyzer.analyze(_build_contexts())
    assert result.summary["coloring"]["coloring"] == "forced"
    labels = {clip.clip_id: clip.cluster_id for clip in result.clips}
    assert set(labels.values()) == {0, 1}
    assert result.summary["chosen_k"] == 2
