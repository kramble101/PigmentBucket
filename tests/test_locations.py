from __future__ import annotations

import numpy as np

from src.pipeline.analyzer import Analyzer, AnalyzerConfig
from src.pipeline.locations import LocationClip, LocationConfig, LocationGrouper
from src.pipeline.types import (
    ClipContext as PipelineClipContext,
    ClipFeatures,
    ClusterResult,
    FrameSample,
)


def test_consecutive_clusters_form_distinct_locations() -> None:
    clips = [
        LocationClip("c1", cluster_id=0, duration_seconds=5.0, features=[1.0, 0.0]),
        LocationClip("c2", cluster_id=0, duration_seconds=4.0, features=[1.0, 0.1]),
        LocationClip("c3", cluster_id=1, duration_seconds=6.0, features=[0.0, 1.0]),
        LocationClip("c4", cluster_id=1, duration_seconds=2.5, features=[0.0, 0.9]),
    ]

    result = LocationGrouper(LocationConfig(min_duration_sec=3.0)).assign(clips)

    assert result.assignments == {
        "c1": 0,
        "c2": 0,
        "c3": 1,
        "c4": 1,
    }
    assert result.location_count == 2


def test_short_leading_segment_merges_into_next_location() -> None:
    clips = [
        LocationClip("c1", cluster_id=5, duration_seconds=1.0, features=[1.0, 0.0]),
        LocationClip("c2", cluster_id=7, duration_seconds=5.0, features=[0.9, 0.1]),
    ]

    result = LocationGrouper(LocationConfig(min_duration_sec=3.0)).assign(clips)

    assert result.assignments["c1"] == 0
    assert result.assignments["c2"] == 0
    assert result.location_clusters[0] == 7


def test_short_middle_segment_merges_with_previous() -> None:
    clips = [
        LocationClip("c1", cluster_id=1, duration_seconds=5.0, features=[1.0, 0.0]),
        LocationClip("c2", cluster_id=2, duration_seconds=1.5, features=[0.5, 0.5]),
        LocationClip("c3", cluster_id=3, duration_seconds=5.0, features=[0.0, 1.0]),
    ]

    result = LocationGrouper(LocationConfig(min_duration_sec=3.0)).assign(clips)

    assert result.assignments == {"c1": 0, "c2": 0, "c3": 1}
    assert result.location_clusters[0] == 1
    assert result.location_clusters[1] == 3


def test_hysteresis_absorbs_short_middle_segment() -> None:
    clips = [
        LocationClip("c1", cluster_id=1, duration_seconds=4.0, features=[1.0, 0.0]),
        LocationClip("c2", cluster_id=2, duration_seconds=0.5, features=[0.8, 0.2]),
        LocationClip("c3", cluster_id=1, duration_seconds=4.0, features=[1.0, 0.1]),
    ]

    config = LocationConfig(min_duration_sec=3.0, hysteresis=True)
    result = LocationGrouper(config).assign(clips)

    assert result.assignments == {"c1": 0, "c2": 0, "c3": 0}
    assert result.location_clusters[0] == 1


def test_similarity_threshold_merges_close_segments() -> None:
    clips = [
        LocationClip("c1", cluster_id=0, duration_seconds=4.0, features=[1.0, 0.0]),
        LocationClip("c2", cluster_id=1, duration_seconds=4.0, features=[0.99, 0.01]),
    ]

    result = LocationGrouper(LocationConfig(similarity_threshold=0.95)).assign(clips)

    assert result.assignments == {"c1": 0, "c2": 0}
    assert result.location_clusters[0] == 0


def test_similarity_threshold_prevents_merge_when_low_similarity() -> None:
    clips = [
        LocationClip("c1", cluster_id=0, duration_seconds=4.0, features=[1.0, 0.0]),
        LocationClip("c2", cluster_id=1, duration_seconds=4.0, features=[0.0, 1.0]),
    ]

    result = LocationGrouper(LocationConfig(similarity_threshold=0.99)).assign(clips)

    assert result.assignments == {"c1": 0, "c2": 1}
    assert result.location_clusters[0] == 0
    assert result.location_clusters[1] == 1


def test_location_spans_cover_full_range(tmp_path) -> None:
    config = AnalyzerConfig(ignore_cache=True, cache_dir=tmp_path / "cache")
    analyzer = Analyzer(config)

    def fake_sample(context: PipelineClipContext) -> list[FrameSample]:
        return [
            FrameSample(
                clip_id=context.clip_id,
                frame_index=0,
                timestamp_seconds=0.0,
                data=np.zeros((2, 2, 3), dtype=np.uint8),
            )
        ]

    def fake_extract(clip_id: str, samples) -> ClipFeatures:
        return ClipFeatures(
            clip_id=clip_id,
            vector=np.array([0.1, 0.2], dtype=np.float32),
            metadata={"feature_dim": 2, "frame_count": len(samples)},
        )

    analyzer._sampler.sample = fake_sample  # type: ignore[attr-defined]
    analyzer._feature_extractor.extract = fake_extract  # type: ignore[attr-defined]

    def fake_cluster(features) -> ClusterResult:  # type: ignore[name-defined]
        ids = [feature.clip_id for feature in features]
        return ClusterResult(labels={clip_id: 0 for clip_id in ids}, chosen_k=1, silhouette=None)

    analyzer._clusterer.cluster = fake_cluster  # type: ignore[attr-defined]

    contexts = [
        PipelineClipContext(
            clip_id="clip-a",
            clip_name="Clip A",
            duration_seconds=5.0,
            duration_frames=120,
            media_path=None,
            timeline_in="00:00:00:00",
            timeline_out="00:00:05:00",
        ),
        PipelineClipContext(
            clip_id="clip-b",
            clip_name="Clip B",
            duration_seconds=5.0,
            duration_frames=120,
            media_path=None,
            timeline_in="00:00:05:00",
            timeline_out="00:00:10:00",
        ),
    ]

    result = analyzer.analyze(contexts)
    detail = result.summary["locations_detail"][0]
    assert detail["start_tc"] == "00:00:00:00"
    assert detail["end_tc"] == "00:00:10:00"
    assert detail["start_tc"] != detail["end_tc"]
