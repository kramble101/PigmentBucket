from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.analyzer import Analyzer, AnalyzerConfig
from src.pipeline.types import ClipContext, ClipFeatures, ClusterResult, FrameSample


def build_contexts() -> list[ClipContext]:
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


def setup_analyzer(
    tmp_path,
    cluster_sequence: list[ClusterResult],
    *,
    auto_mode: bool = True,
    auto_profiles: list[dict] | None = None,
) -> Analyzer:
    config = AnalyzerConfig(
        ignore_cache=True,
        cache_dir=tmp_path / "cache",
        auto_mode=auto_mode,
        auto_target_k=2,
        auto_max_iters=5,
        auto_profiles=auto_profiles or [],
    )
    analyzer = Analyzer(config)

    vectors = {
        "clip-1": np.array([0.1, 0.2], dtype=np.float32),
        "clip-2": np.array([0.1, 0.2], dtype=np.float32),
    }

    def fake_sample(context: ClipContext) -> list[FrameSample]:
        return [
            FrameSample(
                clip_id=context.clip_id,
                frame_index=0,
                timestamp_seconds=0.0,
                data=np.zeros((4, 4, 3), dtype=np.uint8),
            )
        ]

    def fake_extract(clip_id: str, samples) -> ClipFeatures:
        vector = vectors[clip_id]
        return ClipFeatures(
            clip_id=clip_id,
            vector=vector.copy(),
            metadata={"feature_dim": vector.shape[0], "frame_count": len(samples)},
        )

    iterator = iter(cluster_sequence)

    def fake_cluster(features, random_state):  # noqa: ARG001 - signature matches hook
        try:
            return next(iterator)
        except StopIteration:
            return cluster_sequence[-1]

    analyzer._sampler.sample = fake_sample  # type: ignore[attr-defined]
    analyzer._feature_extractor.extract = fake_extract  # type: ignore[attr-defined]
    analyzer._cluster_with_random_state = fake_cluster  # type: ignore[attr-defined]
    return analyzer


def test_auto_mode_retries_until_target_k(tmp_path) -> None:
    first = ClusterResult(labels={"clip-1": 0, "clip-2": 0}, chosen_k=1, silhouette=None)
    second = ClusterResult(labels={"clip-1": 0, "clip-2": 1}, chosen_k=2, silhouette=0.12)
    analyzer = setup_analyzer(tmp_path, [first, second])

    result = analyzer.analyze(build_contexts())

    assert result.summary["auto_mode"] is True
    assert result.summary["auto_iterations"] == 2
    assert result.summary["chosen_k"] == 2
    assert result.summary["clusters"] == 2
    assert pytest.approx(result.summary["auto_similarity_threshold"], rel=1e-3) == 0.95
    assert pytest.approx(result.summary["auto_min_duration_sec"], rel=1e-3) == 2.5
    history = result.summary["auto_history"]
    assert isinstance(history, list)
    assert len(history) == 2


def test_auto_mode_uses_learned_profile(tmp_path) -> None:
    profile = {
        "similarity": 0.97,
        "min_duration": 1.6,
        "centroid": [0.1, 0.2],
        "updated_at": "2025-01-01T00:00:00Z",
    }
    final = ClusterResult(labels={"clip-1": 0, "clip-2": 1}, chosen_k=2, silhouette=0.15)
    analyzer = setup_analyzer(
        tmp_path,
        [final],
        auto_mode=True,
        auto_profiles=[profile],
    )

    result = analyzer.analyze(build_contexts())

    assert result.summary["auto_mode"] is True
    assert result.summary["auto_iterations"] == 1
    history = result.summary["auto_history"]
    assert isinstance(history, list) and len(history) == 1
    first_attempt = history[0]
    assert pytest.approx(first_attempt["similarity"], rel=1e-3) == 0.97
    assert pytest.approx(first_attempt["min_duration"], rel=1e-3) == 1.6
    assert result.summary["auto_initial_source"] == "learned"
