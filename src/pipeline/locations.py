"""Location grouping utilities for Pigment Bucket."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class LocationConfig:
    """Configuration options controlling location aggregation."""

    min_duration_sec: float = 3.0
    similarity_threshold: float = 0.92
    hysteresis: bool = True


@dataclass
class LocationClip:
    """Minimal clip payload required for location assignment."""

    clip_id: str
    cluster_id: int
    duration_seconds: float
    features: Sequence[float] | None
    timeline_in: str | None = None
    timeline_out: str | None = None


@dataclass
class LocationInfo:
    """Aggregated information about a location."""

    location_id: int
    cluster_id: int
    clip_ids: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    clip_count: int = 0
    start_tc: str | None = None
    end_tc: str | None = None
    centroid: List[float] = field(default_factory=list)


@dataclass
class LocationAssignmentResult:
    """Result bundle returned by the location grouper."""

    assignments: Dict[str, int | None]
    locations: Dict[int, LocationInfo]

    @property
    def location_clusters(self) -> Dict[int, int]:
        return {loc_id: info.cluster_id for loc_id, info in self.locations.items()}

    @property
    def location_count(self) -> int:
        return len(self.locations)


class LocationGrouper:
    """Groups sequential clips with temporal and feature-based heuristics."""

    def __init__(self, config: LocationConfig | None = None) -> None:
        self._config = config or LocationConfig()

    # ------------------------------------------------------------------
    def assign(self, clips: Sequence[LocationClip]) -> LocationAssignmentResult:
        if not clips:
            return LocationAssignmentResult(assignments={}, locations={})

        segments = self._initial_segments(clips)
        segments = self._merge_adjacent(segments)
        if self._config.hysteresis:
            segments = self._apply_hysteresis(segments)
        segments = self._merge_short_segments(segments)
        segments = self._merge_by_similarity(segments)

        locations: Dict[int, LocationInfo] = {}
        assignments: Dict[str, int | None] = {}

        location_id = -1
        for segment in segments:
            cluster = segment.get("cluster")
            clip_indices: List[int] = segment.get("indices", [])
            if cluster is None:
                for idx in clip_indices:
                    assignments[clips[idx].clip_id] = None
                continue

            location_id += 1
            centroid_array = self._segment_centroid(segment)
            centroid_list = (
                [float(round(value, 6)) for value in centroid_array.tolist()]
                if centroid_array is not None
                else []
            )
            info = LocationInfo(
                location_id=location_id,
                cluster_id=cluster,
                clip_ids=[clips[idx].clip_id for idx in clip_indices],
                duration_seconds=float(segment.get("duration", 0.0)),
                clip_count=len(clip_indices),
                start_tc=self._first_timeline(clips, clip_indices),
                end_tc=self._last_timeline(clips, clip_indices),
                centroid=centroid_list,
            )
            locations[location_id] = info

            for idx in clip_indices:
                assignments[clips[idx].clip_id] = location_id

        return LocationAssignmentResult(assignments=assignments, locations=locations)

    # ------------------------------------------------------------------
    def _initial_segments(self, clips: Sequence[LocationClip]) -> List[dict]:
        segments: List[dict] = []
        current: dict | None = None

        for index, clip in enumerate(clips):
            cluster = clip.cluster_id
            duration = max(float(clip.duration_seconds or 0.0), 0.0)
            feature_vec = self._to_vector(clip.features)

            if cluster < 0:
                if current:
                    segments.append(current)
                    current = None
                segments.append(
                    {
                        "cluster": None,
                        "indices": [index],
                        "duration": duration,
                        "feature_sum": None,
                        "feature_count": 0,
                    }
                )
                continue

            if current and current["cluster"] == cluster:
                current["indices"].append(index)
                current["duration"] += duration
                if feature_vec is not None:
                    current.setdefault("feature_sum", np.zeros_like(feature_vec))
                    current["feature_sum"] += feature_vec
                    current["feature_count"] += 1
            else:
                if current:
                    segments.append(current)
                current = {
                    "cluster": cluster,
                    "indices": [index],
                    "duration": duration,
                    "feature_sum": feature_vec.copy() if feature_vec is not None else None,
                    "feature_count": 1 if feature_vec is not None else 0,
                }

        if current:
            segments.append(current)
        return segments

    def _merge_adjacent(self, segments: List[dict]) -> List[dict]:
        if not segments:
            return []
        merged: List[dict] = [segments[0]]
        for segment in segments[1:]:
            last = merged[-1]
            if (
                last.get("cluster") is not None
                and last.get("cluster") == segment.get("cluster")
                and segment.get("cluster") is not None
            ):
                self._merge_segment_data(last, segment)
            else:
                merged.append(segment)
        return merged

    def _apply_hysteresis(self, segments: List[dict]) -> List[dict]:
        changed = True
        while changed and len(segments) >= 3:
            changed = False
            for idx in range(1, len(segments) - 1):
                prev_seg = segments[idx - 1]
                segment = segments[idx]
                next_seg = segments[idx + 1]
                if segment.get("cluster") is None:
                    continue
                if segment.get("duration", 0.0) >= self._config.min_duration_sec:
                    continue
                if (
                    prev_seg.get("cluster") is not None
                    and prev_seg.get("cluster") == next_seg.get("cluster")
                    and prev_seg.get("cluster") != segment.get("cluster")
                ):
                    segment["cluster"] = prev_seg.get("cluster")
                    changed = True
            if changed:
                segments = self._merge_adjacent(segments)
        return segments

    def _merge_short_segments(self, segments: List[dict]) -> List[dict]:
        if not segments:
            return []
        min_duration = max(0.0, float(self._config.min_duration_sec))
        changed = True
        while changed:
            changed = False
            for idx, segment in enumerate(list(segments)):
                if segment.get("cluster") is None:
                    continue
                if segment.get("duration", 0.0) >= min_duration:
                    continue
                left = segments[idx - 1] if idx - 1 >= 0 else None
                right = segments[idx + 1] if idx + 1 < len(segments) else None
                target = None
                if left and left.get("cluster") is not None and right and right.get("cluster") is not None:
                    target = left if left.get("duration", 0.0) >= right.get("duration", 0.0) else right
                elif left and left.get("cluster") is not None:
                    target = left
                elif right and right.get("cluster") is not None:
                    target = right
                if target is None:
                    continue
                self._merge_segment_data(target, segment)
                del segments[idx]
                segments = self._merge_adjacent(segments)
                changed = True
                break
        return segments

    def _merge_by_similarity(self, segments: List[dict]) -> List[dict]:
        if len(segments) <= 1:
            return segments
        threshold = max(0.0, min(1.0, float(self._config.similarity_threshold)))
        changed = True
        while changed:
            changed = False
            for idx in range(len(segments) - 1):
                a = segments[idx]
                b = segments[idx + 1]
                if a.get("cluster") is None or b.get("cluster") is None:
                    continue
                centroid_a = self._segment_centroid(a)
                centroid_b = self._segment_centroid(b)
                if centroid_a is None or centroid_b is None:
                    continue
                cosine = self._cosine_similarity(centroid_a, centroid_b)
                if cosine >= threshold:
                    self._merge_segment_data(a, b)
                    del segments[idx + 1]
                    changed = True
                    break
            if changed:
                segments = self._merge_adjacent(segments)
        return segments

    # ------------------------------------------------------------------
    def _merge_segment_data(self, target: dict, source: dict) -> None:
        target.setdefault("indices", []).extend(source.get("indices", []))
        target["duration"] = target.get("duration", 0.0) + source.get("duration", 0.0)
        feature_sum_target = target.get("feature_sum")
        feature_sum_source = source.get("feature_sum")
        if feature_sum_source is not None:
            if feature_sum_target is None:
                target["feature_sum"] = feature_sum_source.copy()
            else:
                target["feature_sum"] = feature_sum_target + feature_sum_source
            target["feature_count"] = target.get("feature_count", 0) + source.get("feature_count", 0)
        else:
            target.setdefault("feature_sum", feature_sum_target)
            target["feature_count"] = target.get("feature_count", 0)

    def _segment_centroid(self, segment: dict) -> np.ndarray | None:
        feature_sum = segment.get("feature_sum")
        count = segment.get("feature_count", 0)
        if feature_sum is None or not count:
            return None
        return feature_sum / float(count)

    def _to_vector(self, features: Sequence[float] | None) -> np.ndarray | None:
        if features is None:
            return None
        array = np.asarray(features, dtype=np.float32)
        if array.size == 0:
            return None
        return array

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0.0:
            return 0.0
        return float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))

    def _first_timeline(self, clips: Sequence[LocationClip], indices: List[int]) -> str | None:
        for idx in indices:
            value = clips[idx].timeline_in
            if value:
                return value
        return None

    def _last_timeline(self, clips: Sequence[LocationClip], indices: List[int]) -> str | None:
        for idx in reversed(indices):
            value = clips[idx].timeline_out
            if value:
                return value
        return None
