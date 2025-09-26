"""Typed primitives for the Pigment Bucket analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FrameSample:
    """Represents a single sampled frame from a timeline clip."""

    clip_id: str
    frame_index: int
    timestamp_seconds: float
    source_path: str | None = None
    metadata: Dict[str, object] = field(default_factory=dict)
    data: Optional[np.ndarray] = None


@dataclass
class ClipFeatures:
    """Aggregated feature vector for a clip."""

    clip_id: str
    vector: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ClipAnalysis:
    """Final analysis payload for a clip."""

    clip_id: str
    cluster_id: int
    features: ClipFeatures
    samples: Sequence[FrameSample]
    status: str = "ok"
    skip_reason: Optional[str] = None
    location_id: Optional[int] = None
    location_span_start: Optional[str] = None
    location_span_end: Optional[str] = None
    location_clip_count: Optional[int] = None
    location_duration: Optional[float] = None
    location_centroid: Optional[List[float]] = None


@dataclass
class AnalysisResult:
    """Result bundle produced by the analyzer."""

    clips: List[ClipAnalysis]
    summary: Dict[str, float]


@dataclass
class ClipContext:
    """Input context for sampling a clip."""

    clip_id: str
    clip_name: str
    duration_seconds: float
    duration_frames: int | None = None
    media_path: Optional[str]
    timeline_in: Optional[str]
    timeline_out: Optional[str]


@dataclass
class ClusterResult:
    labels: Dict[str, int]
    chosen_k: int
    silhouette: Optional[float]
