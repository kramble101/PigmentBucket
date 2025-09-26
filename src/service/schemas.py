"""Pydantic models for the mock analysis service."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for mock analysis jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNDONE = "undone"


class ClipDescriptor(BaseModel):
    """Minimal information the service expects for each timeline clip."""

    clip_id: str = Field(..., description="Stable identifier for the clip in the timeline")
    clip_name: str = Field(..., description="Human readable name for the clip")
    timeline_in: str = Field(..., description="Timeline in-point in timecode form")
    timeline_out: str = Field(..., description="Timeline out-point in timecode form")
    track_index: int = Field(..., ge=1, description="1-based video track index")
    duration_frames: int = Field(..., ge=0, description="Clip duration expressed in timeline frames")
    duration_seconds: Optional[float] = Field(None, description="Clip duration in seconds")
    media_path: Optional[str] = Field(None, description="Absolute path to underlying media if available")


class PipelineConfig(BaseModel):
    sampler_backend: Optional[str] = Field(None, description="Sampler backend: auto|resolve|ffmpeg")
    max_frames_per_clip: Optional[int] = Field(None, ge=1)
    min_spacing_sec: Optional[float] = Field(None, ge=0.1)
    max_k: Optional[int] = Field(None, ge=2)
    auto_mode: Optional[bool] = Field(None, description="Enable automatic tuning of location parameters")
    auto_target_k: Optional[int] = Field(None, ge=2, description="Desired minimum cluster count when auto mode is enabled")
    auto_max_iters: Optional[int] = Field(None, ge=1, description="Maximum auto-tuning iterations")
    dump_frames: Optional[bool] = Field(None, description="Dump sampled frames to disk for debugging")
    location_store: Optional[str] = Field(
        None,
        description="Preferred location persistence backend (sqlite|json)",
    )
    report_format: Optional[str] = Field(
        None,
        description="Server-side report format: json|csv|both",
    )
    location_min_duration_sec: Optional[float] = Field(
        None,
        ge=0.0,
        description="Minimum duration (seconds) to keep a standalone location segment",
    )
    location_similarity_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for merging adjacent locations",
    )
    location_hysteresis: Optional[bool] = Field(None, description="Enable hysteresis merge for short inserts")
    force_min_k: Optional[int] = Field(None, ge=1)
    ignore_cache: Optional[bool] = Field(None)
    project_id: Optional[str] = Field(None, description="Caller-supplied Resolve project identifier")
    clip_context_hash: Optional[str] = Field(
        None,
        description="Stable hash describing the submitted clip context for auto-learning",
    )


class AnalyzeRequest(BaseModel):
    """Payload for starting a mock analysis job."""

    clips: List[ClipDescriptor] = Field(default_factory=list)
    dry_run: bool = Field(False, description="If true, the adapter should skip SetClipColor")
    selection_mode: Optional[str] = Field(
        None,
        description="Optional selection mode hint provided by the adapter",
    )
    config: Optional[PipelineConfig] = Field(
        None,
        description="Optional overrides for sampler/clusterer configuration",
    )
    run_id: Optional[str] = Field(None, description="Client-supplied run identifier")


class AnalyzeResponse(BaseModel):
    job_id: str
    run_id: str


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    detail: Optional[str] = None
    run_id: Optional[str] = None


class ClipResult(BaseModel):
    clip_id: str
    clip_name: str
    color_name: str
    palette_index: int
    cluster_id: int
    status: str = Field("ok", description="Processing outcome for the clip")
    skip_reason: Optional[str] = None
    timeline_in: Optional[str] = None
    timeline_out: Optional[str] = None
    clip_color: Optional[str] = None
    features: Optional[List[float]] = None
    location_id: Optional[int] = Field(None, description="Assigned location identifier")
    location_span_start: Optional[str] = Field(None, description="Location start timecode")
    location_span_end: Optional[str] = Field(None, description="Location end timecode")
    location_len_clips: Optional[int] = Field(None, description="Number of clips in the location")
    location_duration_sec: Optional[float] = Field(None, description="Total location duration in seconds")
    location_centroid: Optional[List[float]] = Field(None, description="Mean feature vector for the location")
    persisted_location_id: Optional[int] = Field(None, description="Stable location identifier from the persistent store")
    location_color_name: Optional[str] = Field(None, description="Resolve color associated with the location")


class Summary(BaseModel):
    processed: int
    skipped: int
    clusters: int
    feature_dim: Optional[int] = None
    chosen_k: Optional[int] = None
    silhouette: Optional[float] = None
    sampled_frames: Optional[int] = None
    coloring: Optional[Dict[str, str]] = None
    cache_hits: Optional[int] = None
    locations: Optional[int] = None
    locations_stats: Optional[Dict[str, Dict[str, float]]] = None
    locations_detail: Optional[List[Dict[str, object]]] = None
    locations_persisted: Optional[int] = None
    new_locations: Optional[int] = None
    matched_locations: Optional[int] = None
    location_store: Optional[str] = None
    centroid_dim: Optional[int] = None
    report_format: Optional[str] = None
    median_hue_spread: Optional[float] = None
    quality_metric_name: Optional[str] = None
    quality_metric_value: Optional[float] = None
    auto_mode: Optional[bool] = None
    auto_iterations: Optional[int] = None
    auto_similarity_threshold: Optional[float] = None
    auto_min_duration_sec: Optional[float] = None
    auto_history: Optional[List[Dict[str, object]]] = None
    auto_initial_source: Optional[str] = None
    auto_target_k: Optional[int] = None
    auto_reference_centroid: Optional[List[float]] = None
    project_id: Optional[str] = None
    clip_context_hash: Optional[str] = None
    dumped_frames_count: Optional[int] = None
    frames_dir: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    run_id: Optional[str] = None
    status: JobStatus
    summary: Summary
    clips: List[ClipResult] = Field(default_factory=list)


class ArtifactUpdate(BaseModel):
    undo_path: Optional[str] = None


class UndoRequest(BaseModel):
    undo_path: Optional[str] = None
    note: Optional[str] = None
