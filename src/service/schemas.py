"""Pydantic models for the mock analysis service."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for mock analysis jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ClipDescriptor(BaseModel):
    """Minimal information the service expects for each timeline clip."""

    clip_id: str = Field(..., description="Stable identifier for the clip in the timeline")
    clip_name: str = Field(..., description="Human readable name for the clip")
    timeline_in: str = Field(..., description="Timeline in-point in timecode form")
    timeline_out: str = Field(..., description="Timeline out-point in timecode form")
    track_index: int = Field(..., ge=1, description="1-based video track index")
    duration_frames: int = Field(..., ge=0, description="Clip duration expressed in timeline frames")


class AnalyzeRequest(BaseModel):
    """Payload for starting a mock analysis job."""

    clips: List[ClipDescriptor] = Field(default_factory=list)
    dry_run: bool = Field(False, description="If true, the adapter should skip SetClipColor")


class AnalyzeResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    detail: Optional[str] = None


class ClipResult(BaseModel):
    clip_id: str
    clip_name: str
    color_name: str
    palette_index: int
    cluster_id: int
    status: str = Field("ok", description="Processing outcome for the clip")
    skip_reason: Optional[str] = None


class Summary(BaseModel):
    processed: int
    skipped: int
    clusters: int


class ResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    summary: Summary
    clips: List[ClipResult] = Field(default_factory=list)

