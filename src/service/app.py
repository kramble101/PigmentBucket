"""FastAPI mock analysis service for the Pigment Bucket MVP."""
from __future__ import annotations

import asyncio
from typing import Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from . import palette
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClipResult,
    JobStatus,
    ResultResponse,
    StatusResponse,
    Summary,
)

app = FastAPI(title="Pigment Bucket Mock Service", version="0.1.0")

_jobs: Dict[str, ResultResponse] = {}
_jobs_lock = asyncio.Lock()


async def _build_result(job_id: str, request: AnalyzeRequest) -> ResultResponse:
    color_stream = palette.cycling_palette()
    clip_results = []
    for clip in request.clips:
        palette_index, color_name = next(color_stream)
        clip_results.append(
            ClipResult(
                clip_id=clip.clip_id,
                clip_name=clip.clip_name,
                color_name=color_name,
                palette_index=palette_index,
                cluster_id=palette_index,
                status="ok",
                skip_reason=None,
            )
        )

    processed = len(clip_results)
    summary = Summary(
        processed=processed,
        skipped=0,
        clusters=len({result.cluster_id for result in clip_results}) if clip_results else 0,
    )
    return ResultResponse(job_id=job_id, status=JobStatus.COMPLETED, summary=summary, clips=clip_results)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Accept an analysis request and return a job identifier immediately."""

    job_id = uuid4().hex
    result = await _build_result(job_id, payload)
    async with _jobs_lock:
        _jobs[job_id] = result
    return AnalyzeResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str) -> StatusResponse:
    """Return the current status for a job."""

    async with _jobs_lock:
        result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return StatusResponse(job_id=job_id, status=result.status)


@app.get("/result/{job_id}", response_model=ResultResponse)
async def result(job_id: str) -> ResultResponse:
    """Return the final result for a completed job."""

    async with _jobs_lock:
        result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return result

