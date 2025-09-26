"""FastAPI mock analysis service for the Pigment Bucket MVP."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

try:
    from . import __version__
except Exception:  # pragma: no cover - fallback for partial installs
    __version__ = "0.0.0+unknown"

from . import palette
from .auto_store import load_profiles
from .colors import assign_cluster_colors
from .config import (
    ANALYZE_JOB_TIMEOUT_MS,
    ANALYZE_MAX_RETRIES,
    ANALYZE_RETRY_DELAY_MS,
    DEBUG_FAULTS,
    FEATURE_CACHE_DIR,
    LOG_DIR,
    MAX_LOG_BYTES,
    MAX_LOG_FILES,
    ARTIFACTS_DIR,
    ensure_dirs,
)
from .db import get_job, init_db, insert_job, list_jobs, update_job
from .rotation import enforce_log_rotation
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ArtifactUpdate,
    ClipResult,
    JobStatus,
    PipelineConfig,
    ResultResponse,
    StatusResponse,
    Summary,
    UndoRequest,
)
from src.pipeline import (
    Analyzer,
    AnalyzerConfig,
    ClipContext,
    ClustererConfig,
    FeatureExtractorConfig,
    SamplerConfig,
)
from src.pipeline.types import ClipAnalysis
from .locations_store import StoreResult, get_location_store

REPORT_SCHEMA_VERSION = "1.0"
PALETTE_NAME = "default_v1"

ensure_dirs()
init_db()

logger = logging.getLogger("pigmentbucket.service")

app = FastAPI(title="Pigment Bucket Mock Service", version=__version__)


def _utcnow() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _analysis_timeout() -> float | None:
    return ANALYZE_JOB_TIMEOUT_MS / 1000 if ANALYZE_JOB_TIMEOUT_MS > 0 else None


def _retry_delay() -> float:
    return max(0, ANALYZE_RETRY_DELAY_MS) / 1000


def _create_analyzer(config: PipelineConfig | None, job_id: str) -> Analyzer:
    sampler_cfg = SamplerConfig()
    cluster_cfg = ClustererConfig()
    feature_cfg = FeatureExtractorConfig()
    defaults = AnalyzerConfig()
    force_min_k = defaults.force_min_k
    ignore_cache = False
    location_min_duration = defaults.location_min_duration_sec
    location_similarity = defaults.location_similarity_threshold
    location_hysteresis = defaults.location_hysteresis
    auto_mode = defaults.auto_mode
    auto_target_k = defaults.auto_target_k
    auto_max_iters = defaults.auto_max_iters
    project_id = None
    clip_context_hash = None
    auto_profiles: Sequence[Dict[str, object]] = []

    dump_frames = False
    frames_dir: Optional[Path] = None

    if config:
        if config.max_frames_per_clip is not None:
            sampler_cfg.max_frames_per_clip = max(1, config.max_frames_per_clip)
        if config.min_spacing_sec is not None:
            sampler_cfg.min_spacing_sec = max(0.1, config.min_spacing_sec)
        if config.sampler_backend:
            sampler_cfg.backend = config.sampler_backend
        if config.max_k is not None:
            cluster_cfg.max_k = max(2, config.max_k)
        if config.force_min_k is not None:
            force_min_k = max(1, config.force_min_k)
        if config.ignore_cache is not None:
            ignore_cache = bool(config.ignore_cache)
        if config.location_min_duration_sec is not None:
            try:
                location_min_duration = max(0.0, float(config.location_min_duration_sec))
            except (TypeError, ValueError):  # pragma: no cover - invalid input
                location_min_duration = defaults.location_min_duration_sec
        if config.location_similarity_threshold is not None:
            try:
                value = float(config.location_similarity_threshold)
                location_similarity = max(0.0, min(1.0, value))
            except (TypeError, ValueError):  # pragma: no cover
                location_similarity = defaults.location_similarity_threshold
        if config.location_hysteresis is not None:
            location_hysteresis = bool(config.location_hysteresis)
        if config.auto_mode is not None:
            auto_mode = bool(config.auto_mode)
        if config.auto_target_k is not None:
            auto_target_k = max(2, int(config.auto_target_k))
        if config.auto_max_iters is not None:
            auto_max_iters = max(1, int(config.auto_max_iters))
        if config.project_id:
            project_id = config.project_id
        if config.clip_context_hash:
            clip_context_hash = config.clip_context_hash
        if project_id and clip_context_hash:
            auto_profiles = load_profiles(project_id, clip_context_hash)
        if config.dump_frames:
            dump_frames = bool(config.dump_frames)

    if dump_frames:
        frames_dir = (ARTIFACTS_DIR / "frames" / job_id).resolve()

    analyzer_config = AnalyzerConfig(
        sampler=sampler_cfg,
        feature=feature_cfg,
        cluster=cluster_cfg,
        force_min_k=force_min_k,
        ignore_cache=ignore_cache,
        cache_dir=FEATURE_CACHE_DIR,
        location_min_duration_sec=location_min_duration,
        location_similarity_threshold=location_similarity,
        location_hysteresis=location_hysteresis,
        auto_mode=auto_mode,
        auto_target_k=auto_target_k,
        auto_max_iters=auto_max_iters,
        auto_profiles=auto_profiles,
        project_id=project_id,
        clip_context_hash=clip_context_hash,
        dump_frames=dump_frames,
        frames_dir=frames_dir,
    )
    return Analyzer(analyzer_config, logger)


@app.get("/health")
def health():
    return {"status": "ok", "service": "pigmentbucket", "version": __version__}


_jobs: Dict[str, ResultResponse] = {}
_jobs_lock = asyncio.Lock()


def _mock_centroid(cluster_id: int) -> Dict[str, float]:
    base = (cluster_id * 53) % 256
    return {
        "r": (base + 37) % 256,
        "g": (base + 73) % 256,
        "b": (base + 151) % 256,
    }


def _cluster_definitions(cluster_ids: Iterable[int]) -> Iterable[Mapping[str, object]]:
    for cid in sorted(set(cluster_ids)):
        yield {"id": cid, "centroid": _mock_centroid(cid)}


def _clip_to_context(clip) -> ClipContext:
    duration_seconds = clip.duration_seconds
    if duration_seconds is None:
        duration_seconds = float(clip.duration_frames or 0)
    return ClipContext(
        clip_id=clip.clip_id,
        clip_name=clip.clip_name,
        duration_seconds=float(duration_seconds or 0),
        duration_frames=int(clip.duration_frames or 0) if clip.duration_frames is not None else None,
        media_path=clip.media_path,
        timeline_in=clip.timeline_in,
        timeline_out=clip.timeline_out,
    )


async def _build_result(job_id: str, request: AnalyzeRequest) -> ResultResponse:
    if DEBUG_FAULTS and random.random() < 0.1:
        raise RuntimeError(f"Injected analysis fault for job {job_id}")

    analyzer = _create_analyzer(request.config, job_id)
    project_id = request.config.project_id if request.config else None
    clip_context_hash = request.config.clip_context_hash if request.config else None
    contexts = [_clip_to_context(clip) for clip in request.clips]
    analysis = await asyncio.to_thread(analyzer.analyze, contexts)

    clip_lookup = {clip.clip_id: clip for clip in request.clips}
    clip_analysis_map = {clip.clip_id: clip for clip in analysis.clips}
    processed_clips = [clip for clip in analysis.clips if clip.status == "ok" and clip.cluster_id >= 0]
    cluster_ids = [clip.cluster_id for clip in processed_clips]
    color_map = assign_cluster_colors(_cluster_definitions(cluster_ids)) if cluster_ids else {}

    raw_summary = analysis.summary if isinstance(analysis.summary, dict) else {}
    raw_coloring = raw_summary.get("coloring") if isinstance(raw_summary, dict) else {}

    report_format = "both"
    if request.config and getattr(request.config, "report_format", None):
        candidate = str(request.config.report_format).lower()
        if candidate in {"json", "csv", "both"}:
            report_format = candidate

    coloring_state = raw_coloring if isinstance(raw_coloring, dict) else {}
    clip_results: list[ClipResult] = []
    for clip_analysis in analysis.clips:
        clip_model = clip_lookup.get(clip_analysis.clip_id)
        assigned_color = color_map.get(clip_analysis.cluster_id)
        if clip_analysis.status != "ok" or assigned_color is None:
            assigned_color = "Grey"
        features_vector = clip_analysis.features.vector.tolist() if clip_analysis.features.vector.size else []
        features_vector = [round(float(value), 6) for value in features_vector]

        original_status = clip_analysis.status
        status = clip_analysis.status
        skip_reason = clip_analysis.skip_reason
        if coloring_state.get("coloring") == "skipped" and status == "ok":
            status = "skipped"
            skip_reason = coloring_state.get("reason") or "coloring-disabled"

        clip_result = ClipResult(
            clip_id=clip_analysis.clip_id,
            clip_name=clip_model.clip_name if clip_model else clip_analysis.clip_id,
            color_name=assigned_color,
            palette_index=int(clip_analysis.cluster_id),
            cluster_id=int(clip_analysis.cluster_id),
            status=status,
            skip_reason=skip_reason,
            timeline_in=clip_model.timeline_in if clip_model else "",
            timeline_out=clip_model.timeline_out if clip_model else "",
            clip_color=assigned_color,
            features=features_vector,
            location_id=clip_analysis.location_id,
            location_span_start=clip_analysis.location_span_start,
            location_span_end=clip_analysis.location_span_end,
            location_len_clips=clip_analysis.location_clip_count,
            location_duration_sec=clip_analysis.location_duration,
            location_centroid=clip_analysis.location_centroid,
            location_color_name=assigned_color,
        )
        clip_results.append(clip_result)

    # ------------------------------------------------------------------
    store_type = "sqlite"
    if request.config and getattr(request.config, "location_store", None):
        store_type = str(request.config.location_store).lower()

    analyzer_config = getattr(analyzer, "_config", None)
    similarity_threshold = 0.92
    if analyzer_config is not None:
        similarity_threshold = getattr(analyzer_config, "location_similarity_threshold", similarity_threshold)

    raw_locations_detail = raw_summary.get("locations_detail") if isinstance(raw_summary, dict) else None
    baseline_locations = [dict(entry) for entry in raw_locations_detail or [] if isinstance(entry, dict)]
    if not baseline_locations:
        baseline_locations = _build_locations_detail_from_clips(clip_results)

    for entry in baseline_locations:
        if "location_id" in entry:
            try:
                entry["location_id"] = int(entry["location_id"])
            except (TypeError, ValueError):
                entry.pop("location_id", None)

    store = get_location_store(store_type)
    store_result = store.assign(
        job_id=job_id,
        run_id=request.run_id or job_id,
        locations=list(baseline_locations),
        similarity_threshold=similarity_threshold,
    )

    assignments = store_result.assignments
    for clip in clip_results:
        temp_id = clip.location_id
        persisted_id = assignments.get(temp_id) if temp_id is not None else None
        clip.persisted_location_id = persisted_id
        if store_result.store_type == "sqlite" and persisted_id is not None:
            clip.location_id = persisted_id

    updated_locations_detail = store_result.locations_detail or baseline_locations
    if store_result.store_type != "sqlite":
        for entry in updated_locations_detail:
            entry.setdefault("persisted_location_id", None)

    location_color_lookup: Dict[int, str] = {}
    for clip in clip_results:
        if clip.location_id is None:
            continue
        if clip.location_color_name:
            location_color_lookup.setdefault(int(clip.location_id), clip.location_color_name)

    for entry in updated_locations_detail or []:
        location_identifier = entry.get("location_id")
        if isinstance(location_identifier, (int, float)):
            entry.setdefault("color_name", location_color_lookup.get(int(location_identifier)))

    locations_detail = updated_locations_detail or []

    median_hue_spread_value = _median_hue_spread_from_locations(locations_detail, clip_analysis_map)

    processed = int(analysis.summary.get("processed", len(processed_clips)))
    clusters = int(analysis.summary.get("clusters", len(set(cluster_ids))))
    skipped = int(analysis.summary.get("skipped", max(0, len(request.clips) - processed)))
    feature_dim = analysis.summary.get("feature_dim")
    chosen_k = analysis.summary.get("chosen_k")
    silhouette_raw = analysis.summary.get("silhouette")
    silhouette = silhouette_raw
    sampled_frames = analysis.summary.get("sampled_frames")

    locations_persisted = int(store_result.total_persisted)
    locations_new = int(store_result.created)
    locations_matched = int(store_result.matched)
    centroid_dim = int(store_result.centroid_dim) if store_result.centroid_dim else None

    coloring = raw_coloring if isinstance(raw_coloring, dict) else None
    cache_hits = raw_summary.get("cache_hits") if isinstance(raw_summary, dict) else None
    locations = raw_summary.get("locations") if isinstance(raw_summary, dict) else None
    locations_stats = raw_summary.get("locations_stats") if isinstance(raw_summary, dict) else None
    locations_detail = locations_detail or raw_summary.get("locations_detail")

    generated_stats = _summarize_locations(locations_detail)
    if not locations_stats and generated_stats:
        locations_stats = generated_stats

    locations_value = None
    if isinstance(locations, (int, float)):
        locations_value = int(locations)
    elif locations_detail:
        locations_value = len(locations_detail)

    if locations_stats is None and locations_detail:
        clip_counts = [entry.get("clip_count", 0) for entry in locations_detail]
        durations = [entry.get("duration_seconds", 0.0) for entry in locations_detail]
        if clip_counts:
            locations_stats = {
                "clip_count": {
                    "min": int(min(clip_counts)),
                    "median": float(np.median(clip_counts)),
                    "max": int(max(clip_counts)),
                },
                "duration_sec": {
                    "min": float(min(durations)) if durations else 0.0,
                    "median": float(np.median(durations)) if durations else 0.0,
                    "max": float(max(durations)) if durations else 0.0,
                },
            }

    quality_metric_name = None
    quality_metric_value = None
    if silhouette_raw is not None and clusters > 1:
        try:
            quality_metric_value = float(silhouette_raw)
            quality_metric_name = "silhouette"
        except (TypeError, ValueError):
            quality_metric_value = None
            quality_metric_name = None
    elif median_hue_spread_value is not None:
        quality_metric_name = "median_hue_spread"
        quality_metric_value = float(median_hue_spread_value)

    auto_mode_flag = raw_summary.get("auto_mode") if isinstance(raw_summary, dict) else None
    auto_iterations_value = raw_summary.get("auto_iterations") if isinstance(raw_summary, dict) else None
    auto_similarity_value = raw_summary.get("auto_similarity_threshold") if isinstance(raw_summary, dict) else None
    auto_min_duration_value = raw_summary.get("auto_min_duration_sec") if isinstance(raw_summary, dict) else None
    auto_history_value = raw_summary.get("auto_history") if isinstance(raw_summary, dict) else None
    auto_initial_source_value = raw_summary.get("auto_initial_source") if isinstance(raw_summary, dict) else None
    auto_target_k_value = raw_summary.get("auto_target_k") if isinstance(raw_summary, dict) else None
    auto_reference_centroid_value = raw_summary.get("auto_reference_centroid") if isinstance(raw_summary, dict) else None
    summary_project_id = raw_summary.get("project_id") if isinstance(raw_summary, dict) else None
    summary_clip_context_hash = raw_summary.get("clip_context_hash") if isinstance(raw_summary, dict) else None
    if summary_project_id is None:
        summary_project_id = project_id
    if summary_clip_context_hash is None:
        summary_clip_context_hash = clip_context_hash

    summary = Summary(
        processed=processed,
        skipped=skipped,
        clusters=clusters,
        feature_dim=int(feature_dim) if feature_dim else None,
        chosen_k=int(chosen_k) if chosen_k else None,
        silhouette=float(silhouette) if silhouette is not None else None,
        sampled_frames=int(sampled_frames) if sampled_frames else None,
        coloring=coloring if isinstance(coloring, dict) else None,
        cache_hits=int(cache_hits) if isinstance(cache_hits, (int, float)) else None,
        locations=locations_value,
        locations_stats=locations_stats if isinstance(locations_stats, dict) else generated_stats,
        locations_detail=updated_locations_detail if isinstance(updated_locations_detail, list) else locations_detail,
        locations_persisted=locations_persisted,
        new_locations=locations_new,
        matched_locations=locations_matched,
        location_store=store_result.store_type,
        centroid_dim=centroid_dim,
        median_hue_spread=median_hue_spread_value,
        quality_metric_name=quality_metric_name,
        quality_metric_value=quality_metric_value,
        report_format=report_format,
        auto_mode=bool(auto_mode_flag) if auto_mode_flag is not None else None,
        auto_iterations=int(auto_iterations_value) if auto_iterations_value is not None else None,
        auto_similarity_threshold=float(auto_similarity_value) if auto_similarity_value is not None else None,
        auto_min_duration_sec=float(auto_min_duration_value) if auto_min_duration_value is not None else None,
        auto_history=auto_history_value if isinstance(auto_history_value, list) else None,
        auto_initial_source=str(auto_initial_source_value) if auto_initial_source_value is not None else None,
        auto_target_k=int(auto_target_k_value) if auto_target_k_value is not None else None,
        auto_reference_centroid=[float(value) for value in auto_reference_centroid_value]
        if isinstance(auto_reference_centroid_value, list)
        else None,
        project_id=summary_project_id,
        clip_context_hash=summary_clip_context_hash,
        dumped_frames_count=int(raw_summary.get("dumped_frames_count", 0) or 0)
        if isinstance(raw_summary.get("dumped_frames_count"), (int, float))
        else None,
        frames_dir=str(raw_summary.get("frames_dir")) if raw_summary.get("frames_dir") else None,
    )
    return ResultResponse(job_id=job_id, run_id=request.run_id, status=JobStatus.COMPLETED, summary=summary, clips=clip_results)


def _build_locations_detail_from_clips(clips: Sequence[ClipResult]) -> List[Dict[str, object]]:
    grouped: Dict[int, Dict[str, object]] = {}
    for clip in clips:
        location_id = clip.location_id
        if location_id is None:
            continue
        info = grouped.setdefault(
            location_id,
            {
                "location_id": location_id,
                "cluster_id": clip.cluster_id,
                "clip_ids": [],
                "duration_seconds": float(clip.location_duration_sec or 0.0),
                "clip_count": 0,
                "start_tc": clip.location_span_start,
                "end_tc": clip.location_span_end,
                "centroid": clip.location_centroid,
                "color_name": clip.location_color_name or clip.clip_color,
            },
        )
        info["clip_ids"].append(clip.clip_id)
        info["clip_count"] += 1
        if clip.location_duration_sec is not None:
            info["duration_seconds"] = float(clip.location_duration_sec)
        if clip.location_span_start and (
            not info.get("start_tc") or clip.location_span_start < info.get("start_tc")
        ):
            info["start_tc"] = clip.location_span_start
        if clip.location_span_end and (
            not info.get("end_tc") or clip.location_span_end > info.get("end_tc")
        ):
            info["end_tc"] = clip.location_span_end
        if clip.location_centroid and not info.get("centroid"):
            info["centroid"] = clip.location_centroid
        if clip.location_color_name and not info.get("color_name"):
            info["color_name"] = clip.location_color_name

    return [grouped[key] for key in sorted(grouped.keys())]


def _summarize_locations(locations: Optional[List[Dict[str, object]]]) -> Optional[Dict[str, Dict[str, float]]]:
    if not locations:
        return None
    clip_counts: List[int] = []
    durations: List[float] = []
    for entry in locations:
        clip_counts.append(int(entry.get("clip_count", 0)))
        durations.append(float(entry.get("duration_seconds", 0.0)))
    if not clip_counts:
        return None
    return {
        "clip_count": {
            "min": int(min(clip_counts)),
            "median": float(np.median(clip_counts)),
            "max": int(max(clip_counts)),
        },
        "duration_sec": {
            "min": float(min(durations)) if durations else 0.0,
            "median": float(np.median(durations)) if durations else 0.0,
            "max": float(max(durations)) if durations else 0.0,
        },
    }


def _median_hue_spread_from_locations(
    locations: Optional[List[Dict[str, object]]],
    clips: Mapping[str, ClipAnalysis],
) -> float | None:
    if not locations:
        return None
    spreads: List[float] = []
    for location in locations:
        clip_ids = location.get("clip_ids") or []
        hues: List[float] = []
        for clip_id in clip_ids:
            clip = clips.get(clip_id)
            if not clip:
                continue
            vector = clip.features.vector
            if vector.size >= 7:
                hues.append(float(vector[6]))
        if hues:
            arr = np.asarray(hues, dtype=np.float32)
            median = float(np.median(arr))
            spreads.append(float(np.median(np.abs(arr - median))))
    if spreads:
        return float(np.median(spreads))
    return None


async def _run_analysis(job_id: str, payload: AnalyzeRequest) -> ResultResponse:
    attempts = ANALYZE_MAX_RETRIES + 1
    delay = _retry_delay()
    timeout = _analysis_timeout()
    for attempt in range(1, attempts + 1):
        try:
            logger.debug("Analysis attempt %s/%s for job %s", attempt, attempts, job_id)
            if timeout:
                return await asyncio.wait_for(_build_result(job_id, payload), timeout=timeout)
            return await _build_result(job_id, payload)
        except Exception as exc:  # pragma: no cover - integration path
            logger.warning("Analysis attempt %s failed for job %s: %s", attempt, job_id, exc)
            if attempt == attempts:
                raise
            if delay:
                await asyncio.sleep(delay)
    raise RuntimeError("Analysis retries exhausted")


def _store_job_placeholder(job_id: str, selection_mode: str | None, run_id: str) -> None:
    insert_job(
        {
            "id": job_id,
            "run_id": run_id,
            "created_at": _utcnow(),
            "status": JobStatus.PENDING.value,
            "selection_mode": selection_mode,
        }
    )


def _write_reports(job_id: str, report: Dict[str, object], report_format: str) -> Dict[str, str]:
    normalized_format = (report_format or "both").lower()
    if normalized_format not in {"json", "csv", "both"}:
        normalized_format = "both"

    write_json = normalized_format in {"json", "both"}
    write_csv = normalized_format in {"csv", "both"}

    json_path = LOG_DIR / f"{job_id}.json"
    csv_path = LOG_DIR / f"{job_id}.csv"
    location_csv_path = LOG_DIR / f"locations_{job_id}.csv"
    summary_info = report.get("summary", {})
    quality_name = summary_info.get("quality_metric_name")
    quality_value_raw = summary_info.get("quality_metric_value")
    median_hue_spread_raw = summary_info.get("median_hue_spread")

    if write_json:
        try:
            json_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as error:  # pragma: no cover - best effort logging
            logger.warning("Failed to write JSON report for %s: %s", job_id, error)

    if write_csv:
        try:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "job_id",
                        "clip_uid",
                        "clip_name",
                        "start_tc",
                        "end_tc",
                        "cluster_id",
                        "location_id",
                        "persisted_location_id",
                        "location_start_tc",
                        "location_end_tc",
                        "location_len_clips",
                        "location_duration_sec",
                        "location_centroid",
                        "location_color_name",
                        "clip_color",
                        "status",
                        "skip_reason",
                        "quality_metric_name",
                        "quality_metric_value",
                        "median_hue_spread",
                        "auto_mode",
                        "auto_iterations",
                        "auto_similarity_threshold",
                        "auto_min_duration_sec",
                        "auto_initial_source",
                        "apply_action",
                        "apply_reason",
                        "resolve_color_before",
                        "resolve_color_after",
                    ]
                )
                auto_mode_cell = summary_info.get("auto_mode")
                auto_iterations_cell = summary_info.get("auto_iterations")
                auto_similarity_cell = summary_info.get("auto_similarity_threshold")
                auto_min_duration_cell = summary_info.get("auto_min_duration_sec")
                auto_initial_source_cell = summary_info.get("auto_initial_source")
                for clip in report["clips"]:
                    centroid = clip.get("location_centroid")
                    centroid_str = "" if centroid is None else " ".join(
                        f"{float(value):.6f}" for value in centroid
                    )
                    persisted_value = clip.get("persisted_location_id")
                    persisted_cell = "" if persisted_value is None else str(persisted_value)
                    writer.writerow(
                        [
                            job_id,
                            clip["clip_uid"],
                            clip["clip_name"],
                            clip["start_tc"],
                            clip["end_tc"],
                            clip["cluster_id"],
                            clip.get("location_id", ""),
                            persisted_cell,
                            clip.get("location_start_tc", ""),
                            clip.get("location_end_tc", ""),
                            clip.get("location_len_clips", ""),
                            clip.get("location_duration_sec", ""),
                            centroid_str,
                            clip.get("location_color_name", ""),
                            clip["clip_color"],
                            clip.get("status", ""),
                        clip.get("skip_reason", ""),
                        quality_name or "",
                        (
                            f"{float(quality_value_raw):.4f}"
                            if isinstance(quality_value_raw, (int, float))
                            else (quality_value_raw or "")
                        ),
                        (
                            f"{float(median_hue_spread_raw):.4f}"
                            if isinstance(median_hue_spread_raw, (int, float))
                            else (median_hue_spread_raw or "")
                        ),
                        str(bool(auto_mode_cell)) if auto_mode_cell is not None else "",
                        auto_iterations_cell if auto_iterations_cell is not None else "",
                        (
                            f"{float(auto_similarity_cell):.4f}"
                            if isinstance(auto_similarity_cell, (int, float))
                            else (auto_similarity_cell or "")
                        ),
                        (
                            f"{float(auto_min_duration_cell):.3f}"
                            if isinstance(auto_min_duration_cell, (int, float))
                            else (auto_min_duration_cell or "")
                        ),
                        auto_initial_source_cell or "",
                        clip.get("apply_action", ""),
                        clip.get("apply_reason", ""),
                        clip.get("resolve_color_before", ""),
                        clip.get("resolve_color_after", ""),
                    ]
                )
        except Exception as error:  # pragma: no cover - best effort logging
            logger.warning("Failed to write CSV report for %s: %s", job_id, error)
            write_csv = False

        try:
            locations_detail = report.get("locations", [])
            if locations_detail:
                with location_csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "location_id",
                            "persisted_location_id",
                            "cluster_id",
                            "clip_count",
                            "duration_seconds",
                            "start_tc",
                            "end_tc",
                            "clip_ids",
                            "centroid",
                            "color_name",
                        ]
                    )
                    for entry in locations_detail:
                        centroid = entry.get("centroid")
                        centroid_str = "" if centroid is None else " ".join(
                            f"{float(value):.6f}" for value in centroid
                        )
                        persisted_value = entry.get("persisted_location_id")
                        if persisted_value is None:
                            persisted_cell = "-" if report.get("location_store") == "json" else ""
                        else:
                            persisted_cell = str(persisted_value)
                        writer.writerow(
                            [
                                entry.get("location_id"),
                                persisted_cell,
                                entry.get("cluster_id"),
                                entry.get("clip_count"),
                                entry.get("duration_seconds"),
                                entry.get("start_tc"),
                                entry.get("end_tc"),
                                ",".join(entry.get("clip_ids", [])),
                                centroid_str,
                                entry.get("color_name", ""),
                            ]
                        )
        except Exception as error:  # pragma: no cover - best effort logging
            logger.warning("Failed to write location CSV for %s: %s", job_id, error)
            location_csv_path = None
            write_csv = False

    enforce_log_rotation(LOG_DIR, MAX_LOG_FILES, MAX_LOG_BYTES)

    paths: Dict[str, str] = {}
    if write_json and json_path.exists():
        paths["json_path"] = str(json_path)
    if write_csv and csv_path.exists():
        paths["csv_path"] = str(csv_path)
    if write_csv and location_csv_path.exists():
        paths["locations_csv_path"] = str(location_csv_path)
    return paths


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Accept an analysis request and return a job identifier immediately."""

    job_id = uuid4().hex
    run_id = payload.run_id or job_id
    started = time.perf_counter()
    _store_job_placeholder(job_id, payload.selection_mode, run_id)

    async with _jobs_lock:
        _jobs[job_id] = ResultResponse(
            job_id=job_id,
            run_id=run_id,
            status=JobStatus.PENDING,
            summary=Summary(processed=0, skipped=0, clusters=0),
            clips=[],
        )

    update_job(job_id, status=JobStatus.RUNNING.value)
    async with _jobs_lock:
        _jobs[job_id].status = JobStatus.RUNNING

    try:
        payload.run_id = run_id
        result = await _run_analysis(job_id, payload)
    except Exception as error:
        finished_at = _utcnow()
        update_job(job_id, status=JobStatus.FAILED.value, error=str(error), finished_at=finished_at)
        async with _jobs_lock:
            _jobs[job_id] = ResultResponse(
                job_id=job_id,
                run_id=run_id,
                status=JobStatus.FAILED,
                summary=Summary(processed=0, skipped=0, clusters=0),
                clips=[],
            )
        logger.error("Job %s failed: %s", job_id, error)
        logger.info(
            "Job %s failed after %.2fs (clips=%d)",
            job_id,
            time.perf_counter() - started,
            len(payload.clips),
        )
        raise HTTPException(status_code=500, detail="Analysis failed") from error

    async with _jobs_lock:
        _jobs[job_id] = result

    report = {
        "job_id": job_id,
        "run_id": run_id,
        "schema_version": REPORT_SCHEMA_VERSION,
        "service_version": __version__,
        "palette_name": PALETTE_NAME,
        "summary": result.summary.model_dump(),
        "feature_dim": result.summary.feature_dim,
        "chosen_k": result.summary.chosen_k,
        "silhouette": result.summary.silhouette,
        "sampled_frames": result.summary.sampled_frames,
        "clips": [
            {
                "clip_uid": clip.clip_id,
                "clip_name": clip.clip_name,
                "start_tc": clip.timeline_in or "",
                "end_tc": clip.timeline_out or "",
                "cluster_id": clip.cluster_id,
                "location_id": clip.location_id,
                "location_start_tc": clip.location_span_start or "",
                "location_end_tc": clip.location_span_end or "",
                "location_len_clips": clip.location_len_clips,
                "location_duration_sec": clip.location_duration_sec,
                "location_centroid": clip.location_centroid,
                "location_color_name": clip.location_color_name,
                "clip_color": clip.clip_color or clip.color_name,
                "status": clip.status,
                "skip_reason": clip.skip_reason,
                "features": clip.features or [],
                "persisted_location_id": clip.persisted_location_id,
            }
            for clip in result.clips
        ],
    }
    if payload.selection_mode:
        report["selection_mode"] = payload.selection_mode
    if result.summary.coloring:
        report["coloring"] = result.summary.coloring
    if result.summary.cache_hits is not None:
        report["cache_hits"] = result.summary.cache_hits
    if result.summary.locations_detail:
        report["locations"] = result.summary.locations_detail
    if result.summary.locations_stats:
        report["locations_stats"] = result.summary.locations_stats
    if result.summary.location_store:
        report["location_store"] = result.summary.location_store
    if result.summary.locations_persisted is not None:
        report["locations_persisted"] = result.summary.locations_persisted
    if result.summary.new_locations is not None:
        report["new_locations"] = result.summary.new_locations
    if result.summary.matched_locations is not None:
        report["matched_locations"] = result.summary.matched_locations
    if result.summary.centroid_dim is not None:
        report["centroid_dim"] = result.summary.centroid_dim
    if result.summary.report_format:
        report["report_format"] = result.summary.report_format

    paths = _write_reports(job_id, report, result.summary.report_format or "both")

    finished_at = _utcnow()
    update_job(
        job_id,
        status=JobStatus.COMPLETED.value,
        finished_at=finished_at,
        run_id=run_id,
        processed=result.summary.processed,
        skipped=result.summary.skipped,
        clusters=result.summary.clusters,
        json_path=paths.get("json_path"),
        csv_path=paths.get("csv_path"),
    )

    logger.info(
        "Job %s completed in %.2fs (clips=%d, chosen_k=%s, locations=%s)",
        job_id,
        time.perf_counter() - started,
        len(payload.clips),
        result.summary.chosen_k,
        result.summary.locations,
    )

    return AnalyzeResponse(job_id=job_id, run_id=run_id)


@app.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str) -> StatusResponse:
    """Return the current status for a job."""

    async with _jobs_lock:
        result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return StatusResponse(job_id=job_id, status=result.status, run_id=result.run_id)


@app.get("/result/{job_id}", response_model=ResultResponse)
async def result(job_id: str) -> ResultResponse:
    """Return the final result for a completed job."""

    async with _jobs_lock:
        result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return result


@app.get("/jobs")
def jobs(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return list_jobs(limit=limit, offset=offset)


@app.get("/jobs/{job_id}")
def job_detail(job_id: str):
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return record


class StopRequest(BaseModel):
    reason: Optional[str] = None


@app.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str, payload: StopRequest | None = None):
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    finished_at = _utcnow()
    update_job(job_id, status=JobStatus.STOPPED.value, finished_at=finished_at, error=(payload.reason if payload else None))
    async with _jobs_lock:
        stored = _jobs.get(job_id)
        if stored:
            stored.status = JobStatus.STOPPED
    return {"job_id": job_id, "status": JobStatus.STOPPED}


@app.post("/jobs/{job_id}/artifacts")
async def attach_artifacts(job_id: str, payload: ArtifactUpdate):
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    update_values = {}
    if payload.undo_path:
        update_values["undo_path"] = payload.undo_path
    if update_values:
        update_job(job_id, **update_values)
    return {"job_id": job_id, **update_values}


@app.post("/jobs/{job_id}/undo")
async def mark_undo(job_id: str, payload: UndoRequest):
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    finished_at = _utcnow()
    update_values = {"status": JobStatus.UNDONE.value, "finished_at": finished_at}
    if payload.undo_path:
        update_values["undo_path"] = payload.undo_path
    update_job(job_id, **update_values)
    async with _jobs_lock:
        stored = _jobs.get(job_id)
        if stored:
            stored.status = JobStatus.UNDONE
    return {"job_id": job_id, "status": JobStatus.UNDONE}
