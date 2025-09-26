#!/usr/bin/env python3
"""DaVinci Resolve adapter for the Pigment Bucket mock analysis service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from pathlib import Path

import requests

try:  # Optional dependency for deterministic numpy-based pipelines
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional for Resolve scripts
    np = None

# Resolve API reference: docs/resolve_api_reference.md
SERVICE_URL_ENV = "PIGMENT_SERVICE_URL"
SERVICE_URL_DEFAULT = "http://127.0.0.1:8765"
STATUS_POLL_SECONDS = 2
RETRY_ATTEMPTS = 3
GLOBAL_TIMEOUT_SECONDS = 15 * 60
PREVIEW_LIMIT = 5
UNDO_DIR = Path("undo")
HTTP_TIMEOUT_DEFAULT = int(os.getenv("PIGMENT_HTTP_TIMEOUT", "60"))
FORCE_MIN_K_DEFAULT = int(os.getenv("PIGMENT_FORCE_MIN_K", "1"))
IGNORE_CACHE_DEFAULT = bool(int(os.getenv("PIGMENT_IGNORE_CACHE", "0")))


@dataclass
class ClipPayload:
    clip_id: str
    clip_uid_source: str
    clip_name: str
    timeline_in: str
    timeline_out: str
    track_index: int
    duration_frames: int
    duration_seconds: float = 0.0
    media_path: Optional[str] = None


@dataclass
class ClipContext:
    payload: ClipPayload
    item: object
    start_frame: int
    end_frame: int


class ResolveNotAvailable(RuntimeError):
    """Raised when the Resolve scripting API cannot be imported."""


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_clip_colors(clips: List[ClipPayload], lookup: Dict[str, object]) -> List[Dict[str, Optional[str]]]:
    snapshot: List[Dict[str, Optional[str]]] = []
    for clip in clips:
        clip_item = lookup.get(clip.clip_id)
        if not clip_item:
            continue
        get_color = getattr(clip_item, "GetClipColor", None)
        color: Optional[str] = None
        if callable(get_color):
            try:
                raw_color = get_color()
                if raw_color:
                    color = raw_color
            except Exception:
                color = None
        snapshot.append(
            {
                "clip_id": clip.clip_id,
                "clip_name": clip.clip_name,
                "timeline_in": clip.timeline_in,
                "timeline_out": clip.timeline_out,
                "original_color": color,
            }
        )
    return snapshot


def restore_clip_color(item, color: Optional[str]) -> None:
    try:
        if color:
            item.SetClipColor(color)
        else:
            if hasattr(item, "ClearClipColor"):
                item.ClearClipColor()
            else:
                item.SetClipColor("")
    except Exception as error:
        print(f"[WARN] Failed to restore color: {error}", file=sys.stderr)


def compute_clip_context_hash(clips: List[ClipPayload]) -> str:
    parts = [
        f"{clip.clip_id}:{clip.timeline_in}:{clip.timeline_out}:{clip.track_index}"
        for clip in clips
    ]
    parts.sort()
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def remove_markers_from_timeline(timeline, markers: List[Dict[str, int]]) -> int:
    if not timeline or not markers:
        return 0
    removed = 0
    for marker in markers:
        frame_value = marker.get("frame") if isinstance(marker, dict) else None
        try:
            frame_int = int(frame_value) if frame_value is not None else None
        except (TypeError, ValueError):
            frame_int = None
        if frame_int is None:
            continue
        try:
            if timeline.DeleteMarkerAtFrame(frame_int):
                removed += 1
        except Exception as error:
            print(f"[WARN] Failed to remove marker at frame {frame_int}: {error}", file=sys.stderr)
    return removed


def write_undo_file(
    run_id: str,
    job_id: str,
    selection_mode: str,
    service_url: str,
    summary: Dict[str, object],
    clip_snapshot: List[Dict[str, Optional[str]]],
    result_clips: List[dict],
    report_path: Path,
    markers: Optional[List[Dict[str, int]]] = None,
) -> Path:
    ensure_directory(UNDO_DIR)
    undo_payload = {
        "run_id": run_id,
        "job_id": job_id,
        "service_url": service_url,
        "selection_mode": selection_mode,
        "created_at": _now_iso(),
        "summary": summary,
        "clips": clip_snapshot,
        "applied": result_clips,
        "report_path": str(report_path),
        "markers": markers or [],
    }
    undo_path = UNDO_DIR / f"{run_id}.json"
    undo_path.write_text(json.dumps(undo_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return undo_path


def stop_remote_job(service_url: str, job_id: str, timeout_sec: int, reason: Optional[str] = None) -> None:
    payload = {"reason": reason} if reason else None
    try:
        response = requests.post(
            f"{service_url}/jobs/{job_id}/stop",
            json=payload,
            timeout=(timeout_sec, timeout_sec),
        )
        response.raise_for_status()
        print(f"Requested stop for job {job_id}")
    except requests.RequestException as error:
        print(f"[WARN] Failed to stop job {job_id}: {error}", file=sys.stderr)


def attach_undo_artifact(service_url: str, job_id: str, undo_path: Path, timeout_sec: int) -> None:
    try:
        response = requests.post(
            f"{service_url}/jobs/{job_id}/artifacts",
            json={"undo_path": str(undo_path)},
            timeout=(timeout_sec, timeout_sec),
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"[WARN] Failed to attach undo artifact for {job_id}: {error}", file=sys.stderr)


def mark_job_undone(service_url: str, job_id: str, undo_path: Path, timeout_sec: int) -> None:
    try:
        response = requests.post(
            f"{service_url}/jobs/{job_id}/undo",
            json={"undo_path": str(undo_path)},
            timeout=(timeout_sec, timeout_sec),
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"[WARN] Failed to mark job {job_id} as undone: {error}", file=sys.stderr)


def perform_undo(run_id: str, service_url: str, timeout_sec: int) -> None:
    undo_path = UNDO_DIR / f"{run_id}.json"
    if not undo_path.exists():
        print(f"Undo file not found: {undo_path}", file=sys.stderr)
        return
    data = json.loads(undo_path.read_text(encoding="utf-8"))
    job_id = data.get("job_id")
    clips_data = data.get("clips", [])
    markers_data = data.get("markers", [])
    if not clips_data:
        print("Undo file has no clip data", file=sys.stderr)
        return

    resolve = load_resolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No active project. Open a project before running undo.")
    contexts, _, _ = gather_clips(project)
    clip_lookup = {ctx.payload.clip_id: ctx.item for ctx in contexts}

    restored = 0
    for entry in clips_data:
        clip_id = entry.get("clip_id")
        if not clip_id:
            continue
        item = clip_lookup.get(clip_id)
        if not item:
            print(f"[WARN] Clip {clip_id} not found in current timeline", file=sys.stderr)
            continue
        restore_clip_color(item, entry.get("original_color"))
        restored += 1

    timeline = project.GetCurrentTimeline()
    markers_removed = remove_markers_from_timeline(timeline, markers_data)

    if timeline:
        name = f"PigmentBucket • undo ({run_id})"
        timeline_start = timeline.GetStartFrame() or 0
        timeline.AddMarker(timeline_start, "Red", name, "", 1, None)

    if markers_removed:
        print(f"Removed {markers_removed} marker(s) from timeline")

    print(f"Restored colors for {restored} clip(s) from run {run_id}")
    if job_id:
        mark_job_undone(service_url, job_id, undo_path, timeout_sec)


def perform_marker_undo(run_id: str, service_url: str, timeout_sec: int) -> None:
    undo_path = UNDO_DIR / f"{run_id}.json"
    if not undo_path.exists():
        print(f"Undo file not found: {undo_path}; attempting timeline cleanup", file=sys.stderr)
        undo_payload = None
    else:
        undo_payload = json.loads(undo_path.read_text(encoding="utf-8"))
    markers_data = (undo_payload or {}).get("markers", []) if undo_path.exists() else []
    if not markers_data:
        resolve = load_resolve()
        project_manager = resolve.GetProjectManager()
        project = project_manager.GetCurrentProject()
        if project is None:
            raise RuntimeError("No active project. Open a project before running undo-markers.")
        timeline = project.GetCurrentTimeline()
        if timeline is None:
            raise RuntimeError("No active timeline. Create or open a timeline before running undo-markers.")
        markers_map = timeline.GetMarkers() if hasattr(timeline, "GetMarkers") else {}
        removed = 0
        if isinstance(markers_map, dict):
            for frame, marker in list(markers_map.items()):
                note = marker.get("note") if isinstance(marker, dict) else ""
                if note and f"run={run_id}" in note:
                    try:
                        if timeline.DeleteMarkerAtFrame(int(frame)):
                            removed += 1
                    except Exception as error:
                        print(f"[WARN] Failed to remove marker at frame {frame}: {error}", file=sys.stderr)
        if removed:
            print(f"Removed {removed} marker(s) from run {run_id}")
        else:
            print("No markers were removed (markers may already be gone)")
        return

    resolve = load_resolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No active project. Open a project before running undo-markers.")

    timeline = project.GetCurrentTimeline()
    removed = remove_markers_from_timeline(timeline, markers_data)
    if removed:
        print(f"Removed {removed} marker(s) from run {run_id}")
    else:
        print("No markers were removed (markers may already be gone)")
def load_resolve():
    try:
        import DaVinciResolveScript as dvr  # type: ignore
    except ImportError as exc:  # pragma: no cover - executed inside Resolve
        raise ResolveNotAvailable(
            "DaVinciResolveScript module is not available. Did you set RESOLVE_SCRIPT_API?"
        ) from exc

    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise ResolveNotAvailable("Unable to acquire Resolve scripting object")
    return resolve


def parse_frame_rate(rate: Optional[str]) -> Fraction:
    if not rate:
        return Fraction(24, 1)
    try:
        return Fraction(rate).limit_denominator(1000)
    except (ValueError, ZeroDivisionError):
        return Fraction(24, 1)


def frames_to_timecode(frame: int, fps: Fraction) -> str:
    fps_fraction = Fraction(fps).limit_denominator()
    total_seconds = Fraction(frame * fps_fraction.denominator, fps_fraction.numerator)
    hours = int(total_seconds // 3600)
    total_seconds -= hours * 3600
    minutes = int(total_seconds // 60)
    total_seconds -= minutes * 60
    seconds = int(total_seconds)
    total_seconds -= seconds
    frames = int(total_seconds * fps_fraction)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def compute_clip_uid(
    item,
    media_path: Optional[str],
    timeline_in: str,
    timeline_out: str,
    track_index: int,
) -> Tuple[str, str]:
    unique_id = item.GetUniqueId() if hasattr(item, "GetUniqueId") else None
    if unique_id:
        return unique_id, "resolve_uid"

    hasher = hashlib.sha1()
    key_parts = [
        media_path or item.GetName() or "unknown",
        timeline_in,
        timeline_out,
        str(track_index),
    ]
    for part in key_parts:
        hasher.update(part.encode("utf-8"))
    return hasher.hexdigest(), "sha1_fallback"


def gather_clips(project) -> Tuple[List[ClipContext], Fraction, int]:
    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise RuntimeError("No active timeline found")

    fps = parse_frame_rate(timeline.GetSetting("timelineFrameRate"))
    timeline_start = timeline.GetStartFrame() or 0

    clip_contexts: List[ClipContext] = []

    video_track_count = timeline.GetTrackCount("video")
    for track_index in range(1, video_track_count + 1):
        items = timeline.GetItemListInTrack("video", track_index) or []
        for item in items:
            start_frame = int(item.GetStart())
            end_frame = int(item.GetEnd())
            duration_frames = max(0, end_frame - start_frame)
            timeline_in_tc = frames_to_timecode(start_frame - timeline_start, fps)
            timeline_out_tc = frames_to_timecode(end_frame - timeline_start, fps)

            media_item = item.GetMediaPoolItem() if hasattr(item, "GetMediaPoolItem") else None
            media_path = None
            if media_item:
                path = media_item.GetClipProperty("File Path")
                if isinstance(path, str) and path:
                    media_path = path

            fps_value = float(fps.numerator) / float(fps.denominator) if fps.denominator else 0.0
            duration_seconds = duration_frames / fps_value if fps_value else 0.0

            clip_uid, source = compute_clip_uid(
                item,
                media_path=media_path,
                timeline_in=timeline_in_tc,
                timeline_out=timeline_out_tc,
                track_index=track_index,
            )

            clip_contexts.append(
                ClipContext(
                    payload=ClipPayload(
                        clip_id=clip_uid,
                        clip_uid_source=source,
                        clip_name=item.GetName() or "Unnamed Clip",
                        timeline_in=timeline_in_tc,
                        timeline_out=timeline_out_tc,
                        track_index=track_index,
                        duration_frames=duration_frames,
                        duration_seconds=duration_seconds,
                        media_path=media_path,
                    ),
                    item=item,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

    return clip_contexts, fps, timeline_start


def log_warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def timecode_to_frames(timecode: str, fps: Fraction) -> Optional[int]:
    parts = timecode.split(":")
    if len(parts) != 4:
        return None
    try:
        hours, minutes, seconds, frames = [int(part) for part in parts]
    except ValueError:
        return None
    fps_fraction = Fraction(fps).limit_denominator()
    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_frames = Fraction(total_seconds) * fps_fraction + frames
    return int(total_frames)


def coerce_mark_frame(value, fps: Fraction) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return timecode_to_frames(value, fps)
    return None


def get_mark_range(timeline, fps: Fraction) -> Optional[Tuple[int, int]]:
    if timeline is None:
        return None

    if hasattr(timeline, "GetInOutRange"):
        try:
            mark_range = timeline.GetInOutRange()
        except TypeError:
            mark_range = None
        except Exception:
            mark_range = None
        if isinstance(mark_range, (list, tuple)) and len(mark_range) == 2:
            start = coerce_mark_frame(mark_range[0], fps)
            end = coerce_mark_frame(mark_range[1], fps)
            if start is not None and end is not None and end > start:
                return int(start), int(end)

    getter_pairs = (
        ("GetMarkIn", "GetMarkOut"),
        ("GetMarkInFrame", "GetMarkOutFrame"),
        ("GetMarkIn", "GetMarkOut"),
        ("GetInPoint", "GetOutPoint"),
        ("GetInTimecode", "GetOutTimecode"),
    )
    for start_attr, end_attr in getter_pairs:
        if not hasattr(timeline, start_attr) or not hasattr(timeline, end_attr):
            continue
        try:
            start_raw = getattr(timeline, start_attr)()
            end_raw = getattr(timeline, end_attr)()
        except TypeError:
            continue
        except Exception:
            continue
        start = coerce_mark_frame(start_raw, fps)
        end = coerce_mark_frame(end_raw, fps)
        if start is not None and end is not None and end > start:
            return int(start), int(end)

    return None


def overlaps(range_a: Tuple[int, int], range_b: Tuple[int, int]) -> bool:
    start_a, end_a = range_a
    start_b, end_b = range_b
    return end_a > start_b and end_b > start_a


def filter_contexts_by_range(
    contexts: List[ClipContext],
    mark_in: int,
    mark_out: int,
    timeline_start: int,
) -> List[ClipContext]:
    # Try relative-to-start filtering first
    selected = [
        ctx
        for ctx in contexts
        if overlaps(
            (ctx.start_frame - timeline_start, ctx.end_frame - timeline_start),
            (mark_in, mark_out),
        )
    ]
    if selected:
        return selected
    # Fallback to absolute frames in case the API returns absolute positions
    return [
        ctx
        for ctx in contexts
        if overlaps((ctx.start_frame, ctx.end_frame), (mark_in, mark_out))
    ]


def _extract_items(raw_selected) -> List[object]:
    items: List[object] = []
    if isinstance(raw_selected, dict):
        for value in raw_selected.values():
            if isinstance(value, list):
                items.extend(value)
            elif value:
                items.append(value)
    elif isinstance(raw_selected, (list, tuple)):
        for value in raw_selected:
            if value:
                items.append(value)
    elif raw_selected:
        items.append(raw_selected)
    return items


def get_selected_contexts(timeline, contexts: List[ClipContext]) -> List[ClipContext]:
    if timeline is None:
        return []

    selected_items: List[object] = []
    getters = [(), ("video",), ("timeline",)]
    if hasattr(timeline, "GetSelectedItems"):
        for args in getters:
            try:
                raw = timeline.GetSelectedItems(*args)
            except TypeError:
                continue
            except Exception:
                continue
            selected_items = _extract_items(raw)
            if selected_items:
                break

    if not selected_items:
        for getter_name in ("GetCurrentTimelineItem", "GetCurrentClip", "GetCurrentVideoItem"):
            if not hasattr(timeline, getter_name):
                continue
            try:
                item = getattr(timeline, getter_name)()
            except Exception:
                continue
            if item:
                selected_items = [item]
                break

    if not selected_items:
        return []

    selected_ids = {id(item) for item in selected_items}
    return [ctx for ctx in contexts if id(ctx.item) in selected_ids]


def resolve_clip_subset(
    timeline,
    contexts: List[ClipContext],
    selection_mode: str,
    fps: Fraction,
) -> Tuple[List[ClipContext], str]:
    requested = selection_mode.lower()

    if requested == "all":
        return contexts, "all"

    if requested == "inout":
        mark_range = get_mark_range(timeline, fps)
        if not mark_range:
            log_warn("Mark In/Out not set; falling back to full timeline selection")
            return contexts, "all"
        selected = filter_contexts_by_range(contexts, mark_range[0], mark_range[1], timeline_start)
        if not selected:
            log_warn("No clips intersect Mark In/Out; falling back to full timeline selection")
            return contexts, "all"
        return selected, "inout"

    if requested == "selected":
        selected_contexts = get_selected_contexts(timeline, contexts)
        if selected_contexts:
            return selected_contexts, "selected"
        log_warn("Selected clips not available via API; falling back to Mark In/Out or full timeline")
        mark_range = get_mark_range(timeline, fps)
        if mark_range:
            selected = filter_contexts_by_range(contexts, mark_range[0], mark_range[1], timeline_start)
            if selected:
                log_warn("Using Mark In/Out range instead of selected clips")
                return selected, "inout"
            log_warn("Mark In/Out range has no clips; falling back to full timeline selection")
        else:
            log_warn("Mark In/Out not set; falling back to full timeline selection")
        return contexts, "all"

    # Unknown mode - treat as full timeline to stay safe
    log_warn(f"Unknown selection mode '{selection_mode}', defaulting to full timeline")
    return contexts, "all"


def post_with_retry(url: str, payload: dict, timeout_sec: int) -> requests.Response:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, timeout=(timeout_sec, timeout_sec))
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            wait_time = 2 ** attempt
            print(f"[WARN] POST failed ({error}); retrying in {wait_time}s", file=sys.stderr)
            time.sleep(wait_time)
    raise RuntimeError("Failed to submit analysis request after retries")


def poll_status(base_url: str, job_id: str, timeout_sec: int) -> dict:
    status_url = f"{base_url}/status/{job_id}"
    deadline = time.time() + GLOBAL_TIMEOUT_SECONDS
    while time.time() < deadline:
        response = requests.get(status_url, timeout=(timeout_sec, timeout_sec))
        if response.status_code == 404:
            raise RuntimeError("Service returned 404 for job status")
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "completed":
            return payload
        if payload.get("status") == "failed":
            raise RuntimeError("Analysis job failed")
        time.sleep(STATUS_POLL_SECONDS)
    raise TimeoutError("Timed out waiting for analysis job to finish")


def fetch_result(base_url: str, job_id: str, timeout_sec: int) -> dict:
    result_url = f"{base_url}/result/{job_id}"
    response = requests.get(result_url, timeout=(timeout_sec, timeout_sec))
    response.raise_for_status()
    return response.json()


def list_remote_jobs(base_url: str, limit: int, timeout_sec: int) -> None:
    params = {"limit": limit}
    response = requests.get(f"{base_url}/jobs", params=params, timeout=(timeout_sec, timeout_sec))
    response.raise_for_status()
    jobs = response.json()
    if not jobs:
        print("No jobs recorded yet.")
        return
    for job in jobs:
        job_id = job.get("id")
        run_id = job.get("run_id") or "-"
        status = job.get("status")
        created_at = job.get("created_at")
        finished_at = job.get("finished_at") or "-"
        processed = job.get("processed")
        skipped = job.get("skipped")
        clusters = job.get("clusters")
        undo_path = job.get("undo_path") or "-"
        print(
            f"{job_id} run={run_id} status={status} processed={processed} skipped={skipped} clusters={clusters} "
            f"created={created_at} finished={finished_at}"
            f" undo={undo_path}"
        )


def show_remote_job(base_url: str, job_id: str, timeout_sec: int) -> None:
    response = requests.get(f"{base_url}/jobs/{job_id}", timeout=(timeout_sec, timeout_sec))
    if response.status_code == 404:
        print(f"Job {job_id} not found.")
        return
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def apply_colors(
    report_clips: List[dict],
    clip_contexts: List[ClipContext],
    clip_lookup: Dict[str, object],
    fps: Fraction,
    timeline_start: int,
    allow_apply: bool,
    diagnose: bool,
    reapply_colors: bool,
    dry_run: bool,
    export_only: bool,
) -> Dict[str, int]:
    context_by_id = {ctx.payload.clip_id: ctx for ctx in clip_contexts}
    contexts_by_name = defaultdict(list)
    tolerance_frames = 2
    for ctx in clip_contexts:
        start_rel = timecode_to_frames(ctx.payload.timeline_in, fps) or 0
        end_rel = timecode_to_frames(ctx.payload.timeline_out, fps) or start_rel
        contexts_by_name[ctx.payload.clip_name].append((ctx, start_rel, end_rel))

    used_fallback_ids: set[str] = set()
    stats_action = Counter()
    stats_reason = Counter()

    for clip in report_clips:
        clip_id = clip.get("clip_id")
        clip_name = clip.get("clip_name") or clip_id or "<unknown>"
        timeline_in = clip.get("timeline_in") or ""
        timeline_out = clip.get("timeline_out") or ""
        status = clip.get("status") or "ok"
        target_color = clip.get("color_name") or ""

        matched_context = context_by_id.get(clip_id)
        clip_item = clip_lookup.get(clip_id)

        if matched_context is None or clip_item is None:
            # fallback by name + tc
            candidates = contexts_by_name.get(clip_name, [])
            target_start = timecode_to_frames(timeline_in, fps)
            target_end = timecode_to_frames(timeline_out, fps)
            best_candidate = None
            best_delta = None
            if target_start is not None:
                for candidate_ctx, start_rel, _ in candidates:
                    if candidate_ctx.payload.clip_id in used_fallback_ids:
                        continue
                    delta = abs(start_rel - target_start)
                    if delta <= tolerance_frames and (best_delta is None or delta < best_delta):
                        best_candidate = candidate_ctx
                        best_delta = delta
            if best_candidate:
                matched_context = best_candidate
                clip_item = clip_lookup.get(best_candidate.payload.clip_id)
                if clip_item:
                    clip_id = best_candidate.payload.clip_id
                    clip["clip_id"] = clip_id
                    clip_name = best_candidate.payload.clip_name
                    clip["clip_name"] = clip_name
                    clip["clip_uid"] = clip_id
                    used_fallback_ids.add(best_candidate.payload.clip_id)

        current_color = None
        if clip_item:
            get_color = getattr(clip_item, "GetClipColor", None)
            if callable(get_color):
                try:
                    current_color = get_color()
                except Exception:
                    current_color = None

        action = "miss"
        reason = "no_match"
        after_color = current_color

        if clip_item is None or matched_context is None:
            pass
        elif status != "ok":
            action = "skip"
            reason = f"status_{status}"
        elif not target_color:
            action = "skip"
            reason = "no_target"
        else:
            if not allow_apply:
                action = "skip"
                reason = "dry_run" if dry_run else "export_only"
            else:
                colors_match = (current_color or "").lower() == target_color.lower()
                if reapply_colors or not colors_match:
                    try:
                        success = clip_item.SetClipColor(target_color)
                        if success:
                            action = "set"
                            reason = "applied"
                            after_color = target_color
                        else:
                            action = "skip"
                            reason = "apply_failed"
                    except Exception as error:
                        action = "skip"
                        reason = "apply_error"
                        print(f"[WARN] Failed to set color for {clip_name}: {error}", file=sys.stderr)
                else:
                    action = "skip"
                    reason = "same_color"

        clip["apply_action"] = action
        clip["apply_reason"] = reason
        clip["resolve_color_before"] = current_color
        clip["resolve_color_after"] = after_color

        stats_action[action] += 1
        stats_reason[f"{action}.{reason}"] += 1

        if diagnose:
            log_line = (
                f"[APPLY] clip_uid={clip.get('clip_id')} name=\"{clip_name}\" "
                f"tc={timeline_in}-{timeline_out} current={current_color or '-'} "
                f"target={target_color or '-'} action={action} reason={reason}"
            )
            print(log_line)

    summary: Dict[str, int] = {
        "set": stats_action.get("set", 0),
        "skip": stats_action.get("skip", 0),
        "miss": stats_action.get("miss", 0),
        "matched": len(report_clips) - stats_action.get("miss", 0),
    }
    for key, value in stats_reason.items():
        summary[f"{key}"] = value
    return summary


def build_cluster_color_map(clips: List[dict]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for clip in clips:
        cluster_id = clip.get("cluster_id")
        color_name = clip.get("color_name")
        if cluster_id is None or color_name in (None, ""):
            continue
        try:
            cluster_key = int(cluster_id)
        except (TypeError, ValueError):
            continue
        mapping.setdefault(cluster_key, str(color_name))
    return mapping


def _gather_locations_from_clips(clips: List[dict]) -> List[dict]:
    grouped: Dict[int, dict] = {}
    for clip in clips:
        location_id = clip.get("location_id")
        if location_id is None:
            continue
        try:
            location_key = int(location_id)
        except (TypeError, ValueError):
            continue
        entry = grouped.setdefault(
            location_key,
            {
                "location_id": location_key,
                "cluster_id": clip.get("cluster_id"),
                "clip_ids": [],
                "duration_seconds": clip.get("location_duration_sec"),
                "clip_count": 0,
                "start_tc": clip.get("location_span_start"),
                "end_tc": clip.get("location_span_end"),
                "centroid": clip.get("location_centroid"),
                "persisted_location_id": clip.get("persisted_location_id"),
            },
        )
        entry["clip_ids"].append(clip.get("clip_id"))
        entry["clip_count"] += 1
        span_start = clip.get("location_span_start")
        span_end = clip.get("location_span_end")
        if span_start and (not entry.get("start_tc") or span_start < entry.get("start_tc")):
            entry["start_tc"] = span_start
        if span_end and (not entry.get("end_tc") or span_end > entry.get("end_tc")):
            entry["end_tc"] = span_end
        if entry.get("duration_seconds") is None and clip.get("location_duration_sec") is not None:
            entry["duration_seconds"] = clip.get("location_duration_sec")
        if entry.get("centroid") is None and clip.get("location_centroid") is not None:
            entry["centroid"] = clip.get("location_centroid")
        if entry.get("persisted_location_id") is None and clip.get("persisted_location_id") is not None:
            entry["persisted_location_id"] = clip.get("persisted_location_id")
    return [grouped[key] for key in sorted(grouped.keys())]


def resolve_location_detail(summary: Dict[str, object], clips: List[dict]) -> List[dict]:
    locations_raw = summary.get("locations_detail") if isinstance(summary, dict) else None
    if isinstance(locations_raw, list) and locations_raw:
        detail: List[dict] = []
        for entry in locations_raw:
            if isinstance(entry, dict):
                detail.append(dict(entry))
        if detail:
            return detail
    return _gather_locations_from_clips(clips)


def add_location_markers(
    timeline,
    fps: Fraction,
    timeline_start: int,
    locations_detail: List[dict],
    cluster_colors: Dict[int, str],
    run_id: Optional[str],
) -> List[Dict[str, int]]:
    if not timeline or not locations_detail:
        return []

    undo_markers: List[Dict[str, int]] = []
    for entry in locations_detail:
        start_tc = entry.get("start_tc") or entry.get("location_span_start")
        location_label = entry.get("persisted_location_id")
        if location_label is None:
            location_label = entry.get("location_id")
        if start_tc is None:
            continue
        start_rel = timecode_to_frames(str(start_tc), fps)
        if start_rel is None:
            continue
        start_frame = timeline_start + int(start_rel)
        cluster_id = entry.get("cluster_id")
        color = cluster_colors.get(int(cluster_id)) if isinstance(cluster_id, (int, float)) else None
        marker_color = color or "Yellow"
        label_text = f"Loc {location_label}" if location_label is not None else "Loc"
        note = f"Location start"
        if run_id:
            note += f" (run={run_id})"
        try:
            added_start = timeline.AddMarker(start_frame, marker_color, label_text, note, 1, None)
        except Exception as error:
            print(f"[WARN] Failed to add start marker for {label_text}: {error}", file=sys.stderr)
            added_start = False
        if added_start:
            undo_markers.append({"frame": start_frame})

    if undo_markers:
        print(f"[INFO] Added {len(undo_markers)} location marker(s)")
    else:
        print("[INFO] No location markers added (insufficient data)")
    return undo_markers


def add_completion_marker(
    timeline,
    dry_run: bool,
    clip_count: int,
    cluster_count: int,
    run_id: Optional[str] = None,
) -> None:
    timeline_start = timeline.GetStartFrame() or 0
    name = (
        f"PigmentBucket • dry-run ({clip_count}/{cluster_count})"
        if dry_run
        else f"PigmentBucket • analysis done ({clip_count} clips / {cluster_count} clusters)"
    )
    if run_id:
        name = f"{name} • run {run_id}"
    color = "Lavender" if dry_run else "Cyan"
    timeline.AddMarker(timeline_start, color, name, "", 1, None)


def run(
    service_url: str,
    dry_run: bool,
    report_dir: Path,
    limit: Optional[int],
    selection_mode: str,
    sampler_backend: Optional[str],
    max_k: Optional[int],
    max_frames_per_clip: Optional[int],
    min_spacing_sec: Optional[float],
    http_timeout: int,
    force_min_k: int,
    ignore_cache: bool,
    location_min_duration_sec: Optional[float],
    similarity_threshold: Optional[float],
    hysteresis: Optional[bool],
    location_store: str,
    export_locations_only: bool,
    add_resolve_markers: bool,
    report_format: str,
    auto_mode: bool,
    auto_target_k: Optional[int],
    auto_max_iters: Optional[int],
    diagnose_apply: bool,
    reapply_colors: bool,
    dump_frames: bool,
) -> None:
    resolve = load_resolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No active project. Open a project before running the script.")

    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise RuntimeError("No active timeline. Create or open a timeline before running the script.")

    project_id: Optional[str] = None
    get_name = getattr(project, "GetName", None)
    if callable(get_name):
        try:
            name_candidate = get_name()
            if isinstance(name_candidate, str) and name_candidate:
                project_id = name_candidate
        except Exception:
            project_id = None

    if diagnose_apply:
        timeline_name = timeline.GetName() if hasattr(timeline, "GetName") else "<unknown>"
        project_name = project_id or "<unknown>"
        print(
            f"[INFO] Resolve project={project_name} timeline={timeline_name} selection_mode={selection_mode}"
        )

    if force_min_k is None or force_min_k < 1:
        if force_min_k is not None and force_min_k < 1:
            print("[WARN] --force-min-k must be >= 1; defaulting to 1")
        force_min_k = 1
    else:
        force_min_k = int(force_min_k)

    # Ensure deterministic behavior for preview pipelines
    random.seed(42)
    if np is not None:
        np.random.seed(42)

    clip_contexts, fps, timeline_start = gather_clips(project)
    if diagnose_apply:
        print(f"[INFO] Timeline items={len(clip_contexts)}")

    selected_contexts, effective_mode = resolve_clip_subset(
        timeline=timeline,
        contexts=clip_contexts,
        selection_mode=selection_mode,
        fps=fps,
        timeline_start=timeline_start,
    )

    if limit is not None and limit >= 0:
        limited_contexts = selected_contexts[: limit]
        if len(limited_contexts) < len(selected_contexts):
            print(f"Limiting submission to {len(limited_contexts)} clip(s) (limit={limit})")
        selected_contexts = limited_contexts

    if not selected_contexts:
        print(f"No clips found for selection mode '{selection_mode}'")
        return

    clips = [ctx.payload for ctx in selected_contexts]
    clip_lookup = {ctx.payload.clip_id: ctx.item for ctx in selected_contexts}

    clip_context_hash = compute_clip_context_hash(clips)

    print(f"selection={selection_mode}, effective_mode={effective_mode}, clips={len(clips)}")
    if clips:
        preview_names = ", ".join(clip.clip_name for clip in clips[:PREVIEW_LIMIT])
        if len(clips) > PREVIEW_LIMIT:
            preview_names += ", …"
        print(f"Selected clips: {preview_names}")
    if ignore_cache:
        print("[INFO] Feature cache disabled for this run")
    if force_min_k > 1:
        print(f"[INFO] Minimum clusters enforced: {force_min_k}")
    if location_min_duration_sec is not None:
        print(f"[INFO] Location min duration override: {location_min_duration_sec:.2f}s")
    if similarity_threshold is not None:
        print(f"[INFO] Location similarity threshold: {similarity_threshold:.2f}")
    if hysteresis is not None:
        state = "enabled" if hysteresis else "disabled"
        print(f"[INFO] Location hysteresis: {state}")

    store_choice = (location_store or "sqlite").lower()
    print(f"[INFO] Location store backend: {store_choice}")

    report_choice = (report_format or "both").lower()
    if report_choice not in {"json", "csv", "both"}:
        report_choice = "both"
    print(f"[INFO] Report format: {report_choice}")

    if auto_mode:
        target_display = auto_target_k if auto_target_k is not None else 2
        max_iters_display = auto_max_iters if auto_max_iters is not None else 3
        print(
            "[INFO] Auto mode enabled: target_k=%s max_iters=%s"
            % (target_display, max_iters_display)
        )

    effective_dry_run = dry_run or export_locations_only
    if export_locations_only:
        print("[INFO] Export-only mode: skipping Resolve color and marker updates")

    payload = {
        "clips": [clip.__dict__ for clip in clips],
        "dry_run": effective_dry_run,
    }
    payload["selection_mode"] = effective_mode
    run_id = uuid.uuid4().hex
    payload["run_id"] = run_id

    config_payload: Dict[str, object] = {}
    if sampler_backend:
        config_payload["sampler_backend"] = sampler_backend
    if max_k is not None:
        config_payload["max_k"] = max_k
    if max_frames_per_clip is not None:
        config_payload["max_frames_per_clip"] = max_frames_per_clip
    if min_spacing_sec is not None:
        config_payload["min_spacing_sec"] = min_spacing_sec
    if force_min_k and force_min_k > 1:
        config_payload["force_min_k"] = force_min_k
    if ignore_cache:
        config_payload["ignore_cache"] = True
    if location_min_duration_sec is not None and location_min_duration_sec >= 0:
        config_payload["location_min_duration_sec"] = location_min_duration_sec
    if similarity_threshold is not None:
        config_payload["location_similarity_threshold"] = similarity_threshold
    if hysteresis is not None:
        config_payload["location_hysteresis"] = bool(hysteresis)
    if store_choice:
        config_payload["location_store"] = store_choice
    if report_choice:
        config_payload["report_format"] = report_choice
    if auto_mode:
        config_payload["auto_mode"] = True
    if auto_target_k is not None and auto_target_k >= 2:
        config_payload["auto_target_k"] = int(auto_target_k)
    if auto_max_iters is not None and auto_max_iters >= 1:
        config_payload["auto_max_iters"] = int(auto_max_iters)
    if project_id:
        config_payload["project_id"] = project_id
    if clip_context_hash:
        config_payload["clip_context_hash"] = clip_context_hash
    if dump_frames:
        config_payload["dump_frames"] = True
    if config_payload:
        payload["config"] = config_payload

    print(f"Submitting {len(clips)} clip(s) to analysis service at {service_url}...")
    response = post_with_retry(f"{service_url}/analyze", payload, timeout_sec=http_timeout)
    response_payload = response.json()
    job_id = response_payload["job_id"]
    run_id = response_payload.get("run_id", run_id)
    print(f"Job id: {job_id}  run_id: {run_id}")

    try:
        poll_status(service_url, job_id, timeout_sec=http_timeout)
        result = fetch_result(service_url, job_id, timeout_sec=http_timeout)
    except KeyboardInterrupt:
        stop_remote_job(service_url, job_id, http_timeout, reason="client stop")
        print("[WARN] Job stopped before completion; no changes applied.")
        return
    result_run_id = result.get("run_id")
    if result_run_id:
        run_id = result_run_id
    summary = result.get("summary", {})
    processed = summary.get("processed", 0)
    skipped = summary.get("skipped", 0)
    clusters = summary.get("clusters", 0)

    print(json.dumps(summary, indent=2))
    coloring_decision: Dict[str, object] = {}
    if isinstance(summary, dict):
        coloring_decision = summary.get("coloring") or {}
    if coloring_decision.get("coloring") == "skipped":
        reason = coloring_decision.get("reason", "k==1")
        print(f"[INFO] Coloring skipped: {reason}")
    elif coloring_decision.get("coloring") == "forced":
        reason = coloring_decision.get("reason", "force-min-k")
        print(f"[INFO] Coloring forced via heuristic ({reason})")

    store_backend = summary.get("location_store")
    if store_backend:
        print(f"[INFO] Location store result: {store_backend}")
    report_format_summary = summary.get("report_format")
    if report_format_summary:
        print(f"[INFO] Report format (server): {report_format_summary}")
    persisted_total = summary.get("locations_persisted")
    matched_total = summary.get("matched_locations")
    created_total = summary.get("new_locations")
    if any(value is not None for value in (persisted_total, matched_total, created_total)):
        print(
            "[INFO] Location persistence stats: total={total} matched={matched} new={created}".format(
                total=persisted_total if persisted_total is not None else "?",
                matched=matched_total if matched_total is not None else "?",
                created=created_total if created_total is not None else "?",
            )
        )

    quality_metric_name = summary.get("quality_metric_name")
    quality_metric_value = summary.get("quality_metric_value")
    if quality_metric_name and quality_metric_value is not None:
        print(f"[INFO] Quality metric: {quality_metric_name}={float(quality_metric_value):.4f}")
    elif summary.get("median_hue_spread") is not None:
        print(f"[INFO] Median hue spread: {float(summary['median_hue_spread']):.4f}")

    locations_detected = summary.get("locations") if isinstance(summary, dict) else None
    if locations_detected is not None:
        print(f"[INFO] Locations detected: {locations_detected}")
        locations_stats = summary.get("locations_stats") if isinstance(summary, dict) else None
        if isinstance(locations_stats, dict):
            clip_stats = locations_stats.get("clip_count") or {}
            duration_stats = locations_stats.get("duration_sec") or {}
            clip_stats_str = ", ".join(
                f"{key}={int(value) if isinstance(value, (int, float)) else value}"
                for key, value in clip_stats.items()
            )
            duration_stats_str = ", ".join(
                f"{key}={float(value):.2f}" if isinstance(value, (int, float)) else f"{key}={value}"
                for key, value in duration_stats.items()
            )
            if clip_stats_str:
                print(f"[INFO] Location clip counts: {clip_stats_str}")
            if duration_stats_str:
                print(f"[INFO] Location durations (s): {duration_stats_str}")

    if summary.get("auto_mode"):
        iterations = summary.get("auto_iterations")
        final_similarity = summary.get("auto_similarity_threshold")
        final_min_duration = summary.get("auto_min_duration_sec")
        initial_source = summary.get("auto_initial_source") or "default"
        print(
            "[INFO] Auto mode summary: iterations=%s final sim=%.3f min_dur=%.2fs source=%s"
            % (
                iterations if iterations is not None else "?",
                float(final_similarity) if final_similarity is not None else float("nan"),
                float(final_min_duration) if final_min_duration is not None else float("nan"),
                initial_source,
            )
        )
        history_entries = summary.get("auto_history") or []
        for entry in history_entries:
            iteration = entry.get("iteration")
            sim_value = entry.get("similarity")
            min_value = entry.get("min_duration")
            k_value = entry.get("chosen_k")
            sil_value = entry.get("silhouette")
            decision = entry.get("decision") or ""
            sil_text = "n/a" if sil_value is None else f"{float(sil_value):.3f}"
            print(
                "[INFO] Auto attempt %s: sim=%.3f min_dur=%.2fs k=%s sil=%s decision=%s"
                % (
                    iteration if iteration is not None else "?",
                    float(sim_value) if sim_value is not None else float("nan"),
                    float(min_value) if min_value is not None else float("nan"),
                    k_value if k_value is not None else "?",
                    sil_text,
                    decision,
                )
            )

    allow_apply = not effective_dry_run and not export_locations_only and not coloring_skipped
    apply_stats: Optional[Dict[str, int]] = None
    if diagnose_apply or allow_apply:
        apply_stats = apply_colors(
            result.get("clips", []),
            selected_contexts,
            clip_lookup,
            fps,
            allow_apply=allow_apply,
            diagnose=diagnose_apply,
            reapply_colors=reapply_colors,
            dry_run=effective_dry_run,
            export_only=export_locations_only,
        )
        summary["apply_stats"] = apply_stats
        if diagnose_apply and apply_stats:
            reason_breakdown = [
                f"{key.split('.', 1)[1]}={value}"
                for key, value in apply_stats.items()
                if "." in key and value
            ]
            totals = f"set={apply_stats.get('set', 0)} skip={apply_stats.get('skip', 0)} miss={apply_stats.get('miss', 0)}"
            extra = f" ({', '.join(reason_breakdown)})" if reason_breakdown else ""
            print(f"[INFO] Apply summary: {totals}{extra}")
        if allow_apply and not diagnose_apply:
            print(f"Applied colors to {apply_stats.get('set', 0)} clip(s)")

    if summary.get("frames_dir"):
        frames_dir = summary.get("frames_dir")
        frames_count = summary.get("dumped_frames_count")
        count_text = f" ({frames_count} files)" if frames_count is not None else ""
        print(f"[INFO] Frames saved to {frames_dir}{count_text}")

    summary_locations = resolve_location_detail(summary, result.get("clips", []))
    cluster_colors = build_cluster_color_map(result.get("clips", []))

    undo_snapshot: List[Dict[str, Optional[str]]] = []
    marker_snapshot: List[Dict[str, int]] = []
    coloring_skipped = coloring_decision.get("coloring") == "skipped"
    capture_colors = not effective_dry_run and not export_locations_only
    if capture_colors and not coloring_skipped:
        undo_snapshot = snapshot_clip_colors(clips, clip_lookup)
    elif capture_colors and coloring_skipped:
        print("[INFO] Undo snapshot skipped because no colors were applied")

    if export_locations_only:
        print("[INFO] Coloring not applied in export-only mode")
    elif effective_dry_run:
        print("Coloring not applied due to dry run")
    elif coloring_skipped:
        print("Coloring not applied due to cluster decision")

    should_add_markers = (
        add_resolve_markers
        and not effective_dry_run
        and not export_locations_only
        and bool(summary_locations)
    )
    if should_add_markers:
        marker_snapshot = add_location_markers(
            timeline,
            fps,
            timeline_start,
            summary_locations,
            cluster_colors,
            run_id,
        )
    elif add_resolve_markers:
        if effective_dry_run or export_locations_only:
            print("[INFO] Location markers disabled (dry-run/export-only)")
        elif not summary_locations:
            print("[INFO] No location detail available; markers not added")

    if timeline:
        add_completion_marker(
            timeline,
            dry_run=effective_dry_run,
            clip_count=processed,
            cluster_count=clusters,
            run_id=run_id,
        )

    report_path = report_dir / f"{job_id}.json"
    try:
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved local report to {report_path}")
    except OSError as error:
        print(f"[WARN] Failed to write report {report_path}: {error}", file=sys.stderr)

    if not effective_dry_run and (undo_snapshot or marker_snapshot):
        undo_path = write_undo_file(
            run_id=run_id,
            job_id=job_id,
            selection_mode=selection_mode,
            service_url=service_url,
            summary=summary,
            clip_snapshot=undo_snapshot,
            result_clips=result.get("clips", []),
            report_path=report_path,
            markers=marker_snapshot,
        )
        attach_undo_artifact(service_url, job_id, undo_path, http_timeout)
        print(f"Undo snapshot saved to {undo_path}")
    elif marker_snapshot:
        print("[WARN] Marker snapshot not recorded; dry-run prevents undo export")

    print(f"Processed {processed} clip(s), skipped {skipped}. Report handled by service.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pigment Bucket analysis from Resolve")
    parser.add_argument(
        "--service-url",
        default=None,
        help="Base URL for the analysis service (default: env PIGMENT_SERVICE_URL or http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="logs",
        help="Directory to store local reports (default: logs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips submitted for analysis (default: no limit)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Maximum clusters considered by KMeans auto-selection (default: 8)",
    )
    parser.add_argument(
        "--sampler-backend",
        choices=["auto", "resolve", "ffmpeg"],
        default=None,
        help="Sampler backend to use (default: auto)",
    )
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=None,
        help="Maximum number of frames sampled per clip (default: 5)",
    )
    parser.add_argument(
        "--min-spacing-sec",
        type=float,
        default=None,
        help="Minimum spacing between sampled frames in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--report-format",
        choices=["json", "csv", "both"],
        default="both",
        help="Server-side report format to persist (default: both)",
    )
    parser.add_argument(
        "--diagnose-apply",
        action="store_true",
        help="Enable verbose diagnostic logging when applying colors",
    )
    parser.add_argument(
        "--reapply-colors",
        action="store_true",
        help="Set clip colors even if they already match the target",
    )
    parser.add_argument(
        "--dump-frames",
        action="store_true",
        help="Dump sampled frames to artifacts/frames/<job_id>/ for verification",
    )
    parser.add_argument(
        "--auto-mode",
        action="store_true",
        help="Enable automatic tuning of location grouping parameters",
    )
    parser.add_argument(
        "--auto-target-k",
        type=int,
        default=None,
        help="Target minimum number of clusters when auto mode is enabled (default: 2)",
    )
    parser.add_argument(
        "--auto-max-iters",
        type=int,
        default=None,
        help="Maximum auto-tuning iterations (default: 3)",
    )
    parser.add_argument(
        "--http-timeout",
        type=int,
        default=HTTP_TIMEOUT_DEFAULT,
        help="HTTP timeout (seconds) for connect/read to the analysis service (env: PIGMENT_HTTP_TIMEOUT, default: 60)",
    )
    parser.add_argument(
        "--min-duration-sec",
        "--location-min-duration-sec",
        dest="location_min_duration_sec",
        type=float,
        default=None,
        help="Minimum duration (seconds) to form a standalone location (default: 3.0)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for merging adjacent segments (default: 0.92)",
    )
    parser.add_argument(
        "--hysteresis",
        dest="hysteresis",
        action="store_true",
        default=None,
        help="Enable hysteresis merge for short middle segments",
    )
    parser.add_argument(
        "--no-hysteresis",
        dest="hysteresis",
        action="store_false",
        help="Disable hysteresis merge for short middle segments",
    )
    parser.add_argument(
        "--locations-store",
        choices=["sqlite", "json"],
        default="sqlite",
        help="Location persistence backend (default: sqlite)",
    )
    parser.add_argument(
        "--export-locations-only",
        action="store_true",
        help="Skip Resolve updates and export only location reports",
    )
    parser.add_argument(
        "--add-resolve-markers",
        dest="add_resolve_markers",
        action="store_true",
        help="Add Resolve timeline markers for each detected location",
    )
    parser.add_argument(
        "--add-location-markers",
        dest="add_resolve_markers",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force-min-k",
        type=int,
        default=FORCE_MIN_K_DEFAULT,
        help="Force minimum cluster count when auto-selection yields a single cluster (default: env PIGMENT_FORCE_MIN_K or 1)",
    )
    parser.add_argument(
        "--ignore-cache",
        dest="ignore_cache",
        action="store_true",
        default=IGNORE_CACHE_DEFAULT,
        help="Ignore feature cache and recompute features (env: PIGMENT_IGNORE_CACHE)",
    )
    parser.add_argument(
        "--use-cache",
        dest="ignore_cache",
        action="store_false",
        help="Explicitly enable feature cache even if PIGMENT_IGNORE_CACHE is set",
    )
    parser.set_defaults(ignore_cache=IGNORE_CACHE_DEFAULT, hysteresis=None, add_resolve_markers=False)
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List recent jobs from the service history and exit",
    )
    parser.add_argument(
        "--jobs-limit",
        type=int,
        default=10,
        help="Number of jobs to list when using --list-jobs (default: 10)",
    )
    parser.add_argument(
        "--job",
        dest="job_id",
        type=str,
        default=None,
        help="Show detailed information for the given job id",
    )
    parser.add_argument(
        "--stop-job",
        dest="stop_job_id",
        type=str,
        default=None,
        help="Request job cancellation for the given job id",
    )
    parser.add_argument(
        "--undo",
        dest="undo_run_id",
        type=str,
        default=None,
        help="Restore clip colors using the undo snapshot for the given run id",
    )
    parser.add_argument(
        "--undo-markers",
        dest="undo_markers_run_id",
        type=str,
        default=None,
        help="Remove only Resolve markers using the undo snapshot for the given run id",
    )
    parser.add_argument(
        "--selection",
        choices=["all", "inout", "selected"],
        default="all",
        help="Choose clip subset: all timeline clips, Mark In/Out range, or selected clips",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect results without applying SetClipColor",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    service_url = args.service_url or os.environ.get(SERVICE_URL_ENV, SERVICE_URL_DEFAULT)
    service_url = service_url.rstrip("/")

    if args.undo_run_id:
        perform_undo(args.undo_run_id, service_url, args.http_timeout)
        return 0

    if getattr(args, "undo_markers_run_id", None):
        perform_marker_undo(args.undo_markers_run_id, service_url, args.http_timeout)
        return 0

    if args.stop_job_id:
        stop_remote_job(service_url, args.stop_job_id, args.http_timeout, reason="manual stop command")
        return 0

    if args.list_jobs:
        list_remote_jobs(service_url, max(1, args.jobs_limit), args.http_timeout)
        if args.job_id:
            show_remote_job(service_url, args.job_id, args.http_timeout)
        return 0

    if args.job_id:
        show_remote_job(service_url, args.job_id, args.http_timeout)
        return 0

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(
            service_url=service_url,
            dry_run=args.dry_run,
            report_dir=report_dir,
            limit=args.limit,
            selection_mode=args.selection,
            sampler_backend=args.sampler_backend,
            max_k=args.max_k,
            max_frames_per_clip=args.max_frames_per_clip,
            min_spacing_sec=args.min_spacing_sec,
            http_timeout=args.http_timeout,
            force_min_k=args.force_min_k,
            ignore_cache=args.ignore_cache,
            location_min_duration_sec=args.location_min_duration_sec,
            similarity_threshold=args.similarity_threshold,
            hysteresis=args.hysteresis,
            location_store=args.locations_store,
            export_locations_only=args.export_locations_only,
            add_resolve_markers=args.add_resolve_markers,
            report_format=args.report_format,
            auto_mode=args.auto_mode,
            auto_target_k=args.auto_target_k,
            auto_max_iters=args.auto_max_iters,
            diagnose_apply=args.diagnose_apply,
            reapply_colors=args.reapply_colors,
            dump_frames=args.dump_frames,
        )
    except ResolveNotAvailable as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - integration level logging
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
