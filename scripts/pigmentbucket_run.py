#!/usr/bin/env python3
"""DaVinci Resolve adapter for the Pigment Bucket mock analysis service."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import requests

# Resolve API reference: docs/resolve_api_reference.md
SERVICE_URL_DEFAULT = "http://127.0.0.1:8765"
STATUS_POLL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 5
RETRY_ATTEMPTS = 3
GLOBAL_TIMEOUT_SECONDS = 15 * 60


@dataclass
class ClipPayload:
    clip_id: str
    clip_uid_source: str
    clip_name: str
    timeline_in: str
    timeline_out: str
    track_index: int
    duration_frames: int


class ResolveNotAvailable(RuntimeError):
    """Raised when the Resolve scripting API cannot be imported."""


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


def gather_clips(project) -> Tuple[List[ClipPayload], Dict[str, object]]:
    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise RuntimeError("No active timeline found")

    fps = parse_frame_rate(timeline.GetSetting("timelineFrameRate"))
    timeline_start = timeline.GetStartFrame() or 0

    clip_payloads: List[ClipPayload] = []
    clip_lookup: Dict[str, object] = {}

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

            clip_uid, source = compute_clip_uid(
                item,
                media_path=media_path,
                timeline_in=timeline_in_tc,
                timeline_out=timeline_out_tc,
                track_index=track_index,
            )

            clip_payloads.append(
                ClipPayload(
                    clip_id=clip_uid,
                    clip_uid_source=source,
                    clip_name=item.GetName() or "Unnamed Clip",
                    timeline_in=timeline_in_tc,
                    timeline_out=timeline_out_tc,
                    track_index=track_index,
                    duration_frames=duration_frames,
                )
            )
            clip_lookup[clip_uid] = item

    return clip_payloads, clip_lookup


def post_with_retry(url: str, payload: dict) -> requests.Response:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            wait_time = 2 ** attempt
            print(f"[WARN] POST failed ({error}); retrying in {wait_time}s", file=sys.stderr)
            time.sleep(wait_time)
    raise RuntimeError("Failed to submit analysis request after retries")


def poll_status(base_url: str, job_id: str) -> dict:
    status_url = f"{base_url}/status/{job_id}"
    deadline = time.time() + GLOBAL_TIMEOUT_SECONDS
    while time.time() < deadline:
        response = requests.get(status_url, timeout=REQUEST_TIMEOUT_SECONDS)
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


def fetch_result(base_url: str, job_id: str) -> dict:
    result_url = f"{base_url}/result/{job_id}"
    response = requests.get(result_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def apply_colors(clips: List[dict], lookup: Dict[str, object], dry_run: bool) -> None:
    applied = 0
    for clip in clips:
        if clip.get("status") != "ok":
            continue
        clip_item = lookup.get(clip["clip_id"])
        if not clip_item:
            continue
        if dry_run:
            continue
        color_name = clip.get("color_name")
        if not color_name:
            continue
        success = clip_item.SetClipColor(color_name)
        if success:
            applied += 1
    if not dry_run:
        print(f"Applied colors to {applied} clip(s)")


def add_completion_marker(timeline, dry_run: bool, clip_count: int, cluster_count: int) -> None:
    timeline_start = timeline.GetStartFrame() or 0
    name = (
        f"PigmentBucket • dry-run ({clip_count}/{cluster_count})"
        if dry_run
        else f"PigmentBucket • analysis done ({clip_count} clips / {cluster_count} clusters)"
    )
    color = "Lavender" if dry_run else "Cyan"
    timeline.AddMarker(timeline_start, color, name, "", 1, None)


def run(service_url: str, dry_run: bool) -> None:
    resolve = load_resolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No active project. Open a project before running the script.")

    clips, lookup = gather_clips(project)
    if not clips:
        print("No clips found on the active timeline")
        return

    payload = {
        "clips": [clip.__dict__ for clip in clips],
        "dry_run": dry_run,
    }

    print(f"Submitting {len(clips)} clip(s) to analysis service at {service_url}...")
    response = post_with_retry(f"{service_url}/analyze", payload)
    job_id = response.json()["job_id"]
    print(f"Job id: {job_id}")

    poll_status(service_url, job_id)
    result = fetch_result(service_url, job_id)
    summary = result.get("summary", {})
    processed = summary.get("processed", 0)
    skipped = summary.get("skipped", 0)
    clusters = summary.get("clusters", 0)

    print(json.dumps(summary, indent=2))
    apply_colors(result.get("clips", []), lookup, dry_run=dry_run)

    timeline = project.GetCurrentTimeline()
    if timeline:
        add_completion_marker(timeline, dry_run=dry_run, clip_count=processed, cluster_count=clusters)

    print(f"Processed {processed} clip(s), skipped {skipped}. Report handled by service.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pigment Bucket analysis from Resolve")
    parser.add_argument(
        "--service-url",
        default=SERVICE_URL_DEFAULT,
        help="Base URL for the analysis service (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect results without applying SetClipColor",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(service_url=args.service_url.rstrip("/"), dry_run=args.dry_run)
    except ResolveNotAvailable as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - integration level logging
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

