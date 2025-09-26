"""Frame sampling implementations for Pigment Bucket."""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .types import ClipContext, FrameSample

FFMPEG_PATH = shutil.which("ffmpeg")


class SamplingError(RuntimeError):
    """Raised when frame sampling for a clip fails."""


@dataclass
class SamplerConfig:
    max_frames_per_clip: int = 5
    min_spacing_sec: float = 2.0
    backend: str = "auto"  # auto | resolve | ffmpeg


class Sampler:
    """Abstract sampler interface."""

    def sample(self, clip: ClipContext) -> List[FrameSample]:
        raise NotImplementedError


class FrameSampler(Sampler):
    """Real sampler with Resolve/ffmpeg backends and safe fallbacks."""

    def __init__(self, config: SamplerConfig, logger: Optional[logging.Logger] = None) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)

    def sample(self, clip: ClipContext) -> List[FrameSample]:  # noqa: D401
        backend = self._select_backend()
        self._logger.debug("Sampling clip %s using backend=%s", clip.clip_id, backend)

        if backend == "resolve":
            try:
                return self._sample_via_resolve(clip)
            except SamplingError as error:
                self._logger.warning("Resolve sampling failed for %s: %s; falling back to ffmpeg", clip.clip_id, error)
                backend = "ffmpeg"

        if backend == "ffmpeg":
            return self._sample_via_ffmpeg(clip)

        raise SamplingError(f"No available sampler backend for clip {clip.clip_id}")

    # ------------------------------------------------------------------
    def _select_backend(self) -> str:
        backend = self._config.backend.lower()
        if backend == "auto":
            if self._resolve_available():
                return "resolve"
            if FFMPEG_PATH:
                return "ffmpeg"
            return "resolve"  # last resort; will error with clear message
        if backend == "resolve":
            return "resolve"
        if backend == "ffmpeg":
            if not FFMPEG_PATH:
                raise SamplingError("ffmpeg backend requested but ffmpeg is not available in PATH")
            return "ffmpeg"
        raise SamplingError(f"Unsupported sampler backend '{self._config.backend}'")

    def _resolve_available(self) -> bool:
        # Placeholder: Resolve sampling not available in backend process
        return False

    # ------------------------------------------------------------------
    def _sample_via_resolve(self, clip: ClipContext) -> List[FrameSample]:
        raise SamplingError("Resolve sampling is not available in service environment")

    # ------------------------------------------------------------------
    def _sample_via_ffmpeg(self, clip: ClipContext) -> List[FrameSample]:
        if not clip.media_path:
            raise SamplingError("ffmpeg backend requires media_path")
        source = Path(clip.media_path)
        if not source.exists():
            raise SamplingError(f"Media path does not exist: {source}")
        if not FFMPEG_PATH:
            raise SamplingError("ffmpeg executable not found in PATH")

        timestamps = self._compute_timestamps(max(clip.duration_seconds, 0.0))
        samples: List[FrameSample] = []

        with tempfile.TemporaryDirectory(prefix=f"pb_{clip.clip_id}_") as tmpdir:
            for index, ts in enumerate(timestamps):
                frame_path = Path(tmpdir) / f"frame_{index:03d}.png"
                cmd = [
                    FFMPEG_PATH,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{ts:.3f}",
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ]
                try:
                    subprocess.run(cmd, check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as error:
                    raise SamplingError(f"ffmpeg failed at {ts:.2f}s: {error.stderr.decode().strip()}") from error
                except subprocess.TimeoutExpired as error:
                    raise SamplingError(f"ffmpeg timed out at {ts:.2f}s") from error

                frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise SamplingError(f"Unable to decode frame at {ts:.2f}s")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                samples.append(
                    FrameSample(
                        clip_id=clip.clip_id,
                        frame_index=index,
                        timestamp_seconds=ts,
                        source_path=str(frame_path),
                        metadata={"backend": "ffmpeg"},
                        data=frame_rgb,
                    )
                )
        return samples

    def _compute_timestamps(self, duration: float) -> List[float]:
        max_frames = max(1, self._config.max_frames_per_clip)
        if duration <= 0:
            return [0.0]
        min_spacing = max(0.1, self._config.min_spacing_sec)
        theoretical = int(duration / min_spacing) + 1
        num_frames = min(max_frames, max(1, theoretical))
        span = max(duration - 0.05, 0.0)
        if num_frames <= 1 or span <= 0:
            return [0.0]
        timestamps = [round((span * i) / (num_frames - 1), 3) for i in range(num_frames)]
        return timestamps
