"""Disk-backed feature cache helpers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .features import FeatureExtractorConfig
from .sampler import SamplerConfig
from .types import ClipContext, ClipFeatures


@dataclass(frozen=True)
class CacheDescriptor:
    """Lightweight descriptor used to build a stable cache key."""

    media_path: Path
    size: int
    mtime_ns: int
    sampler: SamplerConfig
    feature: FeatureExtractorConfig

    def digest(self) -> str:
        payload = {
            "path": str(self.media_path.resolve()),
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sampler": {
                "max_frames_per_clip": self.sampler.max_frames_per_clip,
                "min_spacing_sec": round(float(self.sampler.min_spacing_sec), 6),
                "backend": self.sampler.backend,
            },
            "feature": {
                "histogram_bins": self.feature.histogram_bins,
            },
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class FeatureCache:
    """Persists extracted clip features on disk to avoid recomputation."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _descriptor(
        self,
        clip: ClipContext,
        sampler_cfg: SamplerConfig,
        feature_cfg: FeatureExtractorConfig,
    ) -> Optional[CacheDescriptor]:
        if not clip.media_path:
            return None
        path = Path(clip.media_path)
        try:
            stats = path.stat()
        except OSError:
            return None
        return CacheDescriptor(
            media_path=path,
            size=int(stats.st_size),
            mtime_ns=int(getattr(stats, "st_mtime_ns", int(stats.st_mtime * 1e9))),
            sampler=sampler_cfg,
            feature=feature_cfg,
        )

    def _entry_path(self, digest: str) -> Path:
        return self._base_dir / f"{digest}.npz"

    # ------------------------------------------------------------------
    def load(
        self,
        clip: ClipContext,
        sampler_cfg: SamplerConfig,
        feature_cfg: FeatureExtractorConfig,
    ) -> Optional[Tuple[ClipFeatures, int]]:
        descriptor = self._descriptor(clip, sampler_cfg, feature_cfg)
        if not descriptor:
            return None
        entry_path = self._entry_path(descriptor.digest())
        if not entry_path.exists():
            return None
        try:
            with np.load(entry_path, allow_pickle=False) as data:
                vector = data["vector"].astype(np.float32)
                metadata_raw = data["metadata"].item()
                metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else dict(metadata_raw)
                frame_count = int(data["frame_count"][()])
        except Exception:
            # Corrupted entry – remove and treat as miss
            try:
                entry_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        metadata = metadata or {}
        metadata["cache_hit"] = True
        features = ClipFeatures(clip_id=clip.clip_id, vector=vector, metadata=metadata)
        return features, frame_count

    # ------------------------------------------------------------------
    def store(
        self,
        clip: ClipContext,
        sampler_cfg: SamplerConfig,
        feature_cfg: FeatureExtractorConfig,
        features: ClipFeatures,
        frame_count: int,
    ) -> None:
        descriptor = self._descriptor(clip, sampler_cfg, feature_cfg)
        if not descriptor:
            return
        entry_path = self._entry_path(descriptor.digest())
        payload: Dict[str, object] = {
            "vector": features.vector.astype(np.float32),
            "metadata": json.dumps(features.metadata or {}),
            "frame_count": np.array(frame_count, dtype=np.int32),
        }
        tmp_path = entry_path.with_suffix(".tmp.npz")
        try:
            np.savez(tmp_path, **payload)
            os.replace(tmp_path, entry_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
