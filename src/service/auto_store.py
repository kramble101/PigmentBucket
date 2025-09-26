"""Persistence helpers for auto-tuning preferences."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import STATE_DIR

AUTO_PROFILES_PATH = STATE_DIR / "auto_profiles.json"


@dataclass
class AutoProfileEntry:
    similarity: float
    min_duration: float
    centroid: List[float]
    updated_at: str


def _load_store(path: Path = AUTO_PROFILES_PATH) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _write_store(store: Dict[str, Dict[str, List[Dict[str, object]]]], path: Path = AUTO_PROFILES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(store, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def load_profiles(project_id: str, clip_hash: str, path: Path = AUTO_PROFILES_PATH) -> List[Dict[str, object]]:
    store = _load_store(path)
    project_bucket = store.get(project_id) if isinstance(project_id, str) else None
    if not isinstance(project_bucket, dict):
        return []
    entries = project_bucket.get(clip_hash)
    if not isinstance(entries, list):
        return []
    filtered: List[Dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "similarity" not in entry or "min_duration" not in entry or "centroid" not in entry:
            continue
        filtered.append(entry)
    return filtered


def upsert_profile(
    project_id: str,
    clip_hash: str,
    similarity: float,
    min_duration: float,
    centroid: List[float],
    updated_at: str,
    path: Path = AUTO_PROFILES_PATH,
) -> None:
    store = _load_store(path)
    project_bucket = store.setdefault(project_id, {})
    payload = {
        "similarity": float(similarity),
        "min_duration": float(min_duration),
        "centroid": [float(value) for value in centroid],
        "updated_at": updated_at,
    }

    project_bucket[clip_hash] = [payload]

    _write_store(store, path)


__all__ = ["AUTO_PROFILES_PATH", "load_profiles", "upsert_profile", "AutoProfileEntry"]
