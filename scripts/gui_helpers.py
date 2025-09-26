from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SENSITIVITY_DEFAULT = 60
MIN_SENSITIVITY = 0
MAX_SENSITIVITY = 100
AUTO_PROFILES_PATH = Path(__file__).resolve().parents[1] / "state" / "auto_profiles.json"


def sensitivity_to_similarity(value: int) -> float:
    clamped = max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, value))
    return 0.80 + 0.19 * (clamped / 100.0)


def sensitivity_to_min_duration(value: int) -> float:
    clamped = max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, value))
    # round to one decimal place as per specification
    return round(1.0 + 4.0 * (1.0 - clamped / 100.0), 1)


def iso_timestamp() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, payload: dict, max_lines: int = 2000, background: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        lines: list[str] = []
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    lines = handle.readlines()
            except OSError:
                lines = []
        lines.append(json.dumps(payload, ensure_ascii=False) + "\n")
        if max_lines > 0 and len(lines) > max_lines:
            lines = lines[-max_lines:]
        try:
            with path.open("w", encoding="utf-8") as handle:
                handle.writelines(lines)
        except OSError:
            pass

    if background:
        thread = threading.Thread(target=_write, daemon=True)
        thread.start()
    else:
        _write()


def load_gui_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_gui_settings(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(settings, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def _load_auto_profiles(path: Path = AUTO_PROFILES_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_auto_profiles(store: dict, path: Path = AUTO_PROFILES_PATH) -> None:
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


def record_auto_profile(
    project_id: str,
    clip_hash: str,
    similarity: float,
    min_duration: float,
    centroid: list[float],
    updated_at: str,
    path: Path = AUTO_PROFILES_PATH,
) -> None:
    if not project_id or not clip_hash:
        return
    store = _load_auto_profiles(path)
    project_bucket = store.setdefault(project_id, {})
    payload = {
        "similarity": float(similarity),
        "min_duration": float(min_duration),
        "centroid": [float(round(value, 6)) for value in centroid],
        "updated_at": updated_at,
    }
    project_bucket[clip_hash] = [payload]
    _write_auto_profiles(store, path)


__all__ = [
    "SENSITIVITY_DEFAULT",
    "sensitivity_to_similarity",
    "sensitivity_to_min_duration",
    "append_jsonl",
    "load_gui_settings",
    "save_gui_settings",
    "iso_timestamp",
    "record_auto_profile",
]
