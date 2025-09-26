"""Runtime configuration for the Pigment Bucket service."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

_BASE_DIR = Path(os.environ.get("PIGMENT_BASE_DIR", ".")).resolve()

DATA_DIR = Path(os.environ.get("PIGMENT_DATA_DIR", _BASE_DIR / "data")).resolve()
LOG_DIR = Path(os.environ.get("PIGMENT_LOG_DIR", _BASE_DIR / "logs")).resolve()
FEATURE_CACHE_DIR = Path(os.environ.get("PIGMENT_FEATURE_CACHE_DIR", _BASE_DIR / ".cache" / "features")).resolve()
STATE_DIR = Path(os.environ.get("PIGMENT_STATE_DIR", _BASE_DIR / "state" )).resolve()
ARTIFACTS_DIR = Path(os.environ.get("PIGMENT_ARTIFACTS_DIR", _BASE_DIR / "artifacts")).resolve()

MAX_LOG_FILES = int(os.environ.get("PIGMENT_MAX_LOG_FILES", "500"))
MAX_LOG_BYTES = int(os.environ.get("PIGMENT_MAX_LOG_BYTES", str(200 * 1024 * 1024)))
ANALYZE_MAX_RETRIES = int(os.environ.get("PIGMENT_ANALYZE_MAX_RETRIES", "2"))
ANALYZE_RETRY_DELAY_MS = int(os.environ.get("PIGMENT_ANALYZE_RETRY_DELAY_MS", "500"))
ANALYZE_JOB_TIMEOUT_MS = int(os.environ.get("PIGMENT_ANALYZE_JOB_TIMEOUT_MS", "60000"))
DEBUG_FAULTS = bool(int(os.environ.get("PIGMENT_DEBUG_FAULTS", "0")))


def ensure_dirs() -> Tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "frames").mkdir(parents=True, exist_ok=True)
    return DATA_DIR, LOG_DIR


__all__ = [
    "DATA_DIR",
    "LOG_DIR",
    "FEATURE_CACHE_DIR",
    "STATE_DIR",
    "ARTIFACTS_DIR",
    "MAX_LOG_FILES",
    "MAX_LOG_BYTES",
    "ANALYZE_MAX_RETRIES",
    "ANALYZE_RETRY_DELAY_MS",
    "ANALYZE_JOB_TIMEOUT_MS",
    "DEBUG_FAULTS",
    "ensure_dirs",
]
