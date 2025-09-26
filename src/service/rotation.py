"""Log rotation helpers for Pigment Bucket."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def _collect_log_files(log_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    if not log_dir.exists():
        return candidates
    for entry in log_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".json", ".csv"}:
            continue
        candidates.append(entry)
    return candidates


def enforce_log_rotation(log_dir: Path, max_files: int, max_bytes: int) -> None:
    if max_files <= 0 and max_bytes <= 0:
        return

    files = _collect_log_files(log_dir)
    if not files:
        return

    files.sort(key=lambda path: path.stat().st_mtime)

    if max_files > 0:
        while len(files) > max_files:
            victim = files.pop(0)
            try:
                victim.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                files.append(victim)
                break

    if max_bytes > 0:
        def total_size(paths: Iterable[Path]) -> int:
            size = 0
            for path in paths:
                try:
                    size += path.stat().st_size
                except OSError:
                    continue
            return size

        current_size = total_size(files)
        while files and current_size > max_bytes:
            victim = files.pop(0)
            try:
                size_before = victim.stat().st_size
            except OSError:
                size_before = 0
            try:
                victim.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                files.append(victim)
                break
            current_size -= size_before


__all__ = ["enforce_log_rotation"]
