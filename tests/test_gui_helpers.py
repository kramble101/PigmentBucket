from __future__ import annotations

import time
from pathlib import Path

from scripts.gui_helpers import (
    append_jsonl,
    sensitivity_to_min_duration,
    sensitivity_to_similarity,
)


def test_sensitivity_mapping_extremes() -> None:
    assert abs(sensitivity_to_similarity(0) - 0.80) < 1e-6
    assert abs(sensitivity_to_similarity(100) - 0.99) < 1e-6
    mid_similarity = sensitivity_to_similarity(60)
    assert 0.80 < mid_similarity < 0.99

    assert abs(sensitivity_to_min_duration(0) - 5.0) < 1e-6
    assert abs(sensitivity_to_min_duration(100) - 1.0) < 1e-6
    mid_duration = sensitivity_to_min_duration(60)
    assert 1.0 < mid_duration < 5.0


def test_append_jsonl_truncates(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    for i in range(25):
        append_jsonl(path, {"i": i}, max_lines=20, background=False)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    assert lines[0].endswith('"i": 5}')
    assert lines[-1].endswith('"i": 24}')
