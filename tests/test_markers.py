from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pigmentbucket_run.py"
MODULE_NAME = "pigmentbucket_run_test_module"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[MODULE_NAME] = runner
SPEC.loader.exec_module(runner)  # type: ignore[attr-defined]


class StubTimeline:
    def __init__(self) -> None:
        self.added: list[tuple[int, str, str, str]] = []
        self.markers: set[int] = set()

    def AddMarker(self, frame: int, color: str, name: str, note: str, duration: int, custom_data) -> bool:  # noqa: N802 - Resolve API naming
        self.added.append((frame, color, name, note))
        self.markers.add(frame)
        return True

    def DeleteMarkerAtFrame(self, frame: int) -> bool:  # noqa: N802 - Resolve API naming
        if frame in self.markers:
            self.markers.remove(frame)
            return True
        return False


def test_add_location_markers_creates_start_and_end_markers() -> None:
    timeline = StubTimeline()
    locations = [
        {
            "location_id": 10,
            "cluster_id": 1,
            "start_tc": "00:00:00:00",
            "end_tc": "00:00:02:00",
            "clip_ids": ["c1", "c2"],
        }
    ]
    cluster_colors = {1: "Yellow"}

    markers = runner.add_location_markers(
        timeline,
        Fraction(24, 1),
        timeline_start=0,
        locations_detail=locations,
        cluster_colors=cluster_colors,
    )

    assert len(markers) == 2
    assert timeline.added[0][0] == 0
    assert timeline.added[0][1] == "Yellow"
    assert timeline.added[0][2] == "Loc 10"
    # End marker at 48th frame for 2 seconds @24fps
    assert timeline.added[1][0] == 48
    assert timeline.added[1][2] == "Loc 10"


def test_remove_markers_from_timeline_handles_missing_frames() -> None:
    timeline = StubTimeline()
    timeline.markers.update({0, 48})

    removed = runner.remove_markers_from_timeline(
        timeline,
        markers=[{"frame": 48}, {"frame": 99}, {"frame": 0}],
    )

    assert removed == 2
    assert timeline.markers == set()
