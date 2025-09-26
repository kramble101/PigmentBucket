from __future__ import annotations

import importlib
from pathlib import Path
import pytest


@pytest.fixture()
def locations_store_module(tmp_path, monkeypatch):
    monkeypatch.setenv("PIGMENT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIGMENT_STATE_DIR", str(tmp_path / "state"))
    config_module = importlib.import_module("src.service.config")
    importlib.reload(config_module)
    locations_store = importlib.import_module("src.service.locations_store")
    importlib.reload(locations_store)
    return locations_store


def test_sqlite_store_matches_existing(locations_store_module) -> None:
    store = locations_store_module.SQLiteLocationStore()
    locations = [
        {
            "location_id": 0,
            "cluster_id": 1,
            "clip_ids": ["clip1", "clip2"],
            "duration_seconds": 3.5,
            "clip_count": 2,
            "start_tc": "00:00:00:00",
            "end_tc": "00:00:03:12",
            "centroid": [0.1, 0.9, 0.2],
        }
    ]

    first = store.assign("job-a", "run-a", locations, similarity_threshold=0.9)
    assert first.created == 1
    persisted = first.assignments.get(0)
    assert persisted is not None

    second = store.assign("job-b", "run-b", locations, similarity_threshold=0.9)
    assert second.matched == 1
    assert second.created == 0
    assert second.assignments.get(0) == persisted
    assert second.locations_detail[0]["persisted_location_id"] == persisted
    assert second.total_persisted == 1


def test_json_store_writes_payload(locations_store_module, tmp_path) -> None:
    store = locations_store_module.JsonLocationStore()
    locations = [
        {
            "location_id": 0,
            "cluster_id": 1,
            "clip_ids": ["clip1"],
            "duration_seconds": 1.0,
            "clip_count": 1,
            "start_tc": "00:00:00:00",
            "end_tc": "00:00:01:00",
            "centroid": [0.5, 0.5],
        }
    ]

    result = store.assign("job-json", "run-json", locations, similarity_threshold=0.5)
    assert result.created == 1
    assert result.matched == 0
    assert result.total_persisted == 0
    assert result.locations_detail[0]["persisted_location_id"] is None

    output_file = Path(tmp_path / "state" / "locations" / "job-json.json")
    assert output_file.exists()
