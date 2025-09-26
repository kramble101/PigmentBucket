from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def service_app(tmp_path, monkeypatch):
    monkeypatch.setenv("PIGMENT_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIGMENT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PIGMENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PIGMENT_FEATURE_CACHE_DIR", str(tmp_path / "cache"))

    config_module = importlib.import_module("src.service.config")
    importlib.reload(config_module)
    app_module = importlib.import_module("src.service.app")
    importlib.reload(app_module)

    from src.pipeline.types import AnalysisResult, ClipAnalysis, ClipFeatures, ClipContext

    def fake_analyze(self, contexts):  # type: ignore[override]
        clip_results = []
        for ctx in contexts:
            features = ClipFeatures(
                clip_id=ctx.clip_id,
                vector=np.array([0.1, 0.9], dtype=np.float32),
                metadata={"feature_dim": 2, "frame_count": 1},
            )
            analysis = ClipAnalysis(
                clip_id=ctx.clip_id,
                cluster_id=0,
                features=features,
                samples=[],
                status="ok",
            )
            analysis.location_id = 0
            analysis.location_clip_count = len(contexts)
            analysis.location_duration = 1.0
            analysis.location_span_start = "00:00:00:00"
            analysis.location_span_end = "00:00:01:00"
            analysis.location_centroid = features.vector.tolist()
            clip_results.append(analysis)
        summary = {
            "processed": len(contexts),
            "skipped": 0,
            "clusters": 1,
            "feature_dim": 2,
            "chosen_k": 1,
            "silhouette": None,
            "sampled_frames": len(contexts),
            "coloring": {"coloring": "skipped", "reason": "k==1"},
            "locations": 1,
            "locations_detail": [
                {
                    "location_id": 0,
                    "cluster_id": 0,
                    "clip_ids": [ctx.clip_id for ctx in contexts],
                    "duration_seconds": 1.0,
                    "clip_count": len(contexts),
                    "start_tc": "00:00:00:00",
                    "end_tc": "00:00:01:00",
                    "centroid": [0.1, 0.9],
                }
            ],
            "auto_mode": True,
            "auto_iterations": 1,
            "auto_similarity_threshold": 0.92,
            "auto_min_duration_sec": 3.0,
            "auto_history": [
                {
                    "iteration": 1,
                    "similarity": 0.92,
                    "min_duration": 3.0,
                    "random_state": 42,
                    "chosen_k": 1,
                    "unique_clusters": 1,
                    "silhouette": None,
                    "decision": "accept",
                }
            ],
            "auto_initial_source": "default",
            "auto_target_k": 2,
            "auto_reference_centroid": [0.1, 0.9],
            "project_id": "TestProject",
            "clip_context_hash": "hash123",
        }
        return AnalysisResult(clips=clip_results, summary=summary)

    analyzer_cls = importlib.import_module("src.service.app").Analyzer
    monkeypatch.setattr(analyzer_cls, "analyze", fake_analyze)

    return app_module.app


def test_analyze_endpoint_persists_reports(service_app, tmp_path):
    client = TestClient(service_app)

    media_file = tmp_path / "media.mov"
    media_file.write_bytes(b"stub")

    payload = {
        "clips": [
            {
                "clip_id": "c1",
                "clip_name": "First",
                "timeline_in": "00:00:00:00",
                "timeline_out": "00:00:01:00",
                "track_index": 1,
                "duration_frames": 24,
                "duration_seconds": 1.0,
                "media_path": str(media_file),
            }
        ],
        "dry_run": True,
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    result = client.get(f"/result/{job_id}")
    assert result.status_code == 200
    body = result.json()
    assert body["summary"]["coloring"]["coloring"] == "skipped"
    assert body["summary"]["locations"] == 1
    assert body["summary"]["locations_stats"]["clip_count"]["max"] == 1
    assert body["summary"]["location_store"] == "sqlite"
    assert body["summary"]["locations_persisted"] == 1
    assert body["summary"]["new_locations"] == 1
    assert body["summary"]["matched_locations"] == 0
    assert body["summary"]["report_format"] == "both"
    assert body["summary"]["quality_metric_name"] in {"silhouette", "median_hue_spread", None}
    assert "median_hue_spread" in body["summary"]
    assert body["summary"]["auto_mode"] is True
    assert body["summary"]["auto_iterations"] == 1
    assert body["summary"]["auto_initial_source"] == "default"
    assert body["summary"]["auto_reference_centroid"] == [0.1, 0.9]
    assert body["summary"]["project_id"] == "TestProject"
    assert body["summary"]["clip_context_hash"] == "hash123"
    persisted_id = body["clips"][0]["persisted_location_id"]
    assert persisted_id is not None
    assert body["clips"][0]["location_id"] == persisted_id
    assert body["clips"][0]["location_span_start"] == "00:00:00:00"
    assert body["clips"][0]["location_span_end"] == "00:00:01:00"
    assert body["clips"][0]["persisted_location_id"] == persisted_id
    assert body["clips"][0]["location_color_name"] in {"Yellow", "Grey", "Cyan"}

    report_dir = tmp_path / "logs"
    report_json = report_dir / f"{job_id}.json"
    report_csv = report_dir / f"{job_id}.csv"
    assert report_json.exists()
    assert report_csv.exists()

    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_payload["summary"]["feature_dim"] == 2
    assert report_payload["summary"]["locations"] == 1
    assert report_payload["summary"]["location_store"] == "sqlite"
    assert report_payload["summary"]["locations_persisted"] == 1
    assert report_payload["summary"]["report_format"] == "both"
    assert report_payload["summary"].get("quality_metric_name") in {"silhouette", "median_hue_spread", None}
    assert "median_hue_spread" in report_payload["summary"]
    assert report_payload["summary"]["auto_mode"] is True
    assert report_payload["summary"]["auto_iterations"] == 1
    assert report_payload["summary"]["auto_reference_centroid"] == [0.1, 0.9]
    assert report_payload["summary"]["project_id"] == "TestProject"
    assert report_payload["summary"]["clip_context_hash"] == "hash123"
    assert report_payload["coloring"]["coloring"] == "skipped"
    assert report_payload["clips"][0]["location_id"] == persisted_id
    assert report_payload["clips"][0]["location_start_tc"] == "00:00:00:00"
    assert report_payload["clips"][0]["persisted_location_id"] == persisted_id
    assert report_payload["clips"][0]["location_color_name"] in {"Yellow", "Grey", "Cyan"}
    assert report_payload["locations"][0]["clip_ids"] == ["c1"]
    locations_entry = report_payload["locations"][0]
    assert str(locations_entry["persisted_location_id"]) == str(persisted_id)
    assert locations_entry["color_name"] in {"Yellow", "Grey", "Cyan"}

    csv_lines = report_csv.read_text(encoding="utf-8").splitlines()
    header = csv_lines[0].split(",")
    assert "location_color_name" in header
    assert "quality_metric_name" in header
    assert "median_hue_spread" in header
    assert "apply_action" in header
    assert "apply_reason" in header
    assert "auto_mode" in header
    assert "auto_iterations" in header


def test_report_format_csv_only(service_app, tmp_path):
    client = TestClient(service_app)

    payload = {
        "clips": [
            {
                "clip_id": "c1",
                "clip_name": "Clip",
                "timeline_in": "00:00:00:00",
                "timeline_out": "00:00:01:00",
                "track_index": 1,
                "duration_frames": 24,
            }
        ],
        "dry_run": True,
        "config": {"report_format": "csv"},
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    result = client.get(f"/result/{job_id}")
    assert result.status_code == 200
    summary = result.json()["summary"]
    assert summary["report_format"] == "csv"
    assert "quality_metric_name" in summary

    logs_dir = tmp_path / "logs"
    assert (logs_dir / f"{job_id}.csv").exists()
    assert not (logs_dir / f"{job_id}.json").exists()


def test_report_format_json_only(service_app, tmp_path):
    client = TestClient(service_app)

    payload = {
        "clips": [
            {
                "clip_id": "c1",
                "clip_name": "Clip",
                "timeline_in": "00:00:00:00",
                "timeline_out": "00:00:01:00",
                "track_index": 1,
                "duration_frames": 24,
            }
        ],
        "dry_run": True,
        "config": {"report_format": "json"},
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    result = client.get(f"/result/{job_id}")
    assert result.status_code == 200
    summary = result.json()["summary"]
    assert summary["report_format"] == "json"
    assert "quality_metric_value" in summary

    logs_dir = tmp_path / "logs"
    assert (logs_dir / f"{job_id}.json").exists()
    assert not (logs_dir / f"{job_id}.csv").exists()
