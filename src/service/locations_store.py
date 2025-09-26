"""Persistent location store for stable location identifiers."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import STATE_DIR, ensure_dirs


@dataclass
class StoreResult:
    assignments: Dict[int, Optional[int]]
    locations_detail: List[dict]
    matched: int
    created: int
    total_persisted: int
    centroid_dim: Optional[int]
    store_type: str

    @property
    def location_count(self) -> int:
        return len(self.locations_detail)


class BaseLocationStore:
    store_type = "base"

    def assign(
        self,
        job_id: str,
        run_id: str,
        locations: List[dict],
        similarity_threshold: float,
    ) -> StoreResult:
        raise NotImplementedError


class SQLiteLocationStore(BaseLocationStore):
    store_type = "sqlite"

    def __init__(self) -> None:
        ensure_dirs()
        self._db_path = STATE_DIR / "locations.sqlite"
        self._ensure_schema()

    def assign(
        self,
        job_id: str,
        run_id: str,
        locations: List[dict],
        similarity_threshold: float,
    ) -> StoreResult:
        if not locations:
            return StoreResult({}, locations, matched=0, created=0, total_persisted=self._count_locations(), centroid_dim=None, store_type=self.store_type)

        mapping: Dict[int, Optional[int]] = {}
        matched = 0
        created = 0
        centroid_dim: Optional[int] = None

        with self._connect() as conn:
            existing = {
                row["id"]: np.frombuffer(row["centroid"], dtype=np.float32)
                for row in conn.execute("SELECT id, vector_dim, centroid FROM location_centroids")
            }
            dims = {
                row["id"]: row["vector_dim"]
                for row in conn.execute("SELECT id, vector_dim FROM location_centroids")
            }

        used_ids: set[int] = set()
        similarity_threshold = max(0.0, min(1.0, similarity_threshold))

        for entry in locations:
            temp_id = int(entry.get("location_id", -1))
            centroid = np.asarray(entry.get("centroid") or [], dtype=np.float32)
            if centroid.size == 0:
                mapping[temp_id] = None
                continue

            centroid_dim = centroid.size if centroid_dim is None else centroid_dim
            best_id: Optional[int] = None
            best_similarity = -1.0

            for location_id, stored_vector in existing.items():
                if dims.get(location_id) != centroid.size:
                    continue
                denom = np.linalg.norm(stored_vector) * np.linalg.norm(centroid)
                if denom == 0.0:
                    similarity = 0.0
                else:
                    similarity = float(np.clip(np.dot(stored_vector, centroid) / denom, -1.0, 1.0))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = location_id

            if best_id is not None and best_similarity >= similarity_threshold and best_id not in used_ids:
                mapping[temp_id] = best_id
                matched += 1
                used_ids.add(best_id)
                new_vector = (existing[best_id] + centroid) / 2.0
                existing[best_id] = new_vector
                self._update_centroid(best_id, new_vector)
            else:
                new_id = self._insert_centroid(centroid)
                mapping[temp_id] = new_id
                created += 1
                used_ids.add(new_id)
                existing[new_id] = centroid
                dims[new_id] = centroid.size

        self._store_aliases(job_id, run_id, mapping)

        locations_detail = []
        for entry in locations:
            temp_id = int(entry.get("location_id", -1))
            persisted_id = mapping.get(temp_id)
            enriched = dict(entry)
            if persisted_id is not None:
                enriched["location_id"] = persisted_id
            enriched["persisted_location_id"] = persisted_id
            locations_detail.append(enriched)

        total_persisted = self._count_locations()
        return StoreResult(
            assignments=mapping,
            locations_detail=locations_detail,
            matched=matched,
            created=created,
            total_persisted=total_persisted,
            centroid_dim=centroid_dim,
            store_type=self.store_type,
        )

    # ------------------------------------------------------------------
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        ensure_dirs()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS location_centroids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_dim INTEGER NOT NULL,
                    centroid BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS location_aliases (
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    tmp_location_id INTEGER NOT NULL,
                    location_id INTEGER NOT NULL,
                    PRIMARY KEY (job_id, tmp_location_id)
                )
                """
            )

    def _insert_centroid(self, vector: np.ndarray) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO location_centroids (vector_dim, centroid, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(vector.size),
                    vector.astype(np.float32).tobytes(),
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def _update_centroid(self, location_id: int, vector: np.ndarray) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE location_centroids
                SET centroid = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    vector.astype(np.float32).tobytes(),
                    datetime.utcnow().isoformat(),
                    location_id,
                ),
            )

    def _store_aliases(self, job_id: str, run_id: str, mapping: Dict[int, Optional[int]]) -> None:
        rows = [
            (job_id, run_id, temp_id, int(persisted_id))
            for temp_id, persisted_id in mapping.items()
            if persisted_id is not None
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO location_aliases (job_id, run_id, tmp_location_id, location_id)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def _count_locations(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM location_centroids")
            return int(cursor.fetchone()[0])


class JsonLocationStore(BaseLocationStore):
    store_type = "json"

    def __init__(self) -> None:
        ensure_dirs()
        self._dir = STATE_DIR / "locations"
        self._dir.mkdir(parents=True, exist_ok=True)

    def assign(
        self,
        job_id: str,
        run_id: str,
        locations: List[dict],
        similarity_threshold: float,
    ) -> StoreResult:
        output_path = self._dir / f"{job_id}.json"
        payload = {
            "job_id": job_id,
            "run_id": run_id,
            "locations": locations,
        }
        try:
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

        locations_detail = []
        mapping: Dict[int, Optional[int]] = {}
        for entry in locations:
            temp_id = int(entry.get("location_id", -1))
            mapping[temp_id] = None
            enriched = dict(entry)
            enriched["persisted_location_id"] = None
            locations_detail.append(enriched)

        return StoreResult(
            assignments=mapping,
            locations_detail=locations_detail,
            matched=0,
            created=len(locations),
            total_persisted=0,
            centroid_dim=None,
            store_type=self.store_type,
        )


def get_location_store(store_type: str) -> BaseLocationStore:
    store_type = (store_type or "").lower()
    if store_type == "json":
        return JsonLocationStore()
    return SQLiteLocationStore()
