"""SQLite helpers for Pigment Bucket job history."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from .config import DATA_DIR, ensure_dirs

DB_PATH = DATA_DIR / "pigmentbucket.db"


def init_db() -> None:
    ensure_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                selection_mode TEXT,
                processed INTEGER,
                skipped INTEGER,
                clusters INTEGER,
                json_path TEXT,
                csv_path TEXT,
                undo_path TEXT,
                status TEXT NOT NULL,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at
            ON jobs (created_at DESC)
            """
        )
        for column_name, column_def in (
            ("run_id", "ALTER TABLE jobs ADD COLUMN run_id TEXT"),
            ("undo_path", "ALTER TABLE jobs ADD COLUMN undo_path TEXT"),
        ):
            try:
                conn.execute(column_def)
            except sqlite3.OperationalError:
                pass


@contextmanager
def _connect():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_job(job: Dict[str, Any]) -> None:
    columns = ", ".join(job.keys())
    placeholders = ", ".join([":" + key for key in job.keys()])
    query = f"INSERT INTO jobs ({columns}) VALUES ({placeholders})"
    with _connect() as conn:
        conn.execute(query, job)


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join([f"{key} = :{key}" for key in fields.keys()])
    params = dict(fields)
    params["id"] = job_id
    query = f"UPDATE jobs SET {assignments} WHERE id = :id"
    with _connect() as conn:
        conn.execute(query, params)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT id, run_id, created_at, finished_at, status, processed, skipped, clusters, undo_path "
            "FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]


__all__ = ["DB_PATH", "init_db", "insert_job", "update_job", "get_job", "list_jobs"]
