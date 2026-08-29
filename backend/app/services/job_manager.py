"""
JobManager
----------
File de jobs asynchrones (section 18/26 du spec). Persisté dans Postgres
(partagé entre toutes les machines web + worker) au lieu de SQLite local,
car le disque de chaque machine Fly.io n'est ni partagé ni persistant
entre machines : SQLite local causait des 404 aléatoires selon la
machine qui répondait à la requête.
"""
from __future__ import annotations
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras

from app.core.config import settings
from app.models.schemas import JobStatus

DATABASE_URL = os.environ.get("DATABASE_URL")

_lock = threading.Lock()


def _init_db():
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input_key TEXT,
                    result_key TEXT,
                    target_frame INTEGER,
                    target_x REAL,
                    target_y REAL,
                    current_frame INTEGER DEFAULT 0,
                    total_frames INTEGER DEFAULT 0,
                    target_id INTEGER,
                    target_state TEXT,
                    confidence_percent REAL,
                    identity_switches INTEGER DEFAULT 0,
                    error TEXT,
                    claimed_by TEXT,
                    created_at DOUBLE PRECISION,
                    updated_at DOUBLE PRECISION,
                    metrics_json TEXT
                )
                """
            )
        conn.commit()


@contextmanager
def _connect():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


_init_db()


class JobManager:
    def create_job(self, input_key: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO jobs (job_id, status, input_key, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (job_id, JobStatus.QUEUED.value, input_key, now, now),
                )
        return job_id

    def set_target(self, job_id: str, frame: int, x: float, y: float):
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE jobs SET target_frame=%s, target_x=%s, target_y=%s,
                       status=%s, updated_at=%s WHERE job_id=%s""",
                    (frame, x, y, JobStatus.QUEUED.value, time.time(), job_id),
                )

    def claim_next_job(self, worker_id: str) -> Optional[dict]:
        """Utilisé par le worker : récupère atomiquement le prochain job en
        attente et le marque 'processing' pour éviter qu'un autre worker
        ne le traite en double."""
        with _lock, _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM jobs WHERE status=%s AND target_frame IS NOT NULL
                       ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED""",
                    (JobStatus.QUEUED.value,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    "UPDATE jobs SET status=%s, claimed_by=%s, updated_at=%s WHERE job_id=%s",
                    (JobStatus.PROCESSING.value, worker_id, time.time(), row["job_id"]),
                )
                return dict(row)

    def update_progress(
        self,
        job_id: str,
        current_frame: int,
        total_frames: int,
        target_id: Optional[int] = None,
        target_state: Optional[str] = None,
        confidence_percent: Optional[float] = None,
        identity_switches: int = 0,
    ):
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE jobs SET current_frame=%s, total_frames=%s, target_id=%s,
                       target_state=%s, confidence_percent=%s, identity_switches=%s, updated_at=%s
                       WHERE job_id=%s""",
                    (
                        current_frame, total_frames, target_id, target_state,
                        confidence_percent, identity_switches, time.time(), job_id,
                    ),
                )

    def complete_job(self, job_id: str, result_key: str, metrics: dict):
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE jobs SET status=%s, result_key=%s, metrics_json=%s, updated_at=%s
                       WHERE job_id=%s""",
                    (JobStatus.COMPLETED.value, result_key, json.dumps(metrics), time.time(), job_id),
                )

    def fail_job(self, job_id: str, error: str):
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status=%s, error=%s, updated_at=%s WHERE job_id=%s",
                    (JobStatus.FAILED.value, error, time.time(), job_id),
                )

    def get_job(self, job_id: str) -> Optional[dict]:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM jobs WHERE job_id=%s", (job_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def delete_job(self, job_id: str):
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))

    def count_active_jobs(self) -> int:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) as c FROM jobs WHERE status IN (%s, %s)",
                    (JobStatus.QUEUED.value, JobStatus.PROCESSING.value),
                )
                return cur.fetchone()["c"]


job_manager = JobManager()
