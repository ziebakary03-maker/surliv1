"""
JobManager
----------
File de jobs asynchrones (section 18/26 du spec). Le traitement vidéo
peut prendre plusieurs minutes : la requête HTTP d'upload ne doit donc
jamais rester ouverte pendant tout le traitement. On persiste les jobs
dans SQLite (fichier partagé entre le process web et le process worker)
plutôt qu'en mémoire, pour survivre à un redémarrage du serveur web et
pour permettre plusieurs workers.

NOTE PRODUCTION : SQLite convient pour une V1 mono-instance. Pour
plusieurs workers/instances Fly.io en parallèle avec forte charge,
remplacer ce module par Redis + RQ/Celery (l'interface publique
`create_job / claim_next_job / update_progress / get_job` resterait
identique, donc le reste du code n'a pas à changer).
"""
from __future__ import annotations
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Optional

from app.core.config import settings
from app.models.schemas import JobStatus

DB_PATH = f"{settings.LOCAL_STORAGE_PATH}/jobs.sqlite3"

_lock = threading.Lock()


def _init_db():
    import os
    os.makedirs(settings.LOCAL_STORAGE_PATH, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
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
                created_at REAL,
                updated_at REAL,
                metrics_json TEXT
            )
            """
        )
        conn.commit()


_init_db()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class JobManager:
    def create_job(self, input_key: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with _connect() as conn:
            conn.execute(
                """INSERT INTO jobs (job_id, status, input_key, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (job_id, JobStatus.QUEUED.value, input_key, now, now),
            )
        return job_id

    def set_target(self, job_id: str, frame: int, x: float, y: float):
        with _connect() as conn:
            conn.execute(
                """UPDATE jobs SET target_frame=?, target_x=?, target_y=?,
                   status=?, updated_at=? WHERE job_id=?""",
                (frame, x, y, JobStatus.QUEUED.value, time.time(), job_id),
            )

    def claim_next_job(self, worker_id: str) -> Optional[dict]:
        """Utilisé par le worker : récupère atomiquement le prochain job en
        attente et le marque 'processing' pour éviter qu'un autre worker
        ne le traite en double."""
        with _lock, _connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status=? AND target_frame IS NOT NULL ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE jobs SET status=?, claimed_by=?, updated_at=? WHERE job_id=?",
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
            conn.execute(
                """UPDATE jobs SET current_frame=?, total_frames=?, target_id=?,
                   target_state=?, confidence_percent=?, identity_switches=?, updated_at=?
                   WHERE job_id=?""",
                (
                    current_frame, total_frames, target_id, target_state,
                    confidence_percent, identity_switches, time.time(), job_id,
                ),
            )

    def complete_job(self, job_id: str, result_key: str, metrics: dict):
        with _connect() as conn:
            conn.execute(
                """UPDATE jobs SET status=?, result_key=?, metrics_json=?, updated_at=?
                   WHERE job_id=?""",
                (JobStatus.COMPLETED.value, result_key, json.dumps(metrics), time.time(), job_id),
            )

    def fail_job(self, job_id: str, error: str):
        with _connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
                (JobStatus.FAILED.value, error, time.time(), job_id),
            )

    def get_job(self, job_id: str) -> Optional[dict]:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def delete_job(self, job_id: str):
        with _connect() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))

    def count_active_jobs(self) -> int:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM jobs WHERE status IN (?, ?)",
                (JobStatus.QUEUED.value, JobStatus.PROCESSING.value),
            ).fetchone()
            return row["c"]


job_manager = JobManager()
