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
import psycopg2.pool

from app.core.config import settings
from app.models.schemas import JobStatus

DATABASE_URL = os.environ.get("DATABASE_URL")

_lock = threading.Lock()

# Pool de connexions au lieu d'un psycopg2.connect() par appel : ouvrir une
# connexion TCP + authentifier vers Postgres coûte typiquement 10-50ms sur
# le réseau Fly.io. update_progress() étant appelé une fois par frame par le
# worker, ça représentait avant ce changement plusieurs minutes de latence
# réseau pure sur une vidéo de quelques milliers de frames. Le pool garde
# les connexions ouvertes et les réutilise.
_pool: "psycopg2.pool.ThreadedConnectionPool | None" = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("DB_POOL_MAX_CONN", "5")),
            dsn=DATABASE_URL,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
    return _pool


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
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


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
        """
        Ne doit compter que les jobs qui occupent réellement une place de
        traitement (section 26 : limite de concurrence pour le worker).

        Bug corrigé : create_job() insère toujours status="queued", même
        pour un job qui attend encore que l'utilisateur choisisse sa cible
        (target_frame NULL). claim_next_job() n'y touche jamais tant que
        target_frame est NULL, donc un upload abandonné (l'utilisateur
        quitte la page avant de cliquer sur l'objet) restait "queued" pour
        toujours et consommait une place dans MAX_CONCURRENT_JOBS (=2 par
        défaut) définitivement. Il suffisait de 2 uploads abandonnés pour
        bloquer tout le monde avec un 429 permanent. On ne compte donc
        plus que les jobs réellement soumis au worker (queued + target
        choisi) ou en cours de traitement.
        """
        self.expire_stale_jobs()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT COUNT(*) as c FROM jobs
                       WHERE status = %s
                          OR (status = %s AND target_frame IS NOT NULL)""",
                    (JobStatus.PROCESSING.value, JobStatus.QUEUED.value),
                )
                return cur.fetchone()["c"]

    def expire_stale_jobs(
        self,
        processing_timeout_seconds: float = None,
        awaiting_target_timeout_seconds: float = None,
    ) -> int:
        """
        Filet de sécurité complémentaire : si un worker crashe pendant le
        traitement d'un job, celui-ci reste "processing" pour toujours
        (personne ne le marque failed). On le détecte via updated_at (plus
        de mise à jour de progression depuis trop longtemps) et on le
        marque failed pour libérer sa place. Idem pour un job resté
        "queued" sans target_frame pendant très longtemps (upload
        abandonné) : on le marque failed pour ne pas laisser la table
        grossir indéfiniment.
        """
        processing_timeout = processing_timeout_seconds or float(
            os.getenv("JOB_PROCESSING_TIMEOUT_SECONDS", "900")  # 15 min sans update
        )
        awaiting_timeout = awaiting_target_timeout_seconds or float(
            os.getenv("JOB_AWAITING_TARGET_TIMEOUT_SECONDS", "3600")  # 1h sans cible choisie
        )
        now = time.time()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE jobs SET status=%s, error=%s, updated_at=%s
                       WHERE status=%s AND updated_at < %s""",
                    (
                        JobStatus.FAILED.value,
                        "Job expiré: aucune progression reçue (worker probablement interrompu).",
                        now,
                        JobStatus.PROCESSING.value,
                        now - processing_timeout,
                    ),
                )
                processing_expired = cur.rowcount
                cur.execute(
                    """UPDATE jobs SET status=%s, error=%s, updated_at=%s
                       WHERE status=%s AND target_frame IS NULL AND updated_at < %s""",
                    (
                        JobStatus.FAILED.value,
                        "Job expiré: aucune cible sélectionnée par l'utilisateur.",
                        now,
                        JobStatus.QUEUED.value,
                        now - awaiting_timeout,
                    ),
                )
                awaiting_expired = cur.rowcount
                return processing_expired + awaiting_expired


job_manager = JobManager()
