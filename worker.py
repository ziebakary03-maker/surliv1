"""
Worker (section 26 du spec).

Process séparé du serveur web : interroge périodiquement la file de
jobs (JobManager), traite les vidéos une par une, écrit le résultat, et
met à jour la progression consultée par le frontend via polling
GET /api/jobs/{job_id}.

Lancement local :
    python worker/worker.py

Dans Docker/Fly.io : process distinct défini dans fly.toml
(voir section [processes]), ce qui permet d'ajouter plusieurs workers
simplement en augmentant le nombre de machines sur ce process group.
"""
import os
import sys
import time
import uuid
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.job_manager import job_manager  # noqa: E402
from app.services.storage import storage  # noqa: E402
from app.video.processor import process_video  # noqa: E402
from app.core.config import settings  # noqa: E402

WORKER_ID = f"worker_{uuid.uuid4().hex[:8]}"
POLL_INTERVAL_SECONDS = float(os.getenv("WORKER_POLL_INTERVAL", "2"))


def process_one(job: dict):
    job_id = job["job_id"]
    print(f"[{WORKER_ID}] Traitement du job {job_id}")

    input_path = storage.path_for_read(job["input_key"])
    output_path = f"/tmp/{job_id}_result.mp4"

    def on_progress(current_frame, total_frames, result):
        job_manager.update_progress(
            job_id=job_id,
            current_frame=current_frame,
            total_frames=total_frames,
            target_id=result.target_id,
            target_state=result.state.value,
            confidence_percent=result.confidence_percent,
            identity_switches=result.identity_switches_total,
        )

    try:
        metrics = process_video(
            input_path=input_path,
            output_path=output_path,
            click_frame=int(job["target_frame"]),
            click_x=float(job["target_x"]),
            click_y=float(job["target_y"]),
            progress_cb=on_progress,
        )
        result_key = storage.new_key("results", ".mp4")
        storage.save(output_path, result_key)
        job_manager.complete_job(job_id, result_key=result_key, metrics=metrics)
        os.remove(output_path)
        print(f"[{WORKER_ID}] Job {job_id} terminé. Metrics: {metrics}")
    except Exception as e:
        traceback.print_exc()
        job_manager.fail_job(job_id, str(e))
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def main_loop():
    print(f"[{WORKER_ID}] Worker démarré. Device={settings.DEVICE}, Detector={settings.DETECTOR_BACKEND}")
    while True:
        job = job_manager.claim_next_job(WORKER_ID)
        if job:
            process_one(job)
        else:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
