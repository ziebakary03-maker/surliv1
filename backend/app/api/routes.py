cat > backend/app/api/routes.py << 'EOF'
"""
API routes (section 21 du spec).

    POST   /api/upload
    GET    /api/jobs/{job_id}
    POST   /api/jobs/{job_id}/target
    GET    /api/jobs/{job_id}/preview/{frame}   (bonus: pour la sélection UI)
    GET    /api/jobs/{job_id}/result
    DELETE /api/jobs/{job_id}
    GET    /health
"""
from __future__ import annotations
import base64
import os
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import (
    UploadResponse, JobStatus, TargetSelectionRequest, TargetSelectionResponse,
    JobProgress, JobResultResponse, FramesPreviewResponse, DetectedObject,
    ConfidenceLevel, TargetState,
)
from app.services.job_manager import job_manager
from app.services.storage import storage
from app.video.processor import get_frame
from app.tracking.detector import build_detector

router = APIRouter()


def _validate_upload(file: UploadFile, size_bytes: int):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Extension non supportée: {ext}. Autorisées: {settings.ALLOWED_VIDEO_EXTENSIONS}")
    if file.content_type not in settings.ALLOWED_VIDEO_MIME_TYPES:
        raise HTTPException(400, f"Type MIME non supporté: {file.content_type}")
    max_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(400, f"Fichier trop volumineux (max {settings.MAX_VIDEO_SIZE_MB} Mo)")


def _get_warmed_detections(local_path: str, target_frame: int):
    """Crée un détecteur, le fait 'chauffer' en rejouant les frames
    précédentes (nécessaire pour un détecteur à soustraction de fond
    comme OpenCVDetector/MOG2, qui n'a aucune notion de 'fond' tant
    qu'il n'a pas vu plusieurs frames), puis retourne les détections
    de la frame ciblée.
    """
    cap = cv2.VideoCapture(local_path)
    detector = build_detector()

    detections = []
    for i in range(target_frame + 1):
        ok, frame_img = cap.read()
        if not ok:
            break
        detections = detector.detect(frame_img)
    cap.release()
    return detections


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/api/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if job_manager.count_active_jobs() >= settings.MAX_CONCURRENT_JOBS:
        raise HTTPException(429, "Trop de jobs en cours de traitement, réessayez plus tard.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    tmp_path = f"/tmp/{os.urandom(8).hex()}{ext}"
    contents = await file.read()
    _validate_upload(file, len(contents))

    with open(tmp_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps if fps else 0
    if duration > settings.MAX_VIDEO_DURATION_SECONDS:
        os.remove(tmp_path)
        raise HTTPException(400, f"Vidéo trop longue (max {settings.MAX_VIDEO_DURATION_SECONDS}s)")

    key = storage.new_key("uploads", ext)
    storage.save(tmp_path, key)
    os.remove(tmp_path)

    job_id = job_manager.create_job(input_key=key)
    return UploadResponse(job_id=job_id, status=JobStatus.AWAITING_TARGET)


@router.get("/api/jobs/{job_id}/preview/{frame}", response_model=FramesPreviewResponse)
def preview_frame(job_id: str, frame: int):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")

    local_path = storage.path_for_read(job["input_key"])
    image, meta = get_frame(local_path, frame)
    detections = _get_warmed_detections(local_path, frame)

    ok, buf = cv2.imencode(".jpg", image)
    image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return FramesPreviewResponse(
        job_id=job_id,
        frame_index=frame,
        fps=meta.fps,
        total_frames=meta.total_frames,
        detections=detections,
        image_base64=image_b64,
    )


@router.post("/api/jobs/{job_id}/target", response_model=TargetSelectionResponse)
def select_target(job_id: str, req: TargetSelectionRequest):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")

    local_path = storage.path_for_read(job["input_key"])
    detections = _get_warmed_detections(local_path, req.frame)

    best = None
    best_dist = float("inf")
    for det in detections:
        cx = det.bbox.x + det.bbox.width / 2
        cy = det.bbox.y + det.bbox.height / 2
        dist = ((cx - req.x) ** 2 + (cy - req.y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = det

    if best is None:
        raise HTTPException(400, "Aucun objet détecté à cet endroit de la frame.")

    job_manager.set_target(job_id, req.frame, req.x, req.y)

    return TargetSelectionResponse(
        job_id=job_id, target_id=1, bbox=best.bbox, status=JobStatus.QUEUED
    )


@router.get("/api/jobs/{job_id}", response_model=JobProgress)
def get_job_progress(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")

    total = job["total_frames"] or 0
    current = job["current_frame"] or 0
    progress = (current / total * 100) if total else 0.0

    level = None
    if job["confidence_percent"] is not None:
        c = job["confidence_percent"]
        level = ConfidenceLevel.HIGH if c >= 90 else ConfidenceLevel.MEDIUM if c >= 70 else ConfidenceLevel.LOW

    return JobProgress(
        job_id=job_id,
        status=JobStatus(job["status"]),
        current_frame=current,
        total_frames=total,
        progress_percent=progress,
        target_id=job["target_id"],
        target_state=TargetState(job["target_state"]) if job["target_state"] else None,
        confidence_percent=job["confidence_percent"],
        confidence_level=level,
        identity_switches=job["identity_switches"] or 0,
        error=job["error"],
    )


@router.get("/api/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    if job["status"] != JobStatus.COMPLETED.value:
        return JobResultResponse(job_id=job_id, status=JobStatus(job["status"]))

    import json
    metrics = json.loads(job["metrics_json"]) if job["metrics_json"] else None
    return JobResultResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        result_video_url=f"/api/jobs/{job_id}/video",
        metrics=metrics,
    )


@router.get("/api/jobs/{job_id}/video")
def download_result_video(job_id: str):
    job = job_manager.get_job(job_id)
    if not job or not job["result_key"]:
        raise HTTPException(404, "Résultat non disponible")
    local_path = storage.path_for_read(job["result_key"])
    return FileResponse(local_path, media_type="video/mp4", filename="tracked_result.mp4")


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    for key in (job["input_key"], job["result_key"]):
        if key:
            try:
                storage.delete(key)
            except Exception:
                pass
    job_manager.delete_job(job_id)
    return {"deleted": True}
EOF
