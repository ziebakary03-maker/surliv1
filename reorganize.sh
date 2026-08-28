mkdir -p backend/app/api backend/app/core backend/app/models backend/app/services backend/app/tracking backend/app/video
mkdir -p frontend/src/components
mkdir -p worker scripts tests

[ -f api.ts ] && mv api.ts frontend/src/api.ts
[ -f App.tsx ] && mv App.tsx frontend/src/App.tsx
[ -f main.tsx ] && mv main.tsx frontend/src/main.tsx
[ -f styles.css ] && mv styles.css frontend/src/styles.css
[ -f index.html ] && mv index.html frontend/index.html
[ -f package.json ] && mv package.json frontend/package.json
[ -f package-lock.json ] && mv package-lock.json frontend/package-lock.json
[ -f tsconfig.json ] && mv tsconfig.json frontend/tsconfig.json
[ -f vite.config.ts ] && mv vite.config.ts frontend/vite.config.ts
[ -d node_modules ] && mv node_modules frontend/node_modules

[ -f Processing.tsx ] && mv Processing.tsx frontend/src/components/Processing.tsx
[ -f Result.tsx ] && mv Result.tsx frontend/src/components/Result.tsx
[ -f TargetSelector.tsx ] && mv TargetSelector.tsx frontend/src/components/TargetSelector.tsx
[ -f Upload.tsx ] && mv Upload.tsx frontend/src/components/Upload.tsx

[ -f main.py ] && mv main.py backend/app/main.py
[ -f config.py ] && mv config.py backend/app/core/config.py
[ -f job_manager.py ] && mv job_manager.py backend/app/services/job_manager.py
[ -f storage.py ] && mv storage.py backend/app/services/storage.py
[ -f confidence.py ] && mv confidence.py backend/app/tracking/confidence.py
[ -f detector.py ] && mv detector.py backend/app/tracking/detector.py
[ -f identity_manager.py ] && mv identity_manager.py backend/app/tracking/identity_manager.py
[ -f motion_predictor.py ] && mv motion_predictor.py backend/app/tracking/motion_predictor.py
[ -f reidentifier.py ] && mv reidentifier.py backend/app/tracking/reidentifier.py
[ -f tracker.py ] && mv tracker.py backend/app/tracking/tracker.py
[ -f processor.py ] && mv processor.py backend/app/video/processor.py
[ -f requirements.txt ] && mv requirements.txt backend/requirements.txt

[ -f test_video.py ] && mv test_video.py scripts/test_video.py
[ -f synthetic_video.py ] && mv synthetic_video.py tests/synthetic_video.py
[ -f test_tracker.py ] && mv test_tracker.py tests/test_tracker.py

[ -f worker.py ] && mv worker.py worker/worker.py

if [ -f __init__.py ]; then
  cp __init__.py backend/app/__init__.py
  cp __init__.py backend/app/api/__init__.py
  cp __init__.py backend/app/core/__init__.py
  cp __init__.py backend/app/models/__init__.py
  cp __init__.py backend/app/services/__init__.py
  cp __init__.py backend/app/tracking/__init__.py
  cp __init__.py backend/app/video/__init__.py
  rm __init__.py
fi

cat > backend/app/api/routes.py << 'ROUTESEOF'
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

    detector = build_detector()
    detections = detector.detect(image)

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
    image, _ = get_frame(local_path, req.frame)
    detector = build_detector()
    detections = detector.detect(image)

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
ROUTESEOF

cat > backend/app/models/schemas.py << 'SCHEMASEOF'
"""
Schémas Pydantic partagés entre l'API et les services internes.
"""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    AWAITING_TARGET = "awaiting_target"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TargetState(str, Enum):
    """États du cycle de vie de l'identité suivie (section 15 du spec)."""
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    FAST_MOVEMENT = "FAST_MOVEMENT"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REIDENTIFYING = "REIDENTIFYING"
    CONFIDENT = "CONFIDENT"
    AMBIGUOUS = "AMBIGUOUS"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectedObject(BaseModel):
    detection_id: int
    bbox: BoundingBox
    confidence: float


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = "Video uploaded, awaiting target selection."


class TargetSelectionRequest(BaseModel):
    frame: int = Field(..., description="Index de la frame où le clic a eu lieu")
    x: float
    y: float


class TargetSelectionResponse(BaseModel):
    job_id: str
    target_id: int
    bbox: BoundingBox
    status: JobStatus


class FramesPreviewResponse(BaseModel):
    job_id: str
    frame_index: int
    fps: float
    total_frames: int
    detections: List[DetectedObject]
    image_base64: str


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    current_frame: int = 0
    total_frames: int = 0
    progress_percent: float = 0.0
    target_id: Optional[int] = None
    target_state: Optional[TargetState] = None
    confidence_percent: Optional[float] = None
    confidence_level: Optional[ConfidenceLevel] = None
    identity_switches: int = 0
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_video_url: Optional[str] = None
    metrics: Optional[dict] = None
SCHEMASEOF

cat > .gitignore << 'GITEOF'
__pycache__/
*.pyc
.pytest_cache/
.env
node_modules/
frontend/dist/
*.mp4
*.mov
*.webm
!frontend/public/**
output/
test_assets/
tests/_assets/
data/
.DS_Store
*.sqlite3
GITEOF

cat > .env.example << 'ENVEOF'
ENV=development
MAX_VIDEO_SIZE_MB=200
MAX_VIDEO_DURATION_SECONDS=120
MAX_CONCURRENT_JOBS=2
STORAGE_BACKEND=local
LOCAL_STORAGE_PATH=/data/storage
S3_ENDPOINT=
S3_BUCKET=object-tracker
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_REGION=auto
DEVICE=cpu
DETECTOR_BACKEND=opencv
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.4
MAX_OCCLUSION_FRAMES=45
REID_APPEARANCE_WEIGHT=0.4
REID_MOTION_WEIGHT=0.6
WORKER_POLL_INTERVAL=2
ENVEOF

echo "=== TERMINE - structure obtenue : ==="
find . -maxdepth 4 -type f -not -path "./node_modules/*" -not -path "./frontend/node_modules/*" -not -path "./.git/*" | sort
