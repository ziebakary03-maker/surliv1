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
    manual_bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Cadre dessiné manuellement par l'utilisateur (mode sélection libre). "
                    "Si fourni, prime sur la détection automatique.",
    )


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
