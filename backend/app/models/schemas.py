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
    """États du cycle de vie de l'identité suivie.

    Les états historiques (DETECTED/TRACKING/FAST_MOVEMENT/OCCLUDED/
    CONFIDENT) sont conservés pour compatibilité (anciens jobs stockés,
    tests existants) mais le pipeline "boule sous gobelet" utilise
    désormais les états du cahier des charges "container tracking"
    (section 7) :

        VISIBLE -> OCCLUSION_PENDING -> HIDDEN_UNDER_CUP -> CUP_TRACKING
        -> REAPPEARING -> REIDENTIFYING -> (VISIBLE | AMBIGUOUS | LOST)

    avec CROSSING comme état transverse de confiance réduite quand le
    conteneur actuellement suivi croise un autre conteneur.
    """
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    FAST_MOVEMENT = "FAST_MOVEMENT"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REIDENTIFYING = "REIDENTIFYING"
    CONFIDENT = "CONFIDENT"
    AMBIGUOUS = "AMBIGUOUS"

    # --- États spécifiques au tracking "objet caché sous un conteneur" ---
    VISIBLE = "VISIBLE"
    OCCLUSION_PENDING = "OCCLUSION_PENDING"
    HIDDEN_UNDER_CUP = "HIDDEN_UNDER_CUP"
    CUP_TRACKING = "CUP_TRACKING"
    REAPPEARING = "REAPPEARING"
    CROSSING = "CROSSING"


class CupPosition(str, Enum):
    """Position finale du gobelet verrouillé, relative aux autres gobelets
    détectés (ou aux tiers de l'image en repli)."""
    LEFT = "LEFT"
    MIDDLE = "MIDDLE"
    RIGHT = "RIGHT"


class ObjectType(str, Enum):
    """Type sémantique d'un objet suivi (section 15 du cahier des charges).

    UNKNOWN est utilisé par le détecteur OpenCV (pas de compréhension
    sémantique, cf. detector.py) : le reste du pipeline doit alors
    considérer n'importe quelle autre piste comme un conteneur candidat
    plutôt que de supposer à tort qu'elle est un "gobelet".
    """
    BALL = "ball"
    CUP = "cup"
    UNKNOWN = "unknown"


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
    object_type: ObjectType = ObjectType.UNKNOWN


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
    # --- Section 20 : état du conteneur actuellement associé au target ---
    container_id: Optional[int] = None
    container_confidence: Optional[float] = None
    ball_visible: Optional[bool] = None


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_video_url: Optional[str] = None
    metrics: Optional[dict] = None


class TargetFinalResult(BaseModel):
    """Résultat final structuré (section 19 du cahier des charges).
    Sérialisé dans `metrics` par le worker à la fin du traitement."""
    target_id: int
    target_type: ObjectType = ObjectType.BALL
    final_state: TargetState
    final_container_id: Optional[int] = None
    final_position: Optional[CupPosition] = None
    display_state: str = "AMBIGUOUS"  # "FOUND" | valeur brute de final_state sinon
    confidence: float
    identity_switches: int = 0
    occlusion_duration: int = 0
    ambiguous_frames: int = 0
    reidentification_events: int = 0
