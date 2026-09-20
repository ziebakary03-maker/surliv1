"""
Configuration centrale de l'application, lue depuis les variables
d'environnement. Toutes les limites et tous les chemins de stockage
sont pilotés ici pour rester faciles à changer en dev / Docker / Fly.io.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Général ---
    APP_NAME: str = "Object Tracker"
    ENV: str = os.getenv("ENV", "development")

    # --- Limites vidéo ---
    MAX_VIDEO_SIZE_MB: int = int(os.getenv("MAX_VIDEO_SIZE_MB", "200"))
    MAX_VIDEO_DURATION_SECONDS: int = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "120"))
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    ALLOWED_VIDEO_EXTENSIONS: tuple = (".mp4", ".mov", ".webm")
    ALLOWED_VIDEO_MIME_TYPES: tuple = (
        "video/mp4",
        "video/quicktime",
        "video/webm",
        "video/x-matroska",
    )

    # --- Stockage local (fallback dev, jamais utilisé comme stockage
    #     permanent sur Fly.io: le filesystem n'est pas persistant) ---
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "/data/storage")

    # --- Stockage S3 (compatible AWS S3 / MinIO / Tigris sur Fly.io) ---
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")  # "local" | "s3"
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "object-tracker")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "")
    S3_REGION: str = os.getenv("S3_REGION", "auto")

    # --- Device de calcul ---
    DEVICE: str = os.getenv("DEVICE", "cpu")  # "cpu" | "cuda", auto-detecté si "auto"

    # --- Détecteur ---
    # "opencv" = détecteur intégré sans dépendance lourde (fonctionne partout).
    # "yolo"   = utilise ultralytics/YOLO si installé (voir requirements.txt).
    DETECTOR_BACKEND: str = os.getenv("DETECTOR_BACKEND", "opencv")
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4"))
    # Classes du modèle YOLO personnalisé (section 13). Les poids YOLO
    # génériques COCO NE savent PAS distinguer "gobelet cible" / "boule" de
    # ce scénario : un modèle entraîné spécifiquement est nécessaire pour
    # un mode sémantique fiable (voir README "Entraîner le modèle YOLO").
    BALL_CLASS_ID: int = int(os.getenv("BALL_CLASS_ID", "0"))
    CUP_CLASS_ID: int = int(os.getenv("CUP_CLASS_ID", "1"))

    # --- Tracking générique ---
    MAX_OCCLUSION_FRAMES: int = int(os.getenv("MAX_OCCLUSION_FRAMES", "45"))
    REID_APPEARANCE_WEIGHT: float = float(os.getenv("REID_APPEARANCE_WEIGHT", "0.4"))
    REID_MOTION_WEIGHT: float = float(os.getenv("REID_MOTION_WEIGHT", "0.6"))

    # --- Association boule <-> conteneur (section 6/9/12) ---
    # Distance max (en pixels, fraction de la diagonale du conteneur) pour
    # considérer que la boule disparue était "sous" tel conteneur plutôt
    # qu'un autre.
    CONTAINER_ASSOCIATION_THRESHOLD: float = float(os.getenv("CONTAINER_ASSOCIATION_THRESHOLD", "0.75"))
    # Seuil plancher utilisé quand la scène est très rapide (section 24) :
    # on tolère une association moins nette plutôt que de renoncer et de
    # tomber en AMBIGUOUS/LOST alors qu'un conteneur plausible existe.
    CONTAINER_ASSOCIATION_MIN_THRESHOLD: float = float(os.getenv("CONTAINER_ASSOCIATION_MIN_THRESHOLD", "0.35"))
    # Vitesse (px/frame) de la boule au-delà de laquelle le seuil est
    # totalement relâché vers CONTAINER_ASSOCIATION_MIN_THRESHOLD.
    CONTAINER_ASSOCIATION_SPEED_REFERENCE: float = float(os.getenv("CONTAINER_ASSOCIATION_SPEED_REFERENCE", "25.0"))
    # Nombre de frames sans détection avant de considérer la boule
    # "totalement occluse" plutôt qu'un simple raté ponctuel du détecteur.
    OCCLUSION_PENDING_FRAMES: int = int(os.getenv("OCCLUSION_PENDING_FRAMES", "2"))

    # --- Ré-identification (section 11) ---
    REIDENTIFICATION_THRESHOLD: float = float(os.getenv("REIDENTIFICATION_THRESHOLD", "0.75"))
    AMBIGUOUS_THRESHOLD: float = float(os.getenv("AMBIGUOUS_THRESHOLD", "0.45"))

    # --- Poids du score d'identité combiné (section 11) ---
    MOTION_WEIGHT: float = float(os.getenv("MOTION_WEIGHT", "0.25"))
    APPEARANCE_WEIGHT: float = float(os.getenv("APPEARANCE_WEIGHT", "0.30"))
    TRAJECTORY_WEIGHT: float = float(os.getenv("TRAJECTORY_WEIGHT", "0.20"))
    CONTAINER_WEIGHT: float = float(os.getenv("CONTAINER_WEIGHT", "0.25"))
    POSITION_PREDICTION_WEIGHT: float = float(os.getenv("POSITION_PREDICTION_WEIGHT", "0.5"))

    # --- Croisements de conteneurs (section 10) ---
    # IoU des bbox prédites au-delà duquel on considère deux conteneurs
    # "en croisement" (confiance réduite, pas de ré-assignation d'ID).
    CROSSING_IOU_THRESHOLD: float = float(os.getenv("CROSSING_IOU_THRESHOLD", "0.15"))

    # --- Performance (section 22) ---
    # Permet de sauter des frames de détection/tracking pour accélérer un
    # pipeline YOLO coûteux sur CPU ; la position est alors seulement
    # prédite (Kalman) sur les frames sautées. 1 = aucune frame sautée.
    PROCESS_EVERY_N_FRAMES: int = int(os.getenv("PROCESS_EVERY_N_FRAMES", "1"))
    DETECTION_INTERVAL: int = int(os.getenv("DETECTION_INTERVAL", "1"))

    # --- Worker / infra ---
    WORKER_POLL_INTERVAL: int = int(os.getenv("WORKER_POLL_INTERVAL", "2"))
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
