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

    # --- Tracking ---
    MAX_OCCLUSION_FRAMES: int = int(os.getenv("MAX_OCCLUSION_FRAMES", "45"))
    REID_APPEARANCE_WEIGHT: float = float(os.getenv("REID_APPEARANCE_WEIGHT", "0.4"))
    REID_MOTION_WEIGHT: float = float(os.getenv("REID_MOTION_WEIGHT", "0.6"))

    class Config:
        env_file = ".env"


settings = Settings()
