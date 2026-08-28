"""
StorageManager
--------------
Abstraction de stockage (section 22 du spec). Le filesystem local de
Fly.io n'est PAS persistant entre déploiements/redémarrages : il ne doit
servir que de cache temporaire de traitement. Le backend "s3" (compatible
AWS S3, MinIO, ou Tigris — le stockage objet natif de Fly.io) est celui à
utiliser en production. Aucune clé secrète n'est jamais exposée au
frontend : seules des URLs de résultat signées/temporaires ou servies par
le backend transitent côté client.
"""
from __future__ import annotations
import os
import shutil
import uuid
from abc import ABC, abstractmethod

from app.core.config import settings


class BaseStorage(ABC):
    @abstractmethod
    def save(self, local_path: str, key: str) -> str:
        """Copie/upload un fichier local vers le stockage, retourne une clé."""

    @abstractmethod
    def path_for_read(self, key: str) -> str:
        """Retourne un chemin local lisible (télécharge si nécessaire)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @staticmethod
    def new_key(prefix: str, extension: str) -> str:
        # Nom de fichier aléatoire : jamais le nom original de l'utilisateur
        # (protection path traversal + anonymisation, section 29).
        return f"{prefix}/{uuid.uuid4().hex}{extension}"


class LocalStorage(BaseStorage):
    """Stockage disque local — pratique en dev, PAS en production Fly.io."""

    def __init__(self, root: str = None):
        self.root = root or settings.LOCAL_STORAGE_PATH
        os.makedirs(self.root, exist_ok=True)

    def _full_path(self, key: str) -> str:
        full = os.path.normpath(os.path.join(self.root, key))
        if not full.startswith(os.path.normpath(self.root)):
            raise ValueError("Chemin invalide (tentative de path traversal détectée).")
        return full

    def save(self, local_path: str, key: str) -> str:
        dest = self._full_path(key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_path, dest)
        return key

    def path_for_read(self, key: str) -> str:
        return self._full_path(key)

    def delete(self, key: str) -> None:
        full = self._full_path(key)
        if os.path.exists(full):
            os.remove(full)


class S3Storage(BaseStorage):
    """Stockage objet compatible S3 (AWS S3 / MinIO / Tigris)."""

    def __init__(self):
        import boto3

        self.bucket = settings.S3_BUCKET
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT or None,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
        )
        self._local_cache = settings.LOCAL_STORAGE_PATH
        os.makedirs(self._local_cache, exist_ok=True)

    def save(self, local_path: str, key: str) -> str:
        self.client.upload_file(local_path, self.bucket, key)
        return key

    def path_for_read(self, key: str) -> str:
        local_path = os.path.join(self._local_cache, key.replace("/", "_"))
        if not os.path.exists(local_path):
            self.client.download_file(self.bucket, key, local_path)
        return local_path

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


def build_storage() -> BaseStorage:
    if settings.STORAGE_BACKEND == "s3":
        return S3Storage()
    return LocalStorage()


storage = build_storage()
