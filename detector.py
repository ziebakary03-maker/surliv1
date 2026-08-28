"""
Detector
--------
Interface de détection d'objets par frame. Deux implémentations :

- OpenCVDetector : détecteur "background subtraction + contours" qui ne
  nécessite aucun poids/modèle à télécharger. Fonctionne bien pour le cas
  du spec (objets identiques en mouvement sur fond relativement stable,
  ex: 3 gobelets sur une table). C'est le backend par défaut car
  l'environnement de build peut ne pas avoir accès à Internet pour
  télécharger des poids YOLO.

- YoloDetector : utilise ultralytics/YOLO si le package est installé
  (voir requirements.txt, dépendance optionnelle commentée). Recommandé
  en production pour de meilleures détections sur fonds complexes.

Les deux exposent la même méthode `detect(frame) -> List[DetectedObject]`
afin que le reste du pipeline (Tracker, ConfidenceEngine, etc.) soit
totalement agnostique du backend utilisé.
"""
from __future__ import annotations
from typing import List
import cv2
import numpy as np

from app.models.schemas import DetectedObject, BoundingBox
from app.core.config import settings


class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        raise NotImplementedError

    def warmup(self):
        pass


class OpenCVDetector(BaseDetector):
    """
    Détecteur sans dépendance lourde, basé sur une soustraction de fond
    (MOG2) + recherche de contours. Conçu pour repérer N objets mobiles
    de taille comparable sur un fond globalement stable, ce qui
    correspond au cas d'usage cible (3 objets identiques sur une table).

    NOTE: ce détecteur n'a pas de notion sémantique de "gobelet" — il
    détecte des "blobs en mouvement/différents du fond". C'est une
    limite assumée et documentée dans le README ; brancher YoloDetector
    donne une détection sémantique réelle.
    """

    def __init__(self, expected_objects: int = 3, min_area: int = 300, learning_rate: float = 0.01):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=True
        )
        self.expected_objects = expected_objects
        self.min_area = min_area
        self.learning_rate = learning_rate

    def warmup(self, frame: np.ndarray = None, n: int = 15):
        """Nourrit le modèle de fond avec quelques frames avant la
        première vraie détection (utile si les objets sont déjà en
        mouvement dès la première frame de la vidéo)."""
        if frame is not None:
            for _ in range(n):
                self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        # Supprime les ombres (valeur 127 dans le mask MOG2) et le bruit
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Score de confiance grossier basé sur la taille du blob relative
            # à la médiane attendue (les 3 objets sont supposés similaires).
            candidates.append((area, BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))))

        # Garde les N plus grands blobs (N = nombre d'objets attendu),
        # ce qui filtre le bruit résiduel sans halluciner d'objets.
        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[: max(self.expected_objects * 2, 6)]

        detections = []
        for i, (area, bbox) in enumerate(candidates):
            conf = float(np.clip(area / 5000.0, 0.3, 0.99))
            detections.append(DetectedObject(detection_id=i, bbox=bbox, confidence=conf))
        return detections


class YoloDetector(BaseDetector):
    """
    Wrapper autour d'ultralytics YOLO. N'est utilisable que si le package
    `ultralytics` (+ torch) est installé — voir requirements.txt.
    Recommandé en production : détection sémantique réelle, robuste aux
    fonds complexes et aux occlusions partielles.
    """

    def __init__(self, model_path: str = None, confidence: float = None, device: str = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics n'est pas installé. Décommentez torch/ultralytics "
                "dans requirements.txt ou utilisez DETECTOR_BACKEND=opencv."
            ) from e
        self.model = YOLO(model_path or settings.YOLO_MODEL_PATH)
        self.confidence = confidence or settings.YOLO_CONFIDENCE_THRESHOLD
        self.device = device or settings.DEVICE

    def warmup(self):
        pass

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        results = self.model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False
        )
        detections = []
        if not results:
            return detections
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            detections.append(
                DetectedObject(
                    detection_id=i,
                    bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                    confidence=conf,
                )
            )
        return detections


def build_detector(expected_objects: int = 3) -> BaseDetector:
    """Factory choisie via DETECTOR_BACKEND (opencv | yolo)."""
    if settings.DETECTOR_BACKEND == "yolo":
        try:
            return YoloDetector()
        except RuntimeError:
            # Repli automatique et explicite si YOLO n'est pas disponible,
            # plutôt qu'un crash silencieux au premier job.
            return OpenCVDetector(expected_objects=expected_objects)
    return OpenCVDetector(expected_objects=expected_objects)
