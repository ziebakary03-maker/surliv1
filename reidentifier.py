"""
ReIdentifier
------------
Signature d'apparence légère par objet (histogramme couleur HSV de la
région de la bbox). Sert à deux choses :

1. Départager deux détections proches en position lors d'un croisement
   (section 10) en comparant leur apparence à celle enregistrée pour
   chaque identité avant le croisement.
2. Ré-identifier un objet après occlusion (section 9) en comparant les
   candidats réapparus à la dernière signature connue du target perdu.

C'est volontairement simple (pas de réseau de ré-identification profond)
pour rester utilisable en CPU sans dépendance lourde. La section 12 du
spec (signature "invisible" par marquage physique) est prévue comme
extension : voir `signature_id` dans IdentityManager et le README.
"""
from __future__ import annotations
import cv2
import numpy as np


class AppearanceSignature:
    def __init__(self, histogram: np.ndarray):
        self.histogram = histogram
        self.history: list[np.ndarray] = [histogram]
        self.max_history = 20

    def update(self, histogram: np.ndarray, alpha: float = 0.7):
        """Moyenne glissante pour lisser les variations d'éclairage/angle."""
        self.histogram = alpha * self.histogram + (1 - alpha) * histogram
        self.history.append(histogram)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def similarity(self, other_histogram: np.ndarray) -> float:
        """Retourne un score 0..1 (1 = identique) via corrélation d'histogrammes."""
        score = cv2.compareHist(
            self.histogram.astype("float32"), other_histogram.astype("float32"), cv2.HISTCMP_CORREL
        )
        return float(np.clip((score + 1) / 2, 0.0, 1.0))


class ReIdentifier:
    def __init__(self, bins: int = 32):
        self.bins = bins

    def extract_histogram(self, frame: np.ndarray, bbox) -> np.ndarray:
        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
        x, y = max(0, x), max(0, y)
        crop = frame[y:y + max(h, 1), x:x + max(w, 1)]
        if crop.size == 0:
            return np.zeros((self.bins, self.bins), dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.bins, self.bins], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def new_signature(self, frame: np.ndarray, bbox) -> AppearanceSignature:
        return AppearanceSignature(self.extract_histogram(frame, bbox))
