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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings


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

    def extract_contour_signature(self, frame: np.ndarray, bbox) -> float:
        """Ratio (aire du plus grand contour / aire de la bbox) : signal
        de forme grossier mais utile pour départager un objet rond (boule)
        d'un objet plus anguleux (gobelet), sans dépendance lourde."""
        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
        x, y = max(0, x), max(0, y)
        crop = frame[y:y + max(h, 1), x:x + max(w, 1)]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        bbox_area = max(w * h, 1)
        return float(np.clip(cv2.contourArea(largest) / bbox_area, 0.0, 1.0))

    def new_signature(self, frame: np.ndarray, bbox) -> AppearanceSignature:
        return AppearanceSignature(self.extract_histogram(frame, bbox))


# ---------------------------------------------------------------------------
# Signature multi-composantes (sections 4/5/16 du cahier des charges)
# ---------------------------------------------------------------------------
#
# L'histogramme HSV seul (AppearanceSignature ci-dessus) reste l'un des
# signaux mais n'est PAS suffisant pour départager des objets visuellement
# quasi identiques (plusieurs gobelets du même modèle). ObjectSignature
# combine :
#   - apparence : histogramme HSV + ratio de forme (contours)
#   - géométrie : taille, ratio largeur/hauteur
#   - historique : moyenne glissante sur plusieurs observations, jamais un
#     remplacement brutal par une seule observation potentiellement bruitée
#     (cf. section 4.C).


@dataclass
class GeometrySnapshot:
    center: Tuple[float, float]
    size: Tuple[float, float]  # (width, height)
    aspect_ratio: float


def _bbox_geometry(bbox) -> GeometrySnapshot:
    cx = bbox.x + bbox.width / 2.0
    cy = bbox.y + bbox.height / 2.0
    ar = bbox.width / bbox.height if bbox.height > 1e-6 else 1.0
    return GeometrySnapshot(center=(cx, cy), size=(bbox.width, bbox.height), aspect_ratio=ar)


class ObjectSignature:
    """Signature robuste multi-composantes créée lors de la sélection du
    target (section 4) et mise à jour à chaque observation confirmée.

    - `appearance` : AppearanceSignature (histogramme HSV, moyenne glissante)
    - `shape_ratio`: ratio de forme (contours), moyenne glissante
    - `geometry`   : dernière géométrie observée (taille/ratio)
    - `trajectory` : historique de centres (position/vitesse/direction
      dérivées via MotionPredictor, référencées ici uniquement pour le
      score de comparaison)
    - `history`    : observations brutes conservées (bornées) pour ne
      jamais écraser brutalement la signature par une seule mesure douteuse
    """

    def __init__(self, frame: np.ndarray, bbox, reidentifier: Optional[ReIdentifier] = None):
        self._reid = reidentifier or ReIdentifier()
        hist = self._reid.extract_histogram(frame, bbox)
        self.appearance = AppearanceSignature(hist)
        self.shape_ratio = self._reid.extract_contour_signature(frame, bbox)
        self.geometry = _bbox_geometry(bbox)
        self.history: List[GeometrySnapshot] = [self.geometry]
        self.max_history = 30
        self.observation_count = 1

    def update(self, frame: np.ndarray, bbox, alpha: float = 0.7):
        """Ne remplace jamais brutalement l'ancienne signature (section 4.C) :
        moyenne glissante sur l'apparence et la forme, historique borné
        pour la géométrie/trajectoire."""
        hist = self._reid.extract_histogram(frame, bbox)
        self.appearance.update(hist, alpha=alpha)

        shape = self._reid.extract_contour_signature(frame, bbox)
        self.shape_ratio = alpha * self.shape_ratio + (1 - alpha) * shape

        self.geometry = _bbox_geometry(bbox)
        self.history.append(self.geometry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.observation_count += 1

    def average_size(self) -> Tuple[float, float]:
        if not self.history:
            return self.geometry.size
        ws = [g.size[0] for g in self.history]
        hs = [g.size[1] for g in self.history]
        return float(np.mean(ws)), float(np.mean(hs))

    def compare(self, frame: np.ndarray, bbox) -> dict:
        """Compare cette signature à une nouvelle observation candidate.
        Retourne un dict de scores individuels 0..1 ET un score combiné,
        utilisé par IdentityManager.reidentify_target (section 11)."""
        candidate_hist = self._reid.extract_histogram(frame, bbox)
        appearance_similarity = self.appearance.similarity(candidate_hist)

        candidate_shape = self._reid.extract_contour_signature(frame, bbox)
        shape_similarity = 1.0 - min(abs(self.shape_ratio - candidate_shape), 1.0)

        avg_w, avg_h = self.average_size()
        size_similarity = 1.0
        if avg_w > 1e-6 and avg_h > 1e-6:
            w_ratio = min(bbox.width, avg_w) / max(bbox.width, avg_w)
            h_ratio = min(bbox.height, avg_h) / max(bbox.height, avg_h)
            size_similarity = float((w_ratio + h_ratio) / 2.0)

        combined = float(np.clip(
            0.55 * appearance_similarity + 0.20 * shape_similarity + 0.25 * size_similarity,
            0.0, 1.0,
        ))
        return {
            "appearance_similarity": appearance_similarity,
            "shape_similarity": shape_similarity,
            "size_similarity": size_similarity,
            "combined": combined,
        }


class MotionSignature:
    """Compare la vitesse/direction attendues à celles d'une piste
    candidate (section 5/11). Score 0..1, 1 = mouvement parfaitement
    cohérent avec ce qui était attendu."""

    @staticmethod
    def compare(expected_velocity: Tuple[float, float], candidate_velocity: Tuple[float, float]) -> float:
        ex, ey = expected_velocity
        cx, cy = candidate_velocity
        expected_speed = float(np.hypot(ex, ey))
        candidate_speed = float(np.hypot(cx, cy))

        if expected_speed < 1e-3 and candidate_speed < 1e-3:
            return 1.0

        speed_sim = 1.0 - min(abs(expected_speed - candidate_speed) / max(expected_speed, candidate_speed, 1e-3), 1.0)

        if expected_speed < 1e-3 or candidate_speed < 1e-3:
            direction_sim = 0.5  # pas assez d'info directionnelle
        else:
            cos_sim = (ex * cx + ey * cy) / (expected_speed * candidate_speed)
            direction_sim = float(np.clip((cos_sim + 1) / 2, 0.0, 1.0))

        return float(np.clip(0.5 * speed_sim + 0.5 * direction_sim, 0.0, 1.0))


class TrajectorySignature:
    """Compare une position candidate à une position prédite par
    extrapolation de la trajectoire connue (section 11). Score 0..1
    décroissant avec la distance normalisée."""

    @staticmethod
    def compare(predicted_position: Tuple[float, float], candidate_position: Tuple[float, float],
                normalization: float = 200.0) -> float:
        px, py = predicted_position
        cx, cy = candidate_position
        dist = float(np.hypot(px - cx, py - cy))
        return float(np.clip(1.0 - dist / max(normalization, 1e-3), 0.0, 1.0))
