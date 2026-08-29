"""
Tracker (MultiObjectTracker)
----------------------------
Pourquoi cette implémentation plutôt qu'importer ByteTrack/BoT-SORT/DeepSORT
tels quels : ces bibliothèques dépendent presque toutes d'un détecteur YOLO
et/ou de torch/torchvision pour l'extraction d'apparence (ré-id profonde),
qu'il n'est pas possible de télécharger dans cet environnement de build sans
accès réseau. Le tracker ci-dessous reprend cependant leurs idées centrales :

  - Association par coût combiné position/mouvement (comme SORT/ByteTrack) :
    ici via une prédiction de Kalman + distance euclidienne au lieu de
    distance IoU brute, ce qui est plus robuste pour des objets qui se
    croisent (l'IoU seul est ambigu pile au moment du croisement).
  - Association secondaire par apparence (comme DeepSORT / BoT-SORT), via
    histogrammes couleur au lieu d'un ré-id CNN profond (cf. reidentifier.py)
    pour rester utilisable en CPU sans poids à télécharger.
  - Assignment optimal par l'algorithme hongrois (scipy.optimize) plutôt
    qu'un matching glouton, pour minimiser les Identity Switches globaux
    plutôt que de matcher piste par piste dans un ordre arbitraire.

Le module est conçu pour être remplacé par un vrai ByteTrack/BoT-SORT :
il suffit d'implémenter la même interface `step(frame, detections)` dans
un autre fichier et de le brancher dans video/processor.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.models.schemas import DetectedObject, BoundingBox
from app.tracking.motion_predictor import MotionPredictor
from app.tracking.reidentifier import ReIdentifier, AppearanceSignature
from app.core.config import settings


def _centroid(bbox: BoundingBox):
    return bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0


@dataclass
class Track:
    track_id: int
    motion: MotionPredictor
    appearance: AppearanceSignature
    last_bbox: BoundingBox
    last_confidence: float = 1.0
    hits: int = 1
    time_since_update: int = 0
    predicted_position: Optional[tuple] = None
    trajectory_error: float = 0.0  # écart prédiction vs mesure sur le dernier match


class MultiObjectTracker:
    """
    Tracker multi-objets générique (indépendant de la notion de "target
    utilisateur", qui est gérée par IdentityManager). Maintient une liste
    de Track avec des IDs stables tant que les objets restent détectés ou
    en occlusion de courte durée.
    """

    def __init__(self, max_occlusion_frames: int = None, reid: ReIdentifier = None):
        self.tracks: List[Track] = []
        self._next_id = 1
        self.max_occlusion_frames = max_occlusion_frames or settings.MAX_OCCLUSION_FRAMES
        self.reid = reid or ReIdentifier()
        self.identity_switches = 0

    def _cost_matrix(self, frame, detections: List[DetectedObject], det_histograms):
        n_tracks, n_dets = len(self.tracks), len(detections)
        cost = np.full((n_tracks, n_dets), 1e6)
        motion_w = settings.REID_MOTION_WEIGHT
        appearance_w = settings.REID_APPEARANCE_WEIGHT

        for i, track in enumerate(self.tracks):
            px, py = track.predicted_position
            for j, det in enumerate(detections):
                dx, dy = _centroid(det.bbox)
                dist = float(np.hypot(px - dx, py - dy))
                # normalise la distance par une échelle de frame raisonnable
                dist_norm = min(dist / 400.0, 1.0)

                appearance_sim = track.appearance.similarity(det_histograms[j])
                appearance_cost = 1.0 - appearance_sim

                cost[i, j] = motion_w * dist_norm + appearance_w * appearance_cost

        return cost

    def step(self, frame, detections: List[DetectedObject]) -> List[Track]:
        # 1. Prédire la position de chaque piste existante (Kalman)
        for track in self.tracks:
            track.predicted_position = track.motion.predict()

        # Histogramme calculé une seule fois par détection (crop + HSV +
        # calcHist), puis réutilisé à la fois pour le coût d'association et
        # pour l'initialisation des nouvelles pistes — au lieu d'être
        # recalculé plusieurs fois par frame (cf. commentaire ci-dessous).
        det_histograms = [self.reid.extract_histogram(frame, det.bbox) for det in detections]

        matched_track_idx = set()
        matched_det_idx = set()

        if self.tracks and detections:
            cost = self._cost_matrix(frame, detections, det_histograms)
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] > 0.92:  # trop coûteux pour être un vrai match
                    continue
                track = self.tracks[r]
                det = detections[c]
                dx, dy = _centroid(det.bbox)
                px, py = track.predicted_position
                track.trajectory_error = float(np.hypot(px - dx, py - dy))

                track.motion.update(dx, dy)
                track.appearance.update(det_histograms[c])
                track.last_bbox = det.bbox
                track.last_confidence = det.confidence
                track.hits += 1
                track.time_since_update = 0
                matched_track_idx.add(r)
                matched_det_idx.add(c)

        # 2. Pistes non matchées : occlusion (on ne les supprime pas tout de
        #    suite, cf. section 9 du spec — l'identité doit survivre à une
        #    occlusion courte).
        alive_tracks = []
        for i, track in enumerate(self.tracks):
            if i not in matched_track_idx:
                track.time_since_update += 1
                if track.time_since_update <= self.max_occlusion_frames:
                    alive_tracks.append(track)
                # sinon: piste définitivement perdue, supprimée
            else:
                alive_tracks.append(track)
        self.tracks = alive_tracks

        # 3. Détections non matchées : nouvelles pistes candidates
        for j, det in enumerate(detections):
            if j in matched_det_idx:
                continue
            cx, cy = _centroid(det.bbox)
            motion = MotionPredictor(cx, cy)
            appearance = AppearanceSignature(det_histograms[j])
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    motion=motion,
                    appearance=appearance,
                    last_bbox=det.bbox,
                    last_confidence=det.confidence,
                )
            )
            self._next_id += 1

        return self.tracks
