"""
IdentityManager
---------------
Fait le lien entre le MultiObjectTracker (générique) et le concept
métier du spec : un seul "Target" sélectionné par l'utilisateur, dont
on doit connaître l'état à chaque frame (section 15) et le niveau de
confiance (section 14), indépendamment de la position brute.

Ce module NE fait jamais "l'objet le plus à gauche reste l'objet A"
(interdiction explicite section 4) : le lien se fait uniquement via
`track_id` retourné par MultiObjectTracker, qui lui-même est maintenu
par mouvement + apparence, pas par position absolue.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

from app.models.schemas import BoundingBox, TargetState, ConfidenceLevel
from app.tracking.tracker import MultiObjectTracker, Track, _centroid
from app.tracking.confidence import ConfidenceEngine, ConfidenceInputs
from app.core.config import settings


@dataclass
class TargetFrameResult:
    frame_index: int
    target_id: int
    bbox: Optional[BoundingBox]
    state: TargetState
    confidence_percent: float
    confidence_level: ConfidenceLevel
    identity_switches_total: int


class IdentityManager:
    def __init__(self, tracker: MultiObjectTracker):
        self.tracker = tracker
        self.confidence_engine = ConfidenceEngine()
        self.target_track_id: Optional[int] = None
        self.identity_switches = 0
        self._last_known_bbox: Optional[BoundingBox] = None
        self._frames_ambiguous_candidates: List[int] = []

    def select_target(self, click_x: float, click_y: float, tracks: List[Track]) -> int:
        """
        Détermine quelle piste détectée correspond au clic utilisateur
        (section 3) et l'enregistre comme Target ID = 1 (ici, on garde
        l'ID interne du tracker comme "Target ID" pour simplifier —
        l'API expose toujours "Target #1" côté utilisateur).
        """
        best_track = None
        best_dist = float("inf")
        for track in tracks:
            cx, cy = _centroid(track.last_bbox)
            dist = ((cx - click_x) ** 2 + (cy - click_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is None:
            raise ValueError("Aucun objet détecté à proximité des coordonnées cliquées.")

        self.target_track_id = best_track.track_id
        self._last_known_bbox = best_track.last_bbox
        return self.target_track_id

    def process_frame(self, frame_index: int, tracks: List[Track]) -> TargetFrameResult:
        if self.target_track_id is None:
            raise RuntimeError("Aucun target sélectionné. Appelez select_target d'abord.")

        target_track = next((t for t in tracks if t.track_id == self.target_track_id), None)

        if target_track is None:
            # Le tracker a définitivement perdu cette piste (au-delà de
            # MAX_OCCLUSION_FRAMES). On ne réassigne JAMAIS silencieusement
            # un autre track_id ici : c'est au niveau du tracker que la
            # ré-identification par apparence doit avoir déjà eu lieu.
            return TargetFrameResult(
                frame_index=frame_index,
                target_id=self.target_track_id,
                bbox=self._last_known_bbox,
                state=TargetState.LOST,
                confidence_percent=0.0,
                confidence_level=ConfidenceLevel.LOW,
                identity_switches_total=self.identity_switches,
            )

        occluded = target_track.time_since_update > 0
        if occluded and target_track.time_since_update <= settings.MAX_OCCLUSION_FRAMES:
            state = (
                TargetState.REIDENTIFYING
                if target_track.time_since_update > 3
                else TargetState.OCCLUDED
            )
        else:
            speed = target_track.motion.speed
            state = TargetState.FAST_MOVEMENT if speed > 40 else TargetState.TRACKING

        trajectory_consistency = 1.0
        if target_track.trajectory_error:
            trajectory_consistency = max(0.0, 1.0 - min(target_track.trajectory_error / 100.0, 1.0))

        appearance_similarity = 1.0
        if self._last_known_bbox is not None and not occluded:
            appearance_similarity = 1.0  # déjà intégré via appearance.update() dans le tracker

        reid_confidence = 1.0
        if state == TargetState.REIDENTIFYING:
            reid_confidence = max(0.2, 1.0 - target_track.time_since_update / settings.MAX_OCCLUSION_FRAMES)

        inputs = ConfidenceInputs(
            tracking_confidence=target_track.last_confidence,
            motion_consistency=target_track.motion.motion_consistency(),
            trajectory_consistency=trajectory_consistency,
            appearance_similarity=appearance_similarity,
            occlusion_frames=target_track.time_since_update,
            max_occlusion_frames=settings.MAX_OCCLUSION_FRAMES,
            reid_confidence=reid_confidence,
        )
        confidence_percent, confidence_level = self.confidence_engine.compute(inputs)

        # Honnêteté scientifique (section 11) : si la confiance retombe très
        # bas après une occlusion longue, on affiche AMBIGUOUS plutôt que de
        # continuer à afficher TRACKING/CONFIDENT.
        if state == TargetState.REIDENTIFYING and confidence_percent < 55:
            state = TargetState.AMBIGUOUS
        elif state == TargetState.TRACKING and confidence_percent >= 90:
            state = TargetState.CONFIDENT

        if not occluded:
            self._last_known_bbox = target_track.last_bbox

        return TargetFrameResult(
            frame_index=frame_index,
            target_id=self.target_track_id,
            bbox=target_track.last_bbox if not occluded else self._last_known_bbox,
            state=state,
            confidence_percent=confidence_percent,
            confidence_level=confidence_level,
            identity_switches_total=self.identity_switches,
        )
