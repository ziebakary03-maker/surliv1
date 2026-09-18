"""
IdentityManager — moteur "boule cachée sous un gobelet"
--------------------------------------------------------
Fait le lien entre le MultiObjectTracker (générique, indépendant du
scénario métier) et le concept central du cahier des charges :

    TARGET BALL -> BALL IDENTITY -> CURRENT CONTAINER CUP
    -> TRACK CONTAINER -> BALL REMAINS ASSOCIATED WITH CONTAINER
    -> REIDENTIFY BALL WHEN VISIBLE AGAIN

Ce module ne recherche JAMAIS la boule "image par image" : une fois que la
boule disparaît, son identité est reportée sur le conteneur qui la
dissimule (`container_track_id`), qui est lui-même suivi en continu par
`MultiObjectTracker` (mouvement + apparence + Kalman). La boule n'est
"perdue" (LOST) qu'en tout dernier recours ; l'état par défaut en cas de
doute résiduel est AMBIGUOUS (jamais une fausse certitude — section 24).

Méthodes principales (section 14 du cahier des charges) :
    select_target()        : lien initial clic utilisateur -> track_id
    update_target()         : boucle principale appelée une fois par frame
    associate_container()   : détermine sous quel conteneur la boule a disparu
    update_hidden_state()   : fait vivre l'estimation de position pendant l'occlusion
    reidentify_target()     : valide (ou non) la boule réapparue
    get_target_state()      : état exposé au reste du pipeline / à l'API
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

from app.models.schemas import BoundingBox, TargetState, ConfidenceLevel, ObjectType
from app.tracking.tracker import MultiObjectTracker, Track, _centroid, _iou
from app.tracking.reidentifier import ObjectSignature, MotionSignature, TrajectorySignature
from app.tracking.confidence import ConfidenceEngine, ConfidenceInputs
from app.core.config import settings


@dataclass
class HiddenObjectState:
    """État de l'objet cible pendant/à propos d'une occlusion complète
    (section 3 du cahier des charges)."""
    target_id: int
    visible: bool = True
    state: TargetState = TargetState.VISIBLE
    container_id: Optional[int] = None
    last_visible_bbox: Optional[BoundingBox] = None
    last_visible_center: Optional[Tuple[float, float]] = None
    hidden_since_frame: Optional[int] = None
    confidence: float = 1.0
    container_confidence: float = 1.0
    identity_confidence: float = 1.0
    # Décalage boule <-> centre du conteneur au moment de la disparition
    # (section 8) : estimated_ball_position = container_center + offset.
    last_offset: Optional[Tuple[float, float]] = None
    container_label: Optional[str] = None


@dataclass
class TargetFrameResult:
    frame_index: int
    target_id: int
    target_type: ObjectType
    bbox: Optional[BoundingBox]
    state: TargetState
    confidence_percent: float
    confidence_level: ConfidenceLevel
    identity_switches_total: int
    container_id: Optional[int] = None
    estimated: bool = False  # True si bbox est une ESTIMATION (section 18), pas une vraie détection
    container_bbox: Optional[BoundingBox] = None
    container_label: Optional[str] = None
    container_confidence: Optional[float] = None


def ball_track_is_live(tracks: List[Track], target_id: Optional[int]) -> bool:
    return any(t.track_id == target_id and t.time_since_update == 0 for t in tracks)



class IdentityManager:
    def __init__(self, tracker: MultiObjectTracker):
        self.tracker = tracker
        self.confidence_engine = ConfidenceEngine()

        self.target_track_id: Optional[int] = None
        self.target_type: ObjectType = ObjectType.BALL
        self.signature: Optional[ObjectSignature] = None

        self.hidden: HiddenObjectState = HiddenObjectState(target_id=1)

        # --- Compteurs exposés dans le résultat final (section 19) ---
        self.identity_switches = 0
        self.occlusion_duration = 0
        self.ambiguous_frames = 0
        self.reidentification_events = 0

        self._pending_occlusion_frames = 0
        self._frames_since_container_seen = 0
        self._reidentify_candidate: Optional[int] = None
        self._reidentify_streak = 0

        # Verrou métier du gobelet porteur : Cup_A ne change jamais.
        # Le track_id est seulement un handle technique du tracker.
        self._container_label = "Cup_A"
        self._container_signature = None
        self._container_last_center: Optional[Tuple[float, float]] = None
        self._container_miss_frames = 0
        self._container_rebinds = 0

    # ------------------------------------------------------------------
    # Sélection initiale (section 3)
    # ------------------------------------------------------------------
    def select_target(self, click_x: float, click_y: float, tracks: List[Track], frame=None) -> int:
        """Détermine quelle piste détectée correspond au clic utilisateur et
        crée l'identité persistante de la boule cible. Ne se base JAMAIS sur
        "l'objet le plus à gauche" : uniquement sur la proximité au clic à
        l'instant T, puis exclusivement sur le track_id du tracker ensuite."""
        best_track = None
        best_dist = float("inf")
        # On ne sélectionne que parmi les pistes réellement détectées cette
        # frame (time_since_update == 0) : `tracks` peut aussi contenir des
        # pistes "en coasting" (prédites depuis plusieurs frames sans
        # confirmation), qui ne représentent pas ce que l'utilisateur voit
        # réellement à l'écran au moment du clic. On ne retombe sur
        # l'ensemble complet que si, par malchance, aucune piste n'est
        # confirmée cette frame précise.
        live_tracks = [t for t in tracks if t.time_since_update == 0] or tracks
        for track in live_tracks:
            cx, cy = _centroid(track.last_bbox)
            dist = ((cx - click_x) ** 2 + (cy - click_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is None:
            raise ValueError("Aucun objet détecté à proximité des coordonnées cliquées.")

        self.target_track_id = best_track.track_id
        # Le type cible suit ce qu'a observé le détecteur ; par défaut BALL
        # (c'est le rôle métier du target dans ce scénario) si le détecteur
        # ne peut pas le confirmer sémantiquement (OpenCV -> UNKNOWN).
        self.target_type = (
            best_track.object_type if best_track.object_type != ObjectType.UNKNOWN else ObjectType.BALL
        )
        best_track.object_type = self.target_type

        cx, cy = _centroid(best_track.last_bbox)
        if frame is not None:
            self.signature = ObjectSignature(frame, best_track.last_bbox, self.tracker.reid)

        self.hidden = HiddenObjectState(
            target_id=self.target_track_id,
            visible=True,
            state=TargetState.VISIBLE,
            last_visible_bbox=best_track.last_bbox,
            last_visible_center=(cx, cy),
            confidence=1.0,
        )
        self._pending_occlusion_frames = 0
        self._frames_since_container_seen = 0
        self._reidentify_candidate = None
        self._reidentify_streak = 0

        # Verrouillage immédiat : la boule sélectionnée devient Cup_A.
        container_id, container_score = self.associate_container(tracks)
        if container_id is not None:
            container = self.tracker.get_track(container_id)
            if container is not None:
                self._lock_container(container, frame=frame, score=container_score)
        return self.target_track_id

    def _container_candidates(self, tracks: List[Track]) -> List[Track]:
        return [
            t for t in tracks
            if t.time_since_update == 0
            and t.track_id != self.target_track_id
            and not (self.target_type == ObjectType.BALL and t.object_type == ObjectType.BALL)
        ]

    def _lock_container(self, track: Track, frame=None, score: float = 1.0) -> None:
        self.hidden.container_id = track.track_id
        self.hidden.container_label = self._container_label
        self.hidden.container_confidence = float(np.clip(score, 0.0, 1.0))
        self._container_last_center = _centroid(track.last_bbox)
        if frame is not None:
            self._container_signature = ObjectSignature(frame, track.last_bbox, self.tracker.reid)

    def _resolve_locked_container(self, tracks: List[Track], frame=None) -> Tuple[Optional[Track], float]:
        """Conserve Cup_A pendant le mélange avec une hystérésis forte."""
        if self.hidden.container_id is None:
            return None, 0.0

        current = self.tracker.get_track(self.hidden.container_id)
        if current is not None and current.time_since_update <= settings.MAX_OCCLUSION_FRAMES:
            self._container_miss_frames = current.time_since_update
            if current.time_since_update == 0:
                self._container_last_center = _centroid(current.last_bbox)
                return current, 1.0
            # Petit raté de détection : garder le même gobelet et extrapoler
            # sa bbox depuis la prédiction Kalman au lieu de basculer vers un
            # autre gobelet.
            px, py = current.predicted_position or _centroid(current.last_bbox)
            lx, ly = _centroid(current.last_bbox)
            predicted_bbox = BoundingBox(
                x=current.last_bbox.x + (px - lx),
                y=current.last_bbox.y + (py - ly),
                width=current.last_bbox.width,
                height=current.last_bbox.height,
            )
            current.last_bbox = predicted_bbox
            self._container_last_center = (px, py)
            return current, max(0.5, 1.0 - current.time_since_update / max(settings.MAX_OCCLUSION_FRAMES, 1))

        self._container_miss_frames += 1
        candidates = self._container_candidates(tracks)
        if not candidates:
            return None, max(0.0, 1.0 - self._container_miss_frames / max(settings.MAX_OCCLUSION_FRAMES, 1))

        px, py = self._container_last_center or (0.0, 0.0)
        best, best_score = None, -1.0
        for candidate in candidates:
            cx, cy = _centroid(candidate.last_bbox)
            dist = float(np.hypot(cx - px, cy - py))
            spatial = max(0.0, 1.0 - dist / 180.0)
            appearance = 0.5
            if frame is not None and self._container_signature is not None:
                appearance = float(self._container_signature.compare(frame, candidate.last_bbox).get("combined", 0.0))
            motion = candidate.motion.motion_consistency()
            score = 0.65 * spatial + 0.25 * appearance + 0.10 * motion
            if score > best_score:
                best_score, best = score, candidate

        # Très forte hystérésis : aucun changement pendant un croisement
        # ambigu. Le même label Cup_A suit éventuellement un nouveau handle.
        if best is not None and best_score >= 0.78:
            self.hidden.container_id = best.track_id
            self._container_rebinds += 1
            self._container_miss_frames = 0
            self._container_last_center = _centroid(best.last_bbox)
            if frame is not None:
                self._container_signature = ObjectSignature(frame, best.last_bbox, self.tracker.reid)
            return best, best_score
        return None, max(0.0, 1.0 - self._container_miss_frames / max(settings.MAX_OCCLUSION_FRAMES, 1))

    # ------------------------------------------------------------------
    # Association boule <-> conteneur (section 6)
    # ------------------------------------------------------------------
    def associate_container(self, tracks: List[Track]) -> Tuple[Optional[int], float]:
        """Associe la boule au gobelet qui la couvre maintenant OU qui va
        la couvrir selon les trajectoires observées.

        C'est important au moment du clic : la boule peut être visible à côté
        du gobelet quelques frames avant le mélange. On ne doit donc pas
        exiger une superposition immédiate.
        """
        if self.hidden.last_visible_center is None:
            return None, 0.0

        lx, ly = self.hidden.last_visible_center
        ball_track = self.tracker.get_track(self.target_track_id)
        bvx, bvy = ball_track.motion.velocity if ball_track is not None else (0.0, 0.0)
        best_id, best_score = None, 0.0

        ball_area = 0.0
        if ball_track is not None:
            ball_area = max(ball_track.last_bbox.width * ball_track.last_bbox.height, 1.0)

        for track in self._container_candidates(tracks):
            x, y, w, h = track.last_bbox.x, track.last_bbox.y, track.last_bbox.width, track.last_bbox.height
            cx, cy = _centroid(track.last_bbox)
            diag = max(float(np.hypot(w, h)), 1.0)

            # En mode UNKNOWN (OpenCV), les blobs de la boule peuvent être
            # dupliqués. Un gobelet candidat doit avoir une empreinte
            # nettement plus grande que la boule, sauf si YOLO confirme
            # explicitement ObjectType.CUP.
            if track.object_type == ObjectType.UNKNOWN and ball_area > 1.0:
                cup_area = max(w * h, 1.0)
                if cup_area < ball_area * 2.0:
                    continue

            # 1) Recouvrement actuel.
            inside = (x <= lx <= x + w) and (y <= ly <= y + h)
            dist_now = float(np.hypot(cx - lx, cy - ly))
            immediate = 1.0 if inside else max(0.0, 1.0 - dist_now / diag)

            # 2) Projection courte à vitesse constante. On cherche le point
            # de rencontre le plus proche sur les 0..120 prochaines frames.
            cvx, cvy = track.motion.velocity
            rvx, rvy = float(cvx - bvx), float(cvy - bvy)
            rx, ry = float(cx - lx), float(cy - ly)
            denom = rvx * rvx + rvy * rvy
            if denom > 1e-6:
                t = max(0.0, min(120.0, -(rx * rvx + ry * rvy) / denom))
            else:
                t = 0.0
            min_dist = float(np.hypot(rx + rvx * t, ry + rvy * t))
            predicted = max(0.0, 1.0 - min_dist / (diag * 1.15))

            score = max(immediate, predicted)
            if score > best_score:
                best_score, best_id = score, track.track_id

        if best_id is not None and best_score >= settings.CONTAINER_ASSOCIATION_THRESHOLD:
            return best_id, best_score
        return None, best_score

    # ------------------------------------------------------------------
    # Vie de l'estimation pendant l'occlusion (section 8)
    # ------------------------------------------------------------------
    def update_hidden_state(self, frame_index: int, tracks: List[Track]) -> Tuple[Optional[BoundingBox], float]:
        """Retourne (bbox estimée de la boule, confiance du conteneur).
        La position est dérivée du conteneur tant que l'association reste
        fiable : estimated_ball_position = container_center + last_offset.
        Ne prétend jamais "voir à travers" le gobelet (section 24)."""
        container, _ = self._resolve_locked_container(tracks)

        if container is None:
            self._frames_since_container_seen += 1
            # Le conteneur lui-même n'est plus suivi : on ne peut plus rien
            # estimer de fiable -> confiance du conteneur chute à 0.
            self.hidden.container_confidence = max(0.0, 1.0 - self._frames_since_container_seen / 10.0)
            return None, self.hidden.container_confidence

        self._frames_since_container_seen = 0
        # Confiance du conteneur : haute tant qu'il est réellement détecté
        # cette frame, réduite s'il est lui-même en occlusion temporaire.
        self.hidden.container_confidence = float(
            np.clip(1.0 - 0.5 * min(container.time_since_update / max(settings.MAX_OCCLUSION_FRAMES, 1), 1.0), 0.0, 1.0)
        )

        ccx, ccy = _centroid(container.last_bbox)
        if self.hidden.last_offset is None:
            self.hidden.last_offset = (0.0, 0.0)
        ox, oy = self.hidden.last_offset
        est_cx, est_cy = ccx + ox, ccy + oy

        bw = self.hidden.last_visible_bbox.width if self.hidden.last_visible_bbox else container.last_bbox.width * 0.5
        bh = self.hidden.last_visible_bbox.height if self.hidden.last_visible_bbox else container.last_bbox.height * 0.5
        estimated_bbox = BoundingBox(x=est_cx - bw / 2.0, y=est_cy - bh / 2.0, width=bw, height=bh)
        return estimated_bbox, self.hidden.container_confidence

    # ------------------------------------------------------------------
    # Ré-identification de la boule réapparue (section 11)
    # ------------------------------------------------------------------
    def reidentify_target(self, frame_index: int, tracks: List[Track], frame=None) -> Optional[Tuple[int, float]]:
        """Le mode Cup-lock ne ré-identifie pas la boule pendant le mélange."""
        return None

    # ------------------------------------------------------------------
    # Boucle principale (section 17)
    # ------------------------------------------------------------------
    def process_frame(self, frame_index: int, tracks: List[Track], frame=None) -> TargetFrameResult:
        return self.update_target(frame_index, tracks, frame)

    def update_target(self, frame_index: int, tracks: List[Track], frame=None) -> TargetFrameResult:
        if self.target_track_id is None:
            raise RuntimeError("Aucun target sélectionné. Appelez select_target d'abord.")

        # IMPORTANT : après sélection, on suit le gobelet, pas la boule.
        # Si aucun gobelet n'était encore identifiable au clic, on retente
        # uniquement l'association boule->gobelet jusqu'à sa première
        # occlusion, puis Cup_A est définitivement verrouillé.
        if self.hidden.container_id is None and ball_track_is_live(tracks, self.target_track_id):
            candidate_id, candidate_score = self.associate_container(tracks)
            if candidate_id is not None:
                candidate = self.tracker.get_track(candidate_id)
                if candidate is not None:
                    self._lock_container(candidate, frame=frame, score=candidate_score)

        container, container_conf = self._resolve_locked_container(tracks, frame=frame)
        ball_track = next((t for t in tracks if t.track_id == self.target_track_id and t.time_since_update == 0), None)
        ball_visible = ball_track is not None
        crossing = bool(container and self.tracker.is_crossing(container.track_id))

        if container is not None:
            self.hidden.container_id = container.track_id
            self.hidden.container_label = self._container_label
            self.hidden.container_confidence = max(self.hidden.container_confidence, container_conf)
            self._container_last_center = _centroid(container.last_bbox)
            self._frames_since_container_seen = 0
            self.hidden.visible = ball_visible
            self.hidden.state = TargetState.CROSSING if crossing else (TargetState.VISIBLE if ball_visible else TargetState.CUP_TRACKING)
            if ball_visible:
                self.hidden.last_visible_bbox = ball_track.last_bbox
                self.hidden.last_visible_center = _centroid(ball_track.last_bbox)
            else:
                self.occlusion_duration += 1
                if self.hidden.hidden_since_frame is None:
                    self.hidden.hidden_since_frame = frame_index
            bbox = ball_track.last_bbox if ball_visible else container.last_bbox
            estimated = not ball_visible
            confidence = (0.98 if not crossing else 0.90) * max(0.0, min(1.0, container_conf or 1.0))
        else:
            self._frames_since_container_seen += 1
            self.hidden.visible = ball_visible
            if ball_visible:
                self._pending_occlusion_frames = 0
                self.hidden.state = TargetState.VISIBLE
                bbox = ball_track.last_bbox
                self.hidden.last_visible_bbox = bbox
                self.hidden.last_visible_center = _centroid(bbox)
                estimated = False
                confidence = 0.75
            else:
                self.occlusion_duration += 1
                # Petit délai anti-faux-positif lorsque la boule est
                # momentanément ratée mais qu'aucun gobelet n'est encore
                # identifiable. On ne fabrique toujours pas de conteneur.
                self._pending_occlusion_frames += 1
                if self._pending_occlusion_frames <= settings.OCCLUSION_PENDING_FRAMES:
                    self.hidden.state = TargetState.OCCLUSION_PENDING
                    confidence = 0.65
                else:
                    self.hidden.state = TargetState.AMBIGUOUS
                    confidence = max(0.0, 0.40 - self._pending_occlusion_frames * 0.01)
                bbox = self.hidden.last_visible_bbox
                estimated = True

        confidence_percent = float(np.clip(confidence * 100.0, 0.0, 100.0))
        level = ConfidenceLevel.HIGH if confidence_percent >= 90 else ConfidenceLevel.MEDIUM if confidence_percent >= 70 else ConfidenceLevel.LOW
        self.hidden.confidence = confidence_percent / 100.0

        return TargetFrameResult(
            frame_index=frame_index, target_id=self.target_track_id, target_type=self.target_type,
            bbox=bbox, state=self.hidden.state, confidence_percent=confidence_percent,
            confidence_level=level, identity_switches_total=self.identity_switches,
            container_id=self.hidden.container_id, estimated=estimated,
            container_bbox=container.last_bbox if container is not None else None,
            container_label=self.hidden.container_label,
            container_confidence=self.hidden.container_confidence,
        )

    # ------------------------------------------------------------------
    def get_target_state(self) -> HiddenObjectState:
        """Accès en lecture à l'état complet du target (section 14),
        utile pour l'API / les tests sans dupliquer la logique."""
        return self.hidden

