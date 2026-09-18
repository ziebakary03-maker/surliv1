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
from app.tracking.uwb_provider import UWBProvider, NullUWBProvider
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


class IdentityManager:
    def __init__(self, tracker: MultiObjectTracker, uwb_provider: Optional[UWBProvider] = None):
        self.tracker = tracker
        self.confidence_engine = ConfidenceEngine()
        # Section 8 : optionnel. Tant qu'aucun UWBProvider réel n'est
        # injecté, NullUWBProvider garantit un comportement identique à
        # avant (is_available() -> False, donc le bloc ajouté dans
        # update_hidden_state() ci-dessous ne s'exécute jamais).
        self.uwb = uwb_provider or NullUWBProvider()

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
        return self.target_track_id

    # ------------------------------------------------------------------
    # Association boule <-> conteneur (section 6)
    # ------------------------------------------------------------------
    def associate_container(self, tracks: List[Track]) -> Tuple[Optional[int], float]:
        """Détermine quel conteneur (gobelet, ou piste UNKNOWN si détecteur
        non sémantique) est le plus vraisemblablement celui sous lequel la
        boule vient de disparaître, en utilisant sa dernière position/bbox
        connue — jamais une position absolue arbitraire de l'écran."""
        if self.hidden.last_visible_center is None:
            return None, 0.0

        lx, ly = self.hidden.last_visible_center
        best_id, best_score = None, 0.0

        for track in tracks:
            if track.track_id == self.target_track_id:
                continue
            if self.target_type == ObjectType.BALL and track.object_type == ObjectType.BALL:
                continue  # un conteneur ne peut pas être une autre boule
            x, y, w, h = track.last_bbox.x, track.last_bbox.y, track.last_bbox.width, track.last_bbox.height
            cx, cy = _centroid(track.last_bbox)
            diag = max(float(np.hypot(w, h)), 1e-3)

            inside = (x <= lx <= x + w) and (y <= ly <= y + h)
            dist = float(np.hypot(cx - lx, cy - ly))
            proximity_score = 1.0 if inside else max(0.0, 1.0 - dist / diag)

            if proximity_score > best_score:
                best_score = proximity_score
                best_id = track.track_id

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
        # --- Section 8/11 : priorité absolue à la position physique UWB
        # quand elle est disponible. On ne touche à rien d'autre : si le
        # matériel n'est pas connecté ou ne renvoie rien cette frame, on
        # retombe exactement sur le chemin caméra existant ci-dessous.
        if self.uwb.is_available():
            uwb_pos = self.uwb.get_position(frame_index)
            if uwb_pos is not None:
                ux, uy, uwb_confidence = uwb_pos
                bw = (
                    self.hidden.last_visible_bbox.width
                    if self.hidden.last_visible_bbox else 20.0
                )
                bh = (
                    self.hidden.last_visible_bbox.height
                    if self.hidden.last_visible_bbox else 20.0
                )
                self.hidden.container_confidence = float(uwb_confidence)
                return (
                    BoundingBox(x=ux - bw / 2.0, y=uy - bh / 2.0, width=bw, height=bh),
                    float(uwb_confidence),
                )

        container = self.tracker.get_track(self.hidden.container_id) if self.hidden.container_id else None

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
        """Évalue les candidats "boule réapparue" parmi les pistes vues
        cette frame et retourne (track_id, identity_score) du meilleur
        candidat s'il dépasse le seuil AMBIGUOUS_THRESHOLD, sinon None.
        Ne réassigne JAMAIS silencieusement vers une autre boule sans
        score explicite."""
        if self.signature is None or frame is None:
            return None

        container = self.tracker.get_track(self.hidden.container_id) if self.hidden.container_id else None
        predicted_position = None
        container_velocity = (0.0, 0.0)
        if container is not None:
            ccx, ccy = _centroid(container.last_bbox)
            ox, oy = self.hidden.last_offset or (0.0, 0.0)
            predicted_position = (ccx + ox, ccy + oy)
            container_velocity = container.motion.velocity
        elif self.hidden.last_visible_center is not None:
            predicted_position = self.hidden.last_visible_center

        best_candidate, best_score = None, 0.0
        for track in tracks:
            if track.track_id == self.target_track_id:
                continue
            if track.time_since_update != 0:
                continue  # on ne considère que des détections réelles cette frame
            if track.object_type == ObjectType.CUP:
                continue  # un gobelet ne peut pas être la boule réapparue
            if track.track_id == self.hidden.container_id and track.hits > 3:
                # Le conteneur lui-même ne redevient pas soudainement la
                # boule après de nombreuses frames de tracking stable.
                continue

            sig_scores = self.signature.compare(frame, track.last_bbox)
            appearance_similarity = sig_scores["combined"]

            motion_similarity = MotionSignature.compare(container_velocity, track.motion.velocity)

            trajectory_similarity = 1.0
            if predicted_position is not None:
                cx, cy = _centroid(track.last_bbox)
                trajectory_similarity = TrajectorySignature.compare(predicted_position, (cx, cy))
                # Filtre spatial (section 11) : un candidat trop loin de la
                # position prédite (dérivée du conteneur suivi ou de la
                # dernière position connue) n'est pas une "boule réapparue"
                # plausible, même si son apparence ressemble un peu — sans
                # ce filtre, un objet totalement sans rapport ailleurs sur
                # l'image peut fausser le score par la seule composante
                # "container_consistency" (constante). On ne considère
                # donc que les candidats à une distance raisonnable.
                if trajectory_similarity < 0.35:
                    continue

            container_consistency = self.hidden.container_confidence

            identity_score = float(np.clip(
                settings.MOTION_WEIGHT * motion_similarity
                + settings.APPEARANCE_WEIGHT * appearance_similarity
                + settings.TRAJECTORY_WEIGHT * trajectory_similarity
                + settings.CONTAINER_WEIGHT * container_consistency,
                0.0, 1.0,
            ))

            if identity_score > best_score:
                best_score = identity_score
                best_candidate = track.track_id

        if best_candidate is not None and best_score >= settings.AMBIGUOUS_THRESHOLD:
            return best_candidate, best_score
        return None

    # ------------------------------------------------------------------
    # Boucle principale (section 17)
    # ------------------------------------------------------------------
    def process_frame(self, frame_index: int, tracks: List[Track], frame=None) -> TargetFrameResult:
        return self.update_target(frame_index, tracks, frame)

    def update_target(self, frame_index: int, tracks: List[Track], frame=None) -> TargetFrameResult:
        if self.target_track_id is None:
            raise RuntimeError("Aucun target sélectionné. Appelez select_target d'abord.")

        target_track = next((t for t in tracks if t.track_id == self.target_track_id), None)
        estimated = False
        crossing = False

        # ================================================================
        # CAS 1 : la boule est directement visible/trackée cette frame
        # ================================================================
        if target_track is not None and target_track.time_since_update == 0:
            self._pending_occlusion_frames = 0
            cx, cy = _centroid(target_track.last_bbox)
            self.hidden.last_visible_bbox = target_track.last_bbox
            self.hidden.last_visible_center = (cx, cy)

            was_hidden = not self.hidden.visible
            self.hidden.visible = True
            self.hidden.container_id = None
            self.hidden.last_offset = None

            if self.signature is not None and frame is not None:
                self.signature.update(frame, target_track.last_bbox)

            crossing = self.tracker.is_crossing(self.target_track_id)
            self.hidden.state = TargetState.CROSSING if crossing else TargetState.VISIBLE
            self.hidden.identity_confidence = 1.0
            self.hidden.container_confidence = 1.0
            if was_hidden:
                self.reidentification_events += 1
            bbox = target_track.last_bbox

        # ================================================================
        # CAS 2 : la boule n'est pas trackée cette frame -> occlusion /
        #          ré-identification / conteneur (sections 7 à 11)
        # ================================================================
        else:
            self.occlusion_duration += 1

            if self.hidden.visible:
                # Transition VISIBLE -> OCCLUSION_PENDING : grâce de
                # quelques frames avant de conclure à une occlusion réelle
                # (évite de sur-réagir à un raté ponctuel du détecteur).
                self._pending_occlusion_frames += 1
                if self._pending_occlusion_frames < settings.OCCLUSION_PENDING_FRAMES:
                    self.hidden.state = TargetState.OCCLUSION_PENDING
                    self.hidden.confidence = max(0.5, self.hidden.confidence - 0.1)
                    bbox = self.hidden.last_visible_bbox
                    estimated = True
                else:
                    self.hidden.visible = False
                    self.hidden.hidden_since_frame = frame_index
                    container_id, container_score = self.associate_container(tracks)
                    if container_id is not None:
                        self.hidden.container_id = container_id
                        self.hidden.container_confidence = container_score
                        container = self.tracker.get_track(container_id)
                        if container is not None and self.hidden.last_visible_center is not None:
                            ccx, ccy = _centroid(container.last_bbox)
                            lx, ly = self.hidden.last_visible_center
                            # last_ball_offset_inside_container (section 8)
                            self.hidden.last_offset = (lx - ccx, ly - ccy)
                        self.hidden.state = TargetState.HIDDEN_UNDER_CUP
                    else:
                        # Pas de conteneur identifiable avec assez de
                        # confiance : on reste honnête plutôt que de deviner.
                        self.hidden.state = TargetState.AMBIGUOUS
                        self.hidden.container_confidence = 0.0
                    bbox = self.hidden.last_visible_bbox
                    estimated = True
            else:
                # Déjà en occlusion : d'abord tenter la ré-identification
                # (la boule est peut-être réapparue ailleurs sur l'image),
                # sinon continuer à suivre le conteneur.
                reid_result = self.reidentify_target(frame_index, tracks, frame)

                if reid_result is not None:
                    candidate_id, score = reid_result
                    if score >= settings.REIDENTIFICATION_THRESHOLD:
                        # Confirmation directe : identité restaurée.
                        if candidate_id != self.target_track_id:
                            self.identity_switches += 1
                        self.target_track_id = candidate_id
                        new_track = self.tracker.get_track(candidate_id)
                        if new_track is not None:
                            new_track.object_type = self.target_type
                            cx, cy = _centroid(new_track.last_bbox)
                            self.hidden.last_visible_bbox = new_track.last_bbox
                            self.hidden.last_visible_center = (cx, cy)
                            if self.signature is not None and frame is not None:
                                self.signature.update(frame, new_track.last_bbox)
                            bbox = new_track.last_bbox
                        else:
                            bbox = self.hidden.last_visible_bbox
                            estimated = True
                        self.hidden.visible = True
                        self.hidden.container_id = None
                        self.hidden.last_offset = None
                        self.hidden.state = TargetState.VISIBLE
                        self.hidden.identity_confidence = score
                        self.reidentification_events += 1
                        self._reidentify_candidate = None
                        self._reidentify_streak = 0
                    else:
                        # Score intermédiaire : demande confirmation sur
                        # plusieurs frames avant d'accepter (section 11).
                        if self._reidentify_candidate == candidate_id:
                            self._reidentify_streak += 1
                        else:
                            self._reidentify_candidate = candidate_id
                            self._reidentify_streak = 1
                        self.hidden.state = TargetState.REIDENTIFYING
                        self.hidden.identity_confidence = score
                        est_bbox, container_conf = self.update_hidden_state(frame_index, tracks)
                        bbox = est_bbox or self.hidden.last_visible_bbox
                        estimated = True
                else:
                    # Aucun candidat de ré-identification : continuer le
                    # suivi du conteneur (sections 8/9).
                    self._reidentify_candidate = None
                    self._reidentify_streak = 0
                    est_bbox, container_conf = self.update_hidden_state(frame_index, tracks)

                    if self.hidden.container_id is not None:
                        crossing = self.tracker.is_crossing(self.hidden.container_id)
                        if container_conf <= 0.05:
                            self.hidden.state = TargetState.AMBIGUOUS
                        elif crossing:
                            self.hidden.state = TargetState.CROSSING
                        else:
                            self.hidden.state = TargetState.CUP_TRACKING
                    else:
                        self.hidden.state = TargetState.AMBIGUOUS

                    bbox = est_bbox if est_bbox is not None else self.hidden.last_visible_bbox
                    estimated = True

            # Occlusion trop longue sans conteneur fiable ni ré-id -> LOST
            # (dernier recours, jamais la première réaction — section 8).
            occlusion_frames_effective = (
                frame_index - self.hidden.hidden_since_frame
                if self.hidden.hidden_since_frame is not None else 0
            )
            if (
                not self.hidden.visible
                and self.hidden.container_id is None
                and occlusion_frames_effective > settings.MAX_OCCLUSION_FRAMES
            ):
                self.hidden.state = TargetState.LOST
                estimated = False
                bbox = None

        # ================================================================
        # Confiance combinée (section 12)
        # ================================================================
        if target_track is not None and target_track.time_since_update == 0:
            trajectory_consistency = 1.0
            if target_track.trajectory_error:
                trajectory_consistency = max(0.0, 1.0 - min(target_track.trajectory_error / 100.0, 1.0))
            inputs = ConfidenceInputs(
                tracking_confidence=target_track.last_confidence,
                motion_consistency=target_track.motion.motion_consistency(),
                trajectory_consistency=trajectory_consistency,
                appearance_similarity=1.0,
                occlusion_frames=0,
                max_occlusion_frames=settings.MAX_OCCLUSION_FRAMES,
                reid_confidence=1.0,
                container_confidence=1.0,
                crossing=crossing,
            )
        else:
            occlusion_frames_effective = (
                frame_index - self.hidden.hidden_since_frame
                if self.hidden.hidden_since_frame is not None else self._pending_occlusion_frames
            )
            inputs = ConfidenceInputs(
                tracking_confidence=0.9 if self.hidden.container_id is not None else 0.4,
                motion_consistency=1.0,
                trajectory_consistency=1.0,
                appearance_similarity=self.hidden.identity_confidence,
                occlusion_frames=max(occlusion_frames_effective, 0),
                max_occlusion_frames=settings.MAX_OCCLUSION_FRAMES,
                reid_confidence=self.hidden.identity_confidence,
                container_confidence=self.hidden.container_confidence,
                crossing=crossing,
            )

        confidence_percent, confidence_level = self.confidence_engine.compute(inputs)

        if self.hidden.state == TargetState.LOST:
            confidence_percent = 0.0
            confidence_level = ConfidenceLevel.LOW
        elif confidence_percent < settings.AMBIGUOUS_THRESHOLD * 100 and self.hidden.state not in (
            TargetState.VISIBLE, TargetState.LOST,
        ):
            # Honnêteté scientifique (section 11/24) : jamais de fausse
            # confiance affichée quand le score retombe trop bas.
            self.hidden.state = TargetState.AMBIGUOUS

        self.hidden.confidence = confidence_percent / 100.0
        if self.hidden.state == TargetState.AMBIGUOUS:
            self.ambiguous_frames += 1

        return TargetFrameResult(
            frame_index=frame_index,
            target_id=self.target_track_id,
            target_type=self.target_type,
            bbox=bbox,
            state=self.hidden.state,
            confidence_percent=confidence_percent,
            confidence_level=confidence_level,
            identity_switches_total=self.identity_switches,
            container_id=self.hidden.container_id,
            estimated=estimated,
        )

    def select_target_manual(self, bbox: BoundingBox, frame) -> int:
        """Sélection manuelle (cadre dessiné par l'utilisateur) : crée
        directement une piste ET sa signature, sans dépendre d'une
        détection automatique préalable (contrairement à select_target)."""
        track = self.tracker.add_manual_track(frame, bbox, ObjectType.BALL)
        self.target_track_id = track.track_id
        self.target_type = ObjectType.BALL

        cx, cy = _centroid(bbox)
        if frame is not None:
            self.signature = ObjectSignature(frame, bbox, self.tracker.reid)

        self.hidden = HiddenObjectState(
            target_id=self.target_track_id,
            visible=True,
            state=TargetState.VISIBLE,
            last_visible_bbox=bbox,
            last_visible_center=(cx, cy),
            confidence=1.0,
        )
        return self.target_track_id

    # ------------------------------------------------------------------
    def get_target_state(self) -> HiddenObjectState:
        """Accès en lecture à l'état complet du target (section 14),
        utile pour l'API / les tests sans dupliquer la logique."""
        return self.hidden
