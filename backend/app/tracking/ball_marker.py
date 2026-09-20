"""
BallMarker
----------
Couche de présentation pure au-dessus d'IdentityManager. Ne contient
AUCUNE logique de décision : l'association boule<->conteneur, la
ré-identification et l'estimation de position restent entièrement
dans IdentityManager.update_target(). Ce module se contente de
projeter chaque TargetFrameResult vers un objet BALL_MARKER_<id>
persistant, conforme au vocabulaire du cahier des charges :

    BALL_01
      |
    BALL_MARKER_01

et JAMAIS :

    CUP_01
      |
    BALL_MARKER_01

Le marker est créé une seule fois (au moment de select_target) puis
seulement mis à jour ensuite. Il n'est jamais recréé quand la boule
change de conteneur ou traverse une occlusion (section 12 du cahier
des charges) : c'est ce que garantit l'usage de dataclasses mutables
plutôt que de reconstruire l'objet à chaque frame.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from app.models.schemas import TargetState
from app.tracking.identity_manager import TargetFrameResult
from app.tracking.tracker import _centroid


@dataclass
class BallMarker:
    marker_id: str            # ex: "BALL_MARKER_01"
    ball_id: int               # target_track_id stable (= "BALL_01" logique)
    position: Optional[Tuple[float, float]] = None   # centre courant, réel ou estimé
    hidden_under: Optional[int] = None                 # container_id si actuellement caché
    confidence: float = 1.0
    is_estimated: bool = False
    state: TargetState = TargetState.VISIBLE

    @classmethod
    def create_for(cls, ball_track_id: int) -> "BallMarker":
        """Appelé une seule fois, juste après IdentityManager.select_target()."""
        return cls(marker_id=f"BALL_MARKER_{ball_track_id:02d}", ball_id=ball_track_id)

    def update_from_result(self, result: TargetFrameResult) -> None:
        """Met à jour la balise à partir du résultat déjà calculé par
        IdentityManager.update_target(). Ne recalcule RIEN par elle-même :
        la balise suit exactement la position que le tracker a déjà
        déterminée pour la boule, jamais une position dérivée
        directement du gobelet (règle fondamentale, sections 3 et 6).
        """
        if result.bbox is not None:
            self.position = _centroid(result.bbox)
        # Si bbox est None (état LOST), on garde volontairement la
        # dernière position connue plutôt que d'afficher la balise
        # "nulle part" — cf. section 7 : la balise doit rester visible.
        self.hidden_under = result.container_id
        self.confidence = result.confidence_percent / 100.0
        self.is_estimated = result.estimated
        self.state = result.state

    def to_dict(self) -> dict:
        """Représentation exposable à l'API / au frontend (JobProgress,
        websocket de progression, etc.)."""
        return {
            "marker_id": self.marker_id,
            "ball_id": self.ball_id,
            "position": (
                {"x": self.position[0], "y": self.position[1]}
                if self.position is not None else None
            ),
            "hidden_under": self.hidden_under,
            "confidence": self.confidence,
            "is_estimated": self.is_estimated,
            "state": self.state.value,
        }
