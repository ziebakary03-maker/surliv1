"""
ConfidenceEngine
----------------
Combine plusieurs signaux en un score de confiance 0-100% pour le
target suivi (section 14 du spec) :

  tracking_confidence   : confiance brute de la détection assignée
  motion_consistency     : régularité de la trajectoire (MotionPredictor)
  trajectory_consistency : écart entre position prédite et position mesurée
  appearance_similarity  : similarité d'apparence avec la signature enregistrée
  occlusion_penalty      : pénalité croissante avec la durée d'occlusion
  reid_confidence        : confiance du dernier ré-appariement après occlusion

Le score ne doit JAMAIS remonter artificiellement à 100% après une
occlusion longue sans preuve : c'est la garantie "honnêteté
scientifique" du spec (section 11) — si l'information manque, la
confiance reste basse et l'état passe à AMBIGUOUS plutôt que de
deviner silencieusement.
"""
from dataclasses import dataclass
from app.models.schemas import ConfidenceLevel


@dataclass
class ConfidenceInputs:
    tracking_confidence: float       # 0..1, de la détection
    motion_consistency: float        # 0..1
    trajectory_consistency: float    # 0..1 (1 = position prédite ~= mesurée)
    appearance_similarity: float     # 0..1
    occlusion_frames: int            # nombre de frames consécutives sans détection
    max_occlusion_frames: int
    reid_confidence: float = 1.0     # 0..1, 1.0 si pas de ré-id nécessaire


class ConfidenceEngine:
    WEIGHTS = {
        "tracking_confidence": 0.25,
        "motion_consistency": 0.20,
        "trajectory_consistency": 0.20,
        "appearance_similarity": 0.20,
        "reid_confidence": 0.15,
    }

    def compute(self, inputs: ConfidenceInputs) -> tuple[float, ConfidenceLevel]:
        occlusion_ratio = min(inputs.occlusion_frames / max(inputs.max_occlusion_frames, 1), 1.0)
        # La pénalité d'occlusion réduit chaque composant proportionnellement
        # à la durée d'occlusion : plus longtemps l'objet est caché, moins
        # on peut être sûr de son identité au retour.
        occlusion_factor = 1.0 - 0.85 * occlusion_ratio

        score = (
            self.WEIGHTS["tracking_confidence"] * inputs.tracking_confidence
            + self.WEIGHTS["motion_consistency"] * inputs.motion_consistency
            + self.WEIGHTS["trajectory_consistency"] * inputs.trajectory_consistency
            + self.WEIGHTS["appearance_similarity"] * inputs.appearance_similarity
            + self.WEIGHTS["reid_confidence"] * inputs.reid_confidence
        ) * occlusion_factor

        percent = float(max(0.0, min(100.0, score * 100)))

        if percent >= 90:
            level = ConfidenceLevel.HIGH
        elif percent >= 70:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        return percent, level
