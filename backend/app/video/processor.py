"""
VideoProcessor
--------------
Orchestration du pipeline complet décrit section 17 du spec :

Extraction des frames -> Détection -> Tracking -> Sélection du Target
-> Identity Tracking -> Motion Prediction -> Re-identification
-> Confidence Engine -> Rendering -> Génération MP4

Ce module est appelé par le worker (traitement asynchrone, section 18/26).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import cv2

from app.tracking.detector import build_detector
from app.tracking.tracker import MultiObjectTracker
from app.tracking.identity_manager import IdentityManager
from app.models.schemas import TargetState, BoundingBox
from app.core.config import settings


@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    width: int
    height: int


def open_video(path: str) -> tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {path}")
    meta = VideoMeta(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return cap, meta


def get_frame(path: str, frame_index: int):
    """Utilisé par l'étape de sélection du target (section 3)."""
    cap, meta = open_video(path)
    frame_index = max(0, min(frame_index, meta.total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Impossible de lire la frame {frame_index}")
    return frame, meta


STATE_COLORS = {
    TargetState.VISIBLE: (0, 255, 0),
    TargetState.TRACKING: (0, 200, 0),
    TargetState.CONFIDENT: (0, 255, 0),
    TargetState.FAST_MOVEMENT: (0, 165, 255),
    TargetState.OCCLUDED: (0, 140, 255),
    TargetState.OCCLUSION_PENDING: (0, 165, 255),
    TargetState.HIDDEN_UNDER_CUP: (0, 120, 255),
    TargetState.CUP_TRACKING: (0, 120, 255),
    TargetState.REIDENTIFYING: (0, 100, 255),
    TargetState.CROSSING: (0, 80, 255),
    TargetState.LOST: (0, 0, 255),
    TargetState.AMBIGUOUS: (0, 0, 200),
    TargetState.DETECTED: (200, 200, 0),
}

# États pendant lesquels la boule n'est PAS réellement détectée : la bbox
# dessinée est une estimation dérivée du conteneur suivi (section 8/18/24).
# Ne jamais laisser croire que c'est une vraie détection dans ce cas.
HIDDEN_STATES = {
    TargetState.HIDDEN_UNDER_CUP,
    TargetState.CUP_TRACKING,
    TargetState.OCCLUSION_PENDING,
    TargetState.REIDENTIFYING,
}


def draw_target_overlay(frame, result):
    """Dessine le verrou Cup_A au-dessus du gobelet, jamais au-dessus de la
    boule pendant le mélange. La bbox de la boule reste disponible dans le
    résultat pour les états visibles, mais l'overlay métier utilise
    `container_bbox` dès qu'un gobelet porteur est verrouillé."""
    overlay_bbox = getattr(result, "container_bbox", None) or result.bbox
    if overlay_bbox is None:
        return frame

    color = STATE_COLORS.get(result.state, (255, 255, 255))
    x, y, w, h = int(overlay_bbox.x), int(overlay_bbox.y), int(overlay_bbox.width), int(overlay_bbox.height)
    cx = x + w // 2
    top_y = max(0, y - 8)
    label_id = getattr(result, "container_label", None) or (f"CUP #{result.container_id}" if result.container_id is not None else f"TARGET #{result.target_id}")

    # Flèche verticale : son ancrage est le centre du gobelet porteur.
    pin_tip = (cx, top_y)
    pin_top = (cx, max(8, top_y - 42))
    cv2.line(frame, pin_top, pin_tip, color, 4)
    cv2.circle(frame, pin_top, 10, color, -1)
    cv2.line(frame, (cx - 7, top_y - 10), (cx, top_y), color, 3)
    cv2.line(frame, (cx + 7, top_y - 10), (cx, top_y), color, 3)

    # Le gobelet est la cible visuelle. Même si la boule est cachée,
    # l'indicateur reste attaché au gobelet et non à une position écran fixe.
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    state_label = result.state.value
    if result.container_id is not None:
        state_label = f"{label_id} | {state_label}"
    label = f"{state_label} | {result.confidence_percent:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y = max(th + 8, pin_top[1] - 8)
    cv2.rectangle(frame, (cx - tw // 2 - 6, label_y - th - 7), (cx + tw // 2 + 6, label_y + 5), color, -1)
    cv2.putText(frame, label, (cx - tw // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def _draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash_len=8):
    """Rectangle en pointillés : convention visuelle pour "position estimée,
    pas une détection réelle" (section 18)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_len * 2):
        cv2.line(frame, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
    for y in range(y1, y2, dash_len * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)


ProgressCallback = Callable[[int, int, "TargetFrameResult"], None]


def process_video(
    input_path: str,
    output_path: str,
    click_frame: int,
    click_x: float,
    click_y: float,
    expected_objects: int = 3,
    manual_bbox: Optional[BoundingBox] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    """
    Exécute le pipeline complet sur toute la vidéo et écrit la vidéo
    annotée en sortie. Retourne des métriques (identity switches, etc.)
    utilisées pour les tests et le rapport final (section 30).
    """
    cap, meta = open_video(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, meta.fps, (meta.width, meta.height))

    detector = build_detector(expected_objects=expected_objects)
    tracker = MultiObjectTracker()
    identity = IdentityManager(tracker)

    if not detector.semantic_capable:
        print(
            "[process_video] Mode sémantique ball/cup NON disponible avec ce "
            "détecteur (DETECTOR_BACKEND="
            f"{settings.DETECTOR_BACKEND}) : les objets seront différenciés "
            "uniquement par mouvement/apparence/position, pas par classe "
            "sémantique. Voir README pour activer DETECTOR_BACKEND=yolo avec "
            "un modèle entraîné (BALL_CLASS_ID/CUP_CLASS_ID)."
        )

    frame_index = 0
    target_selected = False
    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        tracks = tracker.step(frame, detections)

        if not target_selected and frame_index >= click_frame:
            if manual_bbox is not None:
                # Sélection manuelle : immédiate, ne dépend jamais du
                # détecteur automatique (section 3, mode sélection libre).
                identity.select_target_manual(manual_bbox, frame=frame)
                target_selected = True
            elif tracks:
                # Le détecteur par soustraction de fond a besoin de quelques
                # frames pour "apprendre" le fond avant de détecter des objets
                # de façon fiable. Si aucune piste n'existe encore exactement à
                # click_frame, on attend silencieusement les premières pistes
                # plutôt que d'échouer.
                identity.select_target(click_x, click_y, tracks, frame=frame)
                target_selected = True

        if target_selected:
            result = identity.process_frame(frame_index, tracks, frame=frame)
            last_result = result
            frame = draw_target_overlay(frame, result)
            if progress_cb:
                progress_cb(frame_index, meta.total_frames, result)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    # Résultat final structuré (section 19). Ne jamais inventer un résultat
    # : si l'identité n'a pas pu être confirmée, final_state=AMBIGUOUS/LOST
    # et final_container_id peut rester None.
    final_state = last_result.state.value if last_result else TargetState.AMBIGUOUS.value
    final_position = identity.compute_final_position(tracks, meta.width) if last_result else None
    # "FOUND" n'est affiché que si l'état final est réellement bon avec une
    # confiance suffisante — jamais pour masquer un AMBIGUOUS/LOST honnête
    # (section 24 du cahier des charges : pas de fausse certitude).
    display_state = (
        "FOUND"
        if last_result is not None
        and last_result.state.value in ("VISIBLE", "CUP_TRACKING", "CONFIDENT")
        and last_result.confidence_percent >= 60
        else final_state
    )
    return {
        "total_frames": meta.total_frames,
        "fps": meta.fps,
        "identity_switches": identity.identity_switches,
        "final_confidence": last_result.confidence_percent if last_result else 0.0,
        "final_state": final_state,
        "display_state": display_state,
        "final_position": final_position.value if final_position else None,
        "target_id": identity.target_track_id,
        "target_type": identity.target_type.value if target_selected else None,
        "final_container_id": last_result.container_id if last_result else None,
        "final_container_label": getattr(last_result, "container_label", None) if last_result else None,
        "container_rebinds": getattr(identity, "_container_rebinds", 0),
        "confidence": last_result.confidence_percent if last_result else 0.0,
        "occlusion_duration": identity.occlusion_duration,
        "ambiguous_frames": identity.ambiguous_frames,
        "reidentification_events": identity.reidentification_events,
        "semantic_mode": detector.semantic_capable,
    }

