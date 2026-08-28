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
from app.models.schemas import TargetState


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
    TargetState.TRACKING: (0, 200, 0),
    TargetState.CONFIDENT: (0, 255, 0),
    TargetState.FAST_MOVEMENT: (0, 165, 255),
    TargetState.OCCLUDED: (0, 140, 255),
    TargetState.REIDENTIFYING: (0, 100, 255),
    TargetState.LOST: (0, 0, 255),
    TargetState.AMBIGUOUS: (0, 0, 200),
    TargetState.DETECTED: (200, 200, 0),
}


def draw_target_overlay(frame, result):
    """Dessine l'épingle, le label Target #1 et le score de confiance
    (section 16). L'épingle suit la position calculée par le tracking,
    elle n'est jamais fixée à un endroit constant de l'écran."""
    if result.bbox is None:
        return frame
    color = STATE_COLORS.get(result.state, (255, 255, 255))
    x, y, w, h = int(result.bbox.x), int(result.bbox.y), int(result.bbox.width), int(result.bbox.height)
    cx = x + w // 2
    top_y = max(0, y - 10)

    # Épingle (triangle + tige) au-dessus du target
    pin_tip = (cx, top_y)
    pin_top = (cx, max(0, top_y - 35))
    cv2.line(frame, pin_top, pin_tip, color, 3)
    cv2.circle(frame, pin_top, 9, color, -1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"Target #1 | {result.state.value} | {result.confidence_percent:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(20, pin_top[1] - 10)
    cv2.rectangle(frame, (cx - tw // 2 - 4, label_y - th - 6), (cx + tw // 2 + 4, label_y + 4), color, -1)
    cv2.putText(
        frame, label, (cx - tw // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    return frame


ProgressCallback = Callable[[int, int, "TargetFrameResult"], None]


def process_video(
    input_path: str,
    output_path: str,
    click_frame: int,
    click_x: float,
    click_y: float,
    expected_objects: int = 3,
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
            # Le détecteur par soustraction de fond a besoin de quelques
            # frames pour "apprendre" le fond avant de détecter des objets
            # de façon fiable. Si aucune piste n'existe encore exactement à
            # click_frame, on attend silencieusement les premières pistes
            # plutôt que d'échouer (la position cliquée reste valable tant
            # que les objets n'ont pas eu le temps de beaucoup bouger).
            if tracks:
                identity.select_target(click_x, click_y, tracks)
                target_selected = True

        if target_selected:
            result = identity.process_frame(frame_index, tracks)
            last_result = result
            frame = draw_target_overlay(frame, result)
            if progress_cb:
                progress_cb(frame_index, meta.total_frames, result)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    return {
        "total_frames": meta.total_frames,
        "fps": meta.fps,
        "identity_switches": tracker.identity_switches,
        "final_confidence": last_result.confidence_percent if last_result else None,
        "final_state": last_result.state.value if last_result else None,
    }
