"""
Tests du moteur "boule cachée sous un gobelet" (section 21 du cahier des
charges). Complète tests/test_tracker.py (qui couvre le tracker générique)
sans le dupliquer : ici on exerce IdentityManager de bout en bout sur des
vidéos synthétiques dédiées (tests/synthetic_video.py).

Lancement:
    cd backend && pytest ../tests/test_ball_cup_identity.py -v
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
import cv2

from tests.synthetic_video import (
    make_ball_under_cup_video,
    make_ball_under_cup_with_crossing_video,
    make_ball_disappears_no_container_video,
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "_assets")

# Le détecteur OpenCV a besoin de quelques frames pour que le modèle de
# fond (MOG2) converge ; avant ça, les toutes premières frames peuvent
# produire un unique gros blob confus. On attend cette période avant de
# sélectionner le target, comme le ferait un utilisateur réel qui regarde
# la vidéo quelques instants avant de cliquer sur la boule.
WARMUP_FRAMES = 20


@pytest.fixture(scope="module", autouse=True)
def synthetic_videos():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    make_ball_under_cup_video(f"{ASSETS_DIR}/ball_under_cup.mp4")
    make_ball_under_cup_with_crossing_video(f"{ASSETS_DIR}/ball_under_cup_crossing.mp4")
    make_ball_disappears_no_container_video(f"{ASSETS_DIR}/ball_disappears_no_container.mp4")
    yield


def _run_pipeline(video_path, click_frame=WARMUP_FRAMES, click_xy=(150, 300), max_frames=None):
    """Fait tourner detector -> tracker -> identity_manager frame par frame
    et retourne la liste des TargetFrameResult, comme le ferait
    video/processor.py, mais sans écrire de fichier de sortie (plus rapide
    à exécuter dans les tests)."""
    from app.tracking.detector import OpenCVDetector
    from app.tracking.tracker import MultiObjectTracker
    from app.tracking.identity_manager import IdentityManager

    detector = OpenCVDetector()
    tracker = MultiObjectTracker()
    identity = IdentityManager(tracker)

    cap = cv2.VideoCapture(video_path)
    results = []
    frame_idx = 0
    target_selected = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        detections = detector.detect(frame)
        tracks = tracker.step(frame, detections)

        if not target_selected and frame_idx >= click_frame and tracks:
            identity.select_target(click_xy[0], click_xy[1], tracks, frame=frame)
            target_selected = True

        if target_selected:
            results.append(identity.process_frame(frame_idx, tracks, frame=frame))
        frame_idx += 1
    cap.release()
    assert target_selected, "Le target n'a jamais pu être sélectionné (aucune piste détectée)"
    return results, identity


# ---------------------------------------------------------------------------
# TEST 1 : boule visible pendant toute la vidéo (pas d'occlusion)
# ---------------------------------------------------------------------------

def test_ball_always_visible_stays_visible():
    """Sur les toutes premières frames (bien avant que le gobelet mobile
    n'atteigne la boule, vers la frame ~85), l'état ne doit jamais dégénérer
    vers LOST/AMBIGUOUS : au pire un bref sursaut transitoire dû au bruit
    du détecteur (OCCLUSION_PENDING), jamais un abandon complet — et la
    frame finale de cette fenêtre doit être VISIBLE avec une confiance
    correcte."""
    from app.models.schemas import TargetState

    results, identity = _run_pipeline(
        f"{ASSETS_DIR}/ball_under_cup.mp4", click_frame=WARMUP_FRAMES, max_frames=40,
    )
    assert results, "Le pipeline doit produire des résultats dès la sélection du target"
    assert all(r.state not in (TargetState.LOST, TargetState.AMBIGUOUS) for r in results)
    assert results[-1].state in (TargetState.VISIBLE, TargetState.CROSSING)
    assert results[-1].confidence_percent >= 70


# ---------------------------------------------------------------------------
# TEST 2 : la boule disparaît sous un gobelet -> HIDDEN_UNDER_CUP / CUP_TRACKING
# ---------------------------------------------------------------------------

def test_ball_hidden_under_cup_is_not_immediately_lost():
    from app.models.schemas import TargetState

    results, identity = _run_pipeline(f"{ASSETS_DIR}/ball_under_cup.mp4")

    hidden_states = {
        TargetState.HIDDEN_UNDER_CUP, TargetState.CUP_TRACKING,
        TargetState.OCCLUSION_PENDING, TargetState.CROSSING,
    }
    was_hidden = any(r.state in hidden_states for r in results)
    assert was_hidden, "La boule doit passer par un état d'occlusion suivie pendant que le gobelet passe dessus"

    # Contrainte centrale du spec (section 8) : ne JAMAIS sauter directement
    # à LOST à la première disparition.
    first_hidden_index = next(i for i, r in enumerate(results) if r.state in hidden_states)
    assert results[first_hidden_index].state != TargetState.LOST

    # Un conteneur doit avoir été identifié pendant l'occlusion.
    assert any(r.container_id is not None for r in results)


def test_hidden_ball_position_is_estimated_and_flagged():
    """Section 18/24 : pendant l'occlusion, la position affichée doit être
    marquée comme estimée, jamais présentée comme une vraie détection."""
    from app.models.schemas import TargetState

    results, identity = _run_pipeline(f"{ASSETS_DIR}/ball_under_cup.mp4")
    hidden_results = [
        r for r in results
        if r.state in (TargetState.HIDDEN_UNDER_CUP, TargetState.CUP_TRACKING)
    ]
    assert hidden_results, "La vidéo doit produire au moins quelques frames en occlusion suivie"
    assert all(r.estimated for r in hidden_results)
    assert all(r.bbox is not None for r in hidden_results)


# ---------------------------------------------------------------------------
# TEST : la boule réapparaît -> ré-identification, retour à VISIBLE
# ---------------------------------------------------------------------------

def test_ball_reappears_and_is_reidentified():
    from app.models.schemas import TargetState

    results, identity = _run_pipeline(f"{ASSETS_DIR}/ball_under_cup.mp4")

    # La vidéo se termine avec le gobelet loin de la boule : elle doit être
    # redevenue visible avec une confiance restaurée.
    tail = results[-15:]
    visible_tail = [r for r in tail if r.state == TargetState.VISIBLE]
    assert visible_tail, "La boule doit être redevenue VISIBLE en fin de vidéo"
    assert visible_tail[-1].confidence_percent >= 60

    # La ré-identification a bien été comptabilisée (au moins une fois :
    # sortie d'occlusion -> VISIBLE).
    assert identity.reidentification_events >= 1


# ---------------------------------------------------------------------------
# TEST : croisement de gobelets pendant l'occlusion -> pas de perte d'identité
# ---------------------------------------------------------------------------

def test_container_identity_stable_through_crossing():
    """Section 10 : un second gobelet quasi identique croise le premier
    pendant que la boule est cachée dessous. Le conteneur associé ne doit
    pas changer de façon incohérente à cause du seul chevauchement visuel."""
    from app.models.schemas import TargetState

    results, identity = _run_pipeline(f"{ASSETS_DIR}/ball_under_cup_crossing.mp4")

    container_ids_while_hidden = [
        r.container_id for r in results
        if r.state in (TargetState.HIDDEN_UNDER_CUP, TargetState.CUP_TRACKING, TargetState.CROSSING)
        and r.container_id is not None
    ]
    assert container_ids_while_hidden, "Un conteneur doit être suivi pendant l'occlusion malgré le croisement"

    # Le conteneur ne doit pas changer à chaque frame : une valeur
    # dominante doit représenter la majorité des frames cachées (tolérance
    # pour une éventuelle re-association légitime après un croisement long).
    counts = Counter(container_ids_while_hidden)
    _, most_common_count = counts.most_common(1)[0]
    assert most_common_count / len(container_ids_while_hidden) >= 0.5

    # Pas de changement d'identité "gratuit" en cascade (section 12 : ne
    # jamais réassigner en silence). Une poignée d'événements légitimes de
    # ré-identification est acceptable, une explosion ne l'est pas.
    assert identity.identity_switches <= 5


# ---------------------------------------------------------------------------
# TEST : disparition sans conteneur plausible -> AMBIGUOUS, jamais une fausse
# certitude
# ---------------------------------------------------------------------------

def test_ball_disappears_without_container_becomes_ambiguous():
    from app.models.schemas import TargetState

    results, identity = _run_pipeline(
        f"{ASSETS_DIR}/ball_disappears_no_container.mp4",
        click_frame=15, click_xy=(150, 240),
    )
    assert results

    tail = results[-15:]
    # Aucun conteneur plausible dans cette vidéo : le système ne doit
    # jamais prétendre suivre un gobelet contenant la boule.
    assert all(r.container_id is None for r in tail)
    # L'état final doit refléter l'incertitude plutôt qu'une fausse
    # confiance (AMBIGUOUS ou LOST en dernier recours), jamais VISIBLE.
    assert tail[-1].state in (TargetState.AMBIGUOUS, TargetState.LOST)
    assert tail[-1].confidence_percent < 70


# ---------------------------------------------------------------------------
# TEST : le résultat final structuré (section 19) ne contient jamais
# d'invention
# ---------------------------------------------------------------------------

def test_final_result_never_invents_a_container():
    from app.models.schemas import TargetState

    results, identity = _run_pipeline(
        f"{ASSETS_DIR}/ball_disappears_no_container.mp4",
        click_frame=15, click_xy=(150, 240),
    )
    last = results[-1]
    if last.state == TargetState.AMBIGUOUS:
        assert last.container_id is None


# ---------------------------------------------------------------------------
# TEST : select_target choisit l'objet le plus proche du clic, pas "le
# premier" ou "le plus à gauche" (section 3)
# ---------------------------------------------------------------------------

def test_select_target_uses_click_proximity():
    from app.tracking.detector import OpenCVDetector
    from app.tracking.tracker import MultiObjectTracker, _centroid
    from app.tracking.identity_manager import IdentityManager

    detector = OpenCVDetector()
    tracker = MultiObjectTracker()
    identity = IdentityManager(tracker)

    cap = cv2.VideoCapture(f"{ASSETS_DIR}/ball_under_cup.mp4")
    tracks = []
    frame = None
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        assert ok
        detections = detector.detect(frame)
        tracks = tracker.step(frame, detections)
    cap.release()

    assert tracks, "Le détecteur doit avoir trouvé au moins un objet après la période de chauffe"

    # Clique tout près de la boule (~150, ~300 avec un léger jitter).
    target_id = identity.select_target(150, 300, tracks, frame=frame)
    chosen = next(t for t in tracks if t.track_id == target_id)
    cx, cy = _centroid(chosen.last_bbox)
    assert ((cx - 150) ** 2 + (cy - 300) ** 2) ** 0.5 < 60
