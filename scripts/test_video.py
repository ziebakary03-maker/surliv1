"""
Mode développement (section 31 du spec) : teste le pipeline complet sur
une vidéo locale sans passer par l'API/les jobs.

Usage:
    python scripts/test_video.py --input test.mp4 --frame 0 --x 320 --y 240

Si --x/--y ne sont pas fournis, le script détecte les objets sur la
frame choisie et sélectionne automatiquement le premier détecté (utile
pour un test rapide de bout en bout).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.video.processor import process_video, get_frame  # noqa: E402
from app.tracking.detector import build_detector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Chemin vers la vidéo d'entrée")
    parser.add_argument("--output", default="output/tracked.mp4")
    parser.add_argument("--frame", type=int, default=0, help="Frame où sélectionner le target")
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    x, y = args.x, args.y
    if x is None or y is None:
        frame, _ = get_frame(args.input, args.frame)
        detector = build_detector()
        detections = detector.detect(frame)
        if not detections:
            print("Aucun objet détecté sur cette frame, essayez une autre frame.")
            return
        bbox = detections[0].bbox
        x, y = bbox.x + bbox.width / 2, bbox.y + bbox.height / 2
        print(f"Aucune coordonnée fournie: sélection automatique du 1er objet détecté à ({x:.0f}, {y:.0f})")

    def on_progress(current, total, result):
        if current % 30 == 0:
            print(f"Frame {current}/{total} | state={result.state.value} | confidence={result.confidence_percent:.1f}%")

    metrics = process_video(
        input_path=args.input,
        output_path=args.output,
        click_frame=args.frame,
        click_x=x,
        click_y=y,
        progress_cb=on_progress,
    )
    print("Terminé.")
    print(metrics)
    print(f"Vidéo résultat: {args.output}")


if __name__ == "__main__":
    main()
