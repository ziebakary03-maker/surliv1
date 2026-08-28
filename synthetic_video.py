"""
Génère des vidéos synthétiques pour tester les scénarios critiques du
spec sans dépendre d'une vraie vidéo : croisement, occlusion, mouvement
rapide (section 30). Chaque objet est un cercle de couleur légèrement
différente (pour permettre au ReIdentifier de les différencier), mais
de taille identique, ce qui reproduit fidèlement "trois objets
quasi identiques" tout en restant testable de façon déterministe.
"""
import cv2
import numpy as np


def make_crossing_video(path: str, width=640, height=480, n_frames=90, fps=30):
    """Trois objets qui se croisent au centre de l'écran."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = [(60, 60, 220), (60, 200, 60), (220, 160, 60)]  # BGR, légèrement différents
    starts = [(50, 240), (590, 240), (320, 50)]
    ends = [(590, 240), (50, 240), (320, 430)]

    for i in range(n_frames):
        t = i / (n_frames - 1)
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        for (sx, sy), (ex, ey), color in zip(starts, ends, colors):
            x = int(sx + (ex - sx) * t)
            y = int(sy + (ey - sy) * t)
            cv2.circle(frame, (x, y), 22, color, -1)
        writer.write(frame)
    writer.release()


def make_occlusion_video(path: str, width=640, height=480, n_frames=90, fps=30):
    """Un objet passe derrière un obstacle opaque pendant plusieurs frames."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = [(60, 60, 220), (60, 200, 60), (220, 160, 60)]
    obstacle = (280, 150, 80, 200)  # x, y, w, h

    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        x0 = int(40 + i * (560 / n_frames))
        positions = [(x0, 240), (int(600 - i * 2), 120), (int(300 + 100 * np.sin(i / 10)), 380)]
        for (x, y), color in zip(positions, colors):
            cv2.circle(frame, (x, y), 22, color, -1)
        ox, oy, ow, oh = obstacle
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (10, 10, 10), -1)
        writer.write(frame)
    writer.release()


def make_fast_movement_video(path: str, width=640, height=480, n_frames=60, fps=30):
    """Un objet se déplace très rapidement d'un bord à l'autre en quelques frames."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = [(60, 60, 220), (60, 200, 60), (220, 160, 60)]

    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        # Mouvement rapide en aller-retour (triangle wave), pas de saut brutal
        period = width - 44
        phase = (i * 45) % (2 * period)
        fast_x = 22 + (phase if phase <= period else 2 * period - phase)
        positions = [(fast_x, 100), (200, 240), (450, 380)]
        for (x, y), color in zip(positions, colors):
            cv2.circle(frame, (x, y), 22, color, -1)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    import os
    os.makedirs("test_assets", exist_ok=True)
    make_crossing_video("test_assets/crossing.mp4")
    make_occlusion_video("test_assets/occlusion.mp4")
    make_fast_movement_video("test_assets/fast_movement.mp4")
    print("Vidéos synthétiques générées dans test_assets/")
