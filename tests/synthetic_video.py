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


def make_ball_under_cup_video(path: str, width=640, height=480, n_frames=200, fps=30):
    """Scénario central du spec : une boule qui dérive lentement (un objet
    parfaitement immobile finirait par être absorbé dans le modèle de
    fond d'un détecteur par soustraction de fond, comme n'importe quelle
    caméra de surveillance réelle — elle a donc besoin d'un mouvement
    propre continu, pas d'un simple jitter sur place) pendant qu'un
    gobelet (carré), plus rapide, la rattrape, la recouvre complètement
    pendant plusieurs dizaines de frames, puis la dépasse — la boule
    redevient visible, toujours en train de dériver. Deux autres gobelets
    bougent ailleurs sur l'image sans jamais toucher la boule (bruit
    réaliste, sections 2 et 6)."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    ball_color = (60, 60, 220)
    ball_r = 18
    ball_vx, ball_vy = 1.0, 0.3
    ball_x0, ball_y0 = 100, 290

    cup_size = 120
    cup_speed = 4.0
    cup_over_x0 = -260  # rattrape et dépasse la boule autour de la frame 100 (cf. calcul en commentaire)
    cup_colors = [(140, 140, 140), (150, 150, 150), (130, 130, 130)]

    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)

        ball_x = int(ball_x0 + ball_vx * i)
        ball_y = int(ball_y0 + ball_vy * i)

        # Boule dessinée en premier : le gobelet passé PAR DESSUS la
        # masque complètement, exactement comme dans le scénario réel.
        cv2.circle(frame, (ball_x, ball_y), ball_r, ball_color, -1)

        cup_over_x = int(cup_over_x0 + cup_speed * i)
        _draw_cup(frame, cup_over_x, ball_y0 - cup_size // 2, cup_size, cup_colors[0])

        # Deux gobelets "décor", loin de la trajectoire de la boule.
        cup2_x = int(width + cup_size - cup_speed * i)
        _draw_cup(frame, cup2_x, 40, cup_size, cup_colors[1])

        cup3_x = int(420 + 50 * np.sin(i / 15))
        _draw_cup(frame, cup3_x, height - cup_size - 20, cup_size, cup_colors[2])

        writer.write(frame)
    writer.release()


def make_ball_under_cup_with_crossing_video(path: str, width=640, height=480, n_frames=200, fps=30):
    """Variante section 10 : pendant que la boule est cachée sous le
    gobelet A, un second gobelet B (taille/couleur quasi identique)
    traverse son chemin exactement pendant la fenêtre d'occlusion.
    L'identité du conteneur doit rester stable pendant et après le
    croisement (pas de changement d'ID au simple chevauchement)."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    ball_color = (60, 60, 220)
    ball_r = 18
    ball_vx, ball_vy = 1.0, 0.3
    ball_x0, ball_y0 = 100, 290

    cup_size = 120
    cup_a_speed = 4.0
    cup_a_x0 = -260  # identique à make_ball_under_cup_video : cache la boule vers i≈100

    # cup_b : croise cup_a exactement à i=100 (centre de la fenêtre
    # d'occlusion), à la même hauteur.
    cross_frame = 100
    cup_a_center_at_cross = cup_a_x0 + cup_a_speed * cross_frame + cup_size / 2
    cup_b_speed = 6.0
    cup_b_x0 = cup_a_center_at_cross - cup_size / 2 + cup_b_speed * cross_frame

    # Couleurs volontairement très proches (gobelets "quasi identiques").
    cup_colors = [(140, 140, 140), (144, 144, 144)]

    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        ball_x = int(ball_x0 + ball_vx * i)
        ball_y = int(ball_y0 + ball_vy * i)
        cv2.circle(frame, (ball_x, ball_y), ball_r, ball_color, -1)

        cup_a_x = int(cup_a_x0 + cup_a_speed * i)
        cup_a_y = ball_y0 - cup_size // 2

        cup_b_x = int(cup_b_x0 - cup_b_speed * i)
        cup_b_y = cup_a_y

        # Le gobelet dessiné en dernier "gagne" visuellement le chevauchement
        # sans que cela n'ait de sens physique particulier : c'est le point
        # justement testé (le tracker ne doit pas se fier au chevauchement
        # visuel seul pour décider de l'identité).
        _draw_cup(frame, cup_a_x, cup_a_y, cup_size, cup_colors[0])
        _draw_cup(frame, cup_b_x, cup_b_y, cup_size, cup_colors[1])

        writer.write(frame)
    writer.release()


def make_ball_disappears_no_container_video(path: str, width=640, height=480, n_frames=90, fps=30):
    """Section 11/12 : la boule disparaît brutalement (sort du champ, par
    exemple) loin de tout gobelet. Aucun conteneur plausible ne doit être
    trouvé : le système doit reconnaître qu'il ne sait pas plutôt que
    d'inventer une association (état attendu : AMBIGUOUS)."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    ball_color = (60, 60, 220)
    cup_size = 100
    cup_color = (140, 140, 140)

    disappear_at = n_frames // 2
    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        if i < disappear_at:
            bx = int(100 + 1.0 * i)
            by = int(240 + 0.3 * i)
            cv2.circle(frame, (bx, by), 18, ball_color, -1)
        # Gobelet loin de la position de disparition de la boule, mais en
        # mouvement pour rester détectable par la soustraction de fond.
        cup_x = int(480 + 30 * np.sin(i / 10.0))
        _draw_cup(frame, cup_x, height - cup_size - 20, cup_size, cup_color)
        writer.write(frame)
    writer.release()


def _draw_cup(frame, x, y, size, color):
    cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)


if __name__ == "__main__":
    import os
    os.makedirs("test_assets", exist_ok=True)
    make_crossing_video("test_assets/crossing.mp4")
    make_occlusion_video("test_assets/occlusion.mp4")
    make_fast_movement_video("test_assets/fast_movement.mp4")
    make_ball_under_cup_video("test_assets/ball_under_cup.mp4")
    make_ball_under_cup_with_crossing_video("test_assets/ball_under_cup_crossing.mp4")
    make_ball_disappears_no_container_video("test_assets/ball_disappears_no_container.mp4")
    print("Vidéos synthétiques générées dans test_assets/")
