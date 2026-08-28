"""
MotionPredictor
---------------
Filtre de Kalman par objet : conserve position, vitesse et estime
l'accélération, et prédit la position probable à la frame suivante.
C'est ce qui permet de garder une identité stable pendant les
mouvements rapides (section 7/8 du spec) et pendant les croisements
(section 10), en donnant au tracker une estimation de "où l'objet
devrait être" même quand la détection brute est bruitée ou absente
une frame.

État du filtre : [x, y, vx, vy]
Mesure : [x, y] (centre de la bbox)
"""
import numpy as np


class _SimpleKalman:
    """
    Petit filtre de Kalman linéaire (état [x, y, vx, vy], mesure [x, y]),
    équivalent fonctionnel à filterpy.KalmanFilter mais sans dépendance
    externe supplémentaire — tout le reste du projet dépend déjà de
    numpy/scipy. Garde exactement la même interface (F, H, Q, R, P, x,
    predict(), update()) pour rester facilement remplaçable par filterpy
    si besoin.
    """

    def __init__(self, dt: float):
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        self.R = np.eye(2) * 5.0
        self.P = np.eye(4) * 100.0
        # Bruit de process plus élevé sur la vitesse: permet au filtre de
        # s'adapter rapidement à un changement brutal de direction (rebond,
        # freinage soudain) plutôt que de rester "verrouillé" sur l'ancienne
        # vélocité et de perdre la correspondance avec la détection suivante.
        self.Q = np.diag([0.5, 0.5, 8.0, 8.0])
        self.x = np.zeros(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class MotionPredictor:
    def __init__(self, x: float, y: float, dt: float = 1.0):
        self.dt = dt
        kf = _SimpleKalman(dt)
        kf.x = np.array([x, y, 0.0, 0.0])
        self.kf = kf

        # historique borné pour trajectoire / vitesse / accélération
        self.history = [(x, y)]
        self.velocity_history = [(0.0, 0.0)]
        self.max_history = 90

    def predict(self):
        """Avance le filtre d'un pas de temps et retourne (x, y) prédits."""
        self.kf.predict()
        x, y = self.kf.x[0], self.kf.x[1]
        return float(x), float(y)

    def update(self, x: float, y: float):
        """Corrige le filtre avec une mesure réelle (détection)."""
        self.kf.update(np.array([x, y]))

        self.history.append((float(x), float(y)))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        vx, vy = float(self.kf.x[2]), float(self.kf.x[3])
        self.velocity_history.append((vx, vy))
        if len(self.velocity_history) > self.max_history:
            self.velocity_history.pop(0)

    @property
    def position(self):
        return float(self.kf.x[0]), float(self.kf.x[1])

    @property
    def velocity(self):
        return float(self.kf.x[2]), float(self.kf.x[3])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def acceleration(self):
        """Estimation grossière de l'accélération sur les 2 dernières mesures de vitesse."""
        if len(self.velocity_history) < 2:
            return 0.0, 0.0
        (vx0, vy0), (vx1, vy1) = self.velocity_history[-2], self.velocity_history[-1]
        return (vx1 - vx0) / self.dt, (vy1 - vy0) / self.dt

    def direction_degrees(self) -> float:
        vx, vy = self.velocity
        if vx == 0 and vy == 0:
            return 0.0
        return float(np.degrees(np.arctan2(vy, vx)))

    def motion_consistency(self) -> float:
        """
        Score 0..1 mesurant la régularité de la trajectoire récente
        (utilisé par le Confidence Engine). Une trajectoire qui change
        brutalement de direction obtient un score plus bas.
        """
        if len(self.history) < 3:
            return 1.0
        pts = np.array(self.history[-10:])
        deltas = np.diff(pts, axis=0)
        norms = np.linalg.norm(deltas, axis=1)
        norms = norms[norms > 1e-3]
        if len(norms) < 2:
            return 1.0
        dirs = deltas[np.linalg.norm(deltas, axis=1) > 1e-3]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        cos_sims = np.sum(dirs[:-1] * dirs[1:], axis=1)
        # cos_sims proche de 1 = direction stable
        return float(np.clip((np.mean(cos_sims) + 1) / 2, 0.0, 1.0))
