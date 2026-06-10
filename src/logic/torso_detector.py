from typing import Dict, Optional, Tuple

import numpy as np


class TorsoTiltDetector:
    """Evalúa el ángulo del torso respecto a la vertical usando coordenadas world."""

    def __init__(self, max_tilt_deg: float = 40.0):
        """Inicializa el detector.

        Args:
            max_tilt_deg: Ángulo límite en grados antes de marcar fallo.
        """
        self.max_tilt_deg = float(max_tilt_deg)

    def analyze(
        self, world_landmarks: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[bool, float]:
        """Evalúa si el torso se inclina excesivamente hacia adelante.

        Args:
            world_landmarks: Coordenadas world de MediaPipe, o None.

        Returns:
            Tupla (has_error, angle_deg). Devuelve (False, 0.0) si no hay
            landmarks o la visibilidad es insuficiente.
        """
        if world_landmarks is None:
            return False, 0.0
        try:
            l_hip = world_landmarks["LEFT_HIP"]
            r_hip = world_landmarks["RIGHT_HIP"]
            l_shoulder = world_landmarks["LEFT_SHOULDER"]
            r_shoulder = world_landmarks["RIGHT_SHOULDER"]

            if (
                l_hip[3] < 0.5
                or r_hip[3] < 0.5
                or l_shoulder[3] < 0.5
                or r_shoulder[3] < 0.5
            ):
                return False, 0.0

            mid_hip = (l_hip[:3] + r_hip[:3]) / 2.0
            mid_shoulder = (l_shoulder[:3] + r_shoulder[:3]) / 2.0
            torso_vector = mid_shoulder - mid_hip

            vertical_vector = np.array([0.0, -1.0, 0.0])
            dot_product = np.dot(torso_vector, vertical_vector)
            norm_torso = np.linalg.norm(torso_vector)
            if norm_torso == 0:
                return False, 0.0

            cos_theta = dot_product / norm_torso
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            angle_deg = float(np.degrees(angle_rad))

            has_error = bool(angle_deg > self.max_tilt_deg)
            return has_error, angle_deg

        except KeyError:
            return False, 0.0
