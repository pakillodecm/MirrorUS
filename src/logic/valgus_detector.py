from typing import Dict, Optional, Tuple

import numpy as np


class KneeValgusDetector:
    """Detecta el colapso medial de rodillas comparando anchos en el plano XZ."""

    def __init__(self, threshold: float = 0.90):
        """Inicializa el detector.

        Args:
            threshold: Ratio mínimo rodillas/caderas antes de marcar valgo.
                0.90 implica que las rodillas no deben estar más del 10% más
                juntas que las caderas.
        """
        self.threshold = float(threshold)

    def analyze(
        self, world_landmarks: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[bool, float]:
        """Analiza el ratio de alineación rodillas/caderas en el plano XZ.

        Args:
            world_landmarks: Coordenadas world de MediaPipe, o None.

        Returns:
            Tupla (is_valgus, valgus_ratio). Devuelve (False, 1.0) si no hay
            landmarks o la visibilidad es insuficiente.
        """
        if world_landmarks is None:
            return False, 1.0
        try:
            l_hip = world_landmarks["LEFT_HIP"]
            r_hip = world_landmarks["RIGHT_HIP"]
            l_knee = world_landmarks["LEFT_KNEE"]
            r_knee = world_landmarks["RIGHT_KNEE"]

            if l_hip[3] < 0.5 or r_hip[3] < 0.5 or l_knee[3] < 0.5 or r_knee[3] < 0.5:
                return False, 1.0

            hip_distance = np.sqrt(
                (l_hip[0] - r_hip[0]) ** 2 + (l_hip[2] - r_hip[2]) ** 2
            )
            if hip_distance == 0:
                return False, 1.0

            knee_distance = np.sqrt(
                (l_knee[0] - r_knee[0]) ** 2 + (l_knee[2] - r_knee[2]) ** 2
            )
            valgus_ratio = knee_distance / hip_distance
            is_valgus = bool(valgus_ratio < self.threshold)
            return is_valgus, valgus_ratio

        except KeyError:
            return False, 1.0
