from typing import Dict, Optional, Tuple

import numpy as np


class KneeValgusDetector:
    """Detecta el colapso medial de rodillas midiendo la desviación perpendicular
    de cada rodilla respecto a la línea cadera-tobillo en el plano frontal (XY)."""

    def __init__(self, threshold: float = 0.08):
        """Inicializa el detector.

        Args:
            threshold: Desviación medial máxima permitida, normalizada por la
                longitud del segmento cadera-tobillo. 0.08 equivale al 8%.
        """
        self.threshold = float(threshold)

    def _perpendicular_deviation(
        self,
        hip: np.ndarray,
        knee: np.ndarray,
        ankle: np.ndarray,
        sign: float,
    ) -> Optional[float]:
        """Desviación perpendicular normalizada de la rodilla respecto a la
        línea cadera-tobillo en XY. Positiva = colapso medial (valgo)."""
        h, k, a = hip[:2], knee[:2], ankle[:2]
        ha = a - h
        ha_len = np.linalg.norm(ha)
        if ha_len < 0.1:
            return None
        hk = k - h
        cross = float(ha[0] * hk[1] - ha[1] * hk[0])
        return sign * cross / ha_len

    def analyze(
        self, world_landmarks: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[bool, float]:
        """Evalúa si hay colapso medial de rodillas.

        Args:
            world_landmarks: Coordenadas world de MediaPipe, o None.

        Returns:
            Tupla (is_valgus, avg_dev). avg_dev es la desviación medial media
            ponderada por visibilidad; positiva = colapso medial.
            Devuelve (False, 0.0) si no hay landmarks o visibilidad insuficiente.
        """
        if world_landmarks is None:
            return False, 0.0
        try:
            l_hip = world_landmarks["LEFT_HIP"]
            l_knee = world_landmarks["LEFT_KNEE"]
            l_ankle = world_landmarks["LEFT_ANKLE"]
            r_hip = world_landmarks["RIGHT_HIP"]
            r_knee = world_landmarks["RIGHT_KNEE"]
            r_ankle = world_landmarks["RIGHT_ANKLE"]

            if any(
                lm[3] < 0.5 for lm in [l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
            ):
                return False, 0.0

            l_dev = self._perpendicular_deviation(l_hip, l_knee, l_ankle, sign=1.0)
            r_dev = self._perpendicular_deviation(r_hip, r_knee, r_ankle, sign=-1.0)

            if l_dev is None or r_dev is None:
                return False, 0.0

            l_vis = float(l_knee[3])
            r_vis = float(r_knee[3])
            avg_dev = float((l_dev * l_vis + r_dev * r_vis) / (l_vis + r_vis))

            is_valgus = bool(avg_dev > self.threshold)
            return is_valgus, avg_dev

        except KeyError:
            return False, 0.0
