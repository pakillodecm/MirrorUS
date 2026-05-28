from typing import Dict, Optional, Tuple

import numpy as np

from src.logic.angles import calculate_angle


class DepthDetector:
    def __init__(self, down_threshold: float = 90.0, up_threshold: float = 160.0):
        """Inicializa el detector puramente geométrico de profundidad.

        Args:
            down_threshold (float): Ángulo por debajo del cual se rompe el paralelo.
            up_threshold (float): Ángulo por encima del cual se considera erguido.
        """
        self.down_threshold = float(down_threshold)
        self.up_threshold = float(up_threshold)

    def _calculate_weighted_angle(
        self, landmarks: Dict[str, np.ndarray]
    ) -> Optional[float]:
        """Calcula el ángulo biomecánico promediando ambos lados según visibilidad."""
        try:
            l_hip, l_knee, l_ankle = (
                landmarks["LEFT_HIP"],
                landmarks["LEFT_KNEE"],
                landmarks["LEFT_ANKLE"],
            )
            l_angle = calculate_angle(l_hip[:3], l_knee[:3], l_ankle[:3])
            l_vis = l_knee[3]

            r_hip, r_knee, r_ankle = (
                landmarks["RIGHT_HIP"],
                landmarks["RIGHT_KNEE"],
                landmarks["RIGHT_ANKLE"],
            )
            r_angle = calculate_angle(r_hip[:3], r_knee[:3], r_ankle[:3])
            r_vis = r_knee[3]

            total_vis = l_vis + r_vis
            if total_vis < 1.2:
                return None

            return (l_angle * l_vis + r_angle * r_vis) / total_vis
        except KeyError:
            return None

    def analyze(
        self, world_landmarks: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[bool, float]:
        """Evalúa si el esqueleto actual rompe el paralelo en metros reales.

        Args:
            world_landmarks (dict): Coordenadas tridimensionales de MediaPipe.

        Returns:
            tuple: (is_deep (bool), current_angle (float))
        """
        if world_landmarks is None:
            return False, 180.0

        angle = self._calculate_weighted_angle(world_landmarks)
        if angle is None:
            return False, 180.0

        is_deep = bool(angle <= self.down_threshold)
        return is_deep, angle
