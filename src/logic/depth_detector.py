from typing import Dict, Optional, Tuple

import numpy as np

from src.logic.angles import calculate_angle


class DepthDetector:
    """Evalúa si el atleta rompe el paralelo usando coordenadas world de MediaPipe."""

    def __init__(
        self,
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
        min_combined_visibility: float = 1.2,
    ):
        """Inicializa el detector de profundidad.

        Args:
            down_threshold: Ángulo por debajo del cual se rompe el paralelo.
            up_threshold: Ángulo por encima del cual se considera posición erguida.
            min_combined_visibility: Visibilidad mínima combinada de ambas rodillas.
        """
        self.down_threshold = float(down_threshold)
        self.up_threshold = float(up_threshold)
        self.min_combined_visibility = float(min_combined_visibility)

    def _calculate_weighted_angle(
        self, landmarks: Dict[str, np.ndarray]
    ) -> Optional[float]:
        """Promedia el ángulo de ambas rodillas ponderado por su visibilidad."""
        try:
            l_hip, l_knee, l_ankle = (
                landmarks["LEFT_HIP"],
                landmarks["LEFT_KNEE"],
                landmarks["LEFT_ANKLE"],
            )
            r_hip, r_knee, r_ankle = (
                landmarks["RIGHT_HIP"],
                landmarks["RIGHT_KNEE"],
                landmarks["RIGHT_ANKLE"],
            )
            l_vis, r_vis = l_knee[3], r_knee[3]
            total_vis = l_vis + r_vis
            if total_vis < self.min_combined_visibility:
                return None
            l_angle = calculate_angle(l_hip[:3], l_knee[:3], l_ankle[:3])
            r_angle = calculate_angle(r_hip[:3], r_knee[:3], r_ankle[:3])
            return (l_angle * l_vis + r_angle * r_vis) / total_vis
        except KeyError:
            return None

    def analyze(
        self, world_landmarks: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[bool, float]:
        """Evalúa si el esqueleto actual rompe el paralelo.

        Args:
            world_landmarks: Coordenadas world de MediaPipe, o None.

        Returns:
            Tupla (is_deep, current_angle). Si no hay landmarks devuelve (False, 180.0).
        """
        if world_landmarks is None:
            return False, 180.0
        angle = self._calculate_weighted_angle(world_landmarks)
        if angle is None:
            return False, 180.0

        is_deep = bool(angle <= self.down_threshold)
        return is_deep, angle
