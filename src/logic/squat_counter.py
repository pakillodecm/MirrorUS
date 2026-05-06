import numpy as np
from typing import Dict, Optional
from src.logic.angles import calculate_angle


class SquatCounter:
    def __init__(
        self,
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
        hysteresis: float = 10.0,
        persistence_frames: int = 10,
    ):
        self.thr_down = down_threshold
        self.thr_up = up_threshold
        self.hysteresis = hysteresis
        self.max_ghost_frames = persistence_frames

        self.count = 0
        self.state = 0  # 0: STAND, 1: DESCENDING, 2: DEEP, 3: ASCENDING
        self.ghost_counter = 0
        self.last_angle = 180.0

    def _calculate_weighted_angle(
        self, landmarks: Dict[str, np.ndarray]
    ) -> Optional[float]:
        """Calcula el ángulo efectivo promediando ambos lados según su visibilidad."""
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

    def update(self, world_landmarks: Optional[Dict[str, np.ndarray]]) -> int:
        """Actualiza la FSM y devuelve el contador de repeticiones."""

        if world_landmarks is None:
            self.ghost_counter += 1
            if self.ghost_counter > self.max_ghost_frames:
                self.state = 0
            return self.count

        angle = self._calculate_weighted_angle(world_landmarks)

        if angle is None:
            return self.count

        self.ghost_counter = 0
        self.last_angle = angle

        state_changed = True
        while state_changed:
            state_changed = False
            old_state = self.state

            if self.state == 0 and angle < (self.thr_up - self.hysteresis):
                self.state = 1
            elif self.state == 1 and angle <= self.thr_down:
                self.state = 2
            elif self.state == 2 and angle > (self.thr_down + self.hysteresis):
                self.state = 3
            elif self.state == 3 and angle >= self.thr_up:
                self.state = 0
                self.count += 1

            if old_state != self.state:
                state_changed = True

        return self.count
