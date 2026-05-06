import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, NamedTuple, Optional


class PoseResult(NamedTuple):
    """Estructura de datos para los resultados de la detección."""

    normalized: Optional[Dict[str, np.ndarray]]
    world: Optional[Dict[str, np.ndarray]]


class PoseDetector:
    def __init__(
        self,
        static_image_mode: bool = False,  # Por defecto para video
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,  # 0=Rápido, 1=Equilibrado, 2=Pesado
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_landmarks(self, frame: np.ndarray) -> PoseResult:
        """
        Procesa un frame y devuelve un objeto PoseResult con:
        - normalized: Landmarks para dibujo (0.0 a 1.0).
        - world: Landmarks métricos (metros reales).
        """
        if frame is None:
            return PoseResult(None, None)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return PoseResult(None, None)

        norm_dict = {}
        for i, lm in enumerate(results.pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(i).name
            norm_dict[name] = np.array([lm.x, lm.y, lm.z, lm.visibility])

        world_dict = {}
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(i).name
            world_dict[name] = np.array([lm.x, lm.y, lm.z, lm.visibility])

        return PoseResult(normalized=norm_dict, world=world_dict)
