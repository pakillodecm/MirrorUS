import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional


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

    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Procesa un frame y devuelve un diccionario con los landmarks clave.
        """
        if frame is None:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        landmarks_dict = {}
        for lm_id in self.mp_pose.PoseLandmark:
            lm = results.pose_landmarks.landmark[lm_id]
            landmarks_dict[lm_id.name] = np.array([lm.x, lm.y, lm.z])

        return landmarks_dict
