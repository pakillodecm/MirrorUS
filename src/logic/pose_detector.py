import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, NamedTuple, Optional
from src.logic.filters import OneEuroFilter


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
        self.filters_world = {}

    def _filter_dict(self, data_dict, filter_storage):
        if data_dict is None:
            return None

        filtered_data = {}
        for name, coords in data_dict.items():
            if name not in filter_storage:
                # min_cutoff: sube si hay mucho ruido quieto (0.5 - 1.5)
                # beta: sube si notas lag al moverte (0.001 - 0.1)
                filter_storage[name] = OneEuroFilter(min_cutoff=1.0, beta=0.01)

            # Filtramos solo X, Y, Z. La visibilidad (index 3) no se filtra.
            xyz_filtered = filter_storage[name].apply(coords[:3])
            filtered_data[name] = np.array([*xyz_filtered, coords[3]])

        return filtered_data

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

        raw_result = PoseResult(normalized=norm_dict, world=world_dict)
        clean_world = self._filter_dict(raw_result.world, self.filters_world)

        return PoseResult(normalized=raw_result.normalized, world=clean_world)
