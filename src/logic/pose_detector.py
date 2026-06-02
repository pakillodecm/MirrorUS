from typing import Any, Dict, NamedTuple, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.logic.filters import OneEuroFilter


class PoseResult(NamedTuple):
    """Estructura de datos para los resultados de la detección."""

    normalized: Optional[Dict[str, np.ndarray]]
    world: Optional[Dict[str, np.ndarray]]
    raw: Optional[Any]


class PoseDetector:
    def __init__(
        self,
        static_image_mode: bool = False,  # Por defecto para video
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.7,
        model_complexity: int = 1,  # 0=Rápido, 1=Equilibrado, 2=Pesado
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=False,  # Elimina el doble filtrado redundante
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.filters_world = {}

    def _filter_dict(self, data_dict, filter_storage):
        """
        Aplica un filtro de suavizado (OneEuroFilter) a un diccionario de landmarks.

        El filtrado es esencial para evitar que pequeñas vibraciones en la detección
        generen falsos cambios de estado o ruido en el cálculo de ángulos.
        """
        if data_dict is None:
            return None

        filtered_data = {}
        for name, coords in data_dict.items():
            if name not in filter_storage:
                # min_cutoff: Reduce el ruido cuando estamos quietos [0.1 - 5.0].
                # beta: Reduce el retraso cuando nos movemos rápido [0.001 - 0.1].
                filter_storage[name] = OneEuroFilter(min_cutoff=0.8, beta=0.05)

            xyz_filtered = filter_storage[name].apply(coords[:3])
            filtered_data[name] = np.array([*xyz_filtered, coords[3]])

        return filtered_data

    def extract_landmarks(self, frame: np.ndarray) -> PoseResult:
        """
        Procesa un frame y devuelve un objeto PoseResult con:
        - normalized: Landmarks para dibujo (0.0 a 1.0).
        - world: Landmarks métricos (metros reales).
        - raw: Resultados sin procesar de MediaPipe.
        """
        if frame is None:
            return PoseResult(None, None, None)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return PoseResult(None, None, None)

        norm_dict = {}
        for i, lm in enumerate(results.pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(i).name
            norm_dict[name] = np.array([lm.x, lm.y, lm.z, lm.visibility])

        world_dict = {}
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(i).name
            world_dict[name] = np.array([lm.x, lm.y, lm.z, lm.visibility])

        raw_result = PoseResult(normalized=norm_dict, world=world_dict, raw=results)
        clean_world = self._filter_dict(raw_result.world, self.filters_world)

        return PoseResult(
            normalized=raw_result.normalized, world=clean_world, raw=raw_result.raw
        )

    def draw_landmarks(self, frame, results_raw):
        """Dibuja el esqueleto usando los resultados 'raw' de MediaPipe."""
        if results_raw and results_raw.pose_landmarks:
            df_pose_lm = self.mp_drawing_styles.get_default_pose_landmarks_style()
            self.mp_drawing.draw_landmarks(
                frame,
                results_raw.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=df_pose_lm,
            )

    def reset_filters(self) -> None:
        """Reinicia todos los filtros de suavizado."""
        for f in self.filters_world.values():
            f.reset()
        self.filters_world.clear()
