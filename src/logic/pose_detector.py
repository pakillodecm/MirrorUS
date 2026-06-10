from typing import Any, Dict, NamedTuple, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.logic.filters import OneEuroFilter


class PoseResult(NamedTuple):
    """Resultado de detección de pose para un fotograma."""

    world: Optional[Dict[str, np.ndarray]]
    raw: Optional[Any]


class PoseDetector:
    """Wrapper sobre MediaPipe Pose con filtrado One Euro en coordenadas world."""

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.7,
        model_complexity: int = 1,
    ):
        """Inicializa el detector con los parámetros de confianza de MediaPipe.

        Args:
            static_image_mode: True para imágenes estáticas, False para vídeo.
            min_detection_confidence: Umbral mínimo de confianza en detección.
            min_tracking_confidence: Umbral mínimo de confianza en seguimiento.
            model_complexity: 0=Rápido, 1=Equilibrado, 2=Pesado.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=False,  # evita doble filtrado con One Euro
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.filters_world = {}

    def _filter_dict(self, data_dict, filter_storage):
        """Aplica OneEuroFilter a cada coordenada xyz de un diccionario de landmarks."""
        if data_dict is None:
            return None
        filtered = {}
        for name, coords in data_dict.items():
            if name not in filter_storage:
                filter_storage[name] = OneEuroFilter(min_cutoff=0.8, beta=0.05)
            xyz = filter_storage[name].apply(coords[:3])
            filtered[name] = np.array([*xyz, coords[3]])
        return filtered

    def extract_landmarks(self, frame: np.ndarray) -> PoseResult:
        """Procesa un fotograma y devuelve landmarks world filtrados y resultado raw.

        Args:
            frame: Imagen BGR de OpenCV. None produce un PoseResult vacío.

        Returns:
            PoseResult con world (landmarks en metros reales, filtrados con One Euro)
            y raw (resultado directo de MediaPipe, para acceder a pose_landmarks
            normalizados en el dibujo del esqueleto). Ambos campos son None si
            no se detecta ninguna persona en el fotograma.
        """
        if frame is None:
            return PoseResult(None, None)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return PoseResult(None, None)

        world_dict = {
            self.mp_pose.PoseLandmark(i).name: np.array(
                [lm.x, lm.y, lm.z, lm.visibility]
            )
            for i, lm in enumerate(results.pose_world_landmarks.landmark)
        }
        return PoseResult(
            world=self._filter_dict(world_dict, self.filters_world),
            raw=results,
        )

    def reset_filters(self) -> None:
        """Reinicia todos los filtros One Euro para iniciar una sesión limpia."""
        for f in self.filters_world.values():
            f.reset()
        self.filters_world.clear()
