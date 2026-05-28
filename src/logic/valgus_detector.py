from typing import Dict, Optional, Tuple

import numpy as np


class KneeValgusDetector:
    def __init__(self, threshold: float = 0.90):
        """Inicializa el detector de valgo de rodilla.

        Args:
            threshold (float): Umbral por debajo del cual se considera valgo.
                              Por defecto es 0.90 (rodillas un 10% más juntas
                              que las caderas).
        """
        self.threshold = float(threshold)

    def analyze(
        self, world_landmarks: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[bool, float]:
        """Analiza el colapso de rodillas utilizando coordenadas métricas (eje X).

        Args:
            world_landmarks (dict): Diccionario de landmarks de MediaPipe en
                                    metros reales.

        Returns:
            tuple: (is_valgus (bool), valgus_ratio (float))
        """
        if world_landmarks is None:
            return False, 1.0

        try:
            # Extraemos los puntos críticos del plano frontal
            l_hip = world_landmarks["LEFT_HIP"]
            r_hip = world_landmarks["RIGHT_HIP"]
            l_knee = world_landmarks["LEFT_KNEE"]
            r_knee = world_landmarks["RIGHT_KNEE"]

            # Filtro de confianza estructural mínimo para evitar falsos positivos
            if l_hip[3] < 0.5 or r_hip[3] < 0.5 or l_knee[3] < 0.5 or r_knee[3] < 0.5:
                return False, 1.0

            # Distancia horizontal absoluta en el eje X (metros reales)
            hip_distance = abs(l_hip[0] - r_hip[0])
            knee_distance = abs(l_knee[0] - r_knee[0])

            if hip_distance == 0:
                return False, 1.0

            # Índice de alineación: proporción relativa entre rodillas y caderas
            valgus_ratio = knee_distance / hip_distance
            is_valgus = bool(valgus_ratio < self.threshold)

            return is_valgus, valgus_ratio

        except KeyError:
            # Mitigación segura si el diccionario viene incompleto en este frame
            return False, 1.0
