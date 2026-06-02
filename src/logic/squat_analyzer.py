import time
from typing import Dict, List, Optional, TypedDict

import numpy as np

from src.logic.depth_detector import DepthDetector


class RepRecord(TypedDict):
    rep: int
    valid: bool
    errors: List[str]
    descent_duration_sec: float
    ascent_duration_sec: float


class FramePayload(TypedDict):
    rep_valid_count: int
    rep_invalid_count: int
    fsm_state: int
    feedback_message: str
    current_frame_errors: Dict[str, bool]
    metrics: Dict[str, float]
    session_history: List[RepRecord]


class SquatAnalyzer:
    def __init__(
        self, depth_detector: DepthDetector, detectors: dict, hysteresis: float = 10.0
    ):
        """Orquestador central del ciclo de vida, calidad y velocidad de la sentadilla.

        Args:
            depth_detector (DepthDetector): Sensor del plano sagital.
            detectors (dict): Registro de detectores de fallos posturales.
            hysteresis (float): Ventana de mitigación de ruido para transiciones.
        """
        self.depth_detector = depth_detector
        self.detectors = detectors
        self.hysteresis = float(hysteresis)

        # Variables de estado e historial
        self.state = 0  # 0: STAND, 1: DESCENDING, 2: DEEP, 3: ASCENDING
        self.count_valid = 0
        self.count_invalid = 0
        self.current_rep_errors = set()
        self.history = []

        # Anclas de tiempo absoluto para telemetría de velocidad (VBT)
        self.time_start_descent = None
        self.time_reached_deep = None

        # Almacenamiento persistente del último ciclo completado
        self.last_descent_duration = 0.0
        self.last_ascent_duration = 0.0

    def _get_feedback_by_state(self) -> str:
        """Determina el mensaje textual de guía según el estado de la FSM."""
        if self.state == 0:
            return "Sistema listo. Esperando inicio de la bajada..."
        elif self.state == 1:
            return "⬇️ Descendiendo... Mantén las rodillas alineadas hacia fuera."
        elif self.state == 2:
            return "🏋️ Zona Profunda alcanzada. ¡Buen paralelo! Inicia el ascenso."
        elif self.state == 3:
            return "⬆️ Ascendiendo... Controla la estabilidad y empuja el suelo."
        return "Analizando movimiento..."

    def reset_counters(self) -> None:
        """Reinicia el estado interno, cronómetros e historial de la sesión."""
        self.state = 0
        self.count_valid = 0
        self.count_invalid = 0
        self.current_rep_errors = set()
        self.history = []
        self.time_start_descent = None
        self.time_reached_deep = None
        self.last_descent_duration = 0.0
        self.last_ascent_duration = 0.0

    def process_frame(
        self,
        world_landmarks: Optional[Dict[str, np.ndarray]],
        timestamp: Optional[float] = None,
    ) -> FramePayload:
        """Procesa el fotograma actual ejecutando la lógica analítica y temporal.

        Args:
            world_landmarks (dict): Coordenadas métricas del sujeto.
            timestamp (float, opcional): Marca de tiempo para simulaciones de test.

        Returns:
            dict: Payload estructurado genérico para el frontend.
        """
        # Si no se proporciona un timestamp, usamos el reloj del sistema
        current_time = float(timestamp) if timestamp is not None else time.time()

        # 1. Consulta analítica a los sensores geométricos instantáneos
        is_deep, angle = self.depth_detector.analyze(world_landmarks)

        frame_errors = {}
        metrics = {"knee_angle": angle}

        # Bucle dinámico sobre el registro de detectores inyectados
        for name, detector in self.detectors.items():
            if world_landmarks is not None:
                has_error, value = detector.analyze(world_landmarks)
                frame_errors[name] = has_error
                if name == "KNEE_VALGUS":
                    metrics["valgus_ratio"] = value
                elif name == "TORSO_TILT":
                    metrics["torso_tilt_deg"] = value
            else:
                frame_errors[name] = False
                if name == "KNEE_VALGUS":
                    metrics["valgus_ratio"] = 1.0
                elif name == "TORSO_TILT":
                    metrics["torso_tilt_deg"] = 0.0

        # 2. Motor de transiciones de la FSM con captura de marcas temporales
        old_state = self.state
        state_changed = True
        while state_changed:
            state_changed = False
            if self.state == 0 and angle < (
                self.depth_detector.up_threshold - self.hysteresis
            ):
                self.state = 1
                self.time_start_descent = current_time  # ANCLA: Inicio de la repetición
                state_changed = True
            elif self.state == 1 and is_deep:
                self.state = 2
                self.time_reached_deep = (
                    current_time  # ANCLA: Fin de bajada / Inicio de subida
                )
                state_changed = True
            elif self.state == 1 and angle >= self.depth_detector.up_threshold:
                self.state = 0
                state_changed = True
            elif self.state == 2 and angle > (
                self.depth_detector.down_threshold + self.hysteresis
            ):
                self.state = 3
                state_changed = True
            elif self.state == 3 and angle >= self.depth_detector.up_threshold:
                self.state = 0
                state_changed = True
            elif self.state == 3 and angle <= self.depth_detector.down_threshold:
                self.state = 1
                self.current_rep_errors.add("MID_ASCENT_COLLAPSE")
                state_changed = True

        # 3. Gestión del ciclo de vida de los errores de la repetición
        if old_state == 0 and self.state != 0:
            self.current_rep_errors = set()

        if self.state in [1, 2, 3]:
            for name, is_active in frame_errors.items():
                if is_active:
                    self.current_rep_errors.add(name)

        # 4. Cierre del ciclo de movimiento: Cálculo definitivo de duraciones (VBT)
        if (old_state == 3 and self.state == 0) or (old_state == 1 and self.state == 0):
            rep_idx = len(self.history) + 1

            if old_state == 1:
                self.current_rep_errors.add("NO_DEPTH")
                self.last_descent_duration = (
                    current_time - self.time_start_descent
                    if self.time_start_descent
                    else 0.0
                )
                self.last_ascent_duration = 0.0
            else:
                # Caso normal: tramos diferenciados por los checkpoints cronometrados
                self.last_descent_duration = (
                    self.time_reached_deep - self.time_start_descent
                    if (self.time_reached_deep and self.time_start_descent)
                    else 0.0
                )
                self.last_ascent_duration = (
                    current_time - self.time_reached_deep
                    if self.time_reached_deep
                    else 0.0
                )

            if self.current_rep_errors:
                self.count_invalid += 1
                self.history.append(
                    {
                        "rep": rep_idx,
                        "valid": False,
                        "errors": sorted(list(self.current_rep_errors)),
                        "descent_duration_sec": self.last_descent_duration,
                        "ascent_duration_sec": self.last_ascent_duration,
                    }
                )
            else:
                self.count_valid += 1
                self.history.append(
                    {
                        "rep": rep_idx,
                        "valid": True,
                        "errors": [],
                        "descent_duration_sec": self.last_descent_duration,
                        "ascent_duration_sec": self.last_ascent_duration,
                    }
                )
            self.current_rep_errors = set()

        # 5. Inyección de duraciones instantáneas en métricas de control continuo
        metrics["descent_duration_sec"] = self.last_descent_duration
        metrics["ascent_duration_sec"] = self.last_ascent_duration

        return {
            "rep_valid_count": self.count_valid,
            "rep_invalid_count": self.count_invalid,
            "fsm_state": self.state,
            "feedback_message": self._get_feedback_by_state(),
            "current_frame_errors": frame_errors,
            "metrics": metrics,
            "session_history": self.history,
        }
