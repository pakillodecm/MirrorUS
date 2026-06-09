import time
from typing import Dict, List, Optional, TypedDict

import numpy as np

from src.logic.depth_detector import DepthDetector


class RepRecord(TypedDict):
    """Registro de una repetición completada."""

    rep: int
    valid: bool
    errors: List[str]
    descent_duration_sec: float
    ascent_duration_sec: float
    min_knee_angle: float
    depth_threshold: float
    min_valgus_ratio: Optional[float]
    max_torso_tilt: Optional[float]
    torso_threshold: Optional[float]


class FramePayload(TypedDict):
    """Payload estructurado que el frontend consume tras cada fotograma."""

    rep_valid_count: int
    rep_invalid_count: int
    fsm_state: int
    current_frame_errors: Dict[str, bool]
    metrics: Dict[str, float]
    session_history: List[RepRecord]


class SquatAnalyzer:
    """Orquestador de la FSM, telemetría VBT y detección de errores posturales."""

    def __init__(
        self,
        depth_detector: DepthDetector,
        detectors: dict,
        hysteresis: float = 10.0,
        timeout_sec: float = 8.0,
    ):
        """Inicializa el analizador con los detectores y parámetros de control.

        Args:
            depth_detector: Sensor del ángulo de rodilla en el plano sagital.
            detectors: Registro de detectores de fallos posturales frame-level.
            hysteresis: Ventana en grados para suavizar transiciones de la FSM.
            timeout_sec: Segundos máximos en estado DESCENDING antes de cerrar
                la repetición automáticamente con NO_DEPTH. Evita que la FSM
                quede bloqueada con thresholds extremos o movimientos abortados.
        """
        self.depth_detector = depth_detector
        self.detectors = detectors
        self.hysteresis = float(hysteresis)
        self.timeout_sec = float(timeout_sec)

        self.state = 0  # 0: STAND, 1: DESCENDING, 2: DEEP, 3: ASCENDING
        self.count_valid = 0
        self.count_invalid = 0
        self.current_rep_errors: set = set()
        self.history: List[RepRecord] = []

        self.time_start_descent: Optional[float] = None
        self.time_reached_deep: Optional[float] = None
        self.last_descent_duration = 0.0
        self.last_ascent_duration = 0.0

        # Métricas de rep en curso para el historial
        self._min_knee_angle_this_rep: float = 180.0
        self._min_valgus_ratio_this_rep: float = 1.0
        self._max_torso_tilt_this_rep: float = 0.0

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
        self._min_knee_angle_this_rep = 180.0
        self._min_valgus_ratio_this_rep = 1.0
        self._max_torso_tilt_this_rep = 0.0

    def process_frame(
        self,
        world_landmarks: Optional[Dict[str, np.ndarray]],
        timestamp: Optional[float] = None,
    ) -> FramePayload:
        """Procesa un fotograma y devuelve el payload estructurado para el frontend.

        Ejecuta en orden: detección de sensores, transiciones de FSM,
        timeout de descenso, gestión de errores y cierre de repetición.

        Args:
            world_landmarks: Coordenadas métricas de MediaPipe, o None si
                la detección falló en este fotograma.
            timestamp: Marca de tiempo en segundos; si es None usa time.time().

        Returns:
            FramePayload con el estado completo de la sesión.
        """
        current_time = float(timestamp) if timestamp is not None else time.time()

        # 1. Sensores geométricos instantáneos
        is_deep, angle = self.depth_detector.analyze(world_landmarks)
        frame_errors: Dict[str, bool] = {}
        metrics: Dict[str, float] = {"knee_angle": angle}

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

        # 2. Transiciones de la FSM con histéresis
        old_state = self.state
        state_changed = True
        while state_changed:
            state_changed = False
            if self.state == 0 and angle < (
                self.depth_detector.up_threshold - self.hysteresis
            ):
                self.state = 1
                self.time_start_descent = current_time
                state_changed = True
            elif self.state == 1 and is_deep:
                self.state = 2
                self.time_reached_deep = current_time
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

        # 3. Timeout de descenso: cierra la rep si el atleta lleva demasiado
        # tiempo en DESCENDING sin alcanzar profundidad.
        if (
            self.state == 1
            and self.time_start_descent is not None
            and (current_time - self.time_start_descent) > self.timeout_sec
        ):
            self.state = 0

        # 4. Ciclo de vida de errores y métricas por repetición
        if old_state == 0 and self.state != 0:
            self.current_rep_errors = set()
            self._min_knee_angle_this_rep = 180.0
            self._min_valgus_ratio_this_rep = 1.0
            self._max_torso_tilt_this_rep = 0.0

        if self.state in (1, 2, 3):
            self._min_knee_angle_this_rep = min(angle, self._min_knee_angle_this_rep)
            valgus = metrics.get("valgus_ratio")
            if valgus is not None:
                self._min_valgus_ratio_this_rep = min(
                    valgus, self._min_valgus_ratio_this_rep
                )
            torso = metrics.get("torso_tilt_deg")
            if torso is not None:
                self._max_torso_tilt_this_rep = max(
                    torso, self._max_torso_tilt_this_rep
                )
            for name, is_active in frame_errors.items():
                if is_active:
                    self.current_rep_errors.add(name)

        # 5. Cierre del ciclo: VBT y registro en historial
        rep_just_closed = (old_state == 3 and self.state == 0) or (
            old_state == 1 and self.state == 0
        )
        if rep_just_closed:
            rep_idx = len(self.history) + 1

            # Capturar errores posturales antes de añadir errores estructurales
            has_valgus = "KNEE_VALGUS" in self.current_rep_errors
            has_torso = "TORSO_TILT" in self.current_rep_errors
            torso_detector = self.detectors.get("TORSO_TILT")

            if old_state == 1:
                self.current_rep_errors.add("NO_DEPTH")
                self.last_descent_duration = (
                    current_time - self.time_start_descent
                    if self.time_start_descent
                    else 0.0
                )
                self.last_ascent_duration = 0.0
            else:
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

            is_valid = not bool(self.current_rep_errors)
            record: RepRecord = {
                "rep": rep_idx,
                "valid": is_valid,
                "errors": sorted(list(self.current_rep_errors)),
                "descent_duration_sec": self.last_descent_duration,
                "ascent_duration_sec": self.last_ascent_duration,
                "min_knee_angle": round(self._min_knee_angle_this_rep, 1),
                "depth_threshold": self.depth_detector.down_threshold,
                "min_valgus_ratio": (
                    round(self._min_valgus_ratio_this_rep, 2) if has_valgus else None
                ),
                "max_torso_tilt": (
                    round(self._max_torso_tilt_this_rep, 1) if has_torso else None
                ),
                "torso_threshold": (
                    torso_detector.max_tilt_deg
                    if (has_torso and torso_detector)
                    else None
                ),
            }

            if is_valid:
                self.count_valid += 1
            else:
                self.count_invalid += 1

            self.history.append(record)
            self.current_rep_errors = set()

        # 6. Duraciones instantáneas para el indicador de profundidad en tiempo real
        metrics["descent_duration_sec"] = self.last_descent_duration
        metrics["ascent_duration_sec"] = self.last_ascent_duration

        return {
            "rep_valid_count": self.count_valid,
            "rep_invalid_count": self.count_invalid,
            "fsm_state": self.state,
            "current_frame_errors": frame_errors,
            "metrics": metrics,
            "session_history": self.history,
        }
