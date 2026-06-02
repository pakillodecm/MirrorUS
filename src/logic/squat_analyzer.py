from typing import Any, Dict, Optional

import numpy as np

from src.logic.depth_detector import DepthDetector


class SquatAnalyzer:
    def __init__(
        self, depth_detector: DepthDetector, detectors: dict, hysteresis: float = 10.0
    ):
        """Orquestador central del ciclo de vida y calidad de la sentadilla.

        Args:
            depth_detector (DepthDetector): Sensor del plano sagital.
            detectors (dict): Registro de detectores de fallos posturales.
            hysteresis (float): Ventana de mitigación de ruido para transiciones.
        """
        self.depth_detector = depth_detector
        self.detectors = detectors
        self.hysteresis = float(hysteresis)

        self.state = 0  # 0: STAND, 1: DESCENDING, 2: DEEP, 3: ASCENDING
        self.count_valid = 0
        self.count_invalid = 0
        self.current_rep_errors = set()
        self.history = []

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
        """Reinicia el estado interno y el historial de la sesión."""
        self.state = 0
        self.count_valid = 0
        self.count_invalid = 0
        self.current_rep_errors = set()
        self.history = []

    def process_frame(
        self, world_landmarks: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Procesa el fotograma actual ejecutando la lógica analítica completa.

        Args:
            world_landmarks (dict): Coordenadas métricas del sujeto.

        Returns:
            dict: Payload estructurado genérico para el frontend.
        """
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

        # 2. Motor de transiciones de la FSM Plenamente Bidireccional por Umbrales
        old_state = self.state
        state_changed = True
        while state_changed:
            state_changed = False
            if self.state == 0 and angle < (
                self.depth_detector.up_threshold - self.hysteresis
            ):
                self.state = 1
                state_changed = True
            elif self.state == 1 and is_deep:
                self.state = 2
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

        # 3. Inicialización del contenedor de fallos al romper la posición de reposo
        if old_state == 0 and self.state == 1:
            self.current_rep_errors = set()

        # Captura continua de anomalías posturales en cualquier fase del movimiento
        if self.state in [1, 2, 3]:
            for name, is_active in frame_errors.items():
                if is_active:
                    self.current_rep_errors.add(name)

        # 4. Gestión unificada de cierre de ciclo de vida (Normal o Abortado)
        if (old_state == 3 and self.state == 0) or (old_state == 1 and self.state == 0):
            rep_idx = len(self.history) + 1

            if old_state == 1:
                self.current_rep_errors.add("NO_DEPTH")

            if self.current_rep_errors:
                self.count_invalid += 1
                self.history.append(
                    {
                        "rep": rep_idx,
                        "valid": False,
                        "errors": sorted(list(self.current_rep_errors)),
                    }
                )
            else:
                self.count_valid += 1
                self.history.append({"rep": rep_idx, "valid": True, "errors": []})
            self.current_rep_errors = set()

        # 5. Retorno del contrato genérico inalterado
        return {
            "rep_valid_count": self.count_valid,
            "rep_invalid_count": self.count_invalid,
            "fsm_state": self.state,
            "feedback_message": self._get_feedback_by_state(),
            "current_frame_errors": frame_errors,
            "metrics": metrics,
            "session_history": self.history,
        }
