import numpy as np
import pytest

from src.logic.depth_detector import DepthDetector
from src.logic.squat_analyzer import SquatAnalyzer
from src.logic.valgus_detector import KneeValgusDetector


def create_frame_data(
    angle_deg: float, hip_w: float, knee_w: float, vis: float = 0.95
) -> dict:
    """Genera un esqueleto tridimensional con corrección de Pitágoras.

    Garantiza que el ángulo 3D calculado sea exactamente 'angle_deg'
    sin importar las variaciones de anchura en el eje X.
    """
    ankle = np.array([-knee_w / 2.0, 0.0, 0.0, vis])
    knee = np.array([-knee_w / 2.0, 0.5, 0.0, vis])

    dx = (-hip_w / 2.0) - (-knee_w / 2.0)
    dy = -0.5 * np.cos(np.deg2rad(angle_deg))
    base_z = 0.25 * (np.sin(np.deg2rad(angle_deg)) ** 2) - (dx**2)
    dz = np.sqrt(max(0.0, base_z))

    left_hip = np.array([-hip_w / 2.0, 0.5 + dy, dz, vis])
    right_hip = np.array([hip_w / 2.0, 0.5 + dy, dz, vis])
    left_knee = knee
    right_knee = np.array([knee_w / 2.0, 0.5, 0.0, vis])
    left_ankle = ankle
    right_ankle = np.array([knee_w / 2.0, 0.0, 0.0, vis])

    return {
        "LEFT_HIP": left_hip,
        "RIGHT_HIP": right_hip,
        "LEFT_KNEE": left_knee,
        "RIGHT_KNEE": right_knee,
        "LEFT_ANKLE": left_ankle,
        "RIGHT_ANKLE": right_ankle,
    }


def test_analyzer_perfect_lifecycle():
    """Verifica el ciclo completo de una repetición perfecta sin fallos
    y valida la telemetría temporal de velocidad por fases.
    """
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    # 1. STAND (170 grados) en el segundo 0.0
    payload = analyzer.process_frame(
        create_frame_data(170.0, 0.40, 0.40), timestamp=0.0
    )
    assert payload["fsm_state"] == 0

    # 2. DESCENDING (130 grados) en el segundo 1.0 -> Aquí arranca la bajada
    payload = analyzer.process_frame(
        create_frame_data(130.0, 0.40, 0.40), timestamp=1.0
    )
    assert payload["fsm_state"] == 1

    # 3. DEEP (85 grados) en el segundo 2.5 -> Aquí termina la bajada (Duración = 1.5s)
    payload = analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40), timestamp=2.5)
    assert payload["fsm_state"] == 2

    # 4. ASCENDING (120 grados) en el segundo 3.5 -> Aquí arranca la subida
    payload = analyzer.process_frame(
        create_frame_data(120.0, 0.40, 0.40), timestamp=3.5
    )
    assert payload["fsm_state"] == 3

    # 5. STAND (160 grados) en el segundo 4.5 -> Fin de la subida (Duración = 2.0s)
    payload = analyzer.process_frame(
        create_frame_data(160.0, 0.40, 0.40), timestamp=4.5
    )

    # VALIDACIONES DE TELEMETRÍA TEMPORAL (VBT):
    assert payload["fsm_state"] == 0
    assert payload["rep_valid_count"] == 1

    # Exigimos las duraciones en las métricas del último ciclo
    assert pytest.approx(payload["metrics"]["descent_duration_sec"]) == 1.5
    assert pytest.approx(payload["metrics"]["ascent_duration_sec"]) == 2.0

    # Exigimos que el historial guarde el reporte temporal de forma persistente
    last_history = payload["session_history"][-1]
    assert pytest.approx(last_history["descent_duration_sec"]) == 1.5
    assert pytest.approx(last_history["ascent_duration_sec"]) == 2.0


def test_analyzer_failed_lifecycle_with_valgus():
    """Verifica que una repetición con colapso en el ascenso sea inválida."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40))  # STAND
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40))  # DESCENDING
    analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40))  # DEEP

    payload = analyzer.process_frame(create_frame_data(120.0, 0.40, 0.30))  # ASCENDING
    assert payload["current_frame_errors"]["KNEE_VALGUS"] is True

    payload = analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40))  # STAND
    assert payload["rep_valid_count"] == 0
    assert payload["rep_invalid_count"] == 1
    assert payload["session_history"][-1]["valid"] is False
    assert "KNEE_VALGUS" in payload["session_history"][-1]["errors"]


def test_analyzer_aborted_squat_no_depth():
    """Verifica que una bajada parcial sea interceptada por 'NO_DEPTH'."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40))
    payload = analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40))
    assert payload["fsm_state"] == 1

    payload = analyzer.process_frame(create_frame_data(105.0, 0.40, 0.40))
    assert payload["fsm_state"] == 1

    payload = analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40))
    assert payload["fsm_state"] == 0
    assert payload["rep_valid_count"] == 0
    assert payload["rep_invalid_count"] == 1
    assert payload["session_history"][-1]["valid"] is False
    assert "NO_DEPTH" in payload["session_history"][-1]["errors"]


def test_analyzer_ascent_collapse():
    """Verifica que si el atleta vuelve a caer durante la subida, la FSM lo

    degrade a la zona profunda y registre el fallo 'MID_ASCENT_COLLAPSE'.
    """
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40))  # STAND
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40))  # DESCENDING
    analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40))  # DEEP

    payload = analyzer.process_frame(create_frame_data(120.0, 0.40, 0.40))
    assert payload["fsm_state"] == 3

    payload = analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40))
    assert payload["fsm_state"] == 2

    payload = analyzer.process_frame(create_frame_data(120.0, 0.40, 0.40))
    assert payload["fsm_state"] == 3

    payload = analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40))
    assert payload["fsm_state"] == 0
    assert payload["rep_valid_count"] == 0
    assert payload["rep_invalid_count"] == 1
    assert payload["session_history"][-1]["valid"] is False
    assert "MID_ASCENT_COLLAPSE" in payload["session_history"][-1]["errors"]
