import numpy as np
import pytest

from src.logic.depth_detector import DepthDetector
from src.logic.squat_analyzer import SquatAnalyzer
from src.logic.valgus_detector import KneeValgusDetector


def create_frame_data(
    angle_deg: float, hip_w: float, knee_w: float, vis: float = 0.95
) -> dict:
    """Genera un esqueleto tridimensional con ángulo de rodilla controlado.

    Usa corrección de Pitágoras para garantizar que el ángulo 3D calculado
    sea exactamente 'angle_deg' independientemente de la anchura en el eje X.

    Args:
        angle_deg: Ángulo de rodilla deseado en grados.
        hip_w: Anchura entre caderas en metros.
        knee_w: Anchura entre rodillas en metros.
        vis: Visibilidad aplicada a todos los landmarks (0.0 - 1.0).

    Returns:
        Diccionario de landmarks compatible con SquatAnalyzer.process_frame().
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
    """Verifica el ciclo completo de una repetición válida y su telemetría VBT.

    Simula una sentadilla perfecta en 5 frames con timestamps controlados
    y comprueba que:
    - La FSM transita correctamente por los 4 estados (0→1→2→3→0).
    - Se registra una repetición válida sin errores.
    - Las duraciones de bajada y subida son exactas (1.5s y 2.0s).
    - El historial persiste los datos de telemetría correctamente.
    """
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    payload = analyzer.process_frame(
        create_frame_data(170.0, 0.40, 0.40), timestamp=0.0
    )
    assert payload["fsm_state"] == 0

    payload = analyzer.process_frame(
        create_frame_data(130.0, 0.40, 0.40), timestamp=1.0
    )
    assert payload["fsm_state"] == 1

    payload = analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40), timestamp=2.5)
    assert payload["fsm_state"] == 2

    payload = analyzer.process_frame(
        create_frame_data(120.0, 0.40, 0.40), timestamp=3.5
    )
    assert payload["fsm_state"] == 3

    payload = analyzer.process_frame(
        create_frame_data(160.0, 0.40, 0.40), timestamp=4.5
    )

    assert payload["fsm_state"] == 0
    assert payload["rep_valid_count"] == 1
    assert pytest.approx(payload["metrics"]["descent_duration_sec"]) == 1.5
    assert pytest.approx(payload["metrics"]["ascent_duration_sec"]) == 2.0

    last_rep = payload["session_history"][-1]
    assert pytest.approx(last_rep["descent_duration_sec"]) == 1.5
    assert pytest.approx(last_rep["ascent_duration_sec"]) == 2.0


def test_analyzer_failed_lifecycle_with_valgus():
    """Verifica que el valgo de rodilla durante el ascenso invalida la repetición.

    Simula una sentadilla con rodillas colapsadas (knee_w < hip_w) en el frame
    de ascenso y comprueba que el error KNEE_VALGUS queda registrado.
    """
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40))

    payload = analyzer.process_frame(create_frame_data(120.0, 0.40, 0.30))
    assert payload["current_frame_errors"]["KNEE_VALGUS"] is True

    payload = analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40))
    assert payload["rep_valid_count"] == 0
    assert payload["rep_invalid_count"] == 1
    assert payload["session_history"][-1]["valid"] is False
    assert "KNEE_VALGUS" in payload["session_history"][-1]["errors"]


def test_analyzer_aborted_squat_no_depth():
    """Verifica que una bajada parcial sin alcanzar profundidad se marca como NO_DEPTH.

    Simula una sentadilla abortada a 105° (por encima del umbral de 90°)
    y comprueba que la repetición se registra como inválida con error NO_DEPTH.
    """
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
    """Verifica que una caída durante el ascenso se registra como MID_ASCENT_COLLAPSE.

    Simula una sentadilla en la que el atleta vuelve a caer a zona profunda
    durante el ascenso. La FSM debe degradar el estado a DEEP y registrar
    el error MID_ASCENT_COLLAPSE al finalizar la repetición.
    """
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40))

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


def test_analyzer_reset_counters():
    """Verifica que reset_counters() devuelve el estado interno a sus valores iniciales.

    Ejecuta un ciclo completo para ensuciar el estado y comprueba que tras
    el reset todos los contadores, cronómetros e historial quedan a cero.
    """
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth, detectors={"KNEE_VALGUS": valgus}, hysteresis=5.0
    )

    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40), timestamp=0.0)
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40), timestamp=1.0)
    analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40), timestamp=2.0)
    analyzer.process_frame(create_frame_data(120.0, 0.40, 0.40), timestamp=3.0)
    analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40), timestamp=4.0)
    assert analyzer.count_valid == 1

    analyzer.reset_counters()

    assert analyzer.state == 0
    assert analyzer.count_valid == 0
    assert analyzer.count_invalid == 0
    assert analyzer.history == []
    assert analyzer.current_rep_errors == set()
    assert analyzer.time_start_descent is None
    assert analyzer.time_reached_deep is None
    assert analyzer.last_descent_duration == 0.0
    assert analyzer.last_ascent_duration == 0.0


def test_analyzer_feedback_unknown_state():
    """Verifica que un estado FSM desconocido devuelve el mensaje genérico."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(depth_detector=depth, detectors={}, hysteresis=5.0)
    analyzer.state = 99

    assert analyzer._get_feedback_by_state() == "Analizando movimiento..."


def test_analyzer_torso_tilt_detector_registered():
    """Verifica que TORSO_TILT puebla correctamente la métrica torso_tilt_deg."""
    from src.logic.torso_detector import TorsoTiltDetector

    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    torso = TorsoTiltDetector(max_tilt_deg=40.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"TORSO_TILT": torso},
        hysteresis=5.0,
    )
    payload = analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40))

    assert "torso_tilt_deg" in payload["metrics"]


def test_analyzer_torso_tilt_none_landmarks():
    """Verifica que con landmarks None el detector TORSO_TILT usa el valor por defecto.

    Cuando no hay landmarks disponibles, el payload debe incluir torso_tilt_deg=0.0
    sin lanzar ninguna excepción.
    """
    from src.logic.torso_detector import TorsoTiltDetector

    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    torso = TorsoTiltDetector(max_tilt_deg=40.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"TORSO_TILT": torso},
        hysteresis=5.0,
    )
    payload = analyzer.process_frame(None)

    assert payload["metrics"]["torso_tilt_deg"] == 0.0


def test_analyzer_knee_valgus_none_landmarks():
    """Verifica que con landmarks None el detector KNEE_VALGUS usa el valor por defecto.

    Cuando no hay landmarks disponibles, el payload debe incluir valgus_ratio=1.0
    sin lanzar ninguna excepción.
    """
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    valgus = KneeValgusDetector(threshold=0.90)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"KNEE_VALGUS": valgus},
        hysteresis=5.0,
    )
    payload = analyzer.process_frame(None)

    assert payload["metrics"]["valgus_ratio"] == 1.0
