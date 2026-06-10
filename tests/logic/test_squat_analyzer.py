from unittest.mock import MagicMock

import numpy as np
import pytest

from src.logic.depth_detector import DepthDetector
from src.logic.squat_analyzer import SquatAnalyzer
from src.logic.torso_detector import TorsoTiltDetector
from src.logic.valgus_detector import KneeValgusDetector


def create_frame_data(
    angle_deg: float,
    hip_w: float,
    knee_w: float,
    ankle_w: float,
    vis: float = 0.95,
) -> dict:
    """Genera un esqueleto 3D con ángulo de rodilla controlado.

    Args:
        angle_deg: Ángulo de rodilla deseado en grados.
        hip_w: Anchura entre caderas en metros.
        knee_w: Anchura entre rodillas en metros.
        ankle_w: Anchura entre tobillos en metros.
        vis: Visibilidad aplicada a todos los landmarks.

    Returns:
        Diccionario de landmarks compatible con SquatAnalyzer.process_frame().
    """

    ankle = np.array([-ankle_w / 2.0, 0.0, 0.0, vis])
    knee = np.array([-knee_w / 2.0, 0.5, 0.0, vis])
    dx = (-hip_w / 2.0) - (-knee_w / 2.0)
    dy = -0.5 * np.cos(np.deg2rad(angle_deg))
    base_z = 0.25 * (np.sin(np.deg2rad(angle_deg)) ** 2) - (dx**2)
    dz = np.sqrt(max(0.0, base_z))
    left_hip = np.array([-hip_w / 2.0, 0.5 + dy, dz, vis])

    return {
        "LEFT_HIP": left_hip,
        "RIGHT_HIP": np.array([hip_w / 2.0, 0.5 + dy, dz, vis]),
        "LEFT_KNEE": knee,
        "RIGHT_KNEE": np.array([knee_w / 2.0, 0.5, 0.0, vis]),
        "LEFT_ANKLE": ankle,
        "RIGHT_ANKLE": np.array([ankle_w / 2.0, 0.0, 0.0, vis]),
    }


def _make_mock_analyzer(timeout_sec: float = 5.0):
    """Crea un SquatAnalyzer con DepthDetector mockeado y timeout configurable."""
    depth = MagicMock()
    depth.down_threshold = 90.0
    depth.up_threshold = 150.0
    return (
        SquatAnalyzer(depth_detector=depth, detectors={}, timeout_sec=timeout_sec),
        depth,
    )


# ---------------------------------------------------------------------------
# Ciclo de vida completo
# ---------------------------------------------------------------------------


def test_analyzer_perfect_lifecycle():
    """Verifica el ciclo completo de una repetición válida y su telemetría VBT."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"KNEE_VALGUS": KneeValgusDetector(threshold=0.08)},
        hysteresis=5.0,
    )

    assert (
        analyzer.process_frame(
            create_frame_data(170.0, 0.40, 0.40, 0.40), timestamp=0.0
        )["fsm_state"]
        == 0
    )
    assert (
        analyzer.process_frame(
            create_frame_data(130.0, 0.40, 0.40, 0.40), timestamp=1.0
        )["fsm_state"]
        == 1
    )
    assert (
        analyzer.process_frame(
            create_frame_data(85.0, 0.40, 0.40, 0.40), timestamp=2.5
        )["fsm_state"]
        == 2
    )
    assert (
        analyzer.process_frame(
            create_frame_data(120.0, 0.40, 0.40, 0.40), timestamp=3.5
        )["fsm_state"]
        == 3
    )

    payload = analyzer.process_frame(
        create_frame_data(160.0, 0.40, 0.40, 0.40), timestamp=4.5
    )
    assert payload["fsm_state"] == 0
    assert payload["rep_valid_count"] == 1
    assert pytest.approx(payload["metrics"]["descent_duration_sec"]) == 1.5
    assert pytest.approx(payload["metrics"]["ascent_duration_sec"]) == 2.0
    last = payload["session_history"][-1]
    assert pytest.approx(last["descent_duration_sec"]) == 1.5
    assert pytest.approx(last["ascent_duration_sec"]) == 2.0


def test_analyzer_failed_lifecycle_with_valgus():
    """Verifica que el valgo de rodilla durante el ascenso invalida la repetición."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"KNEE_VALGUS": KneeValgusDetector(threshold=0.08)},
        hysteresis=5.0,
    )
    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40, 0.40))

    payload = analyzer.process_frame(create_frame_data(120.0, 0.40, 0.20, 0.50))
    assert payload["current_frame_errors"]["KNEE_VALGUS"] is True

    payload = analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40, 0.40))
    assert payload["rep_invalid_count"] == 1
    assert "KNEE_VALGUS" in payload["session_history"][-1]["errors"]


def test_analyzer_aborted_squat_no_depth():
    """Verifica que bajada parcial sin alcanzar profundidad se marca como NO_DEPTH."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"KNEE_VALGUS": KneeValgusDetector(threshold=0.08)},
        hysteresis=5.0,
    )
    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(105.0, 0.40, 0.40, 0.40))

    payload = analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40, 0.40))
    assert payload["fsm_state"] == 0
    assert payload["rep_invalid_count"] == 1
    assert "NO_DEPTH" in payload["session_history"][-1]["errors"]


def test_analyzer_ascent_collapse():
    """Verifica que caída durante el ascenso se registra como MID_ASCENT_COLLAPSE."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"KNEE_VALGUS": KneeValgusDetector(threshold=0.08)},
        hysteresis=5.0,
    )
    analyzer.process_frame(create_frame_data(170.0, 0.40, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40, 0.40))
    analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40, 0.40))
    assert (
        analyzer.process_frame(create_frame_data(120.0, 0.40, 0.40, 0.40))["fsm_state"]
        == 3
    )
    assert (
        analyzer.process_frame(create_frame_data(85.0, 0.40, 0.40, 0.40))["fsm_state"]
        == 2
    )
    analyzer.process_frame(create_frame_data(120.0, 0.40, 0.40, 0.40))

    payload = analyzer.process_frame(create_frame_data(160.0, 0.40, 0.40, 0.40))
    assert payload["rep_invalid_count"] == 1
    assert "MID_ASCENT_COLLAPSE" in payload["session_history"][-1]["errors"]


def test_analyzer_reset_counters():
    """Verifica que reset_counters() devuelve el estado a sus valores iniciales."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"KNEE_VALGUS": KneeValgusDetector(threshold=0.08)},
        hysteresis=5.0,
    )
    for t, angle in [(0.0, 170), (1.0, 130), (2.0, 85), (3.0, 120), (4.0, 160)]:
        analyzer.process_frame(create_frame_data(angle, 0.40, 0.40, 0.40), timestamp=t)
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


def test_analyzer_torso_tilt_metric():
    """Verifica que TORSO_TILT puebla correctamente la métrica torso_tilt_deg."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={"TORSO_TILT": TorsoTiltDetector(max_tilt_deg=40.0)},
        hysteresis=5.0,
    )
    payload = analyzer.process_frame(create_frame_data(130.0, 0.40, 0.40, 0.40))

    assert "torso_tilt_deg" in payload["metrics"]


def test_analyzer_none_landmarks_defaults():
    """Verifica que con landmarks None los detectores usan valores por defecto."""
    depth = DepthDetector(down_threshold=90.0, up_threshold=150.0)
    analyzer = SquatAnalyzer(
        depth_detector=depth,
        detectors={
            "KNEE_VALGUS": KneeValgusDetector(threshold=0.08),
            "TORSO_TILT": TorsoTiltDetector(max_tilt_deg=40.0),
        },
        hysteresis=5.0,
    )
    payload = analyzer.process_frame(None)

    assert payload["metrics"]["valgus_ratio"] == 0.0
    assert payload["metrics"]["torso_tilt_deg"] == 0.0


# ---------------------------------------------------------------------------
# Timeout de descenso
# ---------------------------------------------------------------------------


def test_rep_closes_with_no_depth_after_timeout():
    """Tras timeout_sec en descenso la rep se cierra con NO_DEPTH."""
    analyzer, depth = _make_mock_analyzer(timeout_sec=5.0)
    depth.analyze.return_value = (False, 120.0)

    analyzer.process_frame(None, timestamp=0.0)
    payload = analyzer.process_frame(None, timestamp=5.1)

    assert payload["fsm_state"] == 0
    assert payload["rep_invalid_count"] == 1
    assert "NO_DEPTH" in payload["session_history"][-1]["errors"]


def test_rep_stays_open_before_timeout():
    """Antes de timeout_sec la FSM permanece en estado DESCENDING."""
    analyzer, depth = _make_mock_analyzer(timeout_sec=5.0)
    depth.analyze.return_value = (False, 120.0)

    analyzer.process_frame(None, timestamp=0.0)
    payload = analyzer.process_frame(None, timestamp=4.9)

    assert payload["fsm_state"] == 1
    assert payload["rep_invalid_count"] == 0


def test_timeout_timer_resets_for_next_rep():
    """Tras un timeout la siguiente bajada inicia un nuevo contador."""
    analyzer, depth = _make_mock_analyzer(timeout_sec=5.0)
    depth.analyze.return_value = (False, 120.0)

    analyzer.process_frame(None, timestamp=0.0)
    analyzer.process_frame(None, timestamp=5.1)  # primera rep cierra
    assert analyzer.count_invalid == 1

    analyzer.process_frame(None, timestamp=5.2)  # nueva bajada (0→1)
    assert analyzer.state == 1

    payload = analyzer.process_frame(None, timestamp=9.0)  # 9.0-5.2=3.8 < 5.0
    assert payload["fsm_state"] == 1

    payload = analyzer.process_frame(None, timestamp=10.4)  # 10.4-5.2=5.2 > 5.0
    assert payload["fsm_state"] == 0
    assert payload["rep_invalid_count"] == 2


def test_valid_rep_not_affected_by_timeout():
    """Una rep que alcanza profundidad y sube no dispara el timeout."""
    analyzer, depth = _make_mock_analyzer(timeout_sec=5.0)

    depth.analyze.return_value = (False, 120.0)
    analyzer.process_frame(None, timestamp=0.0)

    depth.analyze.return_value = (True, 80.0)
    analyzer.process_frame(None, timestamp=1.5)
    assert analyzer.state == 2

    depth.analyze.return_value = (False, 110.0)
    analyzer.process_frame(None, timestamp=2.0)
    assert analyzer.state == 3

    depth.analyze.return_value = (False, 155.0)
    payload = analyzer.process_frame(None, timestamp=3.0)

    assert payload["fsm_state"] == 0
    assert payload["rep_valid_count"] == 1
    assert payload["rep_invalid_count"] == 0
