import numpy as np
import pytest

from src.logic.torso_detector import TorsoTiltDetector


def create_mock_torso_landmarks(lean_forward_meters: float, vis: float = 0.95) -> dict:
    """Genera landmarks sintéticos simulando la inclinación del torso hacia adelante.

    Coloca las caderas en el plano de origen y desplaza los hombros en el
    eje Z (profundidad) para simular la inclinación sagital del torso.

    Args:
        lean_forward_meters: Desplazamiento en Z de los hombros respecto a la cadera.
            Con 0.5m de altura y 0.5m de avance equivale a 45° de inclinación.
        vis: Visibilidad aplicada a todos los landmarks (0.0 - 1.0).

    Returns:
        Diccionario de landmarks compatible con TorsoTiltDetector.analyze().
    """
    left_hip = np.array([-0.20, 0.90, 0.0, vis])
    right_hip = np.array([0.20, 0.90, 0.0, vis])
    left_shoulder = np.array([-0.20, 0.40, lean_forward_meters, vis])
    right_shoulder = np.array([0.20, 0.40, lean_forward_meters, vis])

    return {
        "LEFT_HIP": left_hip,
        "RIGHT_HIP": right_hip,
        "LEFT_SHOULDER": left_shoulder,
        "RIGHT_SHOULDER": right_shoulder,
    }


def test_torso_detector_upright():
    """Verifica que un torso vertical (0° de inclinación) no activa el detector."""
    detector = TorsoTiltDetector(max_tilt_deg=40.0)
    landmarks = create_mock_torso_landmarks(lean_forward_meters=0.0)
    has_error, angle = detector.analyze(landmarks)

    assert has_error is False
    assert angle == pytest.approx(0.0, abs=1.0)


def test_torso_detector_excessive_tilt():
    """Verifica que una inclinación excesiva (45°) activa el detector.

    Con hombros desplazados 0.5m hacia adelante sobre una altura de 0.5m,
    el ángulo resultante es 45°, superando el umbral de 40°.
    """
    detector = TorsoTiltDetector(max_tilt_deg=40.0)
    landmarks = create_mock_torso_landmarks(lean_forward_meters=0.5)
    has_error, angle = detector.analyze(landmarks)

    assert has_error is True
    assert angle == pytest.approx(45.0, abs=1.0)


def test_torso_detector_missing_landmarks():
    """Verifica que con landmarks None se retorna el fallback seguro."""
    detector = TorsoTiltDetector()
    has_error, angle = detector.analyze(None)

    assert has_error is False
    assert angle == 0.0


def test_torso_detector_low_visibility():
    """Verifica que visibilidad baja (<0.5) retorna el fallback seguro."""
    detector = TorsoTiltDetector()
    landmarks = create_mock_torso_landmarks(0.0, vis=0.3)
    has_error, angle = detector.analyze(landmarks)

    assert has_error is False
    assert angle == 0.0


def test_torso_detector_zero_norm_vector():
    """Verifica que un vector de torso nulo no lanza excepción.

    Si todos los puntos coinciden, el vector torso tiene norma cero y el
    detector debe retornar el fallback seguro sin lanzar excepción.
    """
    detector = TorsoTiltDetector()
    point = np.array([0.0, 0.0, 0.0, 0.95])
    landmarks = {
        "LEFT_HIP": point,
        "RIGHT_HIP": point,
        "LEFT_SHOULDER": point,
        "RIGHT_SHOULDER": point,
    }
    has_error, angle = detector.analyze(landmarks)

    assert has_error is False
    assert angle == 0.0


def test_torso_detector_incomplete_landmarks():
    """Verifica que un diccionario incompleto retorna el fallback seguro."""
    detector = TorsoTiltDetector()
    has_error, angle = detector.analyze({"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.95])})

    assert has_error is False
    assert angle == 0.0
