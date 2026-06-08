import numpy as np
import pytest

from src.logic.torso_detector import TorsoTiltDetector


def create_torso_landmarks(lean_z: float, vis: float = 0.95) -> dict:
    """Genera landmarks con inclinación sagital del torso controlada.

    Args:
        lean_z: Desplazamiento en Z de los hombros respecto a las caderas.
            Con 0.5m de altura y 0.5m de avance equivale a 45°.
        vis: Visibilidad de todos los landmarks.
    """
    return {
        "LEFT_HIP": np.array([-0.20, 0.90, 0.0, vis]),
        "RIGHT_HIP": np.array([0.20, 0.90, 0.0, vis]),
        "LEFT_SHOULDER": np.array([-0.20, 0.40, lean_z, vis]),
        "RIGHT_SHOULDER": np.array([0.20, 0.40, lean_z, vis]),
    }


def test_upright_no_error():
    """Verifica que un torso vertical (0°) no activa el detector."""
    has_error, angle = TorsoTiltDetector(max_tilt_deg=40.0).analyze(
        create_torso_landmarks(0.0)
    )
    assert has_error is False
    assert angle == pytest.approx(0.0, abs=1.0)


def test_excessive_tilt_detected():
    """Verifica que 45° de inclinación supera el umbral de 40°."""
    has_error, angle = TorsoTiltDetector(max_tilt_deg=40.0).analyze(
        create_torso_landmarks(0.5)
    )
    assert has_error is True
    assert angle == pytest.approx(45.0, abs=1.0)


def test_none_landmarks_returns_fallback():
    """Verifica que con None se retorna (False, 0.0)."""
    has_error, angle = TorsoTiltDetector().analyze(None)
    assert has_error is False
    assert angle == 0.0


def test_low_visibility_returns_fallback():
    """Verifica que visibilidad < 0.5 retorna (False, 0.0)."""
    has_error, angle = TorsoTiltDetector().analyze(create_torso_landmarks(0.0, vis=0.3))
    assert has_error is False
    assert angle == 0.0


def test_zero_norm_vector_returns_fallback():
    """Verifica que un vector de torso nulo no lanza excepción."""
    point = np.array([0.0, 0.0, 0.0, 0.95])
    has_error, angle = TorsoTiltDetector().analyze(
        {
            "LEFT_HIP": point,
            "RIGHT_HIP": point,
            "LEFT_SHOULDER": point,
            "RIGHT_SHOULDER": point,
        }
    )
    assert has_error is False
    assert angle == 0.0


def test_incomplete_landmarks_returns_fallback():
    """Verifica que un diccionario incompleto retorna (False, 0.0)."""
    has_error, angle = TorsoTiltDetector().analyze(
        {"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.95])}
    )
    assert has_error is False
    assert angle == 0.0
