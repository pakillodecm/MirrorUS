import numpy as np
import pytest

from src.logic.depth_detector import DepthDetector


def create_mock_landmarks(angle_deg: float, vis: float = 0.95) -> dict:
    """Genera landmarks de prueba modulando el ángulo de la rodilla."""
    ankle = np.array([0.0, 0.0, 0.0, vis])
    knee = np.array([0.0, 0.5, 0.0, vis])

    rad = np.deg2rad(angle_deg)
    hip_x = 0.5 * np.sin(np.pi - rad)
    hip_y = 0.5 + 0.5 * np.cos(np.pi - rad)
    hip = np.array([hip_x, hip_y, 0.0, vis])

    return {
        "LEFT_HIP": hip,
        "LEFT_KNEE": knee,
        "LEFT_ANKLE": ankle,
        "RIGHT_HIP": hip,
        "RIGHT_KNEE": knee,
        "RIGHT_ANKLE": ankle,
    }


def test_depth_detector_standing():
    """Verifica que en posición erguida no se detecte profundidad."""
    detector = DepthDetector(down_threshold=90.0, up_threshold=160.0)
    is_deep, angle = detector.analyze(create_mock_landmarks(170.0))
    assert is_deep is False
    assert angle == pytest.approx(170.0, abs=1.0)


def test_depth_detector_broken_parallel():
    """Verifica que al romper el paralelo se active la bandera de profundidad."""
    detector = DepthDetector(down_threshold=90.0, up_threshold=160.0)
    is_deep, angle = detector.analyze(create_mock_landmarks(85.0))
    assert is_deep is True
    assert angle == pytest.approx(85.0, abs=1.0)


def test_depth_detector_missing_data():
    """Verifica mitigación segura ante landmarks vacíos."""
    detector = DepthDetector()
    is_deep, angle = detector.analyze(None)
    assert is_deep is False
    assert angle == 180.0


def test_depth_detector_custom_visibility_threshold():
    """Con visibilidad baja y umbral alto debe devolver None internamente
    y retornar el valor de fallback (180.0).
    """
    detector = DepthDetector(
        down_threshold=90.0,
        up_threshold=160.0,
        min_combined_visibility=1.99,
    )
    is_deep, angle = detector.analyze(create_mock_landmarks(85.0, vis=0.95))
    assert is_deep is False
    assert angle == 180.0
