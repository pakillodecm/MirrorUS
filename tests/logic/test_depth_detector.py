import numpy as np
import pytest

from src.logic.depth_detector import DepthDetector


def create_mock_landmarks(angle_deg: float, vis: float = 0.95) -> dict:
    """Genera landmarks sintéticos con ángulo de rodilla controlado."""
    ankle = np.array([0.0, 0.0, 0.0, vis])
    knee = np.array([0.0, 0.5, 0.0, vis])
    rad = np.deg2rad(angle_deg)
    hip = np.array(
        [
            0.5 * np.sin(np.pi - rad),
            0.5 + 0.5 * np.cos(np.pi - rad),
            0.0,
            vis,
        ]
    )
    return {
        "LEFT_HIP": hip,
        "LEFT_KNEE": knee,
        "LEFT_ANKLE": ankle,
        "RIGHT_HIP": hip,
        "RIGHT_KNEE": knee,
        "RIGHT_ANKLE": ankle,
    }


def test_standing_no_depth():
    """Verifica que en posición erguida (170°) no se detecte profundidad."""
    detector = DepthDetector(down_threshold=90.0, up_threshold=160.0)
    is_deep, angle = detector.analyze(create_mock_landmarks(170.0))
    assert is_deep is False
    assert angle == pytest.approx(170.0, abs=1.0)


def test_broken_parallel_detected():
    """Verifica que al romper el paralelo (85°) se active is_deep."""
    detector = DepthDetector(down_threshold=90.0, up_threshold=160.0)
    is_deep, angle = detector.analyze(create_mock_landmarks(85.0))
    assert is_deep is True
    assert angle == pytest.approx(85.0, abs=1.0)


def test_none_landmarks_returns_fallback():
    """Verifica que con landmarks None se retorna (False, 180.0)."""
    is_deep, angle = DepthDetector().analyze(None)
    assert is_deep is False
    assert angle == 180.0


def test_low_visibility_returns_fallback():
    """Verifica que visibilidad insuficiente retorna (False, 180.0)."""
    detector = DepthDetector(min_combined_visibility=1.99)
    is_deep, angle = detector.analyze(create_mock_landmarks(85.0, vis=0.95))
    assert is_deep is False
    assert angle == 180.0


def test_incomplete_landmarks_returns_fallback():
    """Verifica que un diccionario incompleto retorna (False, 180.0)."""
    is_deep, angle = DepthDetector().analyze(
        {"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.95])}
    )
    assert is_deep is False
    assert angle == 180.0
