import numpy as np
import pytest

from src.logic.depth_detector import DepthDetector


def create_mock_landmarks(angle_deg: float, vis: float = 0.95) -> dict:
    """Genera landmarks sintéticos con un ángulo de rodilla controlado.

    Construye un esqueleto simétrico (mismos valores para ambos lados)
    colocando la cadera a partir del ángulo deseado mediante trigonometría.

    Args:
        angle_deg: Ángulo de rodilla deseado en grados.
        vis: Visibilidad aplicada a todos los landmarks (0.0 - 1.0).

    Returns:
        Diccionario de landmarks compatible con DepthDetector.analyze().
    """
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
    """Verifica que en posición erguida (170°) no se detecte profundidad."""
    detector = DepthDetector(down_threshold=90.0, up_threshold=160.0)
    is_deep, angle = detector.analyze(create_mock_landmarks(170.0))

    assert is_deep is False
    assert angle == pytest.approx(170.0, abs=1.0)


def test_depth_detector_broken_parallel():
    """Verifica que al romper el paralelo (85°) se active la bandera de profundidad."""
    detector = DepthDetector(down_threshold=90.0, up_threshold=160.0)
    is_deep, angle = detector.analyze(create_mock_landmarks(85.0))

    assert is_deep is True
    assert angle == pytest.approx(85.0, abs=1.0)


def test_depth_detector_missing_data():
    """Verifica que con landmarks None se retorna el fallback seguro (180°)."""
    detector = DepthDetector()
    is_deep, angle = detector.analyze(None)

    assert is_deep is False
    assert angle == 180.0


def test_depth_detector_custom_visibility_threshold():
    """Verifica que un umbral de visibilidad muy alto rechaza la medición.

    Con visibilidad de 0.95 por lado (total 1.90) y umbral de 1.99,
    la medición no supera el filtro y se retorna el fallback (180°).
    """
    detector = DepthDetector(
        down_threshold=90.0,
        up_threshold=160.0,
        min_combined_visibility=1.99,
    )
    is_deep, angle = detector.analyze(create_mock_landmarks(85.0, vis=0.95))

    assert is_deep is False
    assert angle == 180.0


def test_depth_detector_incomplete_landmarks():
    """Verifica que un diccionario incompleto retorna el fallback seguro (180°)."""
    detector = DepthDetector()
    incomplete = {"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.95])}
    is_deep, angle = detector.analyze(incomplete)

    assert is_deep is False
    assert angle == 180.0
