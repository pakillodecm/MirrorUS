import numpy as np
import pytest

from src.logic.torso_detector import TorsoTiltDetector


def create_mock_torso_landmarks(lean_forward_meters: float, vis: float = 0.95) -> dict:
    """Genera landmarks de prueba desplazando los hombros en el eje Z

    para simular la inclinación hacia adelante del torso (Sagital).
    """
    # Caderas alineadas en el plano de origen
    left_hip = np.array([-0.20, 0.90, 0.0, vis])
    right_hip = np.array([0.20, 0.90, 0.0, vis])

    # Hombros situados a 0.5 metros de altura sobre la cadera.
    # El parámetro 'lean_forward_meters' desplaza los hombros en el eje Z (profundidad)
    left_shoulder = np.array([-0.20, 0.40, lean_forward_meters, vis])
    right_shoulder = np.array([0.20, 0.40, lean_forward_meters, vis])

    return {
        "LEFT_HIP": left_hip,
        "RIGHT_HIP": right_hip,
        "LEFT_SHOULDER": left_shoulder,
        "RIGHT_SHOULDER": right_shoulder,
    }


def test_torso_detector_upright():
    """Si el atleta mantiene el torso vertical, el detector debe certificar

    que la postura es correcta (has_error = False) y el ángulo es cercano a 0.
    """
    detector = TorsoTiltDetector(max_tilt_deg=40.0)
    landmarks = create_mock_torso_landmarks(lean_forward_meters=0.0)

    has_error, angle = detector.analyze(landmarks)

    assert has_error is False
    assert angle == pytest.approx(0.0, abs=1.0)


def test_torso_detector_excessive_tilt():
    """Si el atleta se inclina excesivamente hacia adelante (45 grados),

    el detector debe activarse inmediatamente (has_error = True).
    """
    detector = TorsoTiltDetector(max_tilt_deg=40.0)
    # 0.5m de altura y 0.5m de avance en Z equivalen a 45 grados de inclinación
    landmarks = create_mock_torso_landmarks(lean_forward_meters=0.5)

    has_error, angle = detector.analyze(landmarks)

    assert has_error is True
    assert angle == pytest.approx(45.0, abs=1.0)


def test_torso_detector_missing_landmarks():
    """Mitigación segura si faltan los hombros o las caderas en el frame."""
    detector = TorsoTiltDetector()
    has_error, angle = detector.analyze(None)
    assert has_error is False
    assert angle == 0.0
