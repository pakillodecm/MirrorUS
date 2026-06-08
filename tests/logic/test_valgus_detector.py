import numpy as np
import pytest

from src.logic.valgus_detector import KneeValgusDetector


def create_landmarks(hip_w: float, knee_w: float) -> dict:
    """Genera landmarks centrados en el origen con anchuras controladas."""
    return {
        "LEFT_HIP": np.array([-hip_w / 2, 1.0, 0.0, 0.9]),
        "RIGHT_HIP": np.array([hip_w / 2, 1.0, 0.0, 0.9]),
        "LEFT_KNEE": np.array([-knee_w / 2, 0.5, 0.0, 0.9]),
        "RIGHT_KNEE": np.array([knee_w / 2, 0.5, 0.0, 0.9]),
    }


def test_normal_alignment_no_valgus():
    """Verifica que rodillas más abiertas que caderas no activan el detector."""
    is_valgus, ratio = KneeValgusDetector(threshold=0.85).analyze(
        create_landmarks(0.40, 0.44)
    )
    assert is_valgus is False
    assert ratio == pytest.approx(1.10, rel=1e-2)


def test_severe_collapse_detected():
    """Verifica que rodillas muy juntas respecto a caderas activan el detector."""
    is_valgus, ratio = KneeValgusDetector(threshold=0.85).analyze(
        create_landmarks(0.40, 0.30)
    )
    assert is_valgus is True
    assert ratio == pytest.approx(0.75, rel=1e-2)


def test_none_and_incomplete_return_fallback():
    """Verifica que None y diccionario incompleto retornan (False, 1.0)."""
    detector = KneeValgusDetector()
    is_valgus, ratio = detector.analyze(None)
    assert is_valgus is False and ratio == 1.0

    is_valgus, ratio = detector.analyze({"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.9])})
    assert is_valgus is False and ratio == 1.0


def test_low_visibility_returns_fallback():
    """Verifica que visibilidad < 0.5 retorna (False, 1.0)."""
    landmarks = create_landmarks(0.40, 0.40)
    for key in landmarks:
        landmarks[key][3] = 0.3
    is_valgus, ratio = KneeValgusDetector().analyze(landmarks)
    assert is_valgus is False and ratio == 1.0


def test_zero_hip_distance_returns_fallback():
    """Verifica que distancia de caderas nula no provoca división por cero."""
    point = np.array([0.0, 0.0, 0.0, 0.95])
    is_valgus, ratio = KneeValgusDetector().analyze(
        {
            "LEFT_HIP": point,
            "RIGHT_HIP": point,
            "LEFT_KNEE": np.array([-0.2, 0.5, 0.0, 0.95]),
            "RIGHT_KNEE": np.array([0.2, 0.5, 0.0, 0.95]),
        }
    )
    assert is_valgus is False and ratio == 1.0
