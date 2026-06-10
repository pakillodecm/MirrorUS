import numpy as np

from src.logic.valgus_detector import KneeValgusDetector


def create_landmarks(
    hip_w: float, knee_w: float, ankle_w: float, vis: float = 0.9
) -> dict:
    """Genera landmarks con caderas, rodillas y tobillos en posiciones controladas.

    Las caderas se sitúan a y=+0.5, las rodillas en y=0 y los tobillos en y=-0.5,
    formando miembros de longitud unitaria. La separación lateral de cada segmento
    se controla independientemente para simular distintos tipos de sentadilla.
    """
    return {
        "LEFT_HIP": np.array([-hip_w / 2, 0.5, 0.0, vis]),
        "RIGHT_HIP": np.array([hip_w / 2, 0.5, 0.0, vis]),
        "LEFT_KNEE": np.array([-knee_w / 2, 0.0, 0.0, vis]),
        "RIGHT_KNEE": np.array([knee_w / 2, 0.0, 0.0, vis]),
        "LEFT_ANKLE": np.array([-ankle_w / 2, -0.5, 0.0, vis]),
        "RIGHT_ANKLE": np.array([ankle_w / 2, -0.5, 0.0, vis]),
    }


def test_perfect_alignment_no_valgus():
    """Rodillas sobre la línea cadera-tobillo: desviación cero, sin valgo."""
    is_valgus, dev = KneeValgusDetector(threshold=0.08).analyze(
        create_landmarks(0.40, 0.40, 0.40)
    )
    assert is_valgus is False
    assert abs(dev) < 0.01


def test_severe_valgus_detected():
    """Rodillas muy hacia adentro respecto al eje cadera-tobillo: valgo detectado."""
    is_valgus, dev = KneeValgusDetector(threshold=0.08).analyze(
        create_landmarks(0.40, 0.10, 0.50)
    )
    assert is_valgus is True
    assert dev > 0.08


def test_sumo_no_false_positive():
    """Sentadilla sumo con rodillas siguiendo los tobillos: sin falso positivo."""
    # Rodillas y tobillos en proporción constante: desviación perpendicular = 0
    is_valgus, dev = KneeValgusDetector(threshold=0.08).analyze(
        create_landmarks(0.40, 0.60, 0.80)
    )
    assert is_valgus is False
    assert abs(dev) < 0.01


def test_none_landmarks_returns_fallback():
    """Con None se retorna (False, 0.0)."""
    is_valgus, dev = KneeValgusDetector().analyze(None)
    assert is_valgus is False
    assert dev == 0.0


def test_incomplete_landmarks_returns_fallback():
    """Diccionario incompleto retorna (False, 0.0)."""
    is_valgus, dev = KneeValgusDetector().analyze(
        {"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.9])}
    )
    assert is_valgus is False
    assert dev == 0.0


def test_low_visibility_returns_fallback():
    """Visibilidad < 0.5 retorna (False, 0.0)."""
    is_valgus, dev = KneeValgusDetector().analyze(
        create_landmarks(0.40, 0.10, 0.50, vis=0.3)
    )
    assert is_valgus is False
    assert dev == 0.0


def test_short_limb_returns_fallback():
    """Segmento cadera-tobillo menor de 0.1 m retorna (False, 0.0)."""
    lm = {
        "LEFT_HIP": np.array([-0.2, 0.04, 0.0, 0.9]),
        "RIGHT_HIP": np.array([0.2, 0.04, 0.0, 0.9]),
        "LEFT_KNEE": np.array([-0.1, 0.02, 0.0, 0.9]),
        "RIGHT_KNEE": np.array([0.1, 0.02, 0.0, 0.9]),
        "LEFT_ANKLE": np.array([-0.2, 0.0, 0.0, 0.9]),
        "RIGHT_ANKLE": np.array([0.2, 0.0, 0.0, 0.9]),
    }
    is_valgus, dev = KneeValgusDetector().analyze(lm)
    assert is_valgus is False
    assert dev == 0.0
