import numpy as np
import pytest

from src.logic.valgus_detector import KneeValgusDetector


def create_mock_world_landmarks(
    hip_width: float, knee_width: float
) -> dict[str, np.ndarray]:
    """Genera landmarks sintéticos centrados en el origen para el detector de valgo.

    Coloca caderas y rodillas simétricamente en el eje X con visibilidad 0.9,
    suficiente para superar el filtro de confianza del detector.

    Args:
        hip_width: Distancia entre caderas en metros.
        knee_width: Distancia entre rodillas en metros.

    Returns:
        Diccionario de landmarks compatible con KneeValgusDetector.analyze().
    """
    half_hip = hip_width / 2.0
    half_knee = knee_width / 2.0

    return {
        "LEFT_HIP": np.array([-half_hip, 1.0, 0.0, 0.9]),
        "RIGHT_HIP": np.array([half_hip, 1.0, 0.0, 0.9]),
        "LEFT_KNEE": np.array([-half_knee, 0.5, 0.0, 0.9]),
        "RIGHT_KNEE": np.array([half_knee, 0.5, 0.0, 0.9]),
    }


def test_valgus_detector_normal_alignment():
    """Verifica que rodillas más abiertas que las caderas no activan el detector.

    Con caderas a 40cm y rodillas a 44cm el ratio es 1.10, por encima
    del umbral de 0.85, por lo que la postura se considera correcta.
    """
    landmarks = create_mock_world_landmarks(hip_width=0.40, knee_width=0.44)
    detector = KneeValgusDetector(threshold=0.85)
    is_valgus, ratio = detector.analyze(landmarks)

    assert is_valgus is False
    assert ratio == pytest.approx(1.10, rel=1e-2)


def test_valgus_detector_severe_collapse():
    """Verifica que rodillas muy juntas respecto a las caderas activan el detector.

    Con caderas a 40cm y rodillas a 30cm el ratio es 0.75, por debajo
    del umbral de 0.85, por lo que se detecta valgo severo.
    """
    landmarks = create_mock_world_landmarks(hip_width=0.40, knee_width=0.30)
    detector = KneeValgusDetector(threshold=0.85)
    is_valgus, ratio = detector.analyze(landmarks)

    assert is_valgus is True
    assert ratio == pytest.approx(0.75, rel=1e-2)


def test_valgus_detector_missing_landmarks():
    """Verifica que entradas inválidas retornan el fallback seguro (False, 1.0).

    Cubre dos escenarios: entrada nula (sin sujeto detectado) y diccionario
    con claves faltantes (detección parcial de MediaPipe).
    """
    detector = KneeValgusDetector()

    is_valgus, ratio = detector.analyze(None)
    assert is_valgus is False
    assert ratio == 1.0

    incomplete_landmarks = {"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.9])}
    is_valgus, ratio = detector.analyze(incomplete_landmarks)
    assert is_valgus is False
    assert ratio == 1.0


def test_valgus_detector_low_visibility():
    """Verifica que visibilidad baja (<0.5) retorna el fallback seguro."""
    detector = KneeValgusDetector()
    landmarks = create_mock_world_landmarks(hip_width=0.40, knee_width=0.40)
    for key in landmarks:
        landmarks[key][3] = 0.3
    is_valgus, ratio = detector.analyze(landmarks)

    assert is_valgus is False
    assert ratio == 1.0


def test_valgus_detector_zero_hip_distance():
    """Verifica que una distancia de caderas nula retorna el fallback seguro.

    Si ambas caderas están en el mismo punto la distancia es cero y el
    detector debe evitar la división por cero retornando (False, 1.0).
    """
    detector = KneeValgusDetector()
    point = np.array([0.0, 0.0, 0.0, 0.95])
    landmarks = {
        "LEFT_HIP": point,
        "RIGHT_HIP": point,
        "LEFT_KNEE": np.array([-0.2, 0.5, 0.0, 0.95]),
        "RIGHT_KNEE": np.array([0.2, 0.5, 0.0, 0.95]),
    }
    is_valgus, ratio = detector.analyze(landmarks)

    assert is_valgus is False
    assert ratio == 1.0
