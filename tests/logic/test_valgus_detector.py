import numpy as np
import pytest

from src.logic.valgus_detector import KneeValgusDetector


def create_mock_world_landmarks(
    hip_width: float, knee_width: float
) -> dict[str, np.ndarray]:
    """Genera un diccionario dummy de world landmarks en metros reales.

    El eje X mide izquierda/derecha. Centramos el esqueleto en X=0.0.
    La visibilidad se fija en 0.9 para pasar los filtros de confianza.
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
    """Si la distancia de las rodillas es igual o mayor a la de las caderas,

    el detector debe certificar que la postura es correcta (is_valgus = False).
    """
    # Caderas separadas 40cm, rodillas abiertas a 44cm (atleta abriendo rodillas)
    landmarks = create_mock_world_landmarks(hip_width=0.40, knee_width=0.44)
    detector = KneeValgusDetector(threshold=0.85)

    is_valgus, ratio = detector.analyze(landmarks)

    assert is_valgus is False
    assert ratio == pytest.approx(1.10, rel=1e-2)


def test_valgus_detector_severe_collapse():
    """Si la distancia de las rodillas se estrecha por debajo del umbral del 85%,

    el detector debe activarse inmediatamente (is_valgus = True).
    """
    # Caderas separadas 40cm, rodillas colapsadas a 30cm (Ratio = 0.75)
    landmarks = create_mock_world_landmarks(hip_width=0.40, knee_width=0.30)
    detector = KneeValgusDetector(threshold=0.85)

    is_valgus, ratio = detector.analyze(landmarks)

    assert is_valgus is True
    assert ratio == pytest.approx(0.75, rel=1e-2)


def test_valgus_detector_missing_landmarks():
    """Si el diccionario de landmarks está incompleto o falta visibilidad,

    el sistema debe abortar de forma segura sin lanzar una excepción de ejecución.
    """
    detector = KneeValgusDetector()

    # Escenario 1: Entrada nula (cámara tapada o sin sujeto)
    is_valgus, ratio = detector.analyze(None)
    assert is_valgus is False
    assert ratio == 1.0

    # Escenario 2: Faltan claves críticas en el diccionario de MediaPipe
    incomplete_landmarks = {"LEFT_HIP": np.array([0.0, 0.0, 0.0, 0.9])}
    is_valgus, ratio = detector.analyze(incomplete_landmarks)
    assert is_valgus is False
    assert ratio == 1.0
