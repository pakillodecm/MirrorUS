from pathlib import Path

import cv2
import numpy as np
import pytest

from src.logic.pose_detector import PoseDetector


def test_extract_landmarks_real_image():
    """Verifica que se extraen landmarks correctos a partir de una imagen real.

    Comprueba que los landmarks normalizados y los world landmarks se generan,
    que las coordenadas Z difieren entre ambos sistemas (distintas unidades),
    y que cada landmark contiene exactamente 4 componentes (x, y, z, visibilidad).
    """
    assets_dir = Path(__file__).parent.parent / "assets"
    image_path = assets_dir / "squat_check.jpg"

    if not image_path.exists():
        pytest.fail(f"Falta el asset de prueba en: {image_path}")

    detector = PoseDetector(static_image_mode=True)
    frame = cv2.imread(str(image_path))
    result = detector.extract_landmarks(frame)

    assert result.normalized is not None
    assert result.world is not None
    assert "LEFT_KNEE" in result.normalized
    assert result.world["LEFT_KNEE"][2] != result.normalized["LEFT_KNEE"][2]
    assert len(result.normalized["LEFT_KNEE"]) == 4


def test_extract_landmarks_no_person():
    """Verifica que un frame sin persona devuelve un PoseResult vacío."""
    detector = PoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.extract_landmarks(frame)

    assert result.normalized is None
    assert result.world is None


def test_extract_landmarks_none_frame():
    """Verifica que un frame None devuelve un PoseResult vacío de forma segura."""
    detector = PoseDetector()
    result = detector.extract_landmarks(None)

    assert result.normalized is None
    assert result.world is None
    assert result.raw is None


def test_reset_filters_clears_state():
    """Verifica que reset_filters() vacía el diccionario interno de filtros.

    Simula que el detector ha procesado landmarks (creando filtros internos)
    y comprueba que tras el reset el diccionario queda completamente vacío.
    """
    detector = PoseDetector()
    dummy_data = {"LEFT_KNEE": np.array([0.1, 0.2, 0.3, 0.9])}
    detector._filter_dict(dummy_data, detector.filters_world)

    assert len(detector.filters_world) > 0

    detector.reset_filters()

    assert len(detector.filters_world) == 0


def test_draw_landmarks_no_results():
    """Verifica que draw_landmarks no lanza excepción si results_raw es None."""
    detector = PoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.draw_landmarks(frame, None)


def test_draw_landmarks_with_results():
    """Verifica que draw_landmarks se ejecuta correctamente con resultados reales.

    Usa la imagen de prueba para obtener un PoseResult con raw válido
    y comprueba que el método de dibujo no lanza ninguna excepción.
    """
    assets_dir = Path(__file__).parent.parent / "assets"
    image_path = assets_dir / "squat_check.jpg"

    if not image_path.exists():
        pytest.skip("Asset de prueba no disponible")

    detector = PoseDetector(static_image_mode=True)
    frame = cv2.imread(str(image_path))
    result = detector.extract_landmarks(frame)

    assert result.raw is not None, "La imagen de prueba debe detectar una persona"
    detector.draw_landmarks(frame, result.raw)


def test_filter_dict_none_input():
    """Verifica que _filter_dict retorna None si el diccionario de entrada es None."""
    detector = PoseDetector()
    result = detector._filter_dict(None, detector.filters_world)

    assert result is None
