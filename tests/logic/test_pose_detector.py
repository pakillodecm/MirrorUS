from pathlib import Path

import cv2
import numpy as np
import pytest

from src.logic.pose_detector import PoseDetector


def test_extract_landmarks_real_image():
    """Verifica que se extraen landmarks world correctos a partir de una imagen real.

    Comprueba que world contiene la clave LEFT_KNEE con 4 componentes
    (x, y, z, visibilidad) y que raw está disponible para el dibujo.
    """
    image_path = Path(__file__).parent.parent / "assets" / "squat_check.jpg"
    if not image_path.exists():
        pytest.fail(f"Falta el asset de prueba en: {image_path}")

    detector = PoseDetector(static_image_mode=True)
    frame = cv2.imread(str(image_path))
    result = detector.extract_landmarks(frame)

    assert result.world is not None
    assert "LEFT_KNEE" in result.world
    assert len(result.world["LEFT_KNEE"]) == 4
    assert result.raw is not None


def test_extract_landmarks_no_person():
    """Verifica que un frame sin persona devuelve un PoseResult vacío."""
    detector = PoseDetector()
    result = detector.extract_landmarks(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result.world is None
    assert result.raw is None


def test_extract_landmarks_none_frame():
    """Verifica que un frame None devuelve un PoseResult vacío de forma segura."""
    result = PoseDetector().extract_landmarks(None)

    assert result.world is None
    assert result.raw is None


def test_reset_filters_clears_state():
    """Verifica que reset_filters() vacía el diccionario interno de filtros."""
    detector = PoseDetector()
    detector._filter_dict(
        {"LEFT_KNEE": np.array([0.1, 0.2, 0.3, 0.9])},
        detector.filters_world,
    )
    assert len(detector.filters_world) > 0

    detector.reset_filters()

    assert len(detector.filters_world) == 0


def test_filter_dict_none_input():
    """Verifica que _filter_dict retorna None si la entrada es None."""
    assert PoseDetector()._filter_dict(None, {}) is None
