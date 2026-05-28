from pathlib import Path

import cv2
import numpy as np
import pytest

from src.logic.pose_detector import PoseDetector


def test_extract_landmarks_real_image():
    assets_dir = Path(__file__).parent.parent / "assets"
    image_path = assets_dir / "squat_check.jpg"

    if not image_path.exists():
        pytest.fail(f"Falta el asset de prueba en: {image_path}")

    detector = PoseDetector(static_image_mode=True)
    frame = cv2.imread(str(image_path))

    result = detector.extract_landmarks(frame)

    assert result.normalized is not None, "Debería devolver landmarks normalizados"
    assert result.world is not None, "Debería devolver world landmarks"

    assert "LEFT_KNEE" in result.normalized
    assert result.world["LEFT_KNEE"][2] != result.normalized["LEFT_KNEE"][2]
    coord_cnt = len(result.normalized["LEFT_KNEE"])
    assert coord_cnt == 4, f"Se esperaban 4 coordenadas pero se obtuvieron {coord_cnt}"


def test_extract_landmarks_no_person():
    detector = PoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.extract_landmarks(frame)

    assert result.normalized is None, "No debería devolver landmarks normalizados"
    assert result.world is None, "No debería devolver world landmarks"
