import cv2
import pytest
import numpy as np
from pathlib import Path
from src.logic.pose_detector import PoseDetector


def test_extract_landmarks_real_image():
    assets_dir = Path(__file__).parent.parent / "assets"
    image_path = assets_dir / "squat_check.jpg"

    if not image_path.exists():
        pytest.fail(f"Falta el asset de prueba en: {image_path}")

    detector = PoseDetector(static_image_mode=True)
    frame = cv2.imread(str(image_path))

    landmarks = detector.extract_landmarks(frame)

    assert landmarks is not None, "El detector no encontró a nadie en la imagen."

    critical_points = [
        "LEFT_HIP",
        "LEFT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_HIP",
        "RIGHT_KNEE",
        "RIGHT_ANKLE",
    ]
    for point in critical_points:
        assert point in landmarks, f"No se detectó el punto clave: {point}"
        assert isinstance(
            landmarks[point], np.ndarray
        ), f"El punto {point} no es un array de NumPy"
        assert (
            len(landmarks[point]) == 3
        ), f"El punto {point} no tiene 3 coordenadas (x, y, z)"


def test_extract_landmarks_no_person():
    detector = PoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    landmarks = detector.extract_landmarks(frame)
    assert landmarks is None
