import pytest
import numpy as np
from src.logic.angles import calculate_angle


def test_calculate_right_angle():
    """Prueba un ángulo de 90 grados exactos."""
    a = np.array([0, 1])
    b = np.array([0, 0])
    c = np.array([1, 0])

    expected = 90.0
    assert pytest.approx(calculate_angle(a, b, c)) == expected


def test_calculate_straight_line():
    """Prueba un ángulo de 180 grados (línea recta)."""
    a = np.array([0, 1])
    b = np.array([0, 0])
    c = np.array([0, -1])

    expected = 180.0
    assert pytest.approx(calculate_angle(a, b, c)) == expected


def test_calculate_acute_angle():
    """Prueba un ángulo de 45 grados."""
    a = np.array([1, 1])
    b = np.array([0, 0])
    c = np.array([1, 0])

    expected = 45.0
    assert pytest.approx(calculate_angle(a, b, c)) == expected


def test_calculate_obtuse_angle():
    """Prueba un ángulo de 135 grados."""
    a = np.array([-1, 1])
    b = np.array([0, 0])
    c = np.array([1, 0])

    expected = 135.0
    assert pytest.approx(calculate_angle(a, b, c)) == expected
