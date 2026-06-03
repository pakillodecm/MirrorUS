import numpy as np
import pytest

from src.logic.angles import calculate_angle


def test_calculate_right_angle():
    """Verifica que el ángulo recto (90°) se calcula correctamente."""
    a = np.array([0, 1])
    b = np.array([0, 0])
    c = np.array([1, 0])

    assert pytest.approx(calculate_angle(a, b, c)) == 90.0


def test_calculate_straight_line():
    """Verifica que una línea recta produce un ángulo de 180°."""
    a = np.array([0, 1])
    b = np.array([0, 0])
    c = np.array([0, -1])

    assert pytest.approx(calculate_angle(a, b, c)) == 180.0


def test_calculate_acute_angle():
    """Verifica que un ángulo agudo (45°) se calcula correctamente."""
    a = np.array([1, 1])
    b = np.array([0, 0])
    c = np.array([1, 0])

    assert pytest.approx(calculate_angle(a, b, c)) == 45.0


def test_calculate_obtuse_angle():
    """Verifica que un ángulo obtuso (135°) se calcula correctamente."""
    a = np.array([-1, 1])
    b = np.array([0, 0])
    c = np.array([1, 0])

    assert pytest.approx(calculate_angle(a, b, c)) == 135.0


def test_calculate_angle_zero_vector():
    """Verifica que un vector nulo (dos puntos iguales) devuelve 0.0 de forma segura."""
    a = np.array([0, 0])
    b = np.array([0, 0])
    c = np.array([1, 0])

    assert calculate_angle(a, b, c) == 0.0
