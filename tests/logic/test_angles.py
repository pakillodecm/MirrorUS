import numpy as np
import pytest

from src.logic.angles import calculate_angle


def test_right_angle():
    """Verifica que el ángulo recto (90°) se calcula correctamente."""
    assert (
        pytest.approx(
            calculate_angle(np.array([0, 1]), np.array([0, 0]), np.array([1, 0]))
        )
        == 90.0
    )


def test_straight_line():
    """Verifica que una línea recta produce 180°."""
    assert (
        pytest.approx(
            calculate_angle(np.array([0, 1]), np.array([0, 0]), np.array([0, -1]))
        )
        == 180.0
    )


def test_acute_angle():
    """Verifica que un ángulo agudo (45°) se calcula correctamente."""
    assert (
        pytest.approx(
            calculate_angle(np.array([1, 1]), np.array([0, 0]), np.array([1, 0]))
        )
        == 45.0
    )


def test_obtuse_angle():
    """Verifica que un ángulo obtuso (135°) se calcula correctamente."""
    assert (
        pytest.approx(
            calculate_angle(np.array([-1, 1]), np.array([0, 0]), np.array([1, 0]))
        )
        == 135.0
    )


def test_zero_vector_returns_zero():
    """Verifica que un vector nulo devuelve 0.0 de forma segura."""
    assert calculate_angle(np.array([0, 0]), np.array([0, 0]), np.array([1, 0])) == 0.0
