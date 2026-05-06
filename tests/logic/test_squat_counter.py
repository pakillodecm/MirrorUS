import numpy as np
from src.logic.squat_counter import SquatCounter


def create_mock_landmarks(target_angle_deg: float, vis: float = 0.9):
    ankle = np.array([0.0, 0.0, 0.0, vis])
    knee = np.array([0.0, 0.5, 0.0, vis])

    rad = np.deg2rad(target_angle_deg)
    hip_x = 0.5 * np.sin(np.pi - rad)
    hip_y = 0.5 + 0.5 * np.cos(np.pi - rad)
    hip = np.array([hip_x, hip_y, 0.0, vis])

    return {
        "LEFT_HIP": hip,
        "LEFT_KNEE": knee,
        "LEFT_ANKLE": ankle,
        "RIGHT_HIP": hip,
        "RIGHT_KNEE": knee,
        "RIGHT_ANKLE": ankle,
    }


def test_full_squat_cycle():
    """Test de repetición perfecta."""
    counter = SquatCounter(down_threshold=90, up_threshold=160, hysteresis=5)

    sequence = [180, 120, 80, 130, 170]
    counts = []
    for a in sequence:
        res = counter.update(create_mock_landmarks(a))
        counts.append(res)

    assert (
        counter.count == 1
    ), f"Se esperaba 1 repetición pero se obtuvieron {counter.count}"


def test_partial_squat():
    """Test de sentadilla parcial."""
    counter = SquatCounter(down_threshold=90, up_threshold=160, hysteresis=5)

    sequence = [180, 140, 100, 130, 170]
    for a in sequence:
        counter.update(create_mock_landmarks(a))

    assert (
        counter.count == 0
    ), f"Se esperaban 0 repeticiones pero se obtuvieron {counter.count}"


def test_hysteresis_noise():
    """Test de ruido en el umbral crítico."""
    counter = SquatCounter(down_threshold=90, hysteresis=10)

    counter.update(create_mock_landmarks(89))
    assert counter.state == 2, f"Se esperaba estado 2 pero se obtuvo {counter.state}"

    counter.update(create_mock_landmarks(95))
    assert counter.state == 2, f"Se esperaba estado 2 pero se obtuvo {counter.state}"

    counter.update(create_mock_landmarks(105))
    assert counter.state == 3, f"Se esperaba estado 3 pero se obtuvo {counter.state}"


def test_ghost_persistence():
    """Test de pérdida temporal de señal (Sujeto Fantasma)."""
    counter = SquatCounter(persistence_frames=5)

    counter.update(create_mock_landmarks(80))
    assert counter.state == 2, f"Se esperaba estado 2 pero se obtuvo {counter.state}"

    for _ in range(3):
        counter.update(None)
    assert counter.state == 2, f"Se esperaba estado 2 pero se obtuvo {counter.state}"

    for _ in range(10):
        counter.update(None)
    assert counter.state == 0, f"Se esperaba estado 0 pero se obtuvo {counter.state}"


def test_weighted_visibility():
    """Test de visibilidad ponderada."""
    counter = SquatCounter(down_threshold=90)

    lm = create_mock_landmarks(110, vis=0.95)
    right_data = create_mock_landmarks(160, vis=0.25)
    lm["RIGHT_KNEE"] = right_data["RIGHT_KNEE"]
    lm["RIGHT_ANKLE"] = right_data["RIGHT_ANKLE"]

    counter.update(lm)
    assert counter.state == 1, f"Se esperaba estado 1 pero se obtuvo {counter.state}"
