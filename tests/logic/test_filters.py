import numpy as np

from src.logic.filters import OneEuroFilter


def test_filter_reduces_noise():
    """Verifica que el filtro reduce la varianza de una señal ruidosa constante."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = [10.0 + np.random.uniform(-0.5, 0.5) for _ in range(100)]
    filtered = [filt.apply(s, t=i / 30.0) for i, s in enumerate(raw)]
    assert np.var(filtered) < np.var(raw)
    assert np.isclose(np.mean(filtered), 10.0, atol=0.1)


def test_filter_reacts_to_step_change():
    """Verifica que con beta > 0 el filtro reacciona ante un cambio brusco."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    filt.apply(0.0, t=0.0)
    assert filt.apply(100.0, t=0.1) > 20.0


def test_reset_clears_state():
    """Verifica que reset() devuelve el filtro a estado inicial."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    filt.apply(50.0, t=0.0)
    filt.apply(55.0, t=0.1)
    filt.reset()
    result = filt.apply(99.0, t=0.2)
    assert result == 99.0
    assert filt.t_prev == 0.2


def test_non_advancing_timestamp_returns_previous():
    """Verifica que dt <= 0 devuelve el valor previo sin modificar."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    filt.apply(10.0, t=1.0)
    assert filt.apply(99.0, t=1.0) == 10.0
