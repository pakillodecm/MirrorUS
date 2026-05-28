import numpy as np

from src.logic.filters import OneEuroFilter


def test_filter_reduces_noise():
    """El filtro debe reducir la varianza de una señal ruidosa."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    constant_value = 10.0
    noise_level = 0.5
    raw_signals = [
        constant_value + np.random.uniform(-noise_level, noise_level)
        for _ in range(100)
    ]

    filtered_signals = [filt.apply(s, t=i / 30.0) for i, s in enumerate(raw_signals)]

    fs_var = np.var(filtered_signals)
    rs_var = np.var(raw_signals)

    assert (
        fs_var < rs_var
    ), f"La varianza raw ({rs_var}) debe ser mayor que la filtrada ({fs_var})"

    fs_mean = np.mean(filtered_signals)

    assert np.isclose(
        fs_mean, constant_value, atol=0.1
    ), f"El valor medio ({fs_mean}) debe ser cercano al original ({constant_value})"


def test_filter_latency_on_step_change():
    """El filtro debe reaccionar a cambios bruscos (no tener demasiado lag)."""
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.1)

    filt.apply(0.0, t=0.0)
    val_after_jump = filt.apply(100.0, t=0.1)

    assert (
        val_after_jump > 20.0
    ), f"El valor tras el salto ({val_after_jump}) debe ser mayor que 20.0"
