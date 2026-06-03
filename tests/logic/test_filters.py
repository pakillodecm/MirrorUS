import numpy as np

from src.logic.filters import OneEuroFilter


def test_filter_reduces_noise():
    """Verifica que el filtro reduce la varianza de una señal ruidosa constante.

    Aplica ruido uniforme sobre un valor constante durante 100 muestras
    y comprueba que la señal filtrada tiene menor varianza y una media
    cercana al valor original.
    """
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    constant_value = 10.0
    noise_level = 0.5
    raw_signals = [
        constant_value + np.random.uniform(-noise_level, noise_level)
        for _ in range(100)
    ]
    filtered_signals = [filt.apply(s, t=i / 30.0) for i, s in enumerate(raw_signals)]

    assert np.var(filtered_signals) < np.var(raw_signals)
    assert np.isclose(np.mean(filtered_signals), constant_value, atol=0.1)


def test_filter_latency_on_step_change():
    """Verifica que el filtro reacciona con rapidez ante un cambio brusco de señal.

    Con beta > 0 el filtro aumenta su frecuencia de corte cuando detecta
    velocidad alta, reduciendo el retraso. El valor filtrado tras el salto
    debe superar el 20% del rango para confirmar la reactividad.
    """
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.1)

    filt.apply(0.0, t=0.0)
    val_after_jump = filt.apply(100.0, t=0.1)

    assert val_after_jump > 20.0


def test_filter_reset_clears_state():
    """Verifica que reset() devuelve el filtro a su estado inicial.

    Tras aplicar varias muestras y llamar a reset(), el siguiente valor
    debe devolverse sin filtrar (igual al valor de entrada) y el timestamp
    debe actualizarse al nuevo valor.
    """
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    filt.apply(50.0, t=0.0)
    filt.apply(55.0, t=0.1)
    filt.reset()

    result = filt.apply(99.0, t=0.2)

    assert result == 99.0
    assert filt.t_prev == 0.2


def test_filter_ignores_non_advancing_timestamp():
    """Verifica que el filtro devuelve el valor previo si el timestamp no avanza.

    Un dt <= 0 provocaría división por cero en el cálculo de la derivada,
    por lo que el filtro debe retornar x_prev sin modificación.
    """
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    filt.apply(10.0, t=1.0)

    result = filt.apply(99.0, t=1.0)

    assert result == 10.0
