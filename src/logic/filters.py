import time

import numpy as np


class OneEuroFilter:
    """Filtro One Euro para suavizado adaptativo de señales en tiempo real.

    Reduce el ruido cuando la señal está quieta y minimiza el retraso
    cuando la señal cambia rápidamente, ajustando dinámicamente la
    frecuencia de corte según la velocidad de la señal.

    Referencia: Géry Casiez et al., "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.
    https://doi.org/10.1145/2207676.2208639

    Args:
        min_cutoff: Frecuencia de corte mínima en Hz. Valores bajos
            reducen más el ruido en reposo (ej. 0.1-2.0).
        beta: Coeficiente de velocidad. Valores altos reducen el retraso
            en movimientos rápidos (ej. 0.001-0.1).
        d_cutoff: Frecuencia de corte para la derivada. Normalmente 1.0.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        """Calcula el factor de suavizado alpha para una frecuencia de corte dada."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def alpha(self, cutoff, dt):
        return self._alpha(cutoff, dt)

    def reset(self) -> None:
        """Reinicia el estado interno del filtro para iniciar una sesión limpia."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def apply(self, x, t=None):
        """Aplica el filtro a un nuevo valor de la señal.

        Args:
            x: Valor actual de la señal (escalar o array de NumPy).
            t: Marca de tiempo en segundos. Si es None, usa el reloj del sistema.

        Returns:
            Valor filtrado con la misma forma que la entrada.
        """
        t = t if t is not None else time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + self.alpha(self.d_cutoff, dt) * (dx - self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(edx)

        x_filtered = self.x_prev + self.alpha(cutoff, dt) * (x - self.x_prev)

        self.x_prev = x_filtered
        self.dx_prev = edx
        self.t_prev = t

        return x_filtered
