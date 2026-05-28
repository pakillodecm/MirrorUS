import time

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, x, t=None):
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

    def alpha(self, cutoff, dt):
        return self._alpha(cutoff, dt)
