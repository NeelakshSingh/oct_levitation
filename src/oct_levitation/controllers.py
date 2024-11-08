import numpy as np
import scipy as sci
from time import perf_counter

import oct_levitation.common as common
import oct_levitation.filters as filters

class PID1D:
    def __init__(self, Kp, Ki, Kd,
                 windup_lim: float = np.inf,
                 clegg_integrator: bool = False,
                 error_filter: filters.LiveFilter = None,
                 d_filter: filters.LiveFilter = None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.e_prev = 0
        self.e_integral = 0
        self.__first_call = True
        self.windup_lim = windup_lim
        self.clegg_integrator = clegg_integrator
        self.error_filter = error_filter
        self.d_filter = d_filter

    def update(self, r, y, dt):
        e = r - y
        if self.error_filter:
            e = self.error_filter(e)
        self.e_integral += e * dt
        if self.clegg_integrator:
            if np.sign(e) != np.sign(self.e_integral):
                self.e_integral = 0
        d = (e - self.e_prev) / dt
        if self.d_filter:
            d = self.d_filter(d)
        self.e_integral = np.clip(self.e_integral, -self.windup_lim, self.windup_lim)
        u = self.Kp * e + self.Ki * self.e_integral + self.Kd * d
        self.e_prev = e
        return u