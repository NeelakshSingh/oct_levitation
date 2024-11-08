import numpy as np
import scipy as sci
from time import perf_counter

import oct_levitation.common as common

class PID1D:
    def __init__(self, Kp, Ki, Kd,
                 windup_lim: float = np.inf,
                 clegg_integrator: bool = False):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.e_prev = 0
        self.e_integral = 0
        self.__first_call = True
        self.windup_lim = windup_lim
        self.clegg_integrator = clegg_integrator

    def update(self, r, y, dt):
        e = r - y
        self.e_integral += e * dt
        if self.clegg_integrator:
            if np.sign(e) != np.sign(self.e_integral):
                self.e_integral = 0
        self.e_integral = np.clip(self.e_integral, -self.windup_lim, self.windup_lim)
        u = self.Kp * e + self.Ki * self.e_integral + self.Kd * (e - self.e_prev) / dt
        self.e_prev = e
        return u