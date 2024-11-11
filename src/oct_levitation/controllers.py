import numpy as np
import scipy as sci
from time import perf_counter

import oct_levitation.common as common
import oct_levitation.filters as filters
from oct_levitation.msg import PID1DState

import rospy

class PID1D:
    def __init__(self, Kp, Ki, Kd,
                 windup_lim: float = np.inf,
                 clegg_integrator: bool = False,
                 error_filter: filters.LiveFilter = None,
                 d_filter: filters.LiveFilter = None,
                 publish_states: bool = False,
                 state_pub_topic: str = "/pid1d_states"):
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
        self.publish_states = publish_states
        if publish_states:
            self.state_pub = rospy.Publisher(state_pub_topic, PID1DState, queue_size=10)

    def update(self, r, y, dt):
        state_msg = PID1DState()
        e = r - y
        if self.error_filter:
            e = self.error_filter(e)
        self.e_integral += e * dt
        if self.clegg_integrator:
            if np.sign(e) != np.sign(self.e_integral):
                self.e_integral = 0
                state_msg.clegg_triggered = True
        d = (e - self.e_prev) / dt
        if self.d_filter:
            d = self.d_filter(d)
        if self.e_integral > self.windup_lim or self.e_integral < -self.windup_lim:
            state_msg.windup_triggered = True
            self.e_integral = 0
        u = self.Kp * e + self.Ki * self.e_integral + self.Kd * d
        self.e_prev = e
        if self.publish_states:
            state_msg.Kp = self.Kp
            state_msg.Ki = self.Ki
            state_msg.Kd = self.Kd
            state_msg.error = e
            state_msg.error_integral = self.e_integral
            state_msg.error_dot = d
            state_msg.control_input = u
            state_msg.header.stamp = rospy.Time.now()
            self.state_pub.publish(state_msg)
        return u