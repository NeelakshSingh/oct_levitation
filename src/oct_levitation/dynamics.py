import numpy as np
import scipy.signal as sig
import rospy
import control as ct
import tf2_ros

import oct_levitation.common as common

from time import perf_counter
from geometry_msgs.msg import TransformStamped
from typing import Union, Optional

class DynamicalSystemInterface:

    sys = None
    u = None
    t = None

    def __init__(self):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def get_output(self):
        raise NotImplementedError
    
    def update(self, u: np.ndarray):
        raise NotImplementedError

class ZLevitatingMassSystem(DynamicalSystemInterface):

    def __init__(self, 
                 m: float, 
                 b: float,
                 discretize: bool = True,
                 dt: Optional[float] = 1e-3,
                 x0: np.ndarray = np.array([0, 0]),
                 lbounds: float = np.array([-np.inf, -np.inf]),
                 ubounds: float = np.array([np.inf, np.inf])):
        """
        This class assumes that the dipole is north down by default.
        Please specify in the constructor if that's not the case.
        """
        self.m = m
        self.b = b
        A = np.array([[0, 1], [0, -b/m]])
        B = np.array([[0], [1/m]])
        self.discretize = discretize
        if discretize:
            Ad, Bd, Cd, Dd, dt = sig.cont2discrete((A, B, np.eye(2), 0), dt, method='zoh')
            self.A = Ad
            self.B = Bd
        else:
            self.A = A
            self.B = B
        self.x0 = x0
        self.x = x0
        self.lbounds = lbounds
        self.ubounds = ubounds

    def update(self, u: Union[np.ndarray, float], dt: Optional[bool]):
        if self.discretize:
            self.x = (self.A @ self.x.reshape(2,1) + self.B * (u - self.m * common.Constants.g)).flatten()
        else:
            self.x = self.x + dt * (self.A @ self.x.reshape(2,1) + self.B * (u - self.m * common.Constants.g)).flatten()
        if self.x[0] > self.ubounds[0]:
            self.x[0] = self.ubounds[0]
            self.x[1] = 0
        elif self.x[0] < self.lbounds[0]:
            self.x[0] = self.lbounds[0]
            self.x[1] = 0
    
    def reset(self):
        self.x = self.x0

    def get_state(self):
        return self.x
    
    def get_output(self):
        return self.x
        
