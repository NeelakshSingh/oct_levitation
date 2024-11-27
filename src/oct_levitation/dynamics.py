import numpy as np
import numpy.typing as np_t
import scipy.signal as sig
import rospy
import control as ct
import casadi as cs

import oct_levitation.common as common

from time import perf_counter
from geometry_msgs.msg import TransformStamped
from typing import Union, Optional, Tuple, Callable

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
        
class WrenchInput6DOFEulerXYZDynamics:

    def __init__(self, d: float, m: float, I_m: np_t.NDArray, g: float = common.Constants.g):
        """
        This class is completely independent of the dipole orientation since we consider the
        forces and torques as control input. The world frame is denoted as {V} and the body
        frame is denoted as {M} (for {V}icon and {M}agnet). The euler convention employed is
        XYZ because then the roll and pitch rates do not depend on the z component of angular
        velocity which becomes uncontrollable for small angles and a diagonal inertia matrix.
        This is because of the fact that the z component of field applied torque will always be
        0 since the torque is a cross product of the dipole moment and the magnetic field. The 
        z axis of the body frame is aligned with the dipole moment vector if its north up; otherwise
        it is anti parallel for a south up dipole. 
        Note the following points regarding the frames
        of reference considered for each vector:
        - The position and velocity vectors are expressed in the world frame.
        - The orientation vector follows the Euler XYZ convention and represents the orientation of the body frame {M} w.r.t world frame {V}.
        - The angular velocity vector is expressed in the body frame.
        - The applied force vector is expressed in the world frame.
        - The applied torque vector is expressed in the body frame. So please design controllers with this in mind and 
          always transform the torque vector the world frame before the field/current allocation step.
        """
        self.d = d
        self.m = m
        self.I_m = cs.SX(I_m)
        self.g = g

        self.__x = cs.SX.sym("x") 
        self.__y = cs.SX.sym("y")
        self.__z = cs.SX.sym("z")
        self.__vx = cs.SX.sym("vx")
        self.__vy = cs.SX.sym("vy")
        self.__vz = cs.SX.sym("vz")
        self.__phi = cs.SX.sym("phi")
        self.__theta = cs.SX.sym("theta")
        self.__psi = cs.SX.sym("psi")
        self.__wx = cs.SX.sym("wx")
        self.__wy = cs.SX.sym("wy")
        self.__wz = cs.SX.sym("wz")
        self.__w_m = cs.vertcat(self.__wx, self.__wy, self.__wz) # Angular velocity vector

        # State vector
        self.__s = cs.vertcat(
            self.__x, self.__y, self.__z, 
            self.__vx, self.__vy, self.__vz, 
            self.__phi, self.__theta, self.__psi, 
            self.__wx, self.__wy, self.__wz
        )  # [x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz].T

        # Input variables
        self.__Fx = cs.SX.sym("Fx")
        self.__Fy = cs.SX.sym("Fy")
        self.__Fz = cs.SX.sym("Fz")
        self.__Tau_m_x = cs.SX.sym("Tau_m_x")
        self.__Tau_m_y = cs.SX.sym("Tau_m_y")

        # Input vector
        self.__u = cs.vertcat(
            self.__Fx, self.__Fy, self.__Fz,
            self.__Tau_m_x, self.__Tau_m_y
        )  # [Fx, Fy, Fz, Tau_m_x, Tau_m_y].T

        self.__F = cs.vertcat(self.__Fx, self.__Fy, self.__Fz)
        self.__Tau_m_tilde = cs.vertcat(self.__Tau_m_x, self.__Tau_m_y)

        # Define f1
        self.__f1 = cs.vertcat(self.__vx, self.__vy, self.__vz)

        # Euler angle to body frame angular velocity map
        self.__E_exyz_inv = cs.SX(3, 3)
        self.__E_exyz_inv[0, 0] = cs.cos(self.__psi) / cs.cos(self.__theta)
        self.__E_exyz_inv[0, 1] = -cs.sin(self.__psi) / cs.cos(self.__theta)
        self.__E_exyz_inv[1, 0] = cs.sin(self.__psi)
        self.__E_exyz_inv[1, 1] = cs.cos(self.__psi)
        self.__E_exyz_inv[2, 0] = -cs.cos(self.__psi) * cs.tan(self.__theta)
        self.__E_exyz_inv[2, 1] = cs.sin(self.__psi) * cs.tan(self.__theta)
        self.__E_exyz_inv[2, 2] = 1

        # Dynamics equations
        self.__f2 = self.g +  self.__F/ m
        self.__f3 = cs.mtimes(self.__E_exyz_inv, cs.vertcat(self.__wx, self.__wy, self.__wz))
        self.__I_m_inv = cs.inv(self.I_m)
        self.__f4 = cs.mtimes(self.__I_m_inv, cs.vertcat(self.__Tau_m_tilde, 0)) \
                    - cs.mtimes(self.__I_m_inv, cs.cross(self.__w_m, cs.mtimes(self.I_m, self.__w_m)))

        self.__f_s_u = cs.vertcat(self.__f1, self.__f2, self.__f3, self.__f4)

        self.__get_linearized_dynamics_impl = cs.Function("get_linearized_system_impl",
                                                          [self.__s, self.__u], 
                                                          [cs.jacobian(self.__f_s_u, self.__s), cs.jacobian(self.__f_s_u, self.__u)],
                                                          ["s", "u"], ["A", "B"])

        self.__get_non_linear_dynamics_impl = cs.Function("get_non_linear_dynamics_impl",
                                                            [self.__s, self.__u], 
                                                            [self.__f_s_u],
                                                            ["s", "u"], ["f_s_u"])
    
    def get_linearized_dynamics(self, s: np_t.ArrayLike, u: np_t.ArrayLike) -> Tuple[np_t.NDArray, np_t.NDArray]:
        A_op, B_op = self.__get_linearized_dynamics_impl(s, u)
        A_op = np.array(A_op).astype(np.float64)
        B_op = np.array(B_op).astype(np.float64)
        return A_op, B_op
    
    def get_non_linear_dynamics_symbolic(self):
        return self.__f_s_u
    
    def eval_non_linear_dynamics(self, s: np_t.ArrayLike, u: np_t.ArrayLike) -> np_t.NDArray:
        f_s_u = self.__get_non_linear_dynamics_impl(s, u)
        return f_s_u
    
    def get_non_linear_dynamics_function(self) -> Callable[[np_t.ArrayLike, np_t.ArrayLike], np_t.NDArray]:
        return self.__get_non_linear_dynamics_impl