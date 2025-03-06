import numpy as np
import numpy.typing as np_t
import scipy.signal as sig
import rospy
import control as ct
import casadi as cs

import oct_levitation.common as common
import oct_levitation.geometry as geometry

from time import perf_counter
from geometry_msgs.msg import TransformStamped
from typing import Union, Optional, Tuple, Callable

EPSILON_TOLERANCE = 1e-14 # High precision tolerance

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
    
    def simulate_step(self, s: np.ndarray, u: np.ndarray, dt: float):
        raise NotImplementedError
    
    @property
    def num_states(self):
        raise NotImplementedError
    
    @property
    def num_inputs(self):
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
    
class NonLinearDynamicalSystem(DynamicalSystemInterface):
    def get_non_linear_dynamics_symbolic(self):
        raise NotImplementedError
    
    def eval_non_linear_dynamics(self, s: np_t.ArrayLike, u: np_t.ArrayLike) -> np_t.NDArray:
        raise NotImplementedError
    
    def get_non_linear_dynamics_function(self) -> Callable[[np_t.ArrayLike, np_t.ArrayLike], np_t.NDArray]:
        raise NotImplementedError
    
class LinearizableNonLinearDynamicalSystem(DynamicalSystemInterface):
    
    def get_linearized_dynamics(self, s: np_t.ArrayLike, u: np_t.ArrayLike) -> Tuple[np_t.NDArray, np_t.NDArray]:
        raise NotImplementedError

    def simulate_step_linearized(self, s: np_t.ArrayLike, u: np_t.ArrayLike,
                                 s_op: np_t.ArrayLike, u_op: np_t.ArrayLike, dt: float) -> np_t.NDArray:
        raise NotImplementedError
    
    def calculate_steady_state_input(self, s: np_t.ArrayLike) -> np_t.NDArray:
        raise NotImplementedError

        
class WrenchInput6DOFDipoleEulerXYZDynamics(LinearizableNonLinearDynamicalSystem):

    __num_states: int = 12
    __num_inputs: int = 5

    def __init__(self, m: float, I_m: np_t.NDArray):
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

        Note on gravity:
        This dynamics model does NOT include the explicit gravity term in both force and torque equations
        explicity for the sake of simplicity. This is because incorporating it in the orientation dynamics
        requires the transform of the gravity vector to the body frame which complicates the dynamics.
        Therefore, please assume that effect of gravity as the steady state control input at all the times.        
        Use the relevant methods from the mechanical module to get the forces and torques due to gravity
        for gravity compensation.
        
        Parameters
        ----------
            m (float) : The mass of the rigid body in kg.
            I_m (np_t.NDArray) : 3x3 numpy array The inertia matrix of the rigid body in the body frame.
        """
        self.m = m
        self.I_m = cs.SX(I_m)

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
        self.__Tau_m_z = cs.SX.sym("Tau_m_z")

        # Input vector
        self.__u = cs.vertcat(
            self.__Fx, self.__Fy, self.__Fz,
            self.__Tau_m_x, self.__Tau_m_y, self.__Tau_m_z
        )  # [Fx, Fy, Fz, Tau_m_x, Tau_m_y].T

        self.__F = cs.vertcat(self.__Fx, self.__Fy, self.__Fz)
        self.__Tau_m_tilde = cs.vertcat(self.__Tau_m_x, self.__Tau_m_y, self.__Tau_m_z)

        # Define f1
        self.__f1 = cs.vertcat(self.__vx, self.__vy, self.__vz)

        # Euler angle to body frame angular velocity map
        self.__E_exyz_inv = cs.SX(3, 3)
        self.__E_exyz_inv[0, 0] = cs.cos(self.__psi) / (cs.cos(self.__theta) + EPSILON_TOLERANCE)
        self.__E_exyz_inv[0, 1] = -cs.sin(self.__psi) / (cs.cos(self.__theta) + EPSILON_TOLERANCE)
        self.__E_exyz_inv[1, 0] = cs.sin(self.__psi)
        self.__E_exyz_inv[1, 1] = cs.cos(self.__psi)
        self.__E_exyz_inv[2, 0] = -cs.cos(self.__psi) * cs.tan(self.__theta)
        self.__E_exyz_inv[2, 1] = cs.sin(self.__psi) * cs.tan(self.__theta)
        self.__E_exyz_inv[2, 2] = 1

        # Dynamics equations
        self.__f2 = self.__F/ m
        self.__f3 = cs.mtimes(self.__E_exyz_inv, cs.vertcat(self.__wx, self.__wy, self.__wz))
        self.__I_m_inv = cs.inv(self.I_m)
        self.__f4 = cs.mtimes(self.__I_m_inv, self.__Tau_m_tilde) \
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
    
    @property
    def num_states(self):
        return self.__num_states
    
    @property
    def num_inputs(self):
        return self.__num_inputs

    def get_linearized_dynamics(self, s: np_t.ArrayLike, u: np_t.ArrayLike) -> Tuple[np_t.NDArray, np_t.NDArray]:
        """
        This function returns the linearized dynamics matrices A and B for the given state and input.
        Note that a default additive steady state gravity term is assumed in the input force vector
        since it is not included in the dynamics model.

        Parameters
        ----------
            s (np_t.ArrayLike) : The state vector.
            u (np_t.ArrayLike) : The input vector.
        
        Returns
        -------
            A_op (np_t.NDArray) : The linearized dynamics matrix A.
            B_op (np_t.NDArray) : The input matrix B.
        """
        A_op, B_op = self.__get_linearized_dynamics_impl(s, u)
        A_op = np.array(A_op).astype(np.float64)
        B_op = np.array(B_op).astype(np.float64)
        return A_op, B_op
    
    def get_non_linear_dynamics_symbolic(self):
        return self.__f_s_u
    
    def eval_non_linear_dynamics(self, s: np_t.ArrayLike, u: np_t.ArrayLike) -> np_t.NDArray:
        f_s_u = self.__get_non_linear_dynamics_impl(s, u)
        f_s_u = np.array(f_s_u).astype(np.float64).flatten()
        return f_s_u
    
    def get_non_linear_dynamics_function(self) -> Callable[[np_t.ArrayLike, np_t.ArrayLike], np_t.NDArray]:
        return self.eval_non_linear_dynamics
    
    def isolate_roll_pitch_dynamics(self, A: np_t.ArrayLike, B: np_t.ArrayLike) -> Tuple[np_t.ArrayLike, np_t.ArrayLike]:
        """
        This function isolates the roll and pitch dynamics from the linearized dynamics matrix A and input matrix B.

        Parameters
        ----------
            A (np_t.ArrayLike) : The linearized dynamics matrix.
            B (np_t.ArrayLike) : The input matrix.

        Returns
        -------
            A_rp (np_t.ArrayLike) : The 4x4 roll and pitch dynamics matrix with wx and wy.
            B_rp (np_t.ArrayLike) : The 4x2 input matrix isolated for Tau_m_x and Tau_m_y.
        """
        A_rp = np.block([[A[6:8, 6:8], A[6:8, 9:11]], 
                         [A[9:11, 6:8], A[9:11, 9:11]]])
        B_rp = np.block([[B[6:8 , 3:]], 
                         [B[9:11, 3:]]])
        return A_rp, B_rp
    
    def remove_yaw_dynamics(self, A: np_t.ArrayLike, B:np_t.ArrayLike) -> Tuple[np_t.ArrayLike, np_t.ArrayLike]:
        """
        This function removes the yaw dynamics and the z angular velocity dynamics from the linearized dynamics
        matrices A and B. This is useful for designing a controller without the uncontrollable at aligned
        orientations.
        """
        A_no_yaw = np.delete(A, [8, 11], axis=0)
        A_no_yaw = np.delete(A_no_yaw, [8, 11], axis=1)
        B_no_yaw = np.delete(B, [8, 11], axis=0)
        return A_no_yaw, B_no_yaw
    
    def simulate_step(self, s: np_t.NDArray, u: np_t.NDArray, dt: float) -> np_t.NDArray:
        """
        This function simulates the dynamics for one time step using the given state and input.
        It will fix the euler angles to be within the range of -pi to pi. Simple first order
        integration is used.

        Parameters
        ----------
            s (np_t.NDArray) : The state vector.
            u (np_t.NDArray) : The input vector.
            dt (float) : The time step duration.
        
        Returns
        -------
            s_next (np_t.NDArray) : The next state vector.
        """
        s_next = s + self.eval_non_linear_dynamics(s, u)*dt
        s_next[6] = geometry.angle_residual(s_next[6], 0)
        s_next[7] = geometry.angle_residual(s_next[7], 0)
        s_next[8] = geometry.angle_residual(s_next[8], 0)
        return s_next
    
    def simulate_step_linearized(self, s: np_t.NDArray, u: np_t.NDArray,
                                s_op: np_t.ArrayLike, u_op: np_t.ArrayLike, dt: float) -> np_t.NDArray:
        """
        This function simulates the dynamics for one time step using the given state and input
        and the linearized system matrices, linearized around the given state and input.
        It will fix the euler angles to be within the range of -pi to pi. Simple first order
        integration is used.

        Parameters
        ----------
            s (np_t.NDArray) : The state vector.
            u (np_t.NDArray) : The input vector.
            s_op (np_t.ArrayLike) : The operating point state vector for linearization.
            u_op (np_t.ArrayLike) : The operating point input vector for linearization.
            dt (float) : The time step duration.

        Returns
        -------
            s_next (np_t.NDArray) : The next state vector.
        """
        A, B = self.get_linearized_dynamics(s_op, u_op)
        delta_s = s - s_op
        delta_u = u - u_op
        delta_s_next = delta_s + (A @ delta_s + B @ delta_u)
        s_next = s_op + delta_s_next
        s_next[6] = geometry.angle_residual(s_next[6], 0)
        s_next[7] = geometry.angle_residual(s_next[7], 0)
        s_next[8] = geometry.angle_residual(s_next[8], 0)
        return s_next
    
    def calculate_steady_state_input(self, s: np_t.NDArray) -> np_t.NDArray:
        """
        Calculates the steady state input to maintain the given state vector.

        Parameters
        ----------
            s (np_t.ArrayLike) : The state vector.
        
        Returns
        -------
            u_op (np_t.ArrayLike) : The steady state input vector.
        """
        