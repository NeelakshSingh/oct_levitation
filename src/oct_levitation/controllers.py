import numpy as np
import scipy.signal as signal
import control as ct
from time import perf_counter
from scipy.linalg import block_diag

import oct_levitation.common as common
import control_utils.general.filters as filters
import control_utils.general.geometry as geometry
from oct_levitation.msg import PID1DState
from typing import Optional, Union
from geometry_msgs.msg import TransformStamped

import rospy

class ControllerInteface:
    def __init__(self):
        raise NotImplementedError

    def update(self, r, y):
        raise NotImplementedError

class PID1D(ControllerInteface):
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
    
class LQR(ControllerInteface):

    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 dt: Optional[float] = None,
                 discretize: bool = True):
        """
        Args:
            A (np.ndarray): The state matrix.
            B (np.ndarray): The input matrix.
            Q (np.ndarray): The state cost matrix.
            R (np.ndarray): The input cost matrix.
            dt (float): The sampling time for discretization. If None, then continuous LQR is used. 
            discretize (bool): Whether to discretize the system. If yes, then the system is discretized using ZOH.
                               This option will only be used if a time has been specified.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        if dt is not None:
            if discretize:
                Ad, Bd, Cd, Dd, dt = signal.cont2discrete((A, B, np.eye(A.shape[0]), 0), dt=dt, method='zoh')
                self.lqr_out = ct.dlqr(Ad, Bd, Q, R)
            else:
                self.lqr_out = ct.lqr(A, B, Q, R)
        self.K = self.lqr_out[0]

    def update(self, r, y, dt):
        """
        Calculates the control input using the LQR controller.
        Args:
            r (np.ndarray): The reference state.
            y (np.ndarray): The current state.
            dt (float): IGNORED. This is a just simple proportional law. Argument is kept for interface compatibility.
        """
        e = r - y
        u = self.K @ e
        return u

    
class IntegralLQR(ControllerInteface):

    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 Qi: np.ndarray,
                 dt: Optional[float] = 1e-3,
                 discretize: bool = True,
                 windup_lim: float = np.inf,
                 clegg_integrator: bool = False):
        """
        This controller assumes that the matrices provided are in continuous time.
        Because then, the discretized system is obtained after augmenting the 
        state with the error integral.

        Args:
            A (np.ndarray): The state matrix.
            B (np.ndarray): The input matrix.
            Q (np.ndarray): The state cost matrix.
            R (np.ndarray): The input cost matrix.
            Qi (np.ndarray): The integral error cost matrix.
            dt (float): The sampling time for discretization.
            windup_lim (float, optional): The windup limit for the integral error. Defaults to np.inf.
            clegg_integrator (bool, optional): Whether to use the Clegg integrator. Defaults to False.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.discretize = discretize
        
        # Designing a state augmented integral LQR controller
        A_aug = np.block([[A, np.zeros((A.shape[0], A.shape[0]))],
                          [np.eye(A.shape[0]), np.zeros((A.shape[0], A.shape[0]))]])
        B_aug = np.block([[B], [np.zeros((B.shape[0], B.shape[1]))]])
        Q_aug = block_diag(Q, Qi)

        # Performing an exact ZOH discretization of the augmented system
        if self.discretize:
            Ad, Bd, Cd, Dd, dt = signal.cont2discrete((A_aug, B_aug, np.eye(A_aug.shape[0]), 0), dt=dt, method='zoh')
            self.lqr_out = ct.dlqr(Ad, Bd, Q_aug, R)
        else:
            self.lqr_out = ct.lqr(A_aug, B_aug, Q_aug, R)
        self.K = self.lqr_out[0]

        self.windup_lim = windup_lim
        self.clegg_integrator = clegg_integrator
        self.e_integral = np.zeros((A.shape[0], 1))
        self.e_prev = 0

    def update(self, r, y, dt):
        e = r - y
        self.e_integral += e * dt
        if self.clegg_integrator:
            if np.sign(e) != np.sign(self.e_prev):
                self.e_integral = 0
        e_integral = np.clip(self.e_integral, -self.windup_lim, self.windup_lim)
        u = -self.K @ np.vstack([e, e_integral])
        self.e_prev = e
        return u

class Vicon6DOFEulerXYZStateEstimator:

    def __init__(self,
                 initial_orientation: Optional[TransformStamped] = TransformStamped(),
                 initial_velocity: Optional[np.ndarray] = np.zeros(3),
                 initial_angular_velocity: Optional[np.ndarray] = np.zeros(3)) -> None:
        """
        This class is a simple state estimator, made mostly for the purpose of estimating
        the velocities and just have a modular code structure. In the future though this 
        class can house more complicated estimation strategies as required.

        Args:
            initial_orientation (TransformStamped, optional): The initial orientation. Defaults to TransformStamped().
            initial_velocity (np.ndarray, optional): The initial velocity. Defaults to np.zeros(3).
            initial_angular_velocity (np.ndarray, optional): The initial angular velocity. Defaults to np.zeros(3).
        """
        self.orientation = initial_orientation
        self.velocity = initial_velocity
        self.angular_velocity = initial_angular_velocity
        self.last_position = np.array([self.orientation.transform.translation.x,
                                        self.orientation.transform.translation.y,
                                        self.orientation.transform.translation.z])
        self.last_euler = geometry.euler_xyz_from_quaternion(np.array([self.orientation.transform.rotation.x,
                                                                      self.orientation.transform.rotation.y,
                                                                      self.orientation.transform.rotation.z,
                                                                      self.orientation.transform.rotation.w]))        
        
    def update(self, orientation: TransformStamped, dt: float) -> np.ndarray:
        """
        Updates the state estimate.

        Args:
            orientation (TransformStamped): The orientation measurement.
            dt (float): The time step.

        Returns:
            np.ndarray: The 12 element vector containing [position, velocity, euler, angular_velocity].
        """
        position = np.array([orientation.transform.translation.x,
                             orientation.transform.translation.y,
                             orientation.transform.translation.z])
        quaternion = np.array([orientation.transform.rotation.x,
                               orientation.transform.rotation.y,
                               orientation.transform.rotation.z,
                               orientation.transform.rotation.w])
        euler = geometry.euler_xyz_from_quaternion(quaternion)
        velocity = (position - self.last_position) / dt
        euler_rate = (euler - self.last_euler) / dt
        Exyz = geometry.euler_xyz_rate_to_local_angular_velocity_map_matrix(euler)
        angular_velocity = Exyz @ euler_rate
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.last_position = position
        self.last_euler = euler

        return np.concatenate([position, velocity, euler, angular_velocity])
    
    def get_latest_state_estimate(self) -> np.ndarray:
        """
        Returns the latest state estimate.

        Returns:
            np.ndarray: The 12 element vector containing [position, velocity, euler, angular_velocity].
        """
        return np.concatenate([self.last_position, self.velocity, self.last_euler, self.angular_velocity])
    
    def get_latest_yaw_removed_state_estimate(self) -> np.ndarray:
        """
        Returns the latest state estimate with yaw and wz removed.

        Returns:
            np.ndarray: The 10 element vector containing [position, velocity, euler, angular_velocity] with yaw removed.
        """
        return np.concatenate([self.last_position, self.velocity, self.last_euler[:2], self.angular_velocity[:2]])