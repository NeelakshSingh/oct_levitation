import numpy as np
import numpy.typing as np_t
import numba

import oct_levitation.geometry as geometry

from typing import Union, Iterable

########################################
# Different solutions for allocation
########################################

def solve_tikhonov_regularization(A: np_t.ArrayLike, b: np_t.ArrayLike, alpha: float) -> np_t.ArrayLike:
    """
    This function solves the Tikhonov regularized least squares problem.

    Parameters:
        A (np_t.ArrayLike): The matrix A in the least squares problem.
        b (np_t.ArrayLike): The vector b in the least squares problem.
        alpha (float): The regularization parameter.

    Returns:
        np_t.ArrayLike: The solution to the least squares problem.
    """
    return np.linalg.solve(A.T @ A + alpha*np.eye(A.shape[1]), A.T @ b)

########################################
# Numerical integration of orientation
########################################
@numba.jit(nopython=True)
def rodrigues_skew_symmetric_expm(vec: np_t.NDArray) -> np_t.NDArray:
    """
    This function takes a vector as input and then computes the matrix exponential
    of its skew symmetric matrix using rodrigues formula. This formula is a consequence
    of the fact that a skew symmetric matrix belongs to the tangent space of SO(3).
    """
    alpha = (np.linalg.norm(vec) + 1e-16)
    vec_hat = vec/alpha
    skmat = np.array([[0, -vec_hat[2], vec_hat[1]],
                     [vec_hat[2], 0, -vec_hat[0]],
                     [-vec_hat[1], vec_hat[0], 0]])
    exp_skmat = np.eye(3) + np.sin(alpha)*skmat + (1 - np.cos(alpha))*(skmat @ skmat)
    return exp_skmat

@numba.jit(nopython=True)
def integrate_R_omega_constant_torque(R: np_t.NDArray, omega: np_t.NDArray, torque_world: np_t.NDArray, I: np_t.NDArray, dt: float,
                                      damping: np_t.NDArray = np.zeros(3)) -> np_t.NDArray:
    """
    This function integrates the orientation of a rigid body with constant angular velocity and torque.
    Geometric integration over SO(3) using the exponential map works for constant angular velocity but not so well under
    angular acceleration. I found a good alternative which assumes constant angular acceleration and uses magnus expansion 
    with rotation matrix kinematics to integrate the orientation. 
    Source1: https://cwzx.wordpress.com/2013/12/16/numerical-integration-for-rotational-dynamics/
    Source2.1: https://github.com/stephane-caron/pymanoid/blob/master/pymanoid/transformations.py#L135
    Source2.2: https://scaron.info/doc/pymanoid/forward-kinematics.html

    Parameters:
        R (np_t.NDArray): The rotation matrix at the current time step.
        omega (np_t.NDArray): The angular velocity at the current time step (in body fixed frame).
        torque_world (np_t.NDArray): The torque at the current time step, applied in ZOH fashion (given in the world frame).
        I (np_t.NDArray): The inertia tensor of the rigid body in body fixed frame.
        dt (float): The time step.
        damping (np_t.NDArray): The damping coefficients along the three axes. Not normalized by inertia (given in body fixed frame).

    Returns:
        (R, omega) Tuple(np_t.NDArray, np_t.NDArray, ): The rotation matrix and body fixed angular velocities at the next time step.
    """
    # First we convert angvel and I to world frame.
    omega_world = R @ omega
    I_world = R @ I @ R.T

    # This computation is not going to be exact since the inertia matrix will actually vary in time.
    # But we are assuming that the inertia matrix is constant for the time step.
    alpha = np.linalg.solve(I_world, torque_world - np.cross(omega_world, I_world @ omega_world) - np.multiply(damping, omega_world))

    omega_new = omega_world + alpha*dt

    # Now we use magnus expansion
    # NOTE: It might actually be simpler to get the magnus expansion for the body fixed frame
    # dynamics and just directly integrate the body fixed frame dynamics.
    Omega1 = 0.5*(omega + omega_new)*dt
    Omega2 = (1/12)*(np.cross(omega_new, omega))*np.power(dt, 2)
    Omega3 = (1/240)*(np.cross( alpha, np.cross(alpha, omega) ))*np.power(dt, 5)

    Omega = Omega1 + Omega2 + Omega3
    R_new = rodrigues_skew_symmetric_expm(Omega) @ R

    omega_new_body = R_new.T @ omega_new

    return R_new, omega_new_body

@numba.jit(nopython=True)
def integrate_linear_dynamics_constant_force_undamped(p: np_t.NDArray, v: np_t.NDArray, F: np_t.NDArray, m: float, dt: float) -> np_t.NDArray:
    """
    This function integrates the linear dynamics of a rigid body with constant force.
    It just uses stuff from high school.

    Parameters:
        p (np_t.NDArray): The position at the current time step.
        v (np_t.NDArray): The velocity at the current time step.
        F (np_t.NDArray): The force at the current time step.
        m (float): The mass of the rigid body.
        dt (float): The time step.
        damping (np_t.NDArray): The damping coefficients along the three axes. Not normalized by mass.
    
    Returns:
        np_t.NDArray: The position at the next time step.
    """
    velocity_new = v + (F/m)*dt
    position_new = p + v*dt + 0.5*(F/m)*dt**2
    return position_new, velocity_new

########################################
# Other utility functions
########################################

class SigmoidSoftStarter:

    def __init__(self, T: float = 1):
        """
        A simple sigmoid function implementation which keeps track of the time elapsed and yields
        a value which can be used as a multiplicative factor in order to soft start the system.
        The function is shifted by 0.5 and along the y axis and multiplied by 2 to yield a function 
        that goes from zero to 1.

        Parameters
        ----------
            T (float): 95% settling time of the sigmoid function.
        """
        self.k = 3.66356/T
        self.t = 0
        self.__sigmoid = lambda x: 2/(1 + np.exp(-self.k*x)) - 1
    
    def update(self, dt: float):
        self.t += dt
        return self.__sigmoid(self.t)
    
    def __call__(self, dt: float):
        return self.update(dt)
    
class LinearSoftStarter:

    def __init__(self, t_start, duration):
        self.t_start = t_start
        self.max_coeff = 1.0
        self.m = self.max_coeff/duration
        self.t = 0.0
    
    def __call__(self, dt: float):
        self.t += dt
        return min(max((self.t - self.t_start) * self.m, 0.0), self.max_coeff)


class FirstOrderDifferentiator:
    def __init__(self, alpha: float):
        """
        This is a simple first order forward difference differentiator.

        Parameters:
            alpha (float): The smoothing factor, 0 < alpha < 1. A higher alpha will result in a smoother output.
        """
        self.alpha = alpha
        self.prev_val = None
    
    def __call__(self, val: np_t.ArrayLike, dt: float) -> np_t.ArrayLike:
        if self.prev_val is None:
            self.prev_val = val
            return np.zeros_like(val)
        smooth_val = self.alpha*self.prev_val + (1-self.alpha)*val
        diff = (smooth_val - self.prev_val)/dt
        self.prev_val = smooth_val
        return diff

class MultiChannelFirstOrderDifferentiator:
    def __init__(self, channels: int, alpha: Union[float, Iterable[float]]):
        """
        This is a simple first order forward difference differentiator. Shouldn't be needed
        usually first order class can handle arrays.

        Parameters:
            channels (int): The number of channels to differentiate.
            alpha (Union[float, Iterable[float]]): The smoothing factor, 0 < alpha < 1. A higher alpha will result in a smoother output.
        """
        if isinstance(alpha, float):
            alpha = [alpha]*channels
        assert len(alpha) == channels, "Number of alphas must match number of channels."
        self.alpha = alpha
        self.diff_objs = [FirstOrderDifferentiator(a) for a in alpha]

    def __call__(self, vals: Iterable[float], dt: float) -> np_t.ArrayLike:
        return np.array([diff(val, dt) for diff, val in zip(self.diff_objs, vals)])

class FirstOrderIntegrator:
    def __init__(self,
                 windup_lim: float = np.inf,
                 clegg_integrator: bool = False):
        """
        This is a simple first order integrator.

        Parameters:
            windup_lim (float): The windup limit for the integrator.
            clegg_integrator (bool): Whether to use the Clegg integrator or not.
        """
        self.windup_lim = windup_lim
        self.integral = 0
        self.prev_val = None
        self.clegg_integrator = clegg_integrator

    
    def __call__(self, val: np_t.ArrayLike, dt: float) -> np_t.ArrayLike:
        if self.clegg_integrator and self.prev_val is not None:
            # Check which terms changed sign and reset the integral term.
            if not np.isscalar(val):
                reset_idx = np.where(np.sign(val) != np.sign(self.prev_val))[0]
                if reset_idx.size > 0:
                    self.integral[reset_idx] = 0
            else:
                if np.sign(val) != np.sign(self.prev_val):
                    self.integral = 0
        self.integral += val*dt
        self.integral = np.clip(self.integral, -self.windup_lim, self.windup_lim)
        self.prev_val = val
        return self.integral
    
#####################################
# Controller Utilities
#####################################

def tustin_pre_warp_discretize_PD_controller(Kp: float, Kd: float, dt: float, omega: float) -> np_t.NDArray:
    """
    This function discretizes a PD controller using Tustin's method.

    Parameters:
        Kp (float): The proportional gain.
        Kd (float): The derivative gain.
        dt (float): The time step.
        omega (float): The pre-warp frequency to remove scale distortion in rad/s.

    Returns:
        np_t.NDArray: The discretized coefficients for the current error and the previous error.
    """
    eta = omega/np.tan(omega*dt/2)
    now_gain = Kp + Kd*eta
    prev_gain = Kp - Kd*eta
    return np.array([now_gain, prev_gain])

@numba.njit(cache=True)
def numba_pinv(A: np_t.NDArray) -> np_t.NDArray:
    """
    This function is expected to provide a 2x speedup over the normal numpy version.
    """
    return np.linalg.pinv(A)

numba_pinv(np.eye(3)) # Force compilation on import.

@numba.njit(cache=True)
def numba_cond(A: np_t.NDArray) -> float:
    """
    This function is expected to provide a 3x speedup over the normal numpy version.
    """
    return np.linalg.cond(A)