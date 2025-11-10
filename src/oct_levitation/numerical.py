import numpy as np
import numpy.typing as np_t
import numba

from typing import Union, Iterable

########################################
# Numerical integration of orientation
########################################

# NOTE: These functions may be used to create basic simulators for rigid body dynamics
#       for testing controllers and state estimators.
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
    
#####################################
# Controller Utilities
#####################################

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

numba_cond(np.eye(3)) # Force compilation on import.

@numba.njit(cache=True)
def numba_clip(a, a_min, a_max):
    a = np.maximum(a, a_min)
    a = np.minimum(a, a_max)
    return a

numba_clip(np.array([1, 2, 3]), 0, 2) # Force compilation on import.

import numpy as np
import rospy

class SimpleDisturbanceIntegrator:
    """
    A simple first-order discrete-time integrator for disturbance compensation.
    Handles position and reduced attitude axes, supports convergence detection,
    and can disable integrators upon convergence.

    Parameters
    ----------
    Ki_lin_xyz : Tuple[float, float, float], optional
        The linear integral gains for the (X, Y, Z) position errors.
        Each element defines how strongly the corresponding axis integrates
        position error into disturbance compensation.
        Default is (0.0, 0.0, 0.0).

    Ki_ang : float, optional
        The angular integral gain for the reduced attitude errors (Nx, Ny).
        Default is 0.0.

    integrator_enable : np.NDArray[Union[int, bool]], optional

    pos_error_tol : float, optional
        The absolute position error tolerance below which an axis is considered
        converged for a sustained duration. Default is 1e-3.

    att_error_tol : float, optional
        The absolute reduced attitude error tolerance below which an angular
        axis is considered converged for a sustained duration. Default is 1e-3.

    integrator_start_time : float, optional
        The simulation time (in seconds) after which the integrator begins
        accumulating error. Default is 0.0.

    integrator_end_time : float, optional
        The simulation time (in seconds) after which the integrator stops
        updating. Default is infinity.

    integrator_convergence_check_time : float, optional
        The duration (in seconds) for which the error must remain below its
        respective tolerance before the corresponding integrator is considered
        converged. Default is 1.0.

    switch_off_integrator_on_convergence : bool, optional
        If True, disables (freezes) individual integrators once convergence
        has been achieved for that axis. Default is True.

    check_convergence : bool, optional
        If True, enables automatic convergence checking and logging.
        If False, integrators will never be marked as converged. Default is True.
    """

    def __init__(
        self,
        Ki_lin_xyz=(0.0, 0.0, 0.0),  # (Ki_x, Ki_y, Ki_z)
        Ki_ang=0.0,
        integrator_enable=np.ones(5, dtype=bool),
        pos_error_tol=1e-3,
        reduced_att_error_tol=1e-3,
        integrator_start_time=0.0,
        integrator_end_time=np.inf,
        integrator_convergence_check_time=1.0,
        switch_off_integrator_on_convergence=True,
        check_convergence=True,
    ):
        # Integration gains
        self.Ki_lin_x, self.Ki_lin_y, self.Ki_lin_z = Ki_lin_xyz
        self.Ki_ang = Ki_ang

        # Timing
        self.integrator_start_time = integrator_start_time
        self.integrator_end_time = integrator_end_time

        # Convergence settings
        self.__pos_error_tol = pos_error_tol
        self.__att_error_tol = reduced_att_error_tol
        self.__integrator_convergence_check_time = integrator_convergence_check_time
        self.switch_off_integrator_on_convergence = switch_off_integrator_on_convergence
        self.__integrator_check_convergence = check_convergence

        # States
        self.__integrator_enable = integrator_enable
        self.__indiv_integrator_converge_state = np.ones(5, dtype=bool)
        self.__indiv_integrator_converge_state[self.__integrator_enable == 1] = False
        self.__convergence_time = np.zeros(5)
        self.__enabled_integrators_converged = False

        # Disturbance compensation: [Nx, Ny, Z, Y, X] ordering
        self.disturbance_rpxyz = np.zeros(5)

        # Internal time
        self.time_elapsed = 0.0

    def reset(self):
        """Resets integrator states and disturbances."""
        self.disturbance_rpxyz[:] = 0.0
        self.__convergence_time[:] = 0.0
        self.__indiv_integrator_converge_state[:] = False
        self.__enabled_integrators_converged = False
        self.__integrator_enable[:] = True
        self.time_elapsed = 0.0

    def step(self, dt, position_error, reduced_attitude_error):
        """
        Executes one integration step.

        Parameters
        ----------
        dt : float
            Time step for integration.
        position_error : np.ndarray shape (3,)
            Cartesian errors.
        reduced_attitude_error : np.ndarray shape (2,)
            Reduced attitude error as a 2D numpy array.

        Returns
        -------
        np.ndarray
            Updated disturbance_rpxyz (size 5).
        """
        self.dt = dt
        self.time_elapsed += dt

        # Integrate only in valid window
        if self.integrator_start_time < self.time_elapsed < self.integrator_end_time:
            # Integrate position and attitude disturbances
            self.disturbance_rpxyz[0] += self.Ki_ang * reduced_attitude_error[0] * dt * self.__integrator_enable[0]
            self.disturbance_rpxyz[1] += self.Ki_ang * reduced_attitude_error[1] * dt * self.__integrator_enable[1]
            self.disturbance_rpxyz[2] += self.Ki_lin_x * position_error[0] * dt * self.__integrator_enable[2]
            self.disturbance_rpxyz[3] += self.Ki_lin_y * position_error[1] * dt * self.__integrator_enable[3]
            self.disturbance_rpxyz[4] += self.Ki_lin_z * position_error[2] * dt * self.__integrator_enable[4]

            # Check convergence
            if not self.__enabled_integrators_converged and self.__integrator_check_convergence:
                self._check_convergence(position_error[0], position_error[1], position_error[2], reduced_attitude_error)

        return self.disturbance_rpxyz

    def _check_convergence(self, x_error, y_error, z_error, reduced_attitude_error):
        """Internal helper to track convergence and disable integrators if needed."""

        if abs(x_error) < self.__pos_error_tol and self.__integrator_enable[2]:
            self.__convergence_time[2] += self.dt
            if self.__convergence_time[2] > self.__integrator_convergence_check_time:
                rospy.logwarn_once(f"X convergence achieved. Compensation force: {self.disturbance_rpxyz[2]}")
                self.__indiv_integrator_converge_state[2] = True
                if self.switch_off_integrator_on_convergence:
                    rospy.logwarn_once("Stopping X integrator.")
                    self.__integrator_enable[2] = False
        else:
            # Reset convergence time if error goes above tolerance
            self.__convergence_time[2] = 0.0

        if abs(y_error) < self.__pos_error_tol and self.__integrator_enable[3]:
            self.__convergence_time[3] += self.dt
            if self.__convergence_time[3] > self.__integrator_convergence_check_time:
                rospy.logwarn_once(f"Y convergence achieved. Compensation force: {self.disturbance_rpxyz[3]}")
                self.__indiv_integrator_converge_state[3] = True
                if self.switch_off_integrator_on_convergence:
                    rospy.logwarn_once("Stopping Y integrator.")
                    self.__integrator_enable[3] = False
        else:
            self.__convergence_time[3] = 0.0

        if abs(z_error) < self.__pos_error_tol and self.__integrator_enable[4]:
            self.__convergence_time[4] += self.dt
            if self.__convergence_time[4] > self.__integrator_convergence_check_time:
                rospy.logwarn_once(f"Z convergence achieved. Compensation force: {self.disturbance_rpxyz[4]}")
                self.__indiv_integrator_converge_state[4] = True
                if self.switch_off_integrator_on_convergence:
                    rospy.logwarn_once("Stopping Z integrator.")
                    self.__integrator_enable[4] = False
        else:
            self.__convergence_time[4] = 0.0

        if abs(reduced_attitude_error[0]) < self.__att_error_tol and self.__integrator_enable[0]:
            self.__convergence_time[0] += self.dt
            if self.__convergence_time[0] > self.__integrator_convergence_check_time:
                rospy.logwarn_once(f"Reduced attitude Nx convergence achieved. Inertia normalized compensation torque: {self.disturbance_rpxyz[0]}") 
                # Inertia normalized because this is before multiplying by inertia in the controller.
                self.__indiv_integrator_converge_state[0] = True
                if self.switch_off_integrator_on_convergence:
                    rospy.logwarn_once("Stopping RA x integrator.")
                    self.__integrator_enable[0] = False
        else:
            self.__convergence_time[0] = 0.0

        if abs(reduced_attitude_error[1]) < self.__att_error_tol and self.__integrator_enable[1]:
            self.__convergence_time[1] += self.dt
            if self.__convergence_time[1] > self.__integrator_convergence_check_time:
                rospy.logwarn_once(f"Reduced attitude Ny convergence achieved. Inertia normalized compensation torque: {self.disturbance_rpxyz[1]}")
                self.__indiv_integrator_converge_state[1] = True
                if self.switch_off_integrator_on_convergence:
                    rospy.logwarn_once("Stopping RA y integrator.")
                    self.__integrator_enable[1] = False
        else:
            self.__convergence_time[1] = 0.0

        # Global convergence
        if np.all(self.__indiv_integrator_converge_state):
            self.__enabled_integrators_converged = True
            rospy.logwarn_once("All enabled integrators converged.")
