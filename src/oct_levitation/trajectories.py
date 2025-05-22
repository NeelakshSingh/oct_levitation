import numpy as np
import numba
import oct_levitation.plotting as plotting
import oct_levitation.geometry_jit as geometry
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Callable, Any, Dict, List, Union
import numpy.typing as np_t

from functools import partial

PositionArray1D = np_t.NDArray[np.float64]
QuaternionArray1D = np_t.NDArray[np.float64]
RPYArray1D = np_t.NDArray[np.float64]
TimeFloat = float

VelocityArray1D = np_t.NDArray[np.float64]
AngularVelocityArray1D = np_t.NDArray[np.float64]
RPYRatesArray1D = np_t.NDArray[np.float64]

TrajectoryPoint = Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, Union[AngularVelocityArray1D, RPYRatesArray1D]]
TrajectoryFunction = Callable[[TimeFloat, Any], TrajectoryPoint]

REGISTERED_TRAJECTORIES : Dict[str, TrajectoryFunction] = {}

"""
Import notes and conventions:
-> All finally registered trajectory functions should preferably just take time as a single parameter. This will keep things easier.
-> All the trajectories should preferably be defined in the inertial frame, but the angular velocities description can be in the body frame and may even be rotation prameter rates.
-> All the trajectory functions should return the pose and the desired twist of the object for a complete description.
"""

def register_trajectory(name: str, trajectory_func: TrajectoryFunction) -> None:
    """
    Register a trajectory function with a name.
    
    Args:
        name (str): The name of the trajectory.
        trajectory_func (TrajectoryFunction): The trajectory function to register.
    """
    if name in REGISTERED_TRAJECTORIES:
        raise ValueError(f"Trajectory '{name}' is already registered.")
    REGISTERED_TRAJECTORIES[name] = trajectory_func

def list_registered_trajectories() -> List[str]:
    """
    List all registered trajectory names.
    
    Returns:
        List[str]: A list of registered trajectory names.
    """
    return list(REGISTERED_TRAJECTORIES.keys())
#################################
# Useful constants and utility functions
#################################

Z_ALIGNED_INERTIAL_REDUCED_ATTITUDE = np.array([0.0, 0.0, 1.0])
IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])

def plot_trajectory(func_name: Union[str, TrajectoryFunction], duration: float = 10.0, start: float = 0.0, step: float = 1e-2, plot_3d_path: bool = False) -> None:
    """
    Plot the trajectory defined by the given function.
    
    Args:
        trajectory_func (TrajectoryFunction): The trajectory function to plot.
        duration (float): Duration of the trajectory in seconds.
        start (float): Start time of the trajectory in seconds.
        step (float): Time step for plotting in seconds.
        plot_3d_path (bool): Whether to plot the 3D path of the trajectory.
    """
    if callable(func_name):
        trajectory_func = func_name
    elif isinstance(func_name, str):
        trajectory_func = REGISTERED_TRAJECTORIES[func_name]
    else:
        raise ValueError("func_name must be a callable or a string representing a registered trajectory name.")
    t = np.arange(start, start + duration, step)
    pose_cols = ['time', 'transform.translation.x', 'transform.translation.y', 'transform.translation.z',
                    'transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']
    
    ref_pose_df = pd.DataFrame(columns=pose_cols)
    actual_pose_df = pd.DataFrame(columns=pose_cols) # Just a pointer to the origin for now
    for i in range(len(t)):
        xyz, _, quaternion, _ = trajectory_func(t[i])
        t_pose = np.array([t[i], xyz[0], xyz[1], xyz[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        t_actual_pose = np.array([t[i], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        ref_pose_df.loc[i] = t_pose
        actual_pose_df.loc[i] = t_actual_pose

    plotting.DISABLE_PLT_SHOW = True
    plotting.plot_poses_variable_reference(actual_pose_df, ref_pose_df)
    if plot_3d_path:
        plotting.plot_3d_poses_with_arrows_constant_reference(ref_pose_df, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), frame_size=0.5)
    plt.show()

#################################
# Trajectory function definitions
#################################

@numba.njit(cache=True)
def sine_z_trajectory_quaternion(t: float, amplitude: float, frequency: float, center: float,
                                       xy_ref = np.array([0.0, 0.0])) -> Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]:
    z = amplitude * np.sin(2 * np.pi * frequency * t) + center
    z_dot = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
    xyz = np.array([xy_ref[0], xy_ref[1], z])
    velocity = np.array([0.0, 0.0, z_dot])
    
    return xyz, velocity, IDENTITY_QUATERNION, np.zeros(3, np.float64)

sine_z_trajectory_quaternion(0.0, 1.0e-3, 1.0, 0.0) # Force compilation on import for expected type signature
register_trajectory("sine_z_trajectory_quaternion_a4c4f0.5", partial(sine_z_trajectory_quaternion, amplitude=4.0e-3, frequency=0.5, center=4.0e-3))
register_trajectory("sine_z_trajectory_quaternion_a10c10f0.5", partial(sine_z_trajectory_quaternion, amplitude=10.0e-3, frequency=0.5, center=10.0e-3))
register_trajectory("sine_z_trajectory_quaternion_a10c20f0.5", partial(sine_z_trajectory_quaternion, amplitude=10.0e-3, frequency=0.5, center=20.0e-3))

@numba.njit(cache=True)
def xy_lissajous_trajectory_quaternion(t: float, A: float, a_hz: float, B: float, b_hz: float, delta: float,
                                       center: np.ndarray = np.zeros(3)) -> Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]:
    x = A * np.sin(2 * np.pi * a_hz * t + delta)
    y = B * np.sin(2 * np.pi * b_hz * t)
    x_dot = 2 * np.pi * a_hz * A * np.cos(2 * np.pi * a_hz * t + delta)
    y_dot = 2 * np.pi * b_hz * B * np.cos(2 * np.pi * b_hz * t)
    z_dot = 0.0
    xyz = np.array([x, y, 0.0]) + center
    velocity = np.array([x_dot, y_dot, z_dot])
    return xyz, velocity, IDENTITY_QUATERNION, np.zeros(3, np.float64)

xy_lissajous_trajectory_quaternion(0.0, 1.0e-3, 1.0, 1.0e-3, 1.0, 0.0) # Force compilation on import for expected type signature
register_trajectory("xy_circle_quaternion_r5_fhz0.5_cz4", partial(xy_lissajous_trajectory_quaternion, A=5.0e-3, a_hz=0.5, B=5.0e-3, b_hz=0.5, delta=np.pi/2, center=np.array([0.0, 0.0, 4.0e-3])))
register_trajectory("xy_circle_quaternion_r10_fhz0.5_cz10", partial(xy_lissajous_trajectory_quaternion, A=10.0e-3, a_hz=0.5, B=10.0e-3, b_hz=0.5, delta=np.pi/2, center=np.array([0.0, 0.0, 10.0e-3])))
register_trajectory("xy_circle_quaternion_r10_fhz1.0_cz10", partial(xy_lissajous_trajectory_quaternion, A=10.0e-3, a_hz=1.0, B=10.0e-3, b_hz=1.0, delta=np.pi/2, center=np.array([0.0, 0.0, 10.0e-3])))
register_trajectory("xy_infty_lissajous_quaternion_amp10_fx0.5_fy1_cz10", partial(xy_lissajous_trajectory_quaternion, A=10.0e-3, a_hz=0.5, B=10.0e-3, b_hz=1.0, delta=np.pi/2, center=np.array([0.0, 0.0, 10.0e-3])))
register_trajectory("xy_infty_lissajous_quaternion_amp10_fx0.25_fy0.5_cz10", partial(xy_lissajous_trajectory_quaternion, A=10.0e-3, a_hz=0.25, B=10.0e-3, b_hz=0.5, delta=np.pi/2, center=np.array([0.0, 0.0, 10.0e-3])))

@numba.njit(cache=True)
def rp_lissajous_trajectory_quaternion(t: float, r_ang_amp: float, r_hz: float, p_ang_amp: float, p_hz: float, delta: float,
                                       position: np.ndarray = np.zeros(3)) -> Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]:
    """
    Generate a Lissajous trajectory in roll and pitch angles.

    Args:
        t (float): Time in seconds.
        r_ang_amp (float): Roll angle amplitude in radians.
        r_hz (float): Roll frequency in Hz.
        p_ang_amp (float): Pitch angle amplitude in radians.
        p_hz (float): Pitch frequency in Hz.
        delta (float): Phase difference between roll and pitch angles in radians.
        position (np.ndarray, optional): Initial position. Defaults to np.zeros(3).
    
    Returns:
        Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]: 
            - Position in the inertial frame.
            - Velocity in the inertial frame.
            - Quaternion representing the orientation.
            - Angular velocity in the inertial frame.
    """
    r = r_ang_amp * np.sin(2 * np.pi * r_hz * t + delta)
    p = p_ang_amp * np.sin(2 * np.pi * p_hz * t)
    r_dot = 2 * np.pi * r_hz * r_ang_amp * np.cos(2 * np.pi * r_hz * t + delta)
    p_dot = 2 * np.pi * p_hz * p_ang_amp * np.cos(2 * np.pi * p_hz * t)
    
    euler = np.array([r, p, 0.0])
    quat = geometry.quaternion_from_euler_xyz(euler)
    euler_rates = np.array([r_dot, p_dot, 0.0])
    return position, np.zeros(3, np.float64), quat, geometry.euler_xyz_rate_to_inertial_angular_velocity(euler_rates, euler)

rp_lissajous_trajectory_quaternion(0.0, 1.0, 1.0, 1.0, 1.0, 0.0) # Force compilation on import for expected type signature

register_trajectory("rp_circle_quaternion_rp45deg_fhz0.5_cz10", partial(rp_lissajous_trajectory_quaternion, r_ang_amp=np.deg2rad(45.0), r_hz=0.5, p_ang_amp=np.deg2rad(45.0), p_hz=0.5, delta=np.pi/2, position=np.array([0.0, 0.0, 10.0e-3])))
register_trajectory("rp_circle_quaternion_rp45deg_fhz0.1_cz10", partial(rp_lissajous_trajectory_quaternion, r_ang_amp=np.deg2rad(45.0), r_hz=0.1, p_ang_amp=np.deg2rad(45.0), p_hz=0.1, delta=np.pi/2, position=np.array([0.0, 0.0, 10.0e-3])))