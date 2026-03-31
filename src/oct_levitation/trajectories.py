import numpy as np
import numba
import oct_levitation.plotting as plotting
import oct_levitation.geometry_jit as geometry
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Callable, Any, Dict, List, Union, Iterable
from copy import deepcopy
import numpy.typing as np_t

from functools import partial
from bisect import bisect_left
from enum import Enum

PositionArray1D = np_t.NDArray[np.float64]
QuaternionArray1D = np_t.NDArray[np.float64]
RPYArray1D = np_t.NDArray[np.float64]
TimeFloat = float
StartTimeFloat = float
DurationTimeFloat = float

VelocityArray1D = np_t.NDArray[np.float64]
AngularVelocityArray1D = np_t.NDArray[np.float64]
RPYRatesArray1D = np_t.NDArray[np.float64]

TrajectoryPoint = Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, Union[AngularVelocityArray1D, RPYRatesArray1D]]
TrajectoryCallable = Callable[[TimeFloat], TrajectoryPoint]

REGISTERED_TRAJECTORIES : Dict[str, TrajectoryCallable] = {}

"""
Import notes and conventions:
-> All finally registered trajectory functions should preferably just take time as a single parameter. This will keep things easier.
-> All the trajectories should preferably be defined in the inertial frame, but the angular velocities description can be in the body frame and may even be rotation prameter rates.
-> All the trajectory functions should return the pose and the desired twist of the object for a complete description.
"""

def register_trajectory(name: str, trajectory_func: TrajectoryCallable) -> None:
    """
    Register a trajectory function with a name.

    Args:
        name (str): The name of the trajectory.
        trajectory_func (TrajectoryCallable): The trajectory function to register.
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

def plot_trajectory(func_name: Union[str, TrajectoryCallable], duration: float, start: float = 0.0, step: float = 1e-2, plot_3d_path: bool = False, frame_size: float = 1.0) -> None:
    """
    Plot the trajectory defined by the given function.

    Args:
        trajectory_func (TrajectoryCallable): The trajectory function to plot.
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
        plotting.plot_3d_poses_with_arrows_constant_reference(ref_pose_df, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), frame_size=frame_size)
    plt.show()

class DiscreteTrajectory:
    """
    Class to represent a periodic discrete trajectory. The basic idea behind this class is very simple. It
    stores the trajectory as an array of points and in each call just return the next point in the array.
    By itself it cannot do much, but it will be paired with other functions which can discretize the trajectory
    for a specific sampling time and period. One can also just inherit from this class and implement several
    different discrete trajectories. It can be helpful to use this class to precompute the trajectory points.

    Args:
        trajectory_func (TrajectoryCallable): The trajectory function to use.
        period (float): The period of the trajectory in seconds.
        start (float): The start time of the trajectory in seconds.
        step (float): The time step for the trajectory in seconds.
    """
    def __init__(self, step: float, loop: bool = False) -> None:
        self.traj_array : List[TrajectoryPoint] = []
        self.__count = 0
        self.__last_traj_point = None
        self.__len = 0
        self.step = step
        self.loop = loop

    def insert_trajectory_point(self, point: TrajectoryPoint) -> None:
        """
        Insert a trajectory point into the array.

        Args:
            point (TrajectoryPoint): The trajectory point to insert.
        """
        self.traj_array.append(point)
        self.__last_traj_point = point
        self.__len += 1

    def __call__(self, t: float) -> TrajectoryPoint:
        """
        Get the next trajectory point based on the current time.

        Args:
            t (float): The current time in seconds.

        Returns:
            TrajectoryPoint: The next trajectory point.
        """
        if self.__len == 0:
            raise ValueError("Trajectory array is empty.")

        if self.loop:
            index = int(t / self.step) % self.__len
            return self.traj_array[index]
        else:
            index = int(t / self.step)
            if index >= self.__len: # Return the last point
                index = self.__len - 1
                return self.__last_traj_point[0], np.zeros(3, np.float64), self.__last_traj_point[2], np.zeros(3, np.float64)
            return self.traj_array[index]

def create_discretized_trajectory(func: TrajectoryCallable, start_time: float, end_time: float, step: float, loop: bool = False) -> DiscreteTrajectory:
    """
    Create a discretized trajectory using the given function.

    Args:
        func (TrajectoryCallable): The trajectory function to use.
        start_time (float): The start time of the trajectory in seconds.
        end_time (float): The end time of the trajectory in seconds.
        step (float): The time step for the trajectory in seconds.
        loop (bool): Whether to loop the trajectory or not.

    Returns:
        DiscreteTrajectory: The discretized trajectory.
    """
    traj = DiscreteTrajectory(step, loop)
    t = start_time
    while t <= end_time:
        traj_point = func(t)
        traj.insert_trajectory_point(traj_point)
        t += step
    return traj

class TrajectoryTransitions(Enum):
    PAUSE_ON_PREV = 0
    PAUSE_ON_NEXT = 1
    LINEAR_TRANSITION = 2

@numba.njit(cache=True)
def const_pose_setpoint(t: float, position_setpoint: PositionArray1D, quaternion_setpoint: QuaternionArray1D) -> TrajectoryPoint:
    """
    Generate a constant pose setpoint trajectory.

    Args:
        t (float): Time in seconds.
        position_setpoint (PositionArray1D): Position setpoint in the inertial frame.
        quaternion_setpoint (QuaternionArray1D): Quaternion setpoint representing the orientation.

    Returns:
        TrajectoryPoint: The constant pose setpoint trajectory point.
    """
    return position_setpoint, np.zeros(3, np.float64), quaternion_setpoint, np.zeros(3, np.float64)

@numba.njit(cache=True)
def simple_linear_trajectory_quaternion(t: float, start_position: PositionArray1D, end_position: PositionArray1D,
                                        start_euler_xyz: RPYArray1D, end_euler_xyz: RPYArray1D, duration: float) -> Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]:
    """
    Generate a simple linear trajectory in the inertial frame.
    Args:
        t (float): Time in seconds.
        start_position (PositionArray1D): Starting position in the inertial frame.
        end_position (PositionArray1D): Ending position in the inertial frame.
        start_euler_xyz (RPYArray1D): Starting roll, pitch, yaw angles in radians.
        end_euler_xyz (RPYArray1D): Ending roll, pitch, yaw angles in radians.
        duration (float): Duration of the trajectory in seconds.
    Returns:
        Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]:
            - Position in the inertial frame.
            - Velocity in the inertial frame.
            - Quaternion representing the orientation.
            - Angular velocity in the inertial frame.
    """
    euler_rates = (end_euler_xyz - start_euler_xyz) / duration
    velocity = (end_position - start_position) / duration
    if t < 0.0:
        t = 0.0
    if t > duration:
        t = duration
        velocity = np.zeros(3, np.float64)
        euler_rates = np.zeros(3, np.float64)
    alpha = t / duration
    position = (1.0 - alpha) * start_position + alpha * end_position
    euler = (1.0 - alpha) * start_euler_xyz + alpha * end_euler_xyz
    quat = geometry.quaternion_from_euler_xyz(euler)
    return position, velocity, quat, geometry.euler_xyz_rate_to_inertial_angular_velocity(euler_rates, euler)

simple_linear_trajectory_quaternion(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 1.0) # Force compilation on import for expected type signature

class ChainedTrajectory:
    """
    This class is used to chain several trajectories together. It is the user's job to make sure that their
    endpoints match in case one wants them to be continuous. This class will not check for it. It is just a
    utility to allow connecting any number of trajectories and expose the same simple TrajectoryCallable
    interface to the controllers.

    Args:
        trajectories (List[Tuple[TrajectoryCallable, StartTimeFloat, DurationTimeFloat]]): List of trajectory functions
            to chain along with the time relative to 0.0 for the trajectory's beginning and the duration of execution from
            when the trajectory is started in the chain. So StartTimeFloat of 1.0 seconds means the trajectory will start from
            its t=1.0 second mark according to its definition whenever it begins in the chain and will run for DurationTimeFloat seconds.
            Trajectories are executed in the order they are added to the list.

        loop (bool): Whether to loop the chained trajectory or not.
    """

    ChainedTrajectoryElement = Tuple[Union[str, TrajectoryCallable, TrajectoryTransitions], StartTimeFloat, DurationTimeFloat]

    def __init__(self, trajectories: Iterable[ChainedTrajectoryElement], loop: bool = False) -> None:
        self.trajectories = deepcopy(trajectories)

        ## Here we initialize all the callables. Transitions depend on callables and will be initialized in the next pass.
        for i, traj_element in enumerate(self.trajectories): # First pass to retrieve all callables
            traj_element = list(traj_element)
            if len(traj_element) < 3: # To be enforced for options too
                    raise ValueError(f"Trajectory element {i} must have at least 3 elements: (callable | name | transition, start_time, duration).")
            if isinstance(traj_element[0], str):
                traj_element[0] = REGISTERED_TRAJECTORIES[traj_element[0]]
            self.trajectories[i] = traj_element

        self.__len = len(self.trajectories)

        ## Second pass for implement transitions. A LOT OF EDGE CASES ARE NOT HANDLED AT THE MOMENT. WHEN TRANSITIONS OCCUR
        ## TOGETHER THERE ARE SEVERAL EDGE CASES AND RESOLVING THEM WILL NEED A VERY PROPER RULE BASED RESOLUTION. FOR NOW
        ## THAT IS NOT DONE.
        for i, traj_element in enumerate(self.trajectories):
            if isinstance(traj_element[0], TrajectoryTransitions):
                if traj_element[0] == TrajectoryTransitions.PAUSE_ON_PREV:
                    if i==0:
                        raise ValueError(f"PAUSE_ON_PREV transition cannot be the first element.")
                    prev_trajectory_endpoint = self.trajectories[i-1][0](self.trajectories[i-1][1] + self.trajectories[i-1][2])
                    traj_element[0] = partial(const_pose_setpoint, position_setpoint=prev_trajectory_endpoint[0], quaternion_setpoint=prev_trajectory_endpoint[2])
                elif traj_element[0] == TrajectoryTransitions.PAUSE_ON_NEXT:
                    if i==self.__len-1:
                        raise ValueError(f"PAUSE_ON_NEXT transition cannot be the last element.")
                    next_trajectory_start = self.trajectories[i+1][0](self.trajectories[i+1][1])
                    traj_element[0] = partial(const_pose_setpoint, position_setpoint=next_trajectory_start[0], quaternion_setpoint=next_trajectory_start[2])
                elif traj_element[0] == TrajectoryTransitions.LINEAR_TRANSITION:
                    if i==0 or i==self.__len-1:
                        raise ValueError(f"LINEAR_TRANSITION transition cannot be the first or last element.")
                    prev_trajectory_endpoint = self.trajectories[i-1][0](self.trajectories[i-1][1] + self.trajectories[i-1][2])
                    next_trajectory_start = self.trajectories[i+1][0](self.trajectories[i+1][1])

                    traj_element[0] = partial(simple_linear_trajectory_quaternion,
                                              start_position=prev_trajectory_endpoint[0],
                                              end_position=next_trajectory_start[0],
                                              start_euler_xyz=geometry.euler_xyz_from_quaternion(prev_trajectory_endpoint[2]),
                                              end_euler_xyz=geometry.euler_xyz_from_quaternion(next_trajectory_start[2]),
                                              duration=traj_element[2])
                    traj_element[1] = 0.0 # The transition will start at t=0.0 always so ignore this arg from the user
                else:
                    raise ValueError(f"This trajectory transition doesn't seem to be implemented by the Chainer yet.")

            traj_element = tuple(traj_element) # make things immutable beyond this point to avoid accidental modifications.
            self.trajectories[i] = traj_element # replace with the tuple

        self.__transition_times = np.cumsum(np.array([0] + [traj[2] for traj in self.trajectories])[:-1])
        self.__endpoint_times = np.cumsum(np.array([traj[2] for traj in self.trajectories]))
        self.__total_duration = self.__endpoint_times[-1]
        self.__last_traj_point = self.trajectories[-1][0](self.trajectories[-1][1] + self.trajectories[-1][2]) # in case we don't loop
        self.__loop = loop

        # I can exploit the fact that I am expecting the trajectories to be contiguously executed and not do any lookups unless needed.
        self.__current_trajectory_idx = 0

        self.__current_TRAJ_CALLABLE : TrajectoryCallable = self.trajectories[0][0]
        self.__current_traj_t0 : StartTimeFloat = self.trajectories[0][1]
        self.__current_traj_start_time : TimeFloat = self.__transition_times[0]
        self.__current_traj_end_time : TimeFloat = self.__endpoint_times[0]

    def update_current_trajectory_info(self, idx: int) -> None:
        """
        Update the current trajectory information based on the index.

        Args:
            idx (int): The index of the current trajectory.
        """
        self.__current_TRAJ_CALLABLE = self.trajectories[idx][0]
        self.__current_traj_t0 = self.trajectories[idx][1]
        self.__current_traj_start_time = self.__transition_times[idx]
        self.__current_traj_end_time = self.__endpoint_times[idx]

    def get_chain_properties(self) -> Tuple[float, bool, int]:
        """
        Get the properties of the chained trajectory.

        Returns:
            Tuple[float, bool, int]: A tuple containing the total duration, loop flag, and number of trajectories.
        """
        return self.__total_duration, self.__loop, self.__len

    def lookup_trajectory_idx(self, t: float) -> int:
        """
        Lookup the trajectory index based on the current time.

        Args:
            t (float): The current time in seconds.

        Returns:
            int: The index of the trajectory.
        """
        if self.__loop:
            t = t % self.__total_duration
        return bisect_left(self.__endpoint_times, t)

    def __call__(self, t: float) -> TrajectoryPoint:
        """
        Get the next trajectory point based on the current time.

        Args:
            t (float): The current time in seconds.

        Returns:
            TrajectoryPoint: The next trajectory point.
        """
        if self.__loop:
            t = t % self.__total_duration
        else:
            if t > self.__total_duration:
                return self.__last_traj_point[0], np.zeros(3, np.float64), self.__last_traj_point[2], np.zeros(3, np.float64)

        # Check if we need to update the trajectory index
        if t < self.__current_traj_start_time:
            self.__current_trajectory_idx = self.lookup_trajectory_idx(t)
            self.update_current_trajectory_info(self.__current_trajectory_idx)

        if t > self.__current_traj_end_time:
            if t > self.__endpoint_times[self.__current_trajectory_idx + 1]:
                # if we are already past the end of the next trajectory, better to check which one we are in
                self.__current_trajectory_idx = self.lookup_trajectory_idx(t)
            else: # exploting the fact that nominally the trajectories are contiguously executed
                self.__current_trajectory_idx += 1

            self.update_current_trajectory_info(self.__current_trajectory_idx)

        return self.__current_TRAJ_CALLABLE(t - self.__current_traj_start_time + self.__current_traj_t0)

import oct_levitation.trajectory_definitions
