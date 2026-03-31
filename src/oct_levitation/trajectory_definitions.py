import numpy as np
import numba
import oct_levitation.geometry_jit as geometry
from functools import partial
from copy import deepcopy
from oct_levitation.trajectories import (
    register_trajectory, REGISTERED_TRAJECTORIES,
    ChainedTrajectory, TrajectoryTransitions,
    create_discretized_trajectory,
    const_pose_setpoint, simple_linear_trajectory_quaternion,
    IDENTITY_QUATERNION,
    TrajectoryPoint, PositionArray1D, VelocityArray1D,
    QuaternionArray1D, AngularVelocityArray1D,
)
from typing import Tuple

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
register_trajectory("sine_z_trajectory_quaternion_a15c25f0.5", partial(sine_z_trajectory_quaternion, amplitude=15.0e-3, frequency=0.5, center=25.0e-3))

@numba.njit(cache=True)
def xy_lissajous_trajectory_quaternion(t: float, A: float, a_hz: float, B: float, b_hz: float, delta: float,
                                       center: np.ndarray = np.zeros(3), shift: float = 0.0) -> Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]:
    x = A * np.sin(2 * np.pi * a_hz * t + delta + shift)
    y = B * np.sin(2 * np.pi * b_hz * t + shift)
    x_dot = 2 * np.pi * a_hz * A * np.cos(2 * np.pi * a_hz * t + delta)
    y_dot = 2 * np.pi * b_hz * B * np.cos(2 * np.pi * b_hz * t)
    z_dot = 0.0
    xyz = np.array([x, y, 0.0]) + center
    velocity = np.array([x_dot, y_dot, z_dot])
    return xyz, velocity, IDENTITY_QUATERNION, np.zeros(3, np.float64)

xy_lissajous_trajectory_quaternion(0.0, 1.0e-3, 1.0, 1.0e-3, 1.0, 0.0) # Force compilation on import for expected type signature
register_trajectory("xy_infty_lissajous_quaternion_ax20_ay10_fx0.25_fy0.5_cz10", partial(xy_lissajous_trajectory_quaternion, A=20.0e-3, a_hz=0.25, B=10.0e-3, b_hz=0.5, delta=0.0, center=np.array([0.0, 0.0, 10.0e-3]), shift=0.0))

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
register_trajectory("rp_circle_quaternion_rp45deg_fhz0.2_cz10", partial(rp_lissajous_trajectory_quaternion, r_ang_amp=np.deg2rad(45.0), r_hz=0.2, p_ang_amp=np.deg2rad(45.0), p_hz=0.2, delta=np.pi/2, position=np.array([0.0, 0.0, 10.0e-3])))

@numba.njit(cache=True)
def xyrp_lissajous_trajectory_quaternion(
        t:float, x_amp: float, x_hz: float, y_amp: float, y_hz: float,
        r_amp: float, r_hz: float, p_amp: float, p_hz: float,
        center: np.ndarray = np.zeros(3), phi_x = 0.0, phi_y = 0.0, phi_r = 0.0, phi_p = 0.0
    ) -> Tuple[PositionArray1D, VelocityArray1D, QuaternionArray1D, AngularVelocityArray1D]:
        """
        Generate a Lissajous trajectory in x, y, roll, and pitch angles.
        """
        x = x_amp * np.sin(2 * np.pi * x_hz * t + phi_x)
        y = y_amp * np.sin(2 * np.pi * y_hz * t + phi_y)
        r = r_amp * np.sin(2 * np.pi * r_hz * t + phi_r)
        p = p_amp * np.sin(2 * np.pi * p_hz * t + phi_p)

        x_dot = 2 * np.pi * x_hz * x_amp * np.cos(2 * np.pi * x_hz * t + phi_x)
        y_dot = 2 * np.pi * y_hz * y_amp * np.cos(2 * np.pi * y_hz * t + phi_y)
        r_dot = 2 * np.pi * r_hz * r_amp * np.cos(2 * np.pi * r_hz * t + phi_r)
        p_dot = 2 * np.pi * p_hz * p_amp * np.cos(2 * np.pi * p_hz * t + phi_p)

        xyz = center + np.array([x, y, 0.0])
        velocity = np.array([x_dot, y_dot, 0.0])

        euler = np.array([r, p, 0.0])
        quat = geometry.quaternion_from_euler_xyz(euler)
        euler_rates = np.array([r_dot, p_dot, 0.0])

        return xyz, velocity, quat, geometry.euler_xyz_rate_to_inertial_angular_velocity(euler_rates, euler)

xyrp_lissajous_trajectory_quaternion(0.0, 1.0e-3, 1.0, 1.0e-3, 1.0, 0.0, 1.0e-3, 1.0, 1.0e-3, 1.0, 0.0) # Force compilation on import for expected type signature

register_trajectory("xyrp_lissajous_eight_T4_x20_y10_rp30_c0010",
                    partial(xyrp_lissajous_trajectory_quaternion,
                            x_amp=10.0e-3, x_hz=0.5, y_amp=15.0e-3, y_hz=0.25,
                            r_amp=np.deg2rad(30.0), r_hz=0.25, p_amp=np.deg2rad(30.0), p_hz=0.5,
                            phi_x=0.0, phi_y=0.0, phi_r=0.0, phi_p=np.pi,
                            center=np.array([0.0, 0.0, 10.0e-3])))

register_trajectory("xyrp_lissajous_eight_T4_x20_y10_rp15_c0010",
                    partial(xyrp_lissajous_trajectory_quaternion,
                            x_amp=10.0e-3, x_hz=0.5, y_amp=15.0e-3, y_hz=0.25,
                            r_amp=np.deg2rad(15.0), r_hz=0.25, p_amp=np.deg2rad(15.0), p_hz=0.5,
                            phi_x=0.0, phi_y=0.0, phi_r=0.0, phi_p=np.pi,
                            center=np.array([0.0, 0.0, 10.0e-3])))

register_trajectory("xyrp_lissajous_eight_T4_x20_y10_rp0_c0010",
                    partial(xyrp_lissajous_trajectory_quaternion,
                            x_amp=10.0e-3, x_hz=0.5, y_amp=15.0e-3, y_hz=0.25,
                            r_amp=0.0, r_hz=0.25, p_amp=0.0, p_hz=0.5,
                            phi_x=0.0, phi_y=0.0, phi_r=0.0, phi_p=np.pi,
                            center=np.array([0.0, 0.0, 10.0e-3])))

# example chained trajectory tracing 10 cm in +Z then +Y then +X
register_trajectory("sample_linear_chained_trajectory",
                    ChainedTrajectory([
                        (partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 0.0]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=5.0), 0.0, 5.0),
                        (partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 10.0e-3]), end_position=np.array([0.0, 10.0e-3, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=5.0), 0.0, 5.0),
                        (partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 10.0e-3, 10.0e-3]), end_position=np.array([10.0e-3, 10.0e-3, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=5.0), 0.0, 5.0)
                    ], loop=False)
                    )

register_trajectory("sample_periodic_z_linear_trajectory_discretized", # This should give a periodic triangular trajectory
                    create_discretized_trajectory(
                        ChainedTrajectory([
                            (partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 0.0]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0), 0.0, 2.0),
                            (TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 2.0),
                            (partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 10.0e-3]), end_position=np.array([0.0, 0.0, 0.0]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0), 0.0, 2.0),
                            (TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 2.0),
                        ], loop=True),
                        start_time=0.0, end_time=8.0, step=1e-3, loop=True
                    ))

###############################################
## Some long chained trajectories for demonstration
###############################################

demo_chain_list_1 = []
pause_time = 2.0 # pause between trajectories
linear_sweep_duration = 0.5

# 1. Z rise from origin to 15 mm
demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0, linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time
    ]
)

# 2. Symmetric Y sweep and back to 15 mm

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 15.0e-3]), end_position=np.array([0.0, 15.0e-3, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        1.0
    ]
)

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 15.0e-3, 15.0e-3]), end_position=np.array([0.0, -15.0e-3, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time
    ]
)

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, -15.0e-3, 15.0e-3]), end_position=np.array([0.0, 0.0, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time
    ]
)

# 3. X sweep and back to 15 mm

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 15.0e-3]), end_position=np.array([15.0e-3, 0.0, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        1.0
    ]
)

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([15.0e-3, 0.0, 15.0e-3]), end_position=np.array([-15.0e-3, 0.0, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time
    ]
)

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([-15.0e-3, 0.0, 15.0e-3]), end_position=np.array([0.0e-3, 0.0, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time
    ]
)

# 4. Z drop to start position of all demo curves

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 15.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 10.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
        0.0,
        pause_time
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.LINEAR_TRANSITION,
        0.0, 3.0
    ]
)

next_lissajous_traj_name = "rp_circle_quaternion_rp45deg_fhz0.2_cz10"
next_lissajous_traj = REGISTERED_TRAJECTORIES[next_lissajous_traj_name]

demo_chain_list_1.append(
    [
        partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 10.0e-3]), quaternion_setpoint=next_lissajous_traj(0.0)[2]),
        0.0,
        pause_time
    ]
)

# 5.Just staying at a single place and tracking the orientation lissajous circle
demo_chain_list_1.append(
    [next_lissajous_traj, 0.0, 10.0]
)

demo_chain_list_1.append(
    [TrajectoryTransitions.PAUSE_ON_PREV, 0.0, pause_time]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.LINEAR_TRANSITION,
        0.0, 3.0
    ]
)

demo_chain_list_1.append(
    [
        partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 10.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
        0.0,
        pause_time
    ]
)

# 6. Execute the lissajous infinity signs starting from origin
demo_chain_list_1.append(
    [
        "xyrp_lissajous_eight_T4_x20_y10_rp0_c0010",
        0.0,
        4.0*4.0
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time
    ]
)


demo_chain_list_1.append(
    [
        "xyrp_lissajous_eight_T4_x20_y10_rp15_c0010",
        0.0,
        4.0*4.0
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time*2.0
    ]
)

demo_chain_list_1.append(
    [
        "xyrp_lissajous_eight_T4_x20_y10_rp30_c0010",
        0.0,
        4.0*4.0
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_NEXT,
        0.0,
        pause_time
    ]
)

# TODO: Finally execute a 3D lissajous with roll and pitch as the last ultimate trajectory before ending for the final video.

# END. Z drop to start point at the end of the episode

demo_chain_list_1.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 10.0e-3]), end_position=np.array([0.0, 0.0, 5.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        linear_sweep_duration
    ]
)

demo_chain_list_1.append(
    [
        TrajectoryTransitions.PAUSE_ON_PREV,
        0.0,
        pause_time
    ]
)

# Another trajectory with just a bit lower endpoint
demo_chain_list_lower_end = deepcopy(demo_chain_list_1[:-2])

demo_chain_list_lower_end.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 10.0e-3]), end_position=np.array([0.0, 0.0, 30.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration),
        0.0,
        3.0
    ]
)

demo_chain_list_lower_end.append(
    [
        partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 30.0e-3]), end_position=np.array([0.0, 0.0, -15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=linear_sweep_duration*3.0*4.0),
        0.0,
        linear_sweep_duration * 3.0 * 4.0
    ]
)

demo_chain_list_lower_end.append(
    [
        TrajectoryTransitions.PAUSE_ON_PREV,
        0.0,
        pause_time
    ]
)

### Create the chained trajectory
register_trajectory("demo_chain_1", ChainedTrajectory(demo_chain_list_1, loop=True))
register_trajectory("demo_chain_1_no_loop", ChainedTrajectory(demo_chain_list_1, loop=False))
register_trajectory("demo_chain_1_no_loop_lower_z_endpoint", ChainedTrajectory(demo_chain_list_lower_end, loop=False))


### Setpoint change trajectories for measuring the step response of each dimension.

register_trajectory("setpoint_change_x_10mm",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([10.0e-3, 0.0, 10.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_y_10mm",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([0.0, 10.0e-3, 10.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_z_10mm",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 20.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_r_30deg",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 10.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([np.deg2rad(-30.0), 0.0, 0.0]))),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_p_30deg",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 10.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([0.0, np.deg2rad(-30.0), 0.0]))),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_rp_30deg",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 10.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([np.deg2rad(-30.0), np.deg2rad(-30.0), 0.0]))),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_xy_10mm",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([10.0e-3, 10.0e-3, 10.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_xy10mm_rp_30deg",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([10.0e-3, 10.0e-3, 10.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([np.deg2rad(-30.0), np.deg2rad(-30.0), 0.0]))),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_xyz_10mm",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([10.0e-3, 10.0e-3, 10.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_xyz5mm_rp30deg",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([5.0e-3, -5.0e-3, 15.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([np.deg2rad(-15.0), np.deg2rad(-15.0), 0.0]))),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("setpoint_change_xyz5mm_rp30deg_2snaps",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([5.0e-3, -5.0e-3, 15.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([np.deg2rad(-30.0), np.deg2rad(-30.0), 0.0]))),
                                0.0, 2.0
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([0.0e-3, 0.0e-3, 10.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([np.deg2rad(0.0), np.deg2rad(0.0), 0.0]))),
                                0.0, 0.5
                            ],
                            [
                                partial(const_pose_setpoint, position_setpoint=np.array([-5.0e-3, 5.0e-3, 15.0e-3]), quaternion_setpoint=geometry.quaternion_from_euler_xyz(np.array([np.deg2rad(30.0), np.deg2rad(30.0), 0.0]))),
                                0.0, 2.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("lissajous_infty_xy_rp0_quaternion",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 5.0
                            ],
                            [
                                partial(xyrp_lissajous_trajectory_quaternion,
                                    x_amp=10.0e-3, x_hz=1.0*1.5, y_amp=15.0e-3, y_hz=0.5*1.5,
                                    r_amp=0.0, r_hz=0.25, p_amp=0.0, p_hz=0.5,
                                    phi_x=0.0, phi_y=0.0, phi_r=0.0, phi_p=np.pi,
                                    center=np.array([0.0, 0.0, 10.0e-3])),
                                0.0,
                                4.0*8.0/1.5
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("lissajous_infty_xy_rp15_quaternion_8cycles",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                "xyrp_lissajous_eight_T4_x20_y10_rp15_c0010",
                                0.0,
                                4.0*8.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("lissajous_infty_xy_rp30_quaternion_8cycles",
                    ChainedTrajectory(
                        [
                            [
                                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 10.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0),
                                0.0, 2.0
                            ],
                            [
                                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 10.0
                            ],
                            [
                                "xyrp_lissajous_eight_T4_x20_y10_rp30_c0010",
                                0.0,
                                4.0*8.0
                            ]
                        ],
                        loop=False
                    )
)

register_trajectory("simple_take_off_to_15mm_discretized", create_discretized_trajectory(partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 5.0e-3]), end_position=np.array([0.0, 0.0, 15.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=2.0), start_time=0.0, end_time=2.0, step=1e-3, loop=False))
register_trajectory("simple_take_off_to_45mm_discretized", create_discretized_trajectory(partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 10.0e-3]), end_position=np.array([0.0, 0.0, 45.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=10.0), start_time=0.0, end_time=10.0, step=1e-3, loop=False))

register_trajectory("z_boundary_touching_sine_trajectory",
                    ChainedTrajectory(
        [
            [
                partial(simple_linear_trajectory_quaternion, start_position=np.array([0.0, 0.0, 10.0e-3]), end_position=np.array([0.0, 0.0, 28.0e-3]), start_euler_xyz=np.zeros(3), end_euler_xyz=np.zeros(3), duration=5.0),
                0.0, 5.0
            ],
            [
                TrajectoryTransitions.PAUSE_ON_PREV, 0.0, 5.0 # time to remove ss error.
            ],
            [
                partial(sine_z_trajectory_quaternion, amplitude=15.0e-3, frequency=0.5, center=28.0e-3),
                0.0,
                2.0*6.0
            ],
            [
                TrajectoryTransitions.LINEAR_TRANSITION, 0.0, 3.0
            ],
            [
                partial(const_pose_setpoint, position_setpoint=np.array([0.0, 0.0, 43.0e-3]), quaternion_setpoint=IDENTITY_QUATERNION),
                0.0, 5.0
            ]
        ],
        loop=False
    )
)
