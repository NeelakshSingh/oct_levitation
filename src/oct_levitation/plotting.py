import numpy as np
import matplotlib.pyplot as plt
import oct_levitation.geometry as geometry
import pandas as pd

from typing import Optional, Tuple

def plot_6DOF_state_history_euler_xyz(state_history: np.ndarray, figsize: tuple = (30, 10)) -> None:
    """
    Plots the 12 states from the state history of the 6DOF rigid body using the XYZ euler angles for orientation.
    Args:
        state_history (np.ndarray): The state history of the body. Should have shape (N, 12).
                each state sample is of the form [x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz]. All frame
                notations are according to the defined conventions throughout this package.
        figsize (tuple): The figure size of the plot.
    """
    assert state_history.shape[1] == 12, "State history should have 12 columns"
    fig, axs = plt.subplots(2, 6, figsize=figsize)
    axs[0, 0].plot(state_history[:, 0], label='x')
    axs[0, 0].set_title('x')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Position (m)')
    axs[0, 1].plot(state_history[:, 1], label='y')
    axs[0, 1].set_title('y')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Position (m)')
    axs[0, 2].plot(state_history[:, 2], label='z')
    axs[0, 2].set_title('z')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Position (m)')
    axs[0, 3].plot(state_history[:, 3], label='vx')
    axs[0, 3].set_title('vx')
    axs[0, 3].set_xlabel('Time')
    axs[0, 3].set_ylabel('Velocity (m/s)')
    axs[0, 4].plot(state_history[:, 4], label='vy')
    axs[0, 4].set_title('vy')
    axs[0, 4].set_xlabel('Time')
    axs[0, 4].set_ylabel('Velocity (m/s)')
    axs[0, 5].plot(state_history[:, 5], label='vz')
    axs[0, 5].set_title('vz')
    axs[0, 5].set_xlabel('Time')
    axs[0, 5].set_ylabel('Velocity (m/s)')
    axs[1, 0].plot(state_history[:, 6], label='phi')
    axs[1, 0].set_title('phi')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Angle (rad)')
    axs[1, 1].plot(state_history[:, 7], label='theta')
    axs[1, 1].set_title('theta')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Angle (rad)')
    axs[1, 2].plot(state_history[:, 8], label='psi')
    axs[1, 2].set_title('psi')
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel('Angle (rad)')
    axs[1, 3].plot(state_history[:, 9], label='wx')
    axs[1, 3].set_title('wx')
    axs[1, 3].set_xlabel('Time')
    axs[1, 3].set_ylabel('Angular Velocity (rad/s)')
    axs[1, 4].plot(state_history[:, 10], label='wy')
    axs[1, 4].set_title('wy')
    axs[1, 4].set_xlabel('Time')
    axs[1, 4].set_ylabel('Angular Velocity (rad/s)')
    axs[1, 5].plot(state_history[:, 11], label='wz')
    axs[1, 5].set_title('wz')
    axs[1, 5].set_xlabel('Time')
    axs[1, 5].set_ylabel('Angular Velocity (rad/s)')
    plt.show()

def plot_state_history_position3D(state_history: np.ndarray, figsize: tuple = (10,10)) -> None:
    assert state_history.shape[1] == 12, "State history should have 12 columns"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_coordinate_frame(axis, T_0f, size=1, linewidth=3, name=None,
                          xscale=1, yscale=1, zscale=1,
                          x_style='r-', y_style='g-', z_style='b-'):
    """
    Source: https://github.com/ethz-asl/kalibr/blob/master/Schweizer-Messer/sm_python/python/sm/plotCoordinateFrame.py

    Plot a coordinate frame on a 3d axis. In the resulting plot,
    x = red, y = green, z = blue.
    
    plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)

    Args:
        axis: an axis of type matplotlib.axes.Axes3D
        T_0f(): The 4x4 transformation matrix that takes points from the frame of interest, to the plotting frame
        size: the length of each line in the coordinate frame
        linewidth: the width of each line in the coordinate frame
        name: the name of the frame
        xscale: scale factor for the x-axis
        yscale: scale factor for the y-axis
        zscale: scale factor for the z-axis

    see http://matplotlib.sourceforge.net/mpl_toolkits/mplot3d/tutorial.html for more details
    """

    p_f = np.array([[0,0,0,1],
                    [size*xscale,0,0,1],
                    [0,size*yscale,0,1],
                    [0,0,size*zscale,1]]).T
    p_0 = np.dot(T_0f,p_f)

    X = np.append([p_0[:,0].T], [p_0[:,1].T], axis=0 )
    Y = np.append([p_0[:,0].T], [p_0[:,2].T], axis=0 )
    Z = np.append([p_0[:,0].T], [p_0[:,3].T], axis=0 )
    axis.plot3D(X[:,0],X[:,1],X[:,2], x_style, linewidth=linewidth)
    axis.plot3D(Y[:,0],Y[:,1],Y[:,2], y_style, linewidth=linewidth)
    axis.plot3D(Z[:,0],Z[:,1],Z[:,2], z_style, linewidth=linewidth)

    if name is not None:
        axis.text(X[0,0],X[0,1],X[0,2], name, zdir='x')

def plot_6DOF_pose_euler_xyz(state_history: np.ndarray, 
                             orientation_plot_frequency: int = 1,
                             figsize: tuple = (10, 10),
                             frame_size: float = 1.0,
                             frame_linewidth: float = 1.0,
                             xlim: Optional[Tuple] = None,
                             ylim: Optional[Tuple] = None,
                             zlim: Optional[Tuple] = None,
                             write_frame_idx: bool = False) -> None:
    """
    Plots the position and the orientation of the body in 3D space using the XYZ euler angles for orientation.
    Coordinate frame drawn is scaled according to the largest limit along any axis.

    Args
    ----
        state_history (np.ndarray): The state history of the body. Should have shape (N, 12). Each state sample is of the form [x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz]. All frame notations are according to the defined conventions throughout this package.
        orientation_plot_frequency (int): The frequency at which the orientation of the body should be plotted.
        a value of N means orientation is plotted for every Nth position sample.
        figsize (tuple): The figure size of the plot.
        frame_size (float): The size of the frame to be plotted.
        frame_linewidth (float): The width of the frame lines to be plotted.
        kwargs: Additional keyword arguments to be passed to the matplotlib axes function.
    """
    assert state_history.shape[1] == 12, "State history should have 12 columns"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2], color='black')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    axes_limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    axes_ranges = np.diff(axes_limits, axis=1).flatten()
    min_range_idx = np.argmin(axes_ranges)
    scales = np.ones(3)
    for idx in range(3):
        if idx != min_range_idx:
            scales[idx] = axes_ranges[idx]/axes_ranges[min_range_idx] # Upscaling the other axes to match the largest range.
    for i in range(state_history.shape[0]):
        if i % orientation_plot_frequency == 0:
            T_0f = geometry.transformation_matrix_from_euler_xyz(state_history[i, 6:9], state_history[i, :3])
            name = None
            if write_frame_idx:
                name = str(i)
            plot_coordinate_frame(ax, T_0f, size=frame_size, linewidth=frame_linewidth, name=None,
                                  xscale=scales[0], yscale=scales[1], zscale=scales[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_poses_constant_reference(actual_poses: pd.DataFrame, reference_pose: np.ndarray):
    """
    Plots target Euler angles and positions from actual poses DataFrame and a constant reference pose.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual poses (positions and quaternions) with time.
    - reference_pose (np.ndarray): Array of size 7 [x, y, z, qx, qy, qz, qw] representing the constant reference pose.
    """
    # Extract time, positions, and orientations
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Extract reference position and orientation
    reference_position = reference_pose[:3]
    reference_orientation = reference_pose[3:]

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array(geometry.euler_xyz_from_quaternion(reference_orientation))

    # Plot positions
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[0, i].axhline(y=reference_position[i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[0, i].set_title(f"Position {axis}")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Position (m)")
        axs[0, i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[1, i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[1, i].axhline(y=reference_euler[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[1, i].set_title(angle)
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Angle (rad)")
        axs[1, i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_positions_constant_reference(actual_poses: pd.DataFrame, reference_position: np.ndarray):
    """
    Plots target positions from actual poses DataFrame and a constant reference position.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual positions with time.
    - reference_position (np.ndarray): Array of size 3 [x, y, z] representing the constant reference position.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values

    # Plot positions
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[i].axhline(y=reference_position[i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[i].set_title(f"Position {axis}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Position (m)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_orientations_constant_reference(actual_poses: pd.DataFrame, reference_orientation: np.ndarray):
    """
    Plots target orientations (Euler angles) from actual poses DataFrame and a constant reference orientation.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual orientations (quaternions) with time.
    - reference_orientation (np.ndarray): Array of size 4 [qx, qy, qz, qw] representing the constant reference quaternion.
    """
    time = actual_poses['time'].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array(geometry.euler_xyz_from_quaternion(reference_orientation))

    # Plot Euler angles
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[i].axhline(y=reference_euler[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[i].set_title(angle)
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Angle (rad)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_poses_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame):
    """
    Plots target Euler angles and positions from actual poses DataFrame and variable reference poses DataFrame.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame with actual poses (positions and quaternions) and time.
    - reference_poses (pd.DataFrame): DataFrame with reference poses (positions and quaternions) and time.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values
    reference_positions = reference_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values
    reference_orientations = reference_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in reference_orientations])

    # Plot positions
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[0, i].plot(time, reference_positions[:, i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[0, i].set_title(f"Position {axis}")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Position (m)")
        axs[0, i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[1, i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[1, i].plot(time, reference_euler[:, i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[1, i].set_title(angle)
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Angle (rad)")
        axs[1, i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_positions_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame):
    """
    Plots target positions from actual poses DataFrame and variable reference positions DataFrame.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame with actual positions and time.
    - reference_poses (pd.DataFrame): DataFrame with reference positions and time.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values
    reference_positions = reference_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values

    # Plot positions
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[i].plot(time, reference_positions[:, i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[i].set_title(f"Position {axis}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Position (m)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_orientations_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame):
    """
    Plots target orientations (Euler angles) from actual poses DataFrame and variable reference orientations DataFrame.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame with actual orientations (quaternions) and time.
    - reference_poses (pd.DataFrame): DataFrame with reference orientations (quaternions) and time.
    """
    time = actual_poses['time'].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values
    reference_orientations = reference_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in reference_orientations])

    # Plot Euler angles
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[i].plot(time, reference_euler[:, i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[i].set_title(angle)
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Angle (rad)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_3d_poses_with_arrows_non_constant_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame, arrow_interval: int = 10, frame_size: float = 0.01, frame_interval: int = 10):
    """
    Plots the actual and reference poses in 3D space with arrows indicating the direction of forward progress in time.
    Reference poses are taken from the provided DataFrame and are non-constant.

    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual poses (positions and quaternions) with time.
    - reference_poses (pd.DataFrame): DataFrame containing reference poses (positions and quaternions) with time.
    - arrow_interval (int): Interval for plotting arrows indicating the direction of motion.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values
    reference_positions = reference_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values
    reference_orientations = reference_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert reference quaternions to Euler angles using `euler_xyz_from_quaternion`
    reference_eulers = np.array([geometry.euler_xyz_from_quaternion(q) for q in reference_orientations])

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot actual positions
    ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], color='black', label='Actual Path')

    # Plot reference positions (non-constant)
    ax.plot(reference_positions[:, 0], reference_positions[:, 1], reference_positions[:, 2], color='red', linestyle='--', label='Reference Path')

    # Plot arrows for actual positions to indicate direction of forward progress
    for i in range(arrow_interval, len(time), arrow_interval):
        ax.quiver(actual_positions[i-1, 0], actual_positions[i-1, 1], actual_positions[i-1, 2],
                  actual_positions[i, 0] - actual_positions[i-1, 0], 
                  actual_positions[i, 1] - actual_positions[i-1, 1], 
                  actual_positions[i, 2] - actual_positions[i-1, 2], 
                  color='black', arrow_length_ratio=0.1)

    # Plot arrows for reference positions to indicate direction of forward progress
    for i in range(arrow_interval, len(time), arrow_interval):
        ax.quiver(reference_positions[i-1, 0], reference_positions[i-1, 1], reference_positions[i-1, 2],
                  reference_positions[i, 0] - reference_positions[i-1, 0], 
                  reference_positions[i, 1] - reference_positions[i-1, 1], 
                  reference_positions[i, 2] - reference_positions[i-1, 2], 
                  color='red', linestyle='--', arrow_length_ratio=0.1)

    # Add reference pose frames (non-constant)
    for i in range(0, len(time), frame_interval):  # plot frames every 10% of time
        reference_T_0f = geometry.transformation_matrix_from_euler_xyz(reference_eulers[i], reference_positions[i])
        plot_coordinate_frame(ax, reference_T_0f, size=frame_size, linewidth=1.5, name='Reference Pose', xscale=1, yscale=1, zscale=1,
                              x_style='r--', y_style='g--', z_style='b--')

    # Add coordinate frames at selected positions in the actual path
    for i in range(0, len(time), frame_interval):  # plot frames every 10% of time
        actual_T_0f = geometry.transformation_matrix_from_quaternion(actual_poses.iloc[i, 4:8], actual_poses.iloc[i, 1:4])
        plot_coordinate_frame(ax, actual_T_0f, size=frame_size, linewidth=1.5, name=None)

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Actual Pose v/s Reference Pose")

    # Show legend
    ax.legend()

    # Show plot
    plt.show()



def plot_3d_poses_with_arrows_constant_reference(actual_poses: pd.DataFrame, reference_pose: np.ndarray, arrow_interval: int = 10, frame_size: float = 0.01, frame_interval: int = 10):
    """
    Plots the actual poses in 3D space with arrows indicating the direction of forward progress in time,
    and a constant reference pose in 3D space. The reference pose is given in terms of x, y, z, qx, qy, qz, qw,
    which will be converted to Euler angles using `euler_xyz_from_quaternion`.

    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual poses (positions and quaternions) with time.
    - reference_pose (np.ndarray): Array containing a constant reference pose in the form [x, y, z, qx, qy, qz, qw].
    - arrow_interval (int): Interval for plotting arrows indicating the direction of motion.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values
    
    # Reference pose (constant)
    reference_position = reference_pose[:3]
    reference_orientation = reference_pose[3:]

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot actual positions
    ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], color='black', label='Actual Path')

    # Plot constant reference position (horizontal line)
    ax.plot(np.full_like(time, reference_position[0]), 
            np.full_like(time, reference_position[1]),
            np.full_like(time, reference_position[2]), 
            color='red', linestyle='--', label='Constant Reference')

    # Plot arrows for actual positions to indicate direction of forward progress
    for i in range(arrow_interval, len(time), arrow_interval):
        ax.quiver(actual_positions[i-1, 0], actual_positions[i-1, 1], actual_positions[i-1, 2],
                  actual_positions[i, 0] - actual_positions[i-1, 0], 
                  actual_positions[i, 1] - actual_positions[i-1, 1], 
                  actual_positions[i, 2] - actual_positions[i-1, 2], 
                  color='black', arrow_length_ratio=0.1)

    # Add reference pose frame (constant)
    reference_T_0f = geometry.transformation_matrix_from_quaternion(reference_orientation, reference_position)
    plot_coordinate_frame(ax, reference_T_0f, size=frame_size, linewidth=1.5, name='Constant Reference Pose', xscale=1, yscale=1, zscale=1,
                          x_style='r--', y_style='g--', z_style='b--')

    # Add coordinate frames at selected positions in the actual path
    for i in range(0, len(time), frame_interval): 
        actual_T_0f = geometry.transformation_matrix_from_quaternion(actual_orientations[i], actual_positions[i])
        plot_coordinate_frame(ax, actual_T_0f, size=frame_size, linewidth=1.5, name=None)

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Actual Pose v/s Reference Pose")

    # Show legend
    ax.legend()

    # Show plot
    plt.show()