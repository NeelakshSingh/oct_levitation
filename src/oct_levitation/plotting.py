import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import oct_levitation.geometry as geometry
import pandas as pd
from control_utils.general.utilities import quaternion_to_normal_vector, angles_from_normal_vector

import os
import subprocess

from typing import Optional, Tuple, List

INKSCAPE_PATH = "/usr/bin/inkscape" # default

######################################
# PLOTTING UTILITIES
######################################

xkcd_contrast_colors = {
    "Blue": "#0343df",
    "Red": "#e50000",
    "Green": "#15b01a",
    "Orange": "#f97306",
    "Purple": "#7e1e9c",
    "Yellow": "#ffff14",
    "Black": "#000000",
    "Cyan": "#00ffff",
    "Pink": "#ff81c0",
    "Brown": "#653700",
    "Light Gray": "#d3d3d3",
    "Teal": "#029386",
}

xkcd_contrast_list = list(xkcd_contrast_colors.items())

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return matplotlib.colormaps[name]

def export_to_emf(svg_file: str, emf_file: str, inkscape_path: str = INKSCAPE_PATH) -> None:
    """
    Converts an SVG file to an EMF file using Inkscape.

    Parameters:
    - svg_file (str): Path to the SVG file to convert.
    - emf_file (str): Path to save the resulting EMF file.

    Returns:
    - None
    """
    if not os.path.exists(inkscape_path):
        raise FileNotFoundError("Inkscape executable not found at the specified path.")
    subprocess.run([inkscape_path, svg_file, '-M', emf_file], check=True)

######################################
# PLOTTING POSES
######################################

def plot_6DOF_state_history_euler_xyz(state_history: np.ndarray, figsize: tuple = (30, 10),
                                      save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs) -> None:
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
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()

def plot_state_history_position3D(state_history: np.ndarray, figsize: tuple = (10,10),
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs) -> None:
    assert state_history.shape[1] == 12, "State history should have 12 columns"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
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

def plot_poses_constant_reference(actual_poses: pd.DataFrame, reference_pose: np.ndarray,
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    """
    Plots target Euler angles and positions from actual poses DataFrame and a constant reference pose.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual poses (positions and quaternions) with time.
    - reference_pose (np.ndarray): Array of size 7 [x, y, z, qx, qy, qz, qw] representing the constant reference pose.

    Returns:
    - fig (plt.Figure)
    """
    # Extract time, positions, and orientations
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000 # in mm
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Extract reference position and orientation
    reference_position = reference_pose[:3]
    reference_orientation = reference_pose[3:]

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array(geometry.euler_xyz_from_quaternion(reference_orientation))

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)
    reference_euler = np.rad2deg(reference_euler)

    # Plot positions
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[0, i].axhline(y=reference_position[i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[0, i].set_title(f"Position {axis} of Dipole Center")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Position (mm)")
        axs[0, i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[1, i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[1, i].axhline(y=reference_euler[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[1, i].set_title(f"{angle} of Dipole Center")
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Angle (deg)")
        axs[1, i].legend()

    # Adjust layout
    plt.tight_layout()
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs


def plot_positions_constant_reference(actual_poses: pd.DataFrame, reference_position: np.ndarray,
                                      save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
    """
    Plots target positions from actual poses DataFrame and a constant reference position.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual positions with time.
    - reference_position (np.ndarray): Array of size 3 [x, y, z] representing the constant reference position.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000

    # Plot positions
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[i].axhline(y=reference_position[i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[i].set_title(f"Position {axis} of Dipole Center Frame")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Position (mm)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()

    return fig, axs

def plot_z_position_constant_reference(actual_poses: pd.DataFrame, reference_z: float,
                                       save_as: str=None,
                                       save_as_emf: bool=False,
                                       inkscape_path: str=None, **kwargs):
    """
    Plots target positions from actual poses DataFrame and a constant reference position.
    All inputs are in SI units.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual positions with time.
    - reference_z (float): The desired z position.
    """
    time = actual_poses['time'].values
    actual_z_position = actual_poses['transform.translation.z'].values*1000 # in mm
    reference_z = reference_z*1000 # in mm

    # Plot positions
    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot()

    ax.plot(time, actual_z_position, label=f"Actual Z", **kwargs)
    ax.axhline(y=reference_z, label=f"Reference Z", linestyle='dashed', color='r')
    ax.set_title(f"Z Position of Dipole Center Frame")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (mm)")
    ax.legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()

    return fig, ax

def plot_alpha_beta_constant_reference(actual_poses: pd.DataFrame, reference_angles: np.ndarray,
                                       save_as: str=None,
                                       save_as_emf: bool=False,
                                       inkscape_path: str=None, **kwargs):
    time = actual_poses['time'].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_yx = np.array([angles_from_normal_vector(
        quaternion_to_normal_vector(quaternion)
    ) for quaternion in actual_orientations])

    actual_xy = np.roll(actual_yx, 1, axis=1)

    # Convert to degrees
    actual_xy = np.rad2deg(actual_xy)
    reference_angles = np.rad2deg(reference_angles)

    # Plot Euler angles
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.5), sharex=True, sharey=True)

    fig.suptitle("Angles of Dipole Fixed Frame Z-Axis with World's Z-Axis")
    for i, angle in enumerate(['Beta', 'Alpha']):
        axs[i].plot(time, actual_xy[:, i], label=f"Actual {angle}")
        axs[i].axhline(y=reference_angles[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[i].set_title(f"{angle} of Dipole Center Frame")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Angle (deg)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_orientations_constant_reference(actual_poses: pd.DataFrame, reference_orientation: np.ndarray,
                                         save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
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

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)
    reference_euler = np.rad2deg(reference_euler)

    # Plot Euler angles
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[i].axhline(y=reference_euler[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[i].set_title(angle + " of Dipole Center Frame")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Angle (deg)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_exyz_roll_pitch_constant_reference(actual_poses: pd.DataFrame, reference_orientation: np.ndarray,
                                            save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
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

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)
    reference_euler = np.rad2deg(reference_euler)

    # Plot Euler angles
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[i].axhline(y=reference_euler[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[i].set_title(angle + " of Dipole Center Frame")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Angle (deg)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_poses_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
    """
    Plots target Euler angles and positions from actual poses DataFrame and variable reference poses DataFrame.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame with actual poses (positions and quaternions) and time.
    - reference_poses (pd.DataFrame): DataFrame with reference poses (positions and quaternions) and time.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000 # in mm
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values
    reference_positions = reference_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000 # in mm
    reference_orientations = reference_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in reference_orientations])

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)
    reference_euler = np.rad2deg(reference_euler)

    # Plot positions
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[0, i].plot(time, reference_positions[:, i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[0, i].set_title(f"Position {axis} of Dipole Center Frame")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Position (mm)")
        axs[0, i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[1, i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[1, i].plot(time, reference_euler[:, i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[1, i].set_title(angle)
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Angle (deg)")
        axs[1, i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs


def plot_positions_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                                      save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
    """
    Plots target positions from actual poses DataFrame and variable reference positions DataFrame.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame with actual positions and time.
    - reference_poses (pd.DataFrame): DataFrame with reference positions and time.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000
    reference_positions = reference_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000

    # Plot positions
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[i].plot(time, reference_positions[:, i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[i].set_title(f"Position {axis} of Dipole Center Frame")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Position (mm)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_orientations_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                                         save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
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

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)
    reference_euler = np.rad2deg(reference_euler)

    # Plot Euler angles
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[i].plot(time, reference_euler[:, i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[i].set_title(angle + " of Dipole Center Frame.")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Angle (deg)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_3d_poses_with_arrows_non_constant_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame, arrow_interval: int = 10, frame_size: float = 0.01, frame_interval: int = 10,
                                                     save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
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
    ax.set_title("Actual Pose v/s Reference Pose of Dipole Center Frame")

    # Show legend
    ax.legend()

    # Show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, ax



def plot_3d_poses_with_arrows_constant_reference(actual_poses: pd.DataFrame, reference_pose: np.ndarray, arrow_interval: int = 10, frame_size: float = 0.01, frame_interval: int = 10,
                                                 save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
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
    ax.set_title("Actual Pose v/s Reference Pose of Dipole Center Frame")

    # Show legend
    ax.legend()

    # Show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, ax

######################################
# PLOTTING CURRENTS, CONTROL INPUTS, FIELDS
######################################

def plot_actual_currents(system_state_df: pd.DataFrame,
                         save_as: str=None,
                         save_as_emf: bool=False,
                         inkscape_path: str=None, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    # Plot each current column in its respective subplot
    # Create subplots in a 2x4 layout
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle("Actual Currents vs Time", fontsize=16)  # Main title for the figure

    # Flatten the 2D axes array for easier iteration
    axs = axs.flatten()
    for i in range(8):
        axs[i].plot(system_state_df['time'], system_state_df[f'currents_reg_{i}'], label=f'Actual Current {i+1}', color='b', **kwargs)
        axs[i].set_title(f'Currents in Coil {i+1}')
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Current (A)")
        axs[i].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()

    return fig, axs

def plot_currents_with_reference(system_state_df: pd.DataFrame, des_currents_df: pd.DataFrame,
                                save_as: str=None,
                                save_as_emf: bool=False,
                                inkscape_path: str=None, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    # Plot each current column in its respective subplot
    # Create subplots in a 2x4 layout
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle("Actual and Desired Currents vs Time", fontsize=16)  # Main title for the figure

    # Flatten the 2D axes array for easier iteration
    axs = axs.flatten()
    for i in range(8):
        axs[i].plot(system_state_df['time'], system_state_df[f'currents_reg_{i}'], label=f'Actual Current {i+1}', color='b', **kwargs)
        axs[i].plot(des_currents_df['time'], des_currents_df[f'des_currents_reg_{i}'], label=f'Desired Current {i+1}', color='g', **kwargs)
        axs[i].set_title(f'Currents in Coil {i+1}')
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Current (A)")
        axs[i].legend()
        axs[i].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()

    return fig, axs

def plot_forces_and_torques(df: pd.DataFrame,
                            title: str="Desired Forces and Torques",
                            save_as: str=None,
                            save_as_emf: bool=False,
                            inkscape_path: str=None, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    """
    Plots forces (Fx, Fy, Fz) and torques (Tx, Ty, Tz) from a DataFrame with time on the x-axis.
    Forces are converted to milliNewtons (mN) and torques to milliNewton-millimeters (mN-mm).

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'time', 'array_0' to 'array_5' columns.
                           'array_0' = Fx, 'array_1' = Fy, 'array_2' = Fz
                           'array_3' = Tx, 'array_4' = Ty, 'array_5' = Tz
    """
    # Extract relevant columns
    time = df['time']
    Fx = df['array_0'] * 1e3  # Convert to mN
    Fy = df['array_1'] * 1e3
    Fz = df['array_2'] * 1e3
    Tx = df['array_3'] * 1e6  # Convert to mN-mm
    Ty = df['array_4'] * 1e6
    Tz = df['array_5'] * 1e6

    # Create subplots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Plot forces (mN)
    axes[0, 0].plot(time, Fx, color='b')
    axes[0, 0].set_ylabel('Force (mN)')
    axes[0, 0].set_title('Fx')

    axes[0, 1].plot(time, Fy, color='g')
    axes[0, 1].set_title('Fy')

    axes[0, 2].plot(time, Fz, color='k')
    axes[0, 2].set_title('Fz')

    # Share y-axis for forces
    axes[0, 1].sharey(axes[0, 0])
    axes[0, 2].sharey(axes[0, 0])

    # Plot torques (mN-mm)
    axes[1, 0].plot(time, Tx, color='b')
    axes[1, 0].set_ylabel('Torque (mN-mm)')
    axes[1, 0].set_title('Tx')

    axes[1, 1].plot(time, Ty, color='g')
    axes[1, 1].set_title('Ty')

    axes[1, 2].plot(time, Tz, color='k')
    axes[1, 2].set_title('Tz')

    # Share y-axis for torques
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 2].sharey(axes[1, 0])

    # Autoscale shared axes
    for ax_row in axes: 
        for ax in ax_row:
            ax.relim()   
            ax.autoscale()

    # Share x-axis for all subplots
    for ax in axes[1, :]:
        ax.set_xlabel('Time (s)')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the title
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axes

def plot_3d_quiver(dataframe: pd.DataFrame, 
                   scale_factor: float=1.0, 
                   save_as: str=None, 
                   save_as_emf: bool=False, 
                   inkscape_path: str=None, **kwargs):
    """
    Plots a 3D quiver plot using position and field vector data from a pandas dataframe.

    Parameters:
        dataframe (pd.DataFrame): A dataframe containing the columns 'Px', 'Py', 'Pz', 'Bx', 'By', 'Bz'.
        scale_factor (float): A scaling factor to adjust the length of the arrows. Default is 1.0.
        save_as (str): File path to save the plot as an SVG. Use '.svg' extension.
        save_as_emf (bool): Whether to save an additional EMF file (requires Inkscape). Default is False.
        inkscape_path (str): If save_as_emf is true then this argument must be set to inkscape's binary path in your system.
        **kwargs: Additional keyword arguments to pass to the `ax.quiver` function.

    Returns:
        None
    """
    required_columns = {'Px', 'Py', 'Pz', 'Bx', 'By', 'Bz'}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(f"Dataframe must contain the columns: {required_columns}")
    
    Px, Py, Pz = dataframe['Px'] * 1000, dataframe['Py'] * 1000, dataframe['Pz'] * 1000
    Bx, By, Bz = dataframe['Bx'] * 1000, dataframe['By'] * 1000, dataframe['Bz'] * 1000
    Bx_scaled, By_scaled, Bz_scaled = Bx * scale_factor, By * scale_factor, Bz * scale_factor
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(Px, Py, Pz, Bx_scaled, By_scaled, Bz_scaled, length=1, normalize=False, **kwargs)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Quiver Plot of Field Vectors (Field in mT)')
    
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, ax

def plot_3d_comparison_quiver(dataframes: List[pd.DataFrame],
                              labels: List[str],
                              scale_factor: float=1.0, 
                              save_as: str=None, 
                              save_as_emf: bool=False, 
                              inkscape_path: str=None, **kwargs):
    """
    Plots a 3D quiver plot using position and field vector data from a pandas dataframe.

    Parameters:
        dataframes (list[pd.DataFrame]): List of dataframes containing the columns 'Px', 'Py', 'Pz', 'Bx', 'By', 'Bz'.
        labels (list[str]): List of dataframe labels for plotting.
        scale_factor (float): A scaling factor to adjust the length of the arrows. Default is 1.0.
        save_as (str): File path to save the plot as an SVG. Use '.svg' extension.
        save_as_emf (bool): Whether to save an additional EMF file (requires Inkscape). Default is False.
        inkscape_path (str): If save_as_emf is true then this argument must be set to inkscape's binary path in your system.
        **kwargs: Additional keyword arguments to pass to the `ax.quiver` function.

    Returns:
        None
    """
    required_columns = {'Px', 'Py', 'Pz', 'Bx', 'By', 'Bz'}
    cmap = get_cmap(len(dataframes))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, (dataframe, label) in enumerate(zip(dataframes, labels)):
        if not (required_columns.issubset(dataframe.columns)):
            raise ValueError(f"Dataframe must contain the columns: {required_columns}")
    
        Px, Py, Pz = dataframe['Px'] * 1000, dataframe['Py'] * 1000, dataframe['Pz'] * 1000
        Bx, By, Bz = dataframe['Bx'] * 1000, dataframe['By'] * 1000, dataframe['Bz'] * 1000
        Bx_scaled, By_scaled, Bz_scaled = Bx * scale_factor, By * scale_factor, Bz * scale_factor
        
        ax.quiver(Px, Py, Pz, Bx_scaled, By_scaled, Bz_scaled, length=1, normalize=False,
                label=label, color=xkcd_contrast_list[i%len(xkcd_contrast_list)], **kwargs)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'3D Quiver Plot of Field Vectors (Field in mT), Scale: {scale_factor}x')
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, ax

def plot_field_components_at_dipole_center(pose_df, desired_currents_df, actual_currents_df, calibration_fn,
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
    """
    Plots the 3 magnetic field components (Bx, By, Bz) over time for both actual and desired fields.
    Shared x and y axes across subplots. Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibration_fn (Callable): Function that maps position to an 8x8 matrix A
        save_as (str): Filename to save the plot as SVG/EMF (without extension).
        **kwargs: Additional parameters for plt.plot().
    """
    combined_desired = pd.merge_asof(pose_df, desired_currents_df, on='time')
    combined_actual = pd.merge_asof(pose_df, actual_currents_df, on='time')

    time = combined_desired['time']
    desired_fields = {'Bx': [], 'By': [], 'Bz': []}
    actual_fields = {'Bx': [], 'By': [], 'Bz': []}

    for i in range(len(combined_desired)):
        position = [
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ]

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        A = calibration_fn(np.array(position))
        desired_field = A @ desired_currents
        actual_field = A @ actual_currents

        for idx, key in enumerate(['Bx', 'By', 'Bz']):
            desired_fields[key].append(desired_field[idx] * 1000)  # Convert to mT
            actual_fields[key].append(actual_field[idx] * 1000)

    # Plot the field components with shared axes
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    for i, key in enumerate(['Bx', 'By', 'Bz']):
        axs[i].plot(time, actual_fields[key], label=f'Actual {key}', linestyle='-', color='tab:red', **kwargs)
        axs[i].plot(time, desired_fields[key], label=f'Desired {key}', linestyle='-', color='tab:blue', **kwargs)
        axs[i].set_ylabel(f'{key} [mT]')
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Actual v/s Desired Field Components at Dipole Center")
    plt.tight_layout()
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_gradient_components_at_dipole_center(pose_df, desired_currents_df, actual_currents_df, calibration_fn,
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
    """
    Plots the 5 gradient components (dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz) over time 
    for both actual and desired gradients. Shared x and y axes across subplots.
    Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibration_fn (Callable): Function that maps position to an 8x8 matrix A
        save_as (str): Filename to save the plot as SVG/EMF (without extension).
        **kwargs: Additional parameters for plt.plot().
    """
    combined_desired = pd.merge_asof(pose_df, desired_currents_df, on='time')
    combined_actual = pd.merge_asof(pose_df, actual_currents_df, on='time')

    time = combined_desired['time']
    desired_gradients = {'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [], 'dBy/dy': [], 'dBy/dz': []}
    actual_gradients = {'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [], 'dBy/dy': [], 'dBy/dz': []}

    for i in range(len(combined_desired)):
        position = [
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ]

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        A = calibration_fn(np.array(position))
        desired_field = A @ desired_currents
        actual_field = A @ actual_currents

        gradient_keys = ['dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']
        for idx, key in enumerate(gradient_keys, start=3):
            desired_gradients[key].append(desired_field[idx] * 1000)
            actual_gradients[key].append(actual_field[idx] * 1000)

    # Plot the gradient components with shared axes
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True, sharey=True)
    for i, key in enumerate(['dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']):
        axs[i].plot(time, actual_gradients[key], label=f'Actual {key}', linestyle='-', color='tab:red', **kwargs)
        axs[i].plot(time, desired_gradients[key], label=f'Desired {key}', linestyle='-', color='tab:blue', **kwargs)
        axs[i].set_ylabel(f'{key} [mT/m]')
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Actual v/s Desired Gradients at Dipole Center")
    plt.tight_layout()

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axs

def plot_field_components_at_dipole_center_const_actuation_position(
                          pose_df, desired_currents_df, actual_currents_df, calibration_fn, des_actuation_pos=np.zeros(3),
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
    """
    Plots the 3 magnetic field components (Bx, By, Bz) over time for both actual and desired fields.
    Where the desired fields are calculated based on the actuation matrix calcualted at the given 
    desired position.
    Shared x and y axes across subplots. Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibration_fn (Callable): Function that maps position to an 8x8 matrix A
        des_actuation_pos (np.ndarray): The position which is used to calculate the desired field values.
            Defaults to origin.
        save_as (str): Filename to save the plot as SVG/EMF (without extension).
        **kwargs: Additional parameters for plt.plot().
    """
    combined_desired = pd.merge_asof(pose_df, desired_currents_df, on='time')
    combined_actual = pd.merge_asof(pose_df, actual_currents_df, on='time')

    time = combined_desired['time']
    desired_fields = {'Bx': [], 'By': [], 'Bz': []}
    actual_fields = {'Bx': [], 'By': [], 'Bz': []}

    for i in range(len(combined_desired)):
        position = [
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ]

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        A = calibration_fn(np.array(position))
        desired_field = calibration_fn(des_actuation_pos) @ desired_currents
        actual_field = A @ actual_currents

        for idx, key in enumerate(['Bx', 'By', 'Bz']):
            desired_fields[key].append(desired_field[idx] * 1000)  # Convert to mT
            actual_fields[key].append(actual_field[idx] * 1000)

    # Plot the field components with shared axes
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True, sharey=True)
    for i, key in enumerate(['Bx', 'By', 'Bz']):
        axs[i].plot(time, actual_fields[key], label=f'Actual {key}', linestyle='-', color='tab:red', **kwargs)
        axs[i].plot(time, desired_fields[key], label=f'Desired {key}', linestyle='-', color='tab:blue', **kwargs)
        axs[i].set_ylabel(f'{key} [mT]')
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Actual v/s Desired Field Components at Dipole Center for A({des_actuation_pos})")
    plt.tight_layout()
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_gradient_components_at_dipole_center_const_actuation_position(
                          pose_df, desired_currents_df, actual_currents_df, calibration_fn, des_actuation_pos=np.zeros(3),
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=None, **kwargs):
    """
    Plots the 5 gradient components (dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz) over time 
    for both actual and desired gradients. Shared x and y axes across subplots.
    Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibration_fn (Callable): Function that maps position to an 8x8 matrix A
        des_actuation_pos (np.ndarray): The position which is used to calculate the desired field values.
            Defaults to origin.
        save_as (str): Filename to save the plot as SVG/EMF (without extension).
        **kwargs: Additional parameters for plt.plot().
    """
    combined_desired = pd.merge_asof(pose_df, desired_currents_df, on='time')
    combined_actual = pd.merge_asof(pose_df, actual_currents_df, on='time')

    time = combined_desired['time']
    desired_gradients = {'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [], 'dBy/dy': [], 'dBy/dz': []}
    actual_gradients = {'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [], 'dBy/dy': [], 'dBy/dz': []}

    for i in range(len(combined_desired)):
        position = [
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ]

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        A = calibration_fn(np.array(position))
        desired_field = calibration_fn(des_actuation_pos) @ desired_currents
        actual_field = A @ actual_currents

        gradient_keys = ['dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']
        for idx, key in enumerate(gradient_keys, start=3):
            desired_gradients[key].append(desired_field[idx] * 1000)
            actual_gradients[key].append(actual_field[idx] * 1000)

    # Plot the gradient components with shared axes
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True, sharey=True)
    for i, key in enumerate(['dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']):
        axs[i].plot(time, actual_gradients[key], label=f'Actual {key}', linestyle='-', color='tab:red', **kwargs)
        axs[i].plot(time, desired_gradients[key], label=f'Desired {key}', linestyle='-', color='tab:blue', **kwargs)
        axs[i].set_ylabel(f'{key} [mT/m]')
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Actual v/s Desired Gradients at Dipole Center For A({des_actuation_pos})")
    plt.tight_layout()

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axs