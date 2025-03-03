import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.figure import Figure

import oct_levitation.geometry as geometry
import oct_levitation.common as common
import oct_levitation.mechanical as mechanical
from oct_levitation.processing_utils import get_signal_fft
import os

import pandas as pd

import subprocess

import scipy.signal as signal
import scipy.fft as scifft

from typing import Optional, Tuple, List, Callable, Dict
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction
from tvtk.api import tvtk
from warnings import warn

INKSCAPE_PATH = "/usr/bin/inkscape" # default

### NOTE: An older version of mayavi i.e. 4.7.2 is used for this library, the docs for which
### are no longer available on the original website. Consider using the wayback machine:
### https://web.archive.org/web/20210220173347/https://docs.enthought.com/mayavi/mayavi/index.html

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

xkcd_contrast_list = list(xkcd_contrast_colors.values())

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

def wrench_stamped_df_to_array_df(ft_df: pd.DataFrame) -> pd.DataFrame:
    ft_df = ft_df.rename(columns={
                            "wrench.force.x": "array_0",
                            "wrench.force.y": "array_1",
                            "wrench.force.z": "array_2",
                            "wrench.torque.x": "array_3",
                            "wrench.torque.y": "array_4",
                            "wrench.torque.z": "array_5"
                        }, errors="raise", inplace=False)
    return ft_df

### Works but doesn't do column spans
# def arrange_subplots(arrangement, x_share=None, y_share=None):
#     """
#     Arrange a nested list of matplotlib Axes objects into a structured subplot layout,
#     while allowing flexible axis sharing configurations. Replots the data into a new figure.
    
#     Parameters:
#         arrangement (list): A nested list defining the layout of subplots.
#         x_share (list, optional): A nested list specifying x-axis sharing (same number means shared).
#         y_share (list, optional): A nested list specifying y-axis sharing (same number means shared).

#     Returns:
#         fig, ax_grid: The created figure and a grid of subplot axes.
#     """
    
#     def count_grid_size(arr, row=0, col=0, max_size=[0, 0]):
#         """Recursively determine the required number of rows and columns."""
#         if isinstance(arr, list):
#             for i, sub_arr in enumerate(arr):
#                 count_grid_size(sub_arr, row + (i if col == 0 else 0), col + (0 if i == 0 else 1), max_size)
#         else:
#             max_size[0] = max(max_size[0], row + 1)
#             max_size[1] = max(max_size[1], col + 1)

#     # Compute required grid size
#     max_size = [0, 0]
#     count_grid_size(arrangement, max_size=max_size)
#     max_rows, max_cols = max_size

#     # Create figure and subplots
#     fig, ax_grid = plt.subplots(max_rows, max_cols, figsize=(5 * max_cols, 3 * max_rows), squeeze=False)

#     # Dicts to track shared axes
#     shared_x = {}
#     shared_y = {}

#     def get_shared_axis(share_dict, share_id, default_ax):
#         """Get or assign a shared axis based on an identifier."""
#         if share_id not in share_dict:
#             share_dict[share_id] = default_ax
#         return share_dict[share_id]

#     # Recursive function to place axes and replot data
#     def place_axes(arr, row=0, col=0):
#         if isinstance(arr, list):
#             for i, sub_arr in enumerate(arr):
#                 place_axes(sub_arr, row + (i if col == 0 else 0), col + (0 if i == 0 else 1))
#         else:
#             target_ax = ax_grid[row, col]
#             source_ax = arr

#             # Share X-axis if required
#             if x_share:
#                 share_id = x_share[row][col] if isinstance(x_share[row], list) else x_share[row]
#                 if share_id != 0:
#                     target_ax.sharex(get_shared_axis(shared_x, share_id, target_ax))

#             # Share Y-axis if required
#             if y_share:
#                 share_id = y_share[row][col] if isinstance(y_share[row], list) else y_share[row]
#                 if share_id != 0:
#                     target_ax.sharey(get_shared_axis(shared_y, share_id, target_ax))

#             # Copy data
#             for line in source_ax.get_lines():
#                 target_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), linestyle=line.get_linestyle(),
#                                linewidth=line.get_linewidth(), color=line.get_color(), marker=line.get_marker())
            
#             # Copy labels, limits, and title
#             target_ax.set_xlabel(source_ax.get_xlabel())
#             target_ax.set_ylabel(source_ax.get_ylabel())
#             target_ax.set_title(source_ax.get_title())
#             target_ax.set_xlim(source_ax.get_xlim())
#             target_ax.set_ylim(source_ax.get_ylim())
#             target_ax.legend()

#     # Place axes and replot everything
#     place_axes(arrangement)

#     plt.tight_layout()
#     plt.show()
#     return fig, ax_grid

def arrange_subplots(arrangement, x_share=None, y_share=None):
    """
    Arrange a nested list of matplotlib Axes objects into a structured subplot layout,
    while allowing flexible axis sharing configurations. Replots the data into a new figure.
    
    Parameters:
        arrangement (list): A nested list defining the layout of subplots.
        x_share (list, optional): A nested list specifying x-axis sharing (same number means shared).
        y_share (list, optional): A nested list specifying y-axis sharing (same number means shared).

    Returns:
        fig, ax_grid: The created figure and a grid of subplot axes.
    """
    num_rows = len(arrangement)
    num_cols = max(len(row) if isinstance(row, list) else 1 for row in arrangement)
    
    fig, ax_grid = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 3 * num_rows), squeeze=False)
    
    shared_x = {}
    shared_y = {}
    
    def get_shared_axis(share_dict, share_id, default_ax):
        if share_id not in share_dict:
            share_dict[share_id] = default_ax
        return share_dict[share_id]
    
    for row_idx, row in enumerate(arrangement):
        if not isinstance(row, list):
            row = [row]
        num_subplots = len(row)
        col_width = num_cols / num_subplots
        
        for col_idx, source_ax in enumerate(row):
            target_ax = fig.add_subplot(num_rows, num_cols, row_idx * num_cols + col_idx + 1)
            
            if x_share:
                share_id = x_share[row_idx][col_idx] if isinstance(x_share[row_idx], list) else x_share[row_idx]
                if share_id != 0:
                    target_ax.sharex(get_shared_axis(shared_x, share_id, target_ax))
            
            if y_share:
                share_id = y_share[row_idx][col_idx] if isinstance(y_share[row_idx], list) else y_share[row_idx]
                if share_id != 0:
                    target_ax.sharey(get_shared_axis(shared_y, share_id, target_ax))
            
            for line in source_ax.get_lines():
                target_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                               linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                               color=line.get_color(), marker=line.get_marker())
            
            target_ax.set_xlabel(source_ax.get_xlabel())
            target_ax.set_ylabel(source_ax.get_ylabel())
            target_ax.set_title(source_ax.get_title())
            target_ax.set_xlim(source_ax.get_xlim())
            target_ax.set_ylim(source_ax.get_ylim())
            target_ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig, ax_grid

######################################
# PLOTTING POSES
######################################

def plot_6DOF_state_history_euler_xyz(state_history: np.ndarray, figsize: tuple = (30, 10),
                                      save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs) -> None:
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
    axs[0, 0].plot(state_history[:, 0]*1e3, label='x')
    axs[0, 0].set_title('x')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Position (mm)')
    axs[0, 1].plot(state_history[:, 1]*1e3, label='y')
    axs[0, 1].set_title('y')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Position (mm)')
    axs[0, 2].plot(state_history[:, 2]*1e3, label='z')
    axs[0, 2].set_title('z')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Position (mm)')
    axs[0, 3].plot(state_history[:, 3]*1e3, label='vx')
    axs[0, 3].set_title('vx')
    axs[0, 3].set_xlabel('Time')
    axs[0, 3].set_ylabel('Velocity (mm/s)')
    axs[0, 4].plot(state_history[:, 4]*1e3, label='vy')
    axs[0, 4].set_title('vy')
    axs[0, 4].set_xlabel('Time')
    axs[0, 4].set_ylabel('Velocity (mm/s)')
    axs[0, 5].plot(state_history[:, 5]*1e3, label='vz')
    axs[0, 5].set_title('vz')
    axs[0, 5].set_xlabel('Time')
    axs[0, 5].set_ylabel('Velocity (mm/s)')
    axs[1, 0].plot(np.rad2deg(state_history[:, 6]), label='phi')
    axs[1, 0].set_title('phi')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Angle (rad)')
    axs[1, 1].plot(np.rad2deg(state_history[:, 7]), label='theta')
    axs[1, 1].set_title('theta')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Angle (rad)')
    axs[1, 2].plot(np.rad2deg(state_history[:, 8]), label='psi')
    axs[1, 2].set_title('psi')
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel('Angle (rad)')
    axs[1, 3].plot(np.rad2deg(state_history[:, 9]), label='wx')
    axs[1, 3].set_title('wx')
    axs[1, 3].set_xlabel('Time')
    axs[1, 3].set_ylabel('Angular Velocity (rad/s)')
    axs[1, 4].plot(np.rad2deg(state_history[:, 10]), label='wy')
    axs[1, 4].set_title('wy')
    axs[1, 4].set_xlabel('Time')
    axs[1, 4].set_ylabel('Angular Velocity (rad/s)')
    axs[1, 5].plot(np.rad2deg(state_history[:, 11]), label='wz')
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
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs) -> None:
    assert state_history.shape[1] == 12, "State history should have 12 columns"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state_history[:, 0]*1e3, state_history[:, 1]*1e3, state_history[:, 2]*1e3)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
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

def plot_poses_constant_reference(actual_poses: pd.DataFrame, reference_pose: np.ndarray, scale_equal: bool = True,
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
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
    reference_position = reference_pose[:3]*1000 # in mm
    reference_orientation = reference_pose[3:] # Reference orientation is taken as quaternion

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
        axs[0, i].set_title(f"Position {axis} of Body Fixed Frame")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Position (mm)")
        axs[0, i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[1, i].plot(time, actual_euler[:, i], label=f"Actual {angle}")
        axs[1, i].axhline(y=reference_euler[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[1, i].set_title(f"{angle} of Body Fixed Frame")
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Angle (deg)")
        axs[1, i].legend()

    if scale_equal:
        axs[0, 2].sharey(axs[0, 0])
        axs[0, 1].sharey(axs[0, 0])
        axs[1, 1].sharey(axs[1, 0])
        axs[1, 2].sharey(axs[1, 0])
        # Autoscale shared axes
        for ax_row in axs: 
            for ax in ax_row:
                ax.relim()   
                ax.autoscale()

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
                                      save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots target positions from actual poses DataFrame and a constant reference position.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame containing actual positions with time.
    - reference_position (np.ndarray): Array of size 3 [x, y, z] representing the constant reference position.
    """
    time = actual_poses['time'].values
    # All positions are in mm
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000
    reference_position = reference_position*1000

    # Plot positions
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[i].axhline(y=reference_position[i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[i].set_title(f"Position {axis} of Body Fixed Frame")
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
                                       inkscape_path: str=INKSCAPE_PATH, **kwargs):
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
    ax.set_title(f"Z Position of Body Fixed Frame")
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

def plot_z_position_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                              save_as: str = None, save_as_emf: bool = False,
                              inkscape_path: str = INKSCAPE_PATH, **kwargs):
    """
    Plots Z positions over time from actual poses and reference poses.
    All inputs are in SI units.

    Parameters:
        - actual_poses (pd.DataFrame): DataFrame containing actual positions with time.
        - reference_poses (pd.DataFrame): DataFrame containing reference positions with time.
        - save_as (str): Filename to save the plot as SVG (optional).
        - save_as_emf (bool): If True, also save the plot as EMF using Inkscape.
        - inkscape_path (str): Path to Inkscape executable.
        - **kwargs: Additional arguments passed to plt.plot().
    """
    time = actual_poses['time'].values
    actual_z_position = actual_poses['transform.translation.z'].values * 1000  # Convert to mm
    reference_z_position = reference_poses['transform.translation.z'].values * 1000  # Convert to mm

    # Plot positions
    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot()

    ax.plot(time, actual_z_position, label="Actual Z", **kwargs)
    ax.plot(time, reference_z_position, label="Reference Z", linestyle='dashed', color='r', **kwargs)

    ax.set_title("Z Position of Body Fixed Frame")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (mm)")
    ax.legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save as SVG/EMF if required
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
                                       inkscape_path: str=INKSCAPE_PATH, **kwargs):
    time = actual_poses['time'].values
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_yx = np.array([geometry.get_normal_alpha_beta_from_quaternion(quaternion/np.linalg.norm(quaternion)) 
                          for quaternion in actual_orientations])

    actual_xy = np.roll(actual_yx, 1, axis=1)

    # Convert to degrees
    actual_xy = np.rad2deg(actual_xy)
    reference_angles = np.rad2deg(reference_angles)

    # Plot Euler angles
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.5), sharex=True, sharey=True)

    fig.suptitle("Angles of Dipole Fixed Frame Z-Axis with World's Z-Axis")
    for i, angle in enumerate(['Alpha', 'Beta']):
        axs[i].plot(time, actual_xy[:, i], label=f"Actual {angle}")
        axs[i].axhline(y=reference_angles[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[i].set_title(f"{angle} of Body Fixed Frame")
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

def plot_alpha_beta_torques_constant_reference(actual_poses: pd.DataFrame, reference_angles: np.ndarray,
                                              ft_df: pd.DataFrame,
                                              scale_equal: bool=True,
                                              save_as: str=None,
                                              save_as_emf: bool=False,
                                              inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots a 2x2 subplot:
    - Row 1: Actual vs Desired Alpha and Beta angles
    - Row 2: Desired Torques Tx and Ty
    """
    time = actual_poses['time'].values
    
    # Compute actual alpha and beta from quaternions
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values
    actual_yx = np.array([geometry.get_normal_alpha_beta_from_quaternion(q/np.linalg.norm(q)) for q in actual_orientations])

    # Convert to degrees
    actual_xy = np.rad2deg(np.roll(actual_yx, 1, axis=1))
    reference_angles = np.rad2deg(reference_angles)
    
    # Extract desired torques Tx and Ty, in mN-m
    torques = ft_df[['wrench.torque.x', 'wrench.torque.y']].values*1e3
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Alpha, Beta, and Torques")
        
    fig.suptitle("Angles of Dipole Fixed Frame Z-Axis with World's Z-Axis")
    for i, angle in enumerate(['Alpha', 'Beta']):
        axs[0, i].plot(time, actual_xy[:, i], label=f"Actual {angle}")
        axs[0, i].axhline(y=reference_angles[i], label=f"Reference {angle}", linestyle='dashed', color='r')
        axs[0, i].set_title(f"{angle} of Body Fixed Frame")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Angle (deg)")
        axs[0, i].legend()
    
    for i, torque in enumerate(['Torque Tx', 'Torque Ty']):
        axs[1, i].plot(time, torques[:, i], label=f"{torque} Desired")
        axs[1, i].set_title(f"Desired {torque} on COM expressed in world frame")
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Torque (mN-m)")
        axs[1, i].legend()

    if scale_equal:
        axs[0, 1].sharey(axs[0, 0])
        axs[1, 1].sharey(axs[1, 0])
        # Autoscale shared axes
        for ax_row in axs: 
            for ax in ax_row:
                ax.relim()   
                ax.autoscale()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_alpha_beta_vel_errors_torques(rp_error_states_df: pd.DataFrame,
                                       ft_df: pd.DataFrame,
                                       angle_error_cols: List[str],
                                       velocity_error_cols: List[str],
                                       scale_equal: bool=True,
                                       save_as: str=None,
                                       save_as_emf: bool=False,
                                       inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots a 2x2 subplot:
    - Row 1: Actual vs Desired Alpha and Beta angles
    - Row 2: Desired Torques Tx and Ty
    """
    time = rp_error_states_df['time'].values
    
    alpha_beta_error = np.rad2deg(rp_error_states_df[angle_error_cols].to_numpy()) # deg
    alpha_beta_velocity_error = np.rad2deg(rp_error_states_df[velocity_error_cols].to_numpy()) # deg/sec
    
    # Extract desired torques Tx and Ty
    torques = ft_df[['wrench.torque.x', 'wrench.torque.y']].to_numpy()*1e3 # Convert to mN-m
    
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
        
    fig.suptitle("Normal Angle, Angular Velocity Tracking Errors, and Desired Torques of Dipole Fixed Frame Z-Axis In World Frame")
    for i, angle in enumerate(['Alpha', 'Beta']):
        axs[0, i].plot(time, alpha_beta_error[:, i], label=f"{angle}")
        axs[0, i].set_title(f"{angle} Error of Body Fixed Frame")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Angle (deg)")
        axs[0, i].legend()
    
    for i, angle in enumerate(['Alpha Dot', 'Beta Dot']):
        axs[1, i].plot(time, alpha_beta_velocity_error[:, i], label=f"{angle}")
        axs[1, i].set_title(f"{angle} Error of Body Fixed Frame")
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Angle (deg/sec)")
        axs[1, i].legend()
    
    for i, torque in enumerate(['Torque Tx', 'Torque Ty']):
        axs[2, i].plot(time, torques[:, i], label=f"{torque} Desired")
        axs[2, i].set_title(f"Desired {torque} on COM expressed in world frame")
        axs[2, i].set_xlabel("Time (s)")
        axs[2, i].set_ylabel("Torque (mN-m)")
        axs[2, i].legend()

    if scale_equal:
        axs[0, 1].sharey(axs[0, 0])
        axs[1, 1].sharey(axs[1, 0])
        axs[2, 1].sharey(axs[2, 0])

        # Autoscale shared axes
        for ax_row in axs: 
            for ax in ax_row:
                ax.relim()   
                ax.autoscale()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_z_position_Fz_constant_reference(actual_poses: pd.DataFrame, reference_z: float,
                                          ft_df: pd.DataFrame,
                                          save_as: str = None,
                                          save_as_emf: bool = False,
                                          inkscape_path: str = INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    """
    Plots Z position (actual and constant reference) along with the desired Fz force in subplots.
    Positions are in mm, forces in mN.

    Parameters:
        - actual_poses (pd.DataFrame): DataFrame containing actual positions with time.
        - reference_z (float): Constant reference Z position in SI units.
        - ft_df (pd.DataFrame): DataFrame containing forces, 'array_2' corresponds to Fz.
        - save_as (str): Filename to save the plot as SVG (optional).
        - save_as_emf (bool): If True, also save the plot as EMF using Inkscape.
        - inkscape_path (str): Path to Inkscape executable.
        - **kwargs: Additional arguments passed to plt.plot().
    """
    # Extract data
    time = actual_poses['time'].values
    actual_z_position = actual_poses['transform.translation.z'].values * 1000  # Convert to mm
    reference_z = reference_z * 1000  # Constant reference in mm
    Fz = ft_df['array_2'].values * 1e3  # Convert to mN

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
    fig.suptitle("Z Position and Desired Fz Force of Rigid Body", fontsize=14)

    # Plot Z position
    axes[0].plot(time, actual_z_position, label="Actual Z", color='tab:blue', **kwargs)
    axes[0].axhline(y=reference_z, label="Reference Z", linestyle='dashed', color='tab:red')
    axes[0].set_ylabel("Position (mm)")
    axes[0].set_title("Z Position of Body Fixed Frame")
    axes[0].legend()

    # Plot Fz
    axes[1].plot(time, Fz, label="Desired Fz", color='k', **kwargs)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Force (mN)")
    axes[1].set_title("Desired Fz Force")
    axes[1].legend()

    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='major', color=mcolors.CSS4_COLORS['lightslategray'], linewidth=0.8)
        ax.grid(which='minor', color=mcolors.CSS4_COLORS['lightslategray'], linestyle=':', linewidth=0.5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as SVG/EMF if needed
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axes

def plot_z_position_Fz_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                                          ft_df: pd.DataFrame,
                                          save_as: str = None,
                                          save_as_emf: bool = False,
                                          inkscape_path: str = INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    """
    Plots Z position (actual and variable reference) along with the desired Fz force in subplots.
    Positions are in mm, forces in mN.

    Parameters:
        - actual_poses (pd.DataFrame): DataFrame containing actual positions with time.
        - reference_poses (pd.DataFrame): DataFrame containing reference positions with time.
        - ft_df (pd.DataFrame): DataFrame containing forces, 'array_2' corresponds to Fz.
        - save_as (str): Filename to save the plot as SVG (optional).
        - save_as_emf (bool): If True, also save the plot as EMF using Inkscape.
        - inkscape_path (str): Path to Inkscape executable.
        - **kwargs: Additional arguments passed to plt.plot().
    """
    # Extract data
    time = actual_poses['time'].values
    actual_z_position = actual_poses['transform.translation.z'].values * 1000  # Convert to mm
    reference_z_position = reference_poses['transform.translation.z'].values * 1000  # Convert to mm
    Fz = ft_df['array_2'].values * 1e3  # Convert to mN

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
    fig.suptitle("Z Position and Desired Fz Force", fontsize=14)

    # Plot Z position
    axes[0].plot(time, actual_z_position, label="Actual Z", color='tab:blue', **kwargs)
    axes[0].plot(time, reference_z_position, label="Reference Z", linestyle='dashed', color='tab:red', **kwargs)
    axes[0].set_ylabel("Position (mm)")
    axes[0].set_title("Z Position of Body Fixed Frame")
    axes[0].legend()

    # Plot Fz
    axes[1].plot(time, Fz, label="Desired Fz", color='k', **kwargs)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Force (mN)")
    axes[1].set_title("Desired Fz Force")
    axes[1].legend()

    for ax in axes:
        ax.minorticks_on()
        ax.grid(which='major', color=mcolors.CSS4_COLORS['lightslategray'], linewidth=0.8)
        ax.grid(which='minor', color=mcolors.CSS4_COLORS['lightslategray'], linestyle=':', linewidth=0.5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as SVG/EMF if needed
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axes



def plot_alpha_beta_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                              save_as: str = None, save_as_emf: bool = False,
                              inkscape_path: str = INKSCAPE_PATH, **kwargs):
    """
    Plots Alpha and Beta angles over time from actual poses and reference poses.
    Angles are computed from quaternions and converted to degrees.

    Parameters:
        - actual_poses (pd.DataFrame): DataFrame containing actual orientations with time.
        - reference_poses (pd.DataFrame): DataFrame containing reference orientations with time.
        - save_as (str): Filename to save the plot as SVG (optional).
        - save_as_emf (bool): If True, also save the plot as EMF using Inkscape.
        - inkscape_path (str): Path to Inkscape executable.
        - **kwargs: Additional arguments passed to plt.plot().
    """
    time = actual_poses['time'].values

    # Extract quaternions and compute Euler angles
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y',
                                        'transform.rotation.z', 'transform.rotation.w']].values
    reference_orientations = reference_poses[['transform.rotation.x', 'transform.rotation.y',
                                              'transform.rotation.z', 'transform.rotation.w']].values

    actual_angles = np.array([
        geometry.get_normal_alpha_beta_from_quaternion(quaternion/np.linalg.norm(quaternion))
        for quaternion in actual_orientations
    ])
    reference_angles = np.array([
        geometry.get_normal_alpha_beta_from_quaternion(quaternion/np.linalg.norm(quaternion))
        for quaternion in reference_orientations
    ])

    # Convert to degrees
    actual_angles_deg = np.rad2deg(np.roll(actual_angles, 1, axis=1))
    reference_angles_deg = np.rad2deg(np.roll(reference_angles, 1, axis=1))

    # Plot Euler angles
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.5), sharex=True, sharey=True)
    fig.suptitle("Angles of Dipole Fixed Frame Z-Axis with World's Z-Axis")

    angle_labels = ['Alpha', 'Beta']
    for i, angle in enumerate(angle_labels):
        axs[i].plot(time, actual_angles_deg[:, i], label=f"Actual {angle}", **kwargs)
        axs[i].plot(time, reference_angles_deg[:, i], label=f"Reference {angle}",
                    linestyle='dashed', color='r', **kwargs)
        axs[i].set_title(f"{angle} of Body Fixed Frame")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Angle (deg)")
        axs[i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save as SVG/EMF if required
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axs

def plot_alpha_beta_torques_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                                              ft_df: pd.DataFrame,
                                              scale_equal: bool=True,
                                              save_as: str=None,
                                              save_as_emf: bool=False,
                                              inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots a 2x2 subplot:
    - Row 1: Actual vs Desired Alpha and Beta angles
    - Row 2: Desired Torques Tx and Ty
    """
    time = actual_poses['time'].values
    
    # Compute actual alpha and beta from quaternions
    # Extract quaternions and compute Euler angles
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y',
                                        'transform.rotation.z', 'transform.rotation.w']].values
    reference_orientations = reference_poses[['transform.rotation.x', 'transform.rotation.y',
                                              'transform.rotation.z', 'transform.rotation.w']].values

    actual_angles = np.array([
        geometry.get_normal_alpha_beta_from_quaternion(quaternion/np.linalg.norm(quaternion))
        for quaternion in actual_orientations
    ])
    reference_angles = np.array([
        geometry.get_normal_alpha_beta_from_quaternion(quaternion/np.linalg.norm(quaternion))
        for quaternion in reference_orientations
    ])

    # Convert to degrees
    actual_angles_deg = np.rad2deg(np.roll(actual_angles, 1, axis=1))
    reference_angles_deg = np.rad2deg(np.roll(reference_angles, 1, axis=1))
    
    # Extract desired torques Tx and Ty
    torques = ft_df[['wrench.torque.x', 'wrench.torque.y']].values*1e3
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Alpha, Beta, and Torques")
        
    fig.suptitle("Angles of Dipole Fixed Frame Z-Axis with World's Z-Axis")
    for i, angle in enumerate(['Alpha', 'Beta']):
        axs[0, i].plot(time, actual_angles_deg[:, i], label=f"Actual {angle}", **kwargs)
        axs[0, i].plot(time, reference_angles_deg[:, i], label=f"Reference {angle}",
                      linestyle='dashed', color='r', **kwargs)
        axs[0, i].set_title(f"{angle} of Body Fixed Frame")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Angle (deg)")
        axs[0, i].legend()
    
    for i, torque in enumerate(['Torque Tx', 'Torque Ty']):
        axs[1, i].plot(time, torques[:, i], label=f"{torque} Desired")
        axs[1, i].set_title(f"Desired {torque} on COM expressed in world frame")
        axs[1, i].set_xlabel("Time (s)")
        axs[1, i].set_ylabel("Torque (mN-m)")
        axs[1, i].legend()

    if scale_equal:
        axs[0, 1].sharey(axs[0, 0])
        axs[1, 1].sharey(axs[1, 0])
        # Autoscale shared axes
        for ax_row in axs: 
            for ax in ax_row:
                ax.relim()   
                ax.autoscale()

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
                                         save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
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
        axs[i].set_title(angle + " of Body Fixed Frame")
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
                                            save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
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
        axs[i].set_title(angle + " of Body Fixed Frame")
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

def plot_poses_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame, scale_equal: bool = True,
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
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
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(time, actual_positions[:, i], label=f"Actual {axis}")
        axs[0, i].plot(time, reference_positions[:, i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[0, i].set_title(f"Position {axis} of Body Fixed Frame")
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

    if scale_equal:
        axs[0, 2].sharey(axs[0, 0])
        axs[0, 1].sharey(axs[0, 0])
        axs[1, 1].sharey(axs[1, 0])
        axs[1, 2].sharey(axs[1, 0])
        # Autoscale shared axes
        for ax_row in axs: 
            for ax in ax_row:
                ax.relim()   
                ax.autoscale()

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
                                      save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
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
        axs[i].set_title(f"Position {axis} of Body Fixed Frame")
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
                                         save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
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
        axs[i].set_title(angle + " of Body Fixed Frame.")
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

def plot_3d_poses_with_arrows_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame, arrow_interval: int = 10, frame_size: float = 0.01, frame_interval: int = 10,
                                                     save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
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

    # Plot actual positions, CONVERTED TO mm
    ax.plot(actual_positions[:, 0]*1000, actual_positions[:, 1]*1000, actual_positions[:, 2]*1000, color='black', label='Actual Path')

    # Plot reference positions (non-constant)
    ax.plot(reference_positions[:, 0]*1000, reference_positions[:, 1]*1000, reference_positions[:, 2]*1000, color='red', linestyle='--', label='Reference Path')

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

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title("Actual Pose v/s Reference Pose of Body Fixed Frame")

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
                                                 save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
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

    # Plot actual positions, CONVERTED TO mm
    ax.plot(actual_positions[:, 0]*1000, actual_positions[:, 1]*1000, actual_positions[:, 2]*1000, color='black', label='Actual Path')

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

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title("Actual Pose v/s Reference Pose of Body Fixed Frame Frame")

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
# PLOTTING CURRENTS, CONTROL INPUTS, FIELDS, CONDITION NUMBERS
######################################

def plot_actual_currents(system_state_df: pd.DataFrame,
                         save_as: str=None,
                         save_as_emf: bool=False,
                         inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
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
                                inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    # Plot each current column in its respective subplot
    # Create subplots in a 2x4 layout
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle("Actual and Desired Currents vs Time", fontsize=16)  # Main title for the figure

    # Flatten the 2D axes array for easier iteration
    axs = axs.flatten()
    for i in range(8):
        axs[i].plot(system_state_df['time'], system_state_df[f'currents_reg_{i}'], label=f'Actual Current {i+1}', color='tab:blue', **kwargs)
        axs[i].plot(des_currents_df['time'], des_currents_df[f'des_currents_reg_{i}'], label=f'Desired Current {i+1}', color='tab:green', **kwargs)
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

def plot_forces_and_torques(ft_df: pd.DataFrame,
                            title: str="Desired Forces and Torques",
                            torque_in_mNm: bool = True,
                            force_in_N: bool = False,
                            save_as: str=None,
                            save_as_emf: bool=False,
                            inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    """
    Plots forces (Fx, Fy, Fz) and torques (Tx, Ty, Tz) from a DataFrame with time on the x-axis.
    Forces are converted to milliNewtons (mN) and torques to milliNewton-millimeters (mN-mm).

    Parameters:
        ft_df (pd.DataFrame): Input DataFrame with 'time', 'array_0' to 'array_5' columns.
                           'array_0' = Fx, 'array_1' = Fy, 'array_2' = Fz
                           'array_3' = Tx, 'array_4' = Ty, 'array_5' = Tz
    """
    force_units = "mN"
    force_multiplier = 1e3
    torque_units = "Nm"
    torque_multipler = 1
    if force_in_N:
        force_units = "N"
        force_multiplier = 1
    if torque_in_mNm:
        torque_units = "mN-m"
        torque_multiplier = 1e3

    # Extract relevant columns
    time = ft_df['time']
    Fx = ft_df['array_0'] * force_multiplier  # Convert to mN
    Fy = ft_df['array_1'] * force_multiplier
    Fz = ft_df['array_2'] * force_multiplier
    Tx = ft_df['array_3'] * torque_multipler  # Convert to mN-mm
    Ty = ft_df['array_4'] * torque_multipler
    Tz = ft_df['array_5'] * torque_multipler

    # Create subplots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Plot forces (mN)
    axes[0, 0].plot(time, Fx, color='b')
    axes[0, 0].set_ylabel(f'Force ({force_units})')
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
    axes[1, 0].set_ylabel(f'Torque ({torque_units})')
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

def plot_forces_and_torques_from_wrench_stamped(ft_df: pd.DataFrame,
                                                title: str,
                                                torque_in_mNm: bool = True,
                                                force_in_N: bool = False,
                                                save_as: str=None,
                                                save_as_emf: bool=False,
                                                inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:

    # Just convert the dataframe to a format accepted by the array msg based plotter.
    ft_df = ft_df.rename(columns={
                            "wrench.force.x": "array_0",
                            "wrench.force.y": "array_1",
                            "wrench.force.z": "array_2",
                            "wrench.torque.x": "array_3",
                            "wrench.torque.y": "array_4",
                            "wrench.torque.z": "array_5"
                        }, errors="raise", inplace=False)
    return plot_forces_and_torques(ft_df, title, torque_in_mNm, force_in_N, save_as, save_as_emf, inkscape_path, **kwargs)

def plot_3d_quiver(dataframe: pd.DataFrame, 
                   scale_factor: float=1.0, 
                   save_as: str=None, 
                   save_as_emf: bool=False, 
                   inkscape_path: str=INKSCAPE_PATH, **kwargs):
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
    
    # Converting distances to mm and fields to mT
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
                              inkscape_path: str=INKSCAPE_PATH, **kwargs):
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

def plot_field_components_linear_des_at_dipole_center(pose_df: pd.DataFrame, desired_currents_df: pd.DataFrame, 
                          actual_currents_df: pd.DataFrame, calibrated_model: common.OctomagCalibratedModel,
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots the 3 magnetic field components (Bx, By, Bz) over time for both actual and desired fields.
    The desired components are calculated using the linear actuation matrix since that's what we use
    during the computation. However, the actual field is computed using the nonlinear model's forward
    computation function available through the calibration model class.
    Shared x and y axes across subplots. Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibrated_model (common.OctomagCalibratedModel): The calibration model used.
        save_as (str): Filename to save the plot as SVG/EMF (without extension).
        **kwargs: Additional parameters for plt.plot().
    """
    combined_desired = pd.merge_asof(pose_df, desired_currents_df, on='time')
    combined_actual = pd.merge_asof(pose_df, actual_currents_df, on='time')

    time = combined_desired['time']
    desired_fields = {'Bx': [], 'By': [], 'Bz': []}
    actual_fields = {'Bx': [], 'By': [], 'Bz': []}

    for i in range(len(combined_desired)):
        position = np.array([
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ])

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        A = calibrated_model.get_actuation_matrix(position)
        desired_field = A @ desired_currents
        actual_field = calibrated_model.get_exact_field_grad5_from_currents(position, actual_currents)

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
    fig.suptitle("Actual (Non-Linear Model) v/s Desired Field (Linear Approx) Components at Dipole Center")
    plt.tight_layout()
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_gradient_components_linear_des_at_dipole_center(pose_df: pd.DataFrame, desired_currents_df: pd.DataFrame, 
                          actual_currents_df: pd.DataFrame, calibrated_model: common.OctomagCalibratedModel,
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots the 5 gradient components (dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz) over time 
    for both actual and desired gradients. The desired components are calculated using 
    the linear actuation matrix since that's what we use during the computation. However,
    the actual field is computed using the nonlinear model's forward computation function 
    available through the calibration model class.
    Shared x and y axes across subplots. Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibrated_model (common.OctomagCalibratedModel): The calibration model used.
        save_as (str): Filename to save the plot as SVG/EMF (without extension).
        **kwargs: Additional parameters for plt.plot().
    """
    combined_desired = pd.merge_asof(pose_df, desired_currents_df, on='time')
    combined_actual = pd.merge_asof(pose_df, actual_currents_df, on='time')

    time = combined_desired['time']
    desired_gradients = {'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [], 'dBy/dy': [], 'dBy/dz': []}
    actual_gradients = {'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [], 'dBy/dy': [], 'dBy/dz': []}

    for i in range(len(combined_desired)):
        position = np.array([
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ])

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        A = calibrated_model.get_actuation_matrix(position)
        desired_field = A @ desired_currents
        actual_field = calibrated_model.get_exact_field_grad5_from_currents(position, actual_currents)

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
    fig.suptitle("Actual (Non-Linear Model) v/s Desired Gradients (Linear Approx) at Dipole Center")
    plt.tight_layout()

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axs

def plot_field_components_at_dipole_center_const_actuation_position(
                          pose_df: pd.DataFrame, desired_currents_df: pd.DataFrame, 
                          actual_currents_df: pd.DataFrame, calibrated_model: common.OctomagCalibratedModel, des_actuation_pos=np.zeros(3),
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots the 3 magnetic field components (Bx, By, Bz) over time for both actual and desired fields.
    Where the desired fields are calculated based on the actuation matrix calcualted at the given 
    desired position. However, the actual field is computed using the nonlinear model's forward
    computation function available through the calibration model class.
    Shared x and y axes across subplots. Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibrated_model (common.OctomagCalibratedModel): The calibration model used.
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

    A = calibrated_model.get_actuation_matrix(des_actuation_pos)

    for i in range(len(combined_desired)):
        position = np.array([
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ])

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        desired_field = A @ desired_currents
        actual_field = calibrated_model.get_exact_field_grad5_from_currents(position, actual_currents)

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
    fig.suptitle(f"Actual (Non-Linear Model) v/s Desired Field Components at Dipole Center for A({des_actuation_pos})")
    plt.tight_layout()
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_gradient_components_at_dipole_center_const_actuation_position(pose_df: pd.DataFrame, desired_currents_df: pd.DataFrame, 
                          actual_currents_df: pd.DataFrame, calibrated_model: common.OctomagCalibratedModel, des_actuation_pos=np.zeros(3),
                          save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots the 5 gradient components (dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz) over time 
    for both actual and desired gradients. However, the actual field is computed using 
    the nonlinear model's forward computation function available through the calibration 
    model class. Shared x and y axes across subplots. Allows saving as SVG/EMF.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        desired_currents_df (pd.DataFrame): Desired currents dataframe with 'des_currents_reg_*' columns
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        model_fn (Callable): Function that maps position to an 8x8 matrix A
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

    A = calibrated_model.get_actuation_matrix(des_actuation_pos)

    for i in range(len(combined_desired)):
        position = np.array([
            combined_desired['transform.translation.x'].iloc[i],
            combined_desired['transform.translation.y'].iloc[i],
            combined_desired['transform.translation.z'].iloc[i]
        ])

        desired_currents = np.array([combined_desired[f'des_currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        desired_field = A @ desired_currents
        actual_field = calibrated_model.get_exact_field_grad5_from_currents(position, actual_currents)

        gradient_keys = ['dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']
        for idx, key in enumerate(gradient_keys, start=3): 
            desired_gradients[key].append(desired_field[idx] * 1000) # Converting all readings to mT/m
            actual_gradients[key].append(actual_field[idx] * 1000) # Converting all readings to mT/m

    # Plot the gradient components with shared axes
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True, sharey=True)
    for i, key in enumerate(['dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']):
        axs[i].plot(time, actual_gradients[key], label=f'Actual {key}', linestyle='-', color='tab:red', **kwargs)
        axs[i].plot(time, desired_gradients[key], label=f'Desired {key}', linestyle='-', color='tab:blue', **kwargs)
        axs[i].set_ylabel(f'{key} [mT/m]')
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Actual (Non-Linear Model) v/s Desired Gradients at Dipole Center For A({des_actuation_pos})")
    plt.tight_layout()

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axs

def plot_actual_wrench_on_dipole_center(dipole_center_pose_df: pd.DataFrame,
                                        actual_currents_df: pd.DataFrame,
                                        desired_wrench: pd.DataFrame,
                                        calibrated_model: common.OctomagCalibratedModel,
                                        dipole_strength: float,
                                        dipole_axis: np.ndarray,
                                        save_as: str = None,
                                        save_as_emf: bool = False,
                                        inkscape_path: str = INKSCAPE_PATH,
                                        **kwargs):
    """
    Plots the actual and desired wrench (force and torque) components over time for a dipole center,
    based on the given pose and current data.

    This function computes the actual wrench exerted by the dipole based on the current data and
    compares it to the desired wrench values. It then plots the components of force (Fx, Fy, Fz) and torque (Taux, Tauy, Tauz) 
    against time. The plot allows visualization of the agreement between the actual and reference values.

    Parameters:
        dipole_center_pose_df (pd.DataFrame):
            DataFrame containing the pose of the dipole center.

        actual_currents_df (pd.DataFrame):
            DataFrame containing the actual currents (currents_reg_0 to currents_reg_7) applied to the dipole over time.
            Each row corresponds to a specific time step.

        desired_wrench (pd.DataFrame):
            DataFrame containing the desired wrench (force and torque) components at each time step. Should have columns 'Fx', 'Fy', 
            'Fz', 'Taux', 'Tauy', 'Tauz' representing the reference force and torque values.

        calibrated_model (OctomagCalibratedModel):
            An instance of a model used to calculate the exact field gradients based on the position and currents.
            It should have a method `get_exact_field_grad5_from_currents(position, currents)` to compute the actual fields.

        dipole_strength (float):
            The strength of the dipole, which is used to compute the interaction matrix.

        dipole_axis (np.ndarray):
            A 3D vector representing the axis of the dipole for torque computation.

        save_as (str):
            The file path where the plot should be saved (in PNG format). If not provided, the plot is not saved.

        save_as_emf (bool):
            If True, the plot will also be saved in EMF format alongside the PNG file. Default is False.

        inkscape_path (str):
            Path to the Inkscape executable, used for converting the EMF file to PNG when `save_as_emf` is True. Default is None.

        **kwargs (additional) keyword arguments
            Additional arguments to be passed to the plotting function (e.g., for customizing the plot appearance).

    Returns:
        A tuple where the first element is the figure object while the second elements is the axes object.
    
    Notes:
    - The plot consists of two rows: the first row for force components (Fx, Fy, Fz) and the second for torque components (Taux, Tauy, Tauz).
    - The actual wrench is computed from the dipole's position, rotation, and current data using the calibrated model and interaction matrix.
    - The plot includes both actual wrench and reference wrench (desired) values for comparison.
    """
    # Combine pose and current data
    combined_pose_currents = pd.merge_asof(dipole_center_pose_df, actual_currents_df, on='time')
    time = dipole_center_pose_df['time']
    actual_wrench_dict = {'wrench.torque.x': [], 'wrench.torque.y': [], 'wrench.torque.z': [],
                          'wrench.force.x': [], 'wrench.force.y': [], 'wrench.force.z': []}

    # Calculate actual wrench
    for i in range(len(combined_pose_currents)):
        position = np.array([
            combined_pose_currents['transform.translation.x'].iloc[i],
            combined_pose_currents['transform.translation.y'].iloc[i],
            combined_pose_currents['transform.translation.z'].iloc[i]
        ])
        actual_currents = np.array([combined_pose_currents[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        actual_fields = calibrated_model.get_exact_field_grad5_from_currents(position, actual_currents)
        
        quaternion = np.array([
            combined_pose_currents['transform.rotation.x'].iloc[i],
            combined_pose_currents['transform.rotation.y'].iloc[i],
            combined_pose_currents['transform.rotation.z'].iloc[i],
            combined_pose_currents['transform.rotation.w'].iloc[i]
        ])

        quaternion = quaternion/np.linalg.norm(quaternion)

        M = geometry.magnetic_interaction_matrix_from_quaternion(dipole_quaternion=quaternion,
                                                                 dipole_strength=dipole_strength,
                                                                 full_mat=True,
                                                                 torque_first=True,
                                                                 dipole_axis=dipole_axis)
        actual_wrench = M @ actual_fields
        for j, key in enumerate(list(actual_wrench_dict.keys())):
            actual_wrench_dict[key].append(actual_wrench[j])

    # Convert wrench dict to DataFrame
    actual_wrench_df = pd.DataFrame(actual_wrench_dict)
    
    # Plot settings
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Force (actual, reference), Torque (actual, reference)

    key_map = {'Fx': 'wrench.force.x', 'Fy': 'wrench.force.y', 'Fz': 'wrench.force.z',
               'Taux': 'wrench.torque.x', 'Tauy': 'wrench.torque.y', 'Tauz': 'wrench.torque.z'}
    
    fig.suptitle('Actual Wrench (Non-Linear Model computed) v/s Desired Wrench')
    
    # Force subplots (columns 0, 1, 2), Forces converted to mN
    for i, force_component in enumerate(['Fx', 'Fy', 'Fz']):
        axes[0, i].plot(time, actual_wrench_df[key_map[force_component]]*1000, label='Actual Force', color=colors[0], **kwargs)
        axes[0, i].plot(time, desired_wrench[key_map[force_component]]*1000, label='Reference Force', color=colors[1], **kwargs)
        axes[0, i].set_title(f'{force_component} - Force')
        axes[0, i].grid(True)
        if i == 0:
            axes[0, i].set_ylabel('Force (mN)')
        if i == 2:
            axes[0, i].legend(loc='upper right')

    # Torque subplots (columns 0, 1, 2), Torques converted to mN-m
    for i, torque_component in enumerate(['Taux', 'Tauy', 'Tauz']):
        axes[1, i].plot(time, actual_wrench_df[key_map[torque_component]]*1e3, label='Actual Torque', color=colors[2], **kwargs)
        axes[1, i].plot(time, desired_wrench[key_map[torque_component]]*1e3, label='Reference Torque', color=colors[3], **kwargs)
        axes[1, i].set_title(f'{torque_component} - Torque')
        axes[1, i].grid(True)
        if i == 0:
            axes[1, i].set_ylabel('Torque (mN-m)')
        if i == 2:
            axes[1, i].legend(loc='upper right')

    # Shared X-axis
    for ax in axes[1, :]:
        ax.set_xlabel('Time (s)')
    
    axes[0, 1].sharey(axes[0, 0])
    axes[0, 2].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 2].sharey(axes[1, 0])

    # Autoscale axes
    for ax_row in axes: 
            for ax in ax_row:
                ax.relim()   
                ax.autoscale()

    plt.tight_layout()

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()    
    return fig, axes

######################################
# SIGNAL PROCESSING RELATED PLOTS
######################################
def plot_fft_from_dataframe(dataframe: pd.DataFrame, 
                            column_names: list,
                            label_list: list,
                            remove_dc: bool = False, 
                            positive_freqs_only: bool = False,
                            share_scales: bool = False,
                            subplot_shape: Optional[Tuple[int]] = None,
                            figsize: Optional[Tuple[int]] = None,
                            suptitle: Optional[str] = None,
                            save_as: str = None, 
                            save_as_emf: bool = False, 
                            inkscape_path: str = INKSCAPE_PATH):
    """
    Computes and plots the Fast Fourier Transform (FFT) for signals from a dataframe.
    The sampling frequency is derived from the 'time' column in the dataframe. 
    Shared x-axis across all subplots, with one subplot per signal column.

    Parameters:

        dataframe (pd.DataFrame): 
            Input dataframe containing the 'time' column and signal columns for FFT computation.
        column_names (list): 
            List of column names in the dataframe representing the signals to compute and plot FFT for.
        save_as (str, optional): 
            Path to save the plot as an SVG or PNG file (without extension).
        save_as_emf (bool, optional): 
            If True, saves the plot as an EMF file. Requires 'inkscape_path' to be provided.
        inkscape_path (str, optional): 
            Path to the Inkscape executable for converting to EMF format.
        **kwargs: 
            Additional keyword arguments passed to plt.plot().
    """
    # Calculate sampling frequency
    dt = np.mean(np.diff(dataframe['time']))  # Calculate time step

    # Prepare the FFT plot
    if subplot_shape is not None:
        sshape = subplot_shape
    else:
        sshape = (len(column_names), 1)
    if figsize is not None:
        fsize = figsize
    else:
        fsize = (10, len(column_names) * 3)
    fig, axs = plt.subplots(sshape[0], sshape[1], figsize=fsize, sharex=share_scales, sharey=share_scales)
    for i, (column, label) in enumerate(zip(column_names, label_list)):
        # Compute FFT
        signal = dataframe[column].values
        fft_values, fft_frequencies = get_signal_fft(signal, dt, remove_dc=remove_dc, positive_freqs_only=positive_freqs_only)

        # Plot FFT
        if 1 in sshape:
            ax = axs[i]
        else:
            j = i // sshape[1]
            k = i - j*sshape[1] - 1
            ax = axs[j, k]
        ax.plot(fft_frequencies, fft_values, label=f'FFT of {label}', color='tab:blue')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        title = f'FFT Plot for {label}'
        if remove_dc:
            title += ' (DC Removed)'
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    
    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()

    # Save the plot if required
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()
    return fig, axs

def plot_time_series_and_fft(dataframe: pd.DataFrame, 
                             column_names: list,
                             label_list: list,
                             remove_dc: bool = False, 
                             save_as: str = None, 
                             save_as_emf: bool = False, 
                             inkscape_path: str = INKSCAPE_PATH, 
                             **kwargs):
    """
    Plots both the time series data and its corresponding FFT for given columns in the dataframe.
    The sampling frequency is derived from the 'time' column. Produces a 2 x n subplot layout,
    where the first row contains the time series and the second row contains the FFT.

    Parameters:
        dataframe (pd.DataFrame): 
            Input dataframe containing the 'time' column and signal columns for FFT computation.
        column_names (list): 
            List of column names in the dataframe representing the signals to compute and plot FFT for.
        save_as (str, optional): 
            Path to save the plot as an SVG or PNG file (without extension).
        save_as_emf (bool, optional): 
            If True, saves the plot as an EMF file. Requires 'inkscape_path' to be provided.
        inkscape_path (str, optional): 
            Path to the Inkscape executable for converting to EMF format.
        **kwargs: 
            Additional keyword arguments passed to plt.plot().
    """
    # Ensure the dataframe has a 'time' column
    if 'time' not in dataframe.columns:
        raise ValueError("The dataframe must contain a 'time' column.")
    
    time = dataframe['time'].values
    dt = np.mean(np.diff(time))  # Calculate time step
    
    # Prepare the figure
    num_columns = len(column_names)
    fig, axes = plt.subplots(2, num_columns, figsize=(6 * num_columns, 8), constrained_layout=True)
    if num_columns == 1:  # If only one signal, axes won't be 2D, so wrap in a list
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, (column, label) in enumerate(zip(column_names, label_list)):
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in the dataframe.")
        
        signal = dataframe[column].values
        
        # Time series plot
        axes[0, idx].plot(time, signal, label=label, **kwargs)
        axes[0, idx].set_title(f"Time Series - {label}")
        axes[0, idx].set_xlabel("Time (s)")
        axes[0, idx].set_ylabel("Amplitude")
        axes[0, idx].grid(True)
        axes[0, idx].legend()

        # FFT calculation
        fft_vals = np.abs(scifft.fft(signal))
        fft_freqs = scifft.fftfreq(len(signal), dt)
        title = f"FFT - {label}"
        if remove_dc:
            fft_vals[0] = 0.0
            title += " (DC Removed)"
        fft_freqs = np.roll(fft_freqs, len(fft_freqs)//2)
        fft_vals = np.roll(fft_vals, len(fft_vals)//2)

        # FFT plot
        axes[1, idx].plot(fft_freqs, fft_vals, label=label, **kwargs)
        axes[1, idx].set_title(title)
        axes[1, idx].set_xlabel("Frequency (Hz)")
        axes[1, idx].set_ylabel("Amplitude")
        axes[1, idx].grid(True)
        axes[1, idx].legend()

    # Adjust layout and titles
    fig.suptitle("Time Series and FFT Plots", fontsize=16, weight='bold')
    
    # Save the figure if specified
    if save_as:
        fig.savefig(f"{save_as}.svg", format='svg')
        if save_as_emf and inkscape_path:
            emf_path = f"{save_as}.emf"
            svg_path = f"{save_as}.svg"
            import subprocess
            subprocess.run([inkscape_path, svg_path, '--export-type=emf', '--export-filename', emf_path])

    plt.show()

def plot_ffts_from_two_dataframes(dataframe1: pd.DataFrame,
                                  dataframe2: pd.DataFrame,
                                  column_names1: List,
                                  label_list1: List,
                                  column_names2: Optional[List] = None,
                                  label_list2: Optional[List] = None,
                                  label_suffix1: Optional[str] = None,
                                  label_suffix2: Optional[str] = None,
                                  clip_frequency: bool = False,
                                  remove_dc: bool = False,
                                  positive_freqs_only: bool = False,
                                  share_scales: bool = False,
                                  figsize: Optional[Tuple[int]] = None,
                                  suptitle: Optional[str] = None,
                                  save_as: str = None,
                                  save_as_emf: bool = False,
                                  inkscape_path: str = INKSCAPE_PATH
                                ):
    """
    Computes and plots the FFT for signals from two dataframes side by side.

    Parameters:
        dataframe1, dataframe2 (pd.DataFrame):
            Input dataframes containing the 'time' column and signal columns for FFT computation.
        column_names1, column_names2 (list, optional):
            Lists of column names in the dataframes representing the signals to compute and plot FFT for.
            If `column_names2` is None, it defaults to `column_names1`.
        label_list1, label_list2 (list, optional):
            Labels for the signals from both dataframes. If `label_list2` is None, it defaults to `label_list1`.
        remove_dc (bool, optional):
            If True, removes the DC component from the signals before FFT computation.
        positive_freqs_only (bool, optional):
            If True, only plots the positive frequencies.
        share_scales (bool, optional):
            If True, shares the x and y axes across all subplots.
        subplot_shape (Tuple[int], optional):
            Tuple specifying the number of rows and columns for subplots. If None, defaults to a 2-row layout.
        figsize (Tuple[int], optional):
            Size of the figure. Defaults to a size proportional to the number of signals.
        suptitle (str, optional):
            Overall title for the combined plot.
        save_as (str, optional):
            Path to save the plot as an SVG or PNG file (without extension).
        save_as_emf (bool, optional):
            If True, saves the plot as an EMF file. Requires 'inkscape_path' to be provided.
        inkscape_path (str, optional):
            Path to the Inkscape executable for converting to EMF format.

    Returns:
        fig, axs: Matplotlib figure and axes objects.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if column_names2 is None:
        column_names2 = column_names1
    if label_list2 is None:
        label_list2 = label_list1

    # Calculate sampling frequency
    dt1 = np.mean(np.diff(dataframe1['time']))
    dt2 = np.mean(np.diff(dataframe2['time']))

    # Determine subplot shape
    num_signals = len(column_names1)
    subplot_shape = (num_signals, 2)  # Default to 2 columns (one per dataframe)
    if figsize is None:
        figsize = (10, num_signals * 3)

    fig, axs = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=figsize, sharex=share_scales, sharey=share_scales)

    for i, (col1, label1, col2, label2) in enumerate(zip(column_names1, label_list1, column_names2, label_list2)):
        # Compute FFT for dataframe1
        signal1 = dataframe1[col1].values
        fft_values1, fft_frequencies1 = get_signal_fft(signal1, dt1, remove_dc=remove_dc, positive_freqs_only=positive_freqs_only)

        # Compute FFT for dataframe2
        signal2 = dataframe2[col2].values
        fft_values2, fft_frequencies2 = get_signal_fft(signal2, dt2, remove_dc=remove_dc, positive_freqs_only=positive_freqs_only)

        # Plot FFT for dataframe1
        ax1 = axs[i, 0] if len(column_names1) > 1 else axs[0]
        if label_suffix1 is not None:
            label1 += " " + label_suffix1
        if label_suffix2 is not None:
            label2 += " " + label_suffix2
        
        if clip_frequency:
            f_min = np.max([np.min(fft_frequencies1), np.min(fft_frequencies2)])
            f_max = np.min([np.max(fft_frequencies1), np.max(fft_frequencies2)])
            
            f_mask1 = np.logical_and(fft_frequencies1 > f_min, fft_frequencies1 < f_max)
            f_mask2 = np.logical_and(fft_frequencies2 > f_min, fft_frequencies2 < f_max)

            fft_frequencies1 = fft_frequencies1[f_mask1]
            fft_frequencies2 = fft_frequencies2[f_mask2]
            fft_values1 = fft_values1[f_mask1]
            fft_values2 = fft_values2[f_mask2]

        ax1.plot(fft_frequencies1, fft_values1, label=f'FFT of {label1}', color='tab:blue')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'{label1} (Dataframe 1)')
        ax1.grid(True)
        ax1.legend()

        # Plot FFT for dataframe2
        ax2 = axs[i, 1] if len(column_names1) > 1 else axs[1]
        ax2.plot(fft_frequencies2, fft_values2, label=f'FFT of {label2}', color='tab:orange')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title(f'{label2} (Dataframe 2)')
        ax2.grid(True)
        ax2.legend()

    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()

    # Save the plot if required
    if save_as:
        plt.savefig(save_as, format='svg', dpi=300)
        if save_as_emf:
            emf_path = save_as.replace('.svg', '.emf')
            plt.savefig(emf_path, format='emf', dpi=300)
            if inkscape_path:
                os.system(f'"{inkscape_path}" "{emf_path}" --export-filename="{save_as}"')

    plt.show()
    return fig, axs

######################################
# MAGNETIC ACTUATION ANALYSIS PLOTS
######################################

def plot_jma_condition_number(jma_cond_df: pd.DataFrame,
                              save_as: str = None,
                              save_as_emf: bool = False,
                              inkscape_path: str = INKSCAPE_PATH,
                              **kwargs):
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(jma_cond_df['time'], jma_cond_df['vector_0'], color='#0343df', label='vector_0', **kwargs)  # Blue
    ax.set_xlabel('Time')
    ax.set_ylabel('Vector 0')
    ax.set_title('JMA Condition Plot')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    plt.show()

    return fig, ax

def plot_6dof_pose_with_jma_condition_number(actual_poses: pd.DataFrame, cond_df: pd.DataFrame, scale_equal: bool = True,
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots target Euler angles and positions from actual poses DataFrame and variable reference poses DataFrame.
    
    Parameters:
    - actual_poses (pd.DataFrame): DataFrame with actual poses (positions and quaternions) and time.
    - reference_poses (pd.DataFrame): DataFrame with reference poses (positions and quaternions) and time.
    """
    time = actual_poses['time'].values
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000 # in mm
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry.euler_xyz_from_quaternion(q) for q in actual_orientations])

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)

    colors = ['#0343df', '#e50000', '#15b01a', '#f97306', '#7e1e9c', '#ffff14', 'k']

    # Plot positions
    fig, axs = plt.subplots(7, 1, figsize=(12, 21), sharex=True)

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(time, actual_positions[:, i], label=f"Actual {axis}", color=colors[i], **kwargs)
        axs[i].set_title(f"Position {axis} of Body Fixed Frame")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Position (mm)")
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[i+3].plot(time, actual_euler[:, i], label=f"Actual {angle}", color=colors[i], **kwargs)
        axs[i+3].set_title(angle)
        axs[i+3].set_xlabel("Time (s)")
        axs[i+3].set_ylabel("Angle (deg)")
        axs[i+3].legend()

    # Finally plotting the condition number.
    # Plot condition number in the last subplot
    axs[6].plot(cond_df['time'], cond_df['condition'], color=colors[6], label="Allocation Condition Number", **kwargs)
    axs[6].set_xlabel('Time')
    axs[6].set_ylabel("Condition Number")
    axs[6].legend()

    for ax in axs:
        ax.minorticks_on()
        ax.grid(which='major', color=mcolors.CSS4_COLORS['lightslategray'], linewidth=0.8)
        ax.grid(which='minor', color=mcolors.CSS4_COLORS['lightslategray'], linestyle=':', linewidth=0.5)

    if scale_equal:
        axs[2].sharey(axs[0])
        axs[1].sharey(axs[0])
        axs[4].sharey(axs[3])
        axs[5].sharey(axs[3])
        # Autoscale shared axes
        for ax in axs: 
            ax.relim()   
            ax.autoscale()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        plt.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)
    plt.show()
    return fig, axs

def plot_volumetric_ma_condition_number_variation(dipole: mechanical.MagneticDipole,
                                                  calibration_model: common.OctomagCalibratedModel,
                                                  orientation_quaternion: np.ndarray,
                                                  cond_threshold: float = 20,
                                                  cond_color_steps: float = 15,
                                                  clip_cond: float = 100,
                                                  cube_x_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                  cube_y_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                  cube_z_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                  reject_M_component: str = "Tz",
                                                  num_samples: int = 20,
                                                  display_interactive_pane : bool = True,
                                                  save_as: str = None,
                                                  save_dataset: bool = False,
                                                  vtk_dataset_save_kwargs: Dict = dict(),
                                                  **save_kwargs):
    """
    For supported formats using mayavi'2 inbuilt save_fig functionality, refer to:
    https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig

    Apart from this, I use tvtk to save .vti data.
    """
    
    # Sampling points according to the desired plot style.
    x_ticks = np.linspace(cube_x_lim[0], cube_x_lim[1], num_samples)
    y_ticks = np.linspace(cube_y_lim[0], cube_y_lim[1], num_samples)
    z_ticks = np.linspace(cube_z_lim[0], cube_z_lim[1], num_samples)
    X, Y, Z = np.meshgrid(x_ticks, y_ticks, z_ticks)

    M = geometry.magnetic_interaction_matrix_from_quaternion(orientation_quaternion,
                                                             dipole.strength,
                                                             full_mat=True,
                                                             torque_first=True,
                                                             dipole_axis=dipole.axis)
    
    reject_row_map = {"Tx": 0, "Ty": 1, "Tz": 2}
    idx_r = reject_row_map[reject_M_component]
    M = np.vstack((M[:idx_r], M[idx_r+1,:]))

    @np.vectorize
    def get_ma_condition(x, y, z):
        A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
        return np.linalg.cond(M @ A)
    
    cond = get_ma_condition(X, Y, Z)
    cond = np.clip(cond, 0.0, clip_cond)

    # Volumetric rendering
    cond_field = mlab.pipeline.scalar_field(cond)

    ctf = ColorTransferFunction()

    # # Add solid green color for very low values
    ctf.add_rgb_point(0,   0, 1, 0)  # Green at lowest value
    # ctf.add_rgb_point(cond_threshold - 0.1, 0, 1, 0)  # Green up to threshold

    # Add colormap for values ≥ 20 (e.g., blue to red gradient)
    ctf.add_rgb_point(cond_threshold,  0, 0, 1)  # Blue at threshold
    ctf.add_rgb_point(cond_threshold + cond_color_steps,  1, 1, 0)  # Yellow for arbitrarily high
    ctf.add_rgb_point(cond_threshold + 2*cond_color_steps,  1, 0, 0)  # Red until cond number steps from threshold


    # Create opacity transfer function (OTF) by manipulating the _volume_property
    opacity_function = tvtk.PiecewiseFunction()
    # Set opacity for values below the threshold to be opaque (1.0)
    opacity_function.add_point(1.0, 0.0)  # Fully opaque at lowest value
    opacity_function.add_point(cond_threshold, 0.2)  # Less opaque at threshold
    # Set opacity to 0.0 for higher values, making them fully transparent
    opacity_function.add_point(cond_threshold + cond_color_steps, 0.1)
    opacity_function.add_point(cond_threshold + 2*cond_color_steps, 0.01)

    cond_vol = mlab.pipeline.volume(cond_field, vmin=0, vmax=clip_cond)

    cond_vol._volume_property.set_color(ctf)  # Set custom color mapping
    cond_vol._volume_property.set_scalar_opacity(opacity_function)
    cond_vol.update_ctf = True  # Update color transfer function
    # Add axes to the plot
    axes = mlab.axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', ranges=np.array([-cube_x_lim[0], 
                                                                cube_x_lim[1], 
                                                                cube_y_lim[0], 
                                                                cube_y_lim[1],
                                                                cube_z_lim[0],
                                                                cube_z_lim[1]])*1e3)
    mlab.outline()
    rpy = np.rad2deg(geometry.euler_xyz_from_quaternion(orientation_quaternion))

    if save_as is not None:
        mlab.savefig(save_as, **save_kwargs)
        if save_dataset:
            base_name, _ = os.path.splitext(save_as)
            vt_name = base_name + ".vti"
            cond_field.save_output(vt_name, **vtk_dataset_save_kwargs)

    if display_interactive_pane:
        mlab.show()

    return cond_field

def plot_slices_ma_condition_number_variation(dipole: mechanical.MagneticDipole,
                                                calibration_model: common.OctomagCalibratedModel,
                                                orientation_quaternion: np.ndarray,
                                                cond_range: np.ndarray = np.array([0.0, 50.0]),
                                                cube_x_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                cube_y_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                cube_z_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                reject_M_component: str = "Tz",
                                                num_samples: int = 20,
                                                x_plane_idx : Optional[int] = None,
                                                y_plane_idx : Optional[int] = None,
                                                display_interactive_pane : bool = True,
                                                save_as: str = None,
                                                **save_kwargs):
    """
    For supported formats using mayavi'2 inbuilt save_fig functionality, refer to:
    https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig

    Apart from this, I use tvtk to save .vti data.
    """
    
    # Sampling points according to the desired plot style.
    x_ticks = np.linspace(cube_x_lim[0], cube_x_lim[1], num_samples)
    y_ticks = np.linspace(cube_y_lim[0], cube_y_lim[1], num_samples)
    z_ticks = np.linspace(cube_z_lim[0], cube_z_lim[1], num_samples)
    X, Y, Z = np.meshgrid(x_ticks, y_ticks, z_ticks)

    M = geometry.magnetic_interaction_matrix_from_quaternion(orientation_quaternion,
                                                             dipole.strength,
                                                             full_mat=True,
                                                             torque_first=True,
                                                             dipole_axis=dipole.axis)
    
    reject_row_map = {"Tx": 0, "Ty": 1, "Tz": 2}
    idx_r = reject_row_map[reject_M_component]
    M = np.vstack((M[:idx_r], M[idx_r+1,:]))

    @np.vectorize
    def get_ma_condition(x, y, z):
        A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
        return np.linalg.cond(M @ A)
    
    cond = get_ma_condition(X, Y, Z)

    # Render x and y aligned planes with a well defined colormap
    if x_plane_idx is None:
        x_plane_idx = num_samples // 2
    if y_plane_idx is None:
        y_plane_idx = num_samples // 2
    slice_x = mlab.volume_slice(cond, plane_orientation='x_axes', slice_index=x_plane_idx, colormap="jet",
                                    vmin=cond_range[0], vmax=cond_range[1])
    slice_y = mlab.volume_slice(cond, plane_orientation='y_axes', slice_index=y_plane_idx, colormap="jet",
                                    vmin=cond_range[0], vmax=cond_range[1])

    axes = mlab.axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', ranges=np.array([-cube_x_lim[0], 
                                                                cube_x_lim[1], 
                                                                cube_y_lim[0], 
                                                                cube_y_lim[1],
                                                                cube_z_lim[0],
                                                                cube_z_lim[1]])*1e3)
    colorbar = mlab.colorbar(orientation='vertical', nb_labels=5)
    mlab.outline()

    if save_as is not None:
        mlab.savefig(save_as, **save_kwargs)

    if display_interactive_pane:
        mlab.show()

    return (slice_x, slice_y, axes, colorbar)

def plot_volumetric_current_allocation_condition_number_variation(calibration_model: common.OctomagCalibratedModel,
                                                                    cond_threshold: float = 20,
                                                                    cond_color_steps: float = 10,
                                                                    cube_x_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                                    cube_y_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                                    cube_z_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                                    num_samples: int = 20,
                                                                    display_interactive_pane : bool = True,
                                                                    save_as: str = None,
                                                                    save_dataset: bool = False,
                                                                    vtk_dataset_save_kwargs: Dict = dict(),
                                                                    **save_kwargs):
    """
    For supported formats using mayavi'2 inbuilt save_fig functionality, refer to:
    https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig

    Apart from this, I use tvtk to save .vti data.
    """
    
    # Sampling points according to the desired plot style.
    x_ticks = np.linspace(cube_x_lim[0], cube_x_lim[1], num_samples)
    y_ticks = np.linspace(cube_y_lim[0], cube_y_lim[1], num_samples)
    z_ticks = np.linspace(cube_z_lim[0], cube_z_lim[1], num_samples)
    X, Y, Z = np.meshgrid(x_ticks, y_ticks, z_ticks)

    @np.vectorize
    def get_ma_condition(x, y, z):
        A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
        return np.linalg.cond(A)
    
    cond = get_ma_condition(X, Y, Z)

    # Volumetric rendering
    cond_field = mlab.pipeline.scalar_field(cond)

    ctf = ColorTransferFunction()

    # # Add solid green color for very low values
    ctf.add_rgb_point(0,   0, 1, 0)  # Green at lowest value
    # ctf.add_rgb_point(cond_threshold - 0.1, 0, 1, 0)  # Green up to threshold

    # Add colormap for values ≥ 20 (e.g., blue to red gradient)
    ctf.add_rgb_point(cond_threshold,  0, 0, 1)  # Blue at threshold
    ctf.add_rgb_point(cond_threshold + cond_color_steps,  1, 1, 0)  # Yellow for arbitrarily high
    ctf.add_rgb_point(cond_threshold + 2*cond_color_steps,  1, 0, 0)  # Red until cond number steps from threshold


    # Create opacity transfer function (OTF) by manipulating the _volume_property
    opacity_function = tvtk.PiecewiseFunction()
    # Set opacity for values below the threshold to be opaque (1.0)
    opacity_function.add_point(1.0, 0.0)  # Fully opaque at lowest value
    opacity_function.add_point(cond_threshold, 0.2)  # Less opaque at threshold
    # Set opacity to 0.0 for higher values, making them fully transparent
    opacity_function.add_point(cond_threshold + cond_color_steps, 0.1)
    opacity_function.add_point(cond_threshold + 2*cond_color_steps, 0.01)

    cond_vol = mlab.pipeline.volume(cond_field, vmin=0, vmax=0.8)

    cond_vol._volume_property.set_color(ctf)  # Set custom color mapping
    cond_vol._volume_property.set_scalar_opacity(opacity_function)
    cond_vol.update_ctf = True  # Update color transfer function
    # Add axes to the plot
    axes = mlab.axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', ranges=np.array([-cube_x_lim[0], 
                                                                cube_x_lim[1], 
                                                                cube_y_lim[0], 
                                                                cube_y_lim[1],
                                                                cube_z_lim[0],
                                                                cube_z_lim[1]])*1e3)
    mlab.outline()

    if save_as is not None:
        mlab.savefig(save_as, **save_kwargs)
        if save_dataset:
            base_name, _ = os.path.splitext(save_as)
            vt_name = base_name + ".vti"
            cond_field.save_output(vt_name, **vtk_dataset_save_kwargs)

    if display_interactive_pane:
        mlab.show()

    return cond_field

def plot_slices_currnet_allocation_condition_number_variation(calibration_model: common.OctomagCalibratedModel,
                                                                cond_range: np.ndarray = np.array([0.0, 50.0]),
                                                                cube_x_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                                cube_y_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                                cube_z_lim: np.ndarray = np.array([-0.06, 0.06]),
                                                                num_samples: int = 20,
                                                                x_plane_idx : Optional[int] = None,
                                                                y_plane_idx : Optional[int] = None,
                                                                display_interactive_pane : bool = True,
                                                                save_as: str = None,
                                                                **save_kwargs):
    """
    For supported formats using mayavi'2 inbuilt save_fig functionality, refer to:
    https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig

    Apart from this, I use tvtk to save .vti data.
    """
    
    # Sampling points according to the desired plot style.
    x_ticks = np.linspace(cube_x_lim[0], cube_x_lim[1], num_samples)
    y_ticks = np.linspace(cube_y_lim[0], cube_y_lim[1], num_samples)
    z_ticks = np.linspace(cube_z_lim[0], cube_z_lim[1], num_samples)
    X, Y, Z = np.meshgrid(x_ticks, y_ticks, z_ticks)

    @np.vectorize
    def get_ma_condition(x, y, z):
        A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
        return np.linalg.cond(A)
    
    cond = get_ma_condition(X, Y, Z)

    # Render x and y aligned planes with a well defined colormap
    if x_plane_idx is None:
        x_plane_idx = num_samples // 2
    if y_plane_idx is None:
        y_plane_idx = num_samples // 2
    slice_x = mlab.volume_slice(cond, plane_orientation='x_axes', slice_index=x_plane_idx, colormap="jet",
                                    vmin=cond_range[0], vmax=cond_range[1])
    slice_y = mlab.volume_slice(cond, plane_orientation='y_axes', slice_index=y_plane_idx, colormap="jet",
                                    vmin=cond_range[0], vmax=cond_range[1])

    axes = mlab.axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', ranges=np.array([-cube_x_lim[0], 
                                                                cube_x_lim[1], 
                                                                cube_y_lim[0], 
                                                                cube_y_lim[1],
                                                                cube_z_lim[0],
                                                                cube_z_lim[1]])*1e3)
    colorbar = mlab.colorbar(orientation='vertical', nb_labels=5)
    mlab.outline()

    if save_as is not None:
        mlab.savefig(save_as, **save_kwargs)

    if display_interactive_pane:
        mlab.show()

    return (slice_x, slice_y, axes, colorbar)