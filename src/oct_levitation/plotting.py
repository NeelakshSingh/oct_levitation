import numpy as np
import numpy.typing as np_t

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

import oct_levitation.geometry_jit as geometry_jit
import oct_levitation.common as common
import oct_levitation.mechanical as mechanical
from geometry_msgs.msg import Transform
import os

import pandas as pd

import subprocess

import pynumdiff.optimize
import pynumdiff.smooth_finite_difference

from typing import Optional, Tuple, List, Dict, Union, Any, Callable
from copy import deepcopy

INKSCAPE_PATH = "/usr/bin/inkscape" # default

DISABLE_PLT_SHOW = False

"""
NOTE: I would like to thank ChatGPT for coming up with some really good ideas for extending this
library and help with writing some cool plotting code I couldn't have written due to my limited
knowledge when I first started to write this library.
"""

######################################
# PLOTTING UTILITIES
######################################

AxesArray = np_t.NDArray[plt.Axes]

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

def save_plot(save_as: Union[List[str], str], fig: Figure, inkscape_path: str = INKSCAPE_PATH, **save_kwargs) -> None:
    ANSI_BLUE = '\033[94m'
    ANSI_RESET = '\033[0m'
    if isinstance(save_as, str):
        save_as = [save_as]
    for path in save_as:
        name, ext = os.path.splitext(path)
        save_format = ext[1:] if ext else 'png'  # Default to PNG if no extension
        fig.savefig(path, format=save_format, **save_kwargs)
        print(f"{ANSI_BLUE}[INFO][oct_levitation.plotting] Saved plot to {path}{ANSI_RESET}")
        if save_format.lower() == 'emf':
            fig.savefig(name + '.svg', format='svg')  # Save as SVG first
            print(f"{ANSI_BLUE}[INFO][oct_levitation.plotting] Saved intermediate SVG for EMF conversion to {name + '.svg'}{ANSI_RESET}")
            emf_file = name + '.emf'
            print(f"{ANSI_BLUE}[INFO][oct_levitation.plotting] Converting {name + '.svg'} to EMF format at {emf_file} using Inkscape{ANSI_RESET}")
            export_to_emf(path, emf_file, inkscape_path=inkscape_path)

######################################
# PLOTTING POSES
######################################

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

    return axis

def plot_poses_constant_reference(actual_poses: pd.DataFrame, reference_pose: np.ndarray, scale_equal: bool = True,
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, AxesArray]:
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
    actual_euler = np.array([geometry_jit.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array(geometry_jit.euler_xyz_from_quaternion(reference_orientation))

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)
    reference_euler = np.rad2deg(reference_euler)

    # Plot positions
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(time, actual_positions[:, i], label=f"Actual {axis}", **kwargs)
        axs[0, i].axhline(y=reference_position[i], label=f"Reference {axis}", linestyle='dashed', color='r')
        axs[0, i].set_title(f"Position {axis} of Body Fixed Frame")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Position (mm)")
        axs[0, i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[1, i].plot(time, actual_euler[:, i], label=f"Actual {angle}", **kwargs)
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
    fig.tight_layout()
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, axs


def plot_z_position_constant_reference(actual_poses: pd.DataFrame, reference_z: float,
                                       save_as: str=None,
                                       save_as_emf: bool=False,
                                       inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, plt.Axes]:
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()

    return fig, ax

def plot_z_position_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame,
                              save_as: str = None, save_as_emf: bool = False,
                              inkscape_path: str = INKSCAPE_PATH, **kwargs) -> Tuple[Figure, plt.Axes]:
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save as SVG/EMF if required
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)


    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, ax

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
    actual_yx = np.array([geometry_jit.get_normal_angles_from_quaternion(q/np.linalg.norm(q)) for q in actual_orientations])

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
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()
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
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as SVG/EMF if needed
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)


    if not DISABLE_PLT_SHOW:
        fig.show()
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
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as SVG/EMF if needed
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)


    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, axes



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
        geometry_jit.get_normal_angles_from_quaternion(quaternion/np.linalg.norm(quaternion))
        for quaternion in actual_orientations
    ])
    reference_angles = np.array([
        geometry_jit.get_normal_angles_from_quaternion(quaternion/np.linalg.norm(quaternion))
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, axs


def plot_poses_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame, scale_equal: bool = True,
                                  save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **kwargs):
    """
    Plots target Euler angles and positions from actual poses DataFrame and variable reference poses DataFrame.

    Parameters:
    - actual_poses (pd.DataFrame): DataFrame with actual poses (positions and quaternions) and time.
    - reference_poses (pd.DataFrame): DataFrame with reference poses (positions and quaternions) and time.
    """
    actual_positions = actual_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000 # in mm
    actual_orientations = actual_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values
    reference_positions = reference_poses[['transform.translation.x', 'transform.translation.y', 'transform.translation.z']].values*1000 # in mm
    reference_orientations = reference_poses[['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']].values

    # Convert quaternions to Euler angles
    actual_euler = np.array([geometry_jit.euler_xyz_from_quaternion(q) for q in actual_orientations])
    reference_euler = np.array([geometry_jit.euler_xyz_from_quaternion(q) for q in reference_orientations])

    # Convert to degrees
    actual_euler = np.rad2deg(actual_euler)
    reference_euler = np.rad2deg(reference_euler)

    # Plot positions
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    # Position plots
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[0, i].plot(actual_poses['time'], actual_positions[:, i], label=f"Actual {axis}", color="tab:blue")
        axs[0, i].plot(reference_poses['time'], reference_positions[:, i], label=f"Reference {axis}", linestyle='dashed', color='tab:red')
        axs[0, i].set_title(f"Position {axis} of Body Fixed Frame")
        axs[0, i].set_xlabel("Time (s)")
        axs[0, i].set_ylabel("Position (mm)")
        axs[0, i].legend()

    # Euler angle plots
    for i, angle in enumerate(['Roll', 'Pitch', 'Yaw']):
        axs[1, i].plot(actual_poses['time'], actual_euler[:, i], label=f"Actual {angle}", color="tab:blue")
        axs[1, i].plot(reference_poses['time'], reference_euler[:, i], label=f"Reference {angle}", linestyle='dashed', color='tab:red')
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, axs

def plot_3d_poses_with_arrows_variable_reference(actual_poses: pd.DataFrame, reference_poses: pd.DataFrame, arrow_interval: int = 10, frame_size: float = 0.01, frame_interval: int = 10,
                                                 plot_reference_arrows: bool = False, plot_reference_frames: bool = True,
                                                     save_as: str=None, save_as_emf: bool=False, inkscape_path: str=INKSCAPE_PATH, **traj_kwargs):
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
    reference_eulers = np.array([geometry_jit.euler_xyz_from_quaternion(q) for q in reference_orientations])

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot actual positions, CONVERTED TO mm
    ax.plot(actual_positions[:, 0]*1000, actual_positions[:, 1]*1000, actual_positions[:, 2]*1000, color='black', label='Actual Path', **traj_kwargs)

    # Plot reference positions (non-constant)
    ax.plot(reference_positions[:, 0]*1000, reference_positions[:, 1]*1000, reference_positions[:, 2]*1000, color='red', linestyle='--', label='Reference Path', **traj_kwargs)

    # Plot arrows for actual positions to indicate direction of forward progress
    for i in range(arrow_interval, len(time), arrow_interval):
        ax.quiver(actual_positions[i-1, 0], actual_positions[i-1, 1], actual_positions[i-1, 2],
                  actual_positions[i, 0] - actual_positions[i-1, 0],
                  actual_positions[i, 1] - actual_positions[i-1, 1],
                  actual_positions[i, 2] - actual_positions[i-1, 2],
                  color='black', arrow_length_ratio=0.1)

    if plot_reference_arrows:
        # Plot arrows for reference positions to indicate direction of forward progress
        for i in range(arrow_interval, len(time), arrow_interval):
            ax.quiver(reference_positions[i-1, 0], reference_positions[i-1, 1], reference_positions[i-1, 2],
                    reference_positions[i, 0] - reference_positions[i-1, 0],
                    reference_positions[i, 1] - reference_positions[i-1, 1],
                    reference_positions[i, 2] - reference_positions[i-1, 2],
                    color='red', linestyle='--', arrow_length_ratio=0.1)

    # Add reference pose frames (non-constant)
    if plot_reference_frames:
        for i in range(0, len(time), frame_interval):  # plot frames every 10% of time
            reference_T_0f = geometry_jit.transformation_matrix_from_euler_xyz(reference_eulers[i], reference_positions[i])
            plot_coordinate_frame(ax, reference_T_0f, size=frame_size, linewidth=1.5, name='Reference Pose', xscale=1, yscale=1, zscale=1,
                                x_style='r--', y_style='g--', z_style='b--')

    # Add coordinate frames at selected positions in the actual path
    for i in range(0, len(time), frame_interval):  # plot frames every 10% of time
        actual_T_0f = geometry_jit.transformation_matrix_from_quaternion(actual_poses.iloc[i, 4:8].to_numpy(), actual_poses.iloc[i, 1:4].to_numpy())
        plot_coordinate_frame(ax, actual_T_0f, size=frame_size, linewidth=1.5, name=None)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title("Actual Pose v/s Reference Pose of Body Fixed Frame")

    # Show legend
    ax.legend()

    # Show plot
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()
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
    reference_T_0f = geometry_jit.transformation_matrix_from_quaternion(reference_orientation, reference_position)
    plot_coordinate_frame(ax, reference_T_0f, size=frame_size, linewidth=1.5, name='Constant Reference Pose', xscale=1, yscale=1, zscale=1,
                          x_style='r--', y_style='g--', z_style='b--')

    # Add coordinate frames at selected positions in the actual path
    for i in range(0, len(time), frame_interval):
        actual_T_0f = geometry_jit.transformation_matrix_from_quaternion(actual_orientations[i], actual_positions[i])
        plot_coordinate_frame(ax, actual_T_0f, size=frame_size, linewidth=1.5, name=None)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title("Actual Pose v/s Reference Pose of Body Fixed Frame Frame")

    # Show legend
    ax.legend()

    # Show plot
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, ax

######################################
# PLOTTING CURRENTS, CONTROL INPUTS, FIELDS, CONDITION NUMBERS
######################################

def plot_dclink_voltages(system_state_df: pd.DataFrame,
                         indices: List[int]=[0, 1, 2],
                         figsize = None,
                         save_as: str=None,
                         save_as_emf: bool=False,
                         inkscape_path: str=INKSCAPE_PATH, **kwargs) -> Tuple[Figure, List[plt.Axes]]:
    # Plot each current column in its respective subplot
    # Create subplots in a 2x4 layout
    if figsize is None:
        figsize = (3*len(indices), 3)
    fig, axs = plt.subplots(1, len(indices), figsize=figsize, sharex=True, sharey=True)
    fig.suptitle("Actual DC Link Voltages vs Time", fontsize=16)  # Main title for the figure

    # Flatten the 2D axes array for easier iteration
    axs = axs.flatten()
    for i in indices:
        axs[i].plot(system_state_df['time'], system_state_df[f'dclink_voltages_{i}'], label=f'Actual DC Link Voltage {i+1}', color='tab:blue', **kwargs)
        axs[i].set_title(f'DC Link Voltage in PCB {i+1}')
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Voltage (V)")
        axs[i].grid(True)

    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()

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
        axs[i].plot(system_state_df['time'].to_numpy(), system_state_df[f'currents_reg_{i}'].to_numpy(), label=f'Actual Current {i+1}', color='tab:blue', **kwargs)
        axs[i].plot(des_currents_df['time'].to_numpy(), des_currents_df[f'des_currents_reg_{i}'].to_numpy(), label=f'Desired Current {i+1}', color='tab:green', **kwargs)
        axs[i].set_title(f'Currents in Coil {i+1}')
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Current (A)")
        axs[i].legend()
        axs[i].grid(True)

    # Adjust layout to prevent overlap
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()

    return fig, axs

def plot_actual_field_and_gradients(pose_df: pd.DataFrame, actual_currents_df: pd.DataFrame,
                                    calibrated_model: common.OctomagCalibratedModel,
                                    save_as: str = None, save_as_emf: bool = False,
                                    inkscape_path: str = INKSCAPE_PATH, **kwargs):
    """
    Plots the 3 magnetic field components (Bx, By, Bz) and the 5 gradient components (dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz)
    over time using the nonlinear model's forward computation function.

    Parameters:
        pose_df (pd.DataFrame): Pose DataFrame with columns: 'time', 'transform.translation.x', 'y', 'z'
        actual_currents_df (pd.DataFrame): Actual currents dataframe with 'currents_reg_*' columns
        calibrated_model (common.OctomagCalibratedModel): The calibration model used.
        save_as (str): Filename to save the plot as SVG/EMF (without extension).
        **kwargs: Additional parameters for plt.plot().
    """
    combined_actual = pd.merge_asof(pose_df, actual_currents_df, on='time')
    time = combined_actual['time']

    actual_values = {'Bx': [], 'By': [], 'Bz': [], 'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [], 'dBy/dy': [], 'dBy/dz': []}

    for i in range(len(combined_actual)):
        position = np.array([
            combined_actual['transform.translation.x'].iloc[i],
            combined_actual['transform.translation.y'].iloc[i],
            combined_actual['transform.translation.z'].iloc[i]
        ])
        actual_currents = np.array([combined_actual[f'currents_reg_{j}'].iloc[i] for j in range(8)])
        actual_field = calibrated_model.get_exact_field_grad5_from_currents(position, actual_currents)

        for idx, key in enumerate(actual_values.keys()):
            actual_values[key].append(actual_field[idx] * 1000)  # Convert to mT and mT/m

    # Plot the components
    fig, axs = plt.subplots(8, 1, figsize=(12, 14), sharex=True, sharey=False)
    for i, key in enumerate(actual_values.keys()):
        axs[i].plot(time, actual_values[key], linestyle='-', color='tab:red', **kwargs)
        axs[i].set_ylabel(f'{key} [mT]' if 'dB' not in key else f'{key} [mT/m]')
        axs[i].grid(True)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Actual Magnetic Field and Gradient Components at Dipole Center")
    fig.tight_layout()

    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)


    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, axs

def plot_actual_wrench_on_dipole_center_from_each_magnet(pose_df: pd.DataFrame,
                                                         actual_currents_df: pd.DataFrame,
                                                         desired_wrench: pd.DataFrame,
                                                         calibrated_model: common.OctomagCalibratedModel,
                                                         dipole: mechanical.MagneticDipole,
                                                         use_local_frame_for_torques: bool,
                                                         dataset_torques_in_local_frame: bool,
                                                         plot_for_each_magnet: bool = True,
                                                         alpha_range: Tuple[float, float] = (0.2, 0.8),
                                                         stack_label_color_map_func: Optional[Callable[[List[Tuple[Transform, mechanical.PermanentMagnet]]], List[Tuple[str, str, str, float]]]] = None,
                                                         plot_overall_magnet_torque_component: bool = True,
                                                         plot_torque_components_separately: bool = False,
                                                         save_as: str = None,
                                                         return_actual_wrench: bool = False,
                                                         component_plot_kwargs: Optional[Dict[str, Any]] = dict(),
                                                         remove_gravity_compensation_force: bool = False,
                                                         fg_comp: Optional[np_t.NDArray[float]] = None,
                                                         plot_mean_values: bool = False,
                                                         figsize: Tuple[float, float] = (15, 8),
                                                         **kwargs) -> Tuple[Figure, np_t.NDArray[plt.Axes]]:
    """
    Plots the actual and desired wrench (force and torque) components over time for a dipole center,
    based on the given body frame pose and current data.

    This function computes the actual wrench exerted by the dipole based on the current data and
    compares it to the desired wrench values. It then plots the components of force (Fx, Fy, Fz) and torque (Taux, Tauy)
    against time. The plot allows visualization of the agreement between the actual and reference values.

    IMPORTANT: PLE

    Parameters:
        pose_df (pd.DataFrame):
            DataFrame containing the pose of the COM.

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
    combined_pose_currents = pd.merge_asof(pose_df, actual_currents_df, on='time')
    actual_wrench_dict = {'wrench.torque.x': [], 'wrench.torque.y': [], 'wrench.torque.z': [],
                          'wrench.force.x': [], 'wrench.force.y': [], 'wrench.force.z': []}

    # This list of dictionaries will help us plot the force and torque contribution of
    # each magnet to the dipole center.
    per_magnet_wrench_contributions = [
        deepcopy(actual_wrench_dict) for _ in dipole.magnet_stack
    ]

    if not plot_overall_magnet_torque_component and not plot_torque_components_separately:
        plot_overall_magnet_torque_component = True

    # This one will just follow the same format as previous, but the force portion will actually
    # contain the torque on the COM contributed by forces applied to the magnet. This is important.
    per_magnet_torque_ft_contribution_components = deepcopy(per_magnet_wrench_contributions)

    key_map = {'Fx': 'wrench.force.x', 'Fy': 'wrench.force.y', 'Fz': 'wrench.force.z',
               'Taux': 'wrench.torque.x', 'Tauy': 'wrench.torque.y', 'Tauz': 'wrench.torque.z'}

    min_alpha, max_alpha = alpha_range

    def z_color_ft_mapping(magnet_stack: List[Tuple[Transform, mechanical.PermanentMagnet]]) -> List[Tuple[Optional[str], str, str, float, int]]:
        n_magnets = len(magnet_stack)
        # Collect z-values and labels for each magnet
        entries = []
        for idx, (magnet_tf, _) in enumerate(magnet_stack):
            position = geometry_jit.numpy_translation_from_tf_msg(magnet_tf)*1000 # in mm
            label = f'P: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) mm'
            entries.append((idx, position[2], label))

        # Sort by z to assign ranks: rank 0 = lowest z, rank n-1 = highest z
        sorted_by_z = sorted(entries, key=lambda x: x[1])
        rank_map = {}  # original_idx -> rank
        for rank, (idx, _, _) in enumerate(sorted_by_z):
            rank_map[idx] = rank

        first_idx = sorted_by_z[0][0]   # lowest z magnet (original index)
        last_idx = sorted_by_z[-1][0]   # highest z magnet (original index)

        # Build result in original magnet_stack order
        # Zorder: lowest z (rank 0) gets highest magnet zorder, highest z gets lowest
        base_magnet_zorder = 8
        result = [None] * n_magnets
        for idx, _, label in entries:
            rank = rank_map[idx]
            # Uniform alpha spacing by rank
            if n_magnets > 1:
                alpha = min_alpha + (max_alpha - min_alpha) * rank / (n_magnets - 1)
            else:
                alpha = max_alpha
            # Only label the first (lowest z) and last (highest z) magnets
            if n_magnets == 2:
                if idx == first_idx:
                    magnet_label = 'Lower magnet'
                elif idx == last_idx:
                    magnet_label = 'Upper magnet'
                else:
                    magnet_label = None
            else:
                if idx == first_idx:
                    magnet_label = 'Lowest magnet'
                elif idx == last_idx:
                    magnet_label = 'Highest magnet'
                else:
                    magnet_label = None
            # Lowest z (rank 0) on top of highest z (rank n-1)
            zorder = base_magnet_zorder - rank
            result[idx] = (magnet_label, "black", "black", alpha, zorder)

        return result


    if stack_label_color_map_func is None:
        stack_label_color_map_func = z_color_ft_mapping

    stack_properties = stack_label_color_map_func(dipole.magnet_stack)
    # Calculate actual wrench
    for i in range(len(combined_pose_currents)):
        position = np.array([
            combined_pose_currents['transform.translation.x'].iloc[i],
            combined_pose_currents['transform.translation.y'].iloc[i],
            combined_pose_currents['transform.translation.z'].iloc[i]
        ])
        actual_currents = np.array([combined_pose_currents[f'currents_reg_{j}'].iloc[i] for j in range(8)])

        quaternion = np.array([
            combined_pose_currents['transform.rotation.x'].iloc[i],
            combined_pose_currents['transform.rotation.y'].iloc[i],
            combined_pose_currents['transform.rotation.z'].iloc[i],
            combined_pose_currents['transform.rotation.w'].iloc[i]
        ])

        quaternion = quaternion/np.linalg.norm(quaternion)

        # Now we will iterate over each magnet in the dipole and we will treat each magnet
        # as an individual dipole. This way, we can get the forces and torques on their own
        # center. And using their pose w.r.t dipole center, we can then calculate the forces
        # and torques applied to the dipole center as a result.
        T_VM = geometry_jit.transformation_matrix_from_quaternion(quaternion, position)
        R_VM = T_VM[:3, :3]
        dipole_quat = geometry_jit.numpy_quaternion_from_tf_msg(dipole.transform)
        dipole_position = geometry_jit.numpy_translation_from_tf_msg(dipole.transform)
        T_MD = geometry_jit.transformation_matrix_from_quaternion(dipole_quat, dipole_position)

        actual_com_force = np.zeros(3)
        actual_com_torque = np.zeros(3)

        for i, (magnet_tf, magnet) in enumerate(dipole.magnet_stack):
            mag_quaternion = geometry_jit.numpy_quaternion_from_tf_msg(magnet_tf)
            mag_position = geometry_jit.numpy_translation_from_tf_msg(magnet_tf)
            T_DG= geometry_jit.transformation_matrix_from_quaternion(mag_quaternion, mag_position)
            T_MG = T_MD @ T_DG
            t_MG_M = T_MG[:3, 3] # relative position of the magnet w.r.t the body fixed frame expressed in the body fixed frame

            T_VG = T_VM @ T_MG
            R_VG = T_VG[:3, :3] # rotmat from magnet frame to world frame
            R_MG = T_MG[:3, :3] # rotmat from magnet frame to body fixed frame
            p_G_V = T_VG[:3, 3] # position of the magnet frame (magnet's dipole center) in world frame (calibration frame)

            bg_V = calibrated_model.get_exact_field_grad5_from_currents(p_G_V, actual_currents)
            b_V = bg_V[:3] # magnetic field in world frame
            g_V = bg_V[3:] # magnetic field gradient in world frame

            mag_dipole_G = magnet.magnetization_axis * magnet.get_dipole_strength()
            mag_dipole_V = R_VG @ mag_dipole_G # magnet's dipole moment expressed in world frame
            mag_dipole_M = R_MG @ mag_dipole_G # magnet's dipole moment expressed in body fixed frame

            Mf = geometry_jit.magnetic_interaction_grad5_to_force(mag_dipole_V) # magnetic interaction from V frame gradients to V frame forces on the magnet center
            magnet_force_V = Mf @ g_V
            magnet_force_M = (R_VM.T @ magnet_force_V).flatten()

            Mbar_tau = geometry_jit.magnetic_interaction_field_to_local_torque_from_rotmat(mag_dipole_M, R_VM) # This will map the V frame field to M frame torques

            magnet_force_world = magnet_force_V.flatten()
            magnet_com_torque_from_torque = (Mbar_tau @ b_V).flatten()

            actual_com_force += magnet_force_world
            magnet_com_torque_from_force = np.cross(t_MG_M, magnet_force_M).flatten()
            magnet_torque_com_M = magnet_com_torque_from_force + magnet_com_torque_from_torque
            actual_com_torque += magnet_torque_com_M

            # let's convert the contribution to its respective frame too.
            magnet_com_torque_contribution = magnet_torque_com_M
            if not use_local_frame_for_torques:
                magnet_com_torque_contribution = R_VM @ magnet_com_torque_contribution
                magnet_com_torque_from_torque = R_VM @ magnet_com_torque_from_torque
                magnet_com_torque_from_force = R_VM @ magnet_com_torque_from_force

            # Explicit appending to avoid confusion. There were indexing errors with the overall
            # contribution dictionary. So I won't do that for this edit.
            magnet_contribution_dict = per_magnet_wrench_contributions[i]
            magnet_contribution_dict[key_map['Fx']].append(magnet_force_world[0])
            magnet_contribution_dict[key_map['Fy']].append(magnet_force_world[1])
            magnet_contribution_dict[key_map['Fz']].append(magnet_force_world[2])
            magnet_contribution_dict[key_map['Taux']].append(magnet_com_torque_contribution[0])
            magnet_contribution_dict[key_map['Tauy']].append(magnet_com_torque_contribution[1])
            magnet_contribution_dict[key_map['Tauz']].append(magnet_com_torque_contribution[2])
            magnet_torque_ft_contribution_dict = per_magnet_torque_ft_contribution_components[i]
            magnet_torque_ft_contribution_dict[key_map['Fx']].append(magnet_com_torque_from_force[0])
            magnet_torque_ft_contribution_dict[key_map['Fy']].append(magnet_com_torque_from_force[1])
            magnet_torque_ft_contribution_dict[key_map['Fz']].append(magnet_com_torque_from_force[2])
            magnet_torque_ft_contribution_dict[key_map['Taux']].append(magnet_com_torque_from_torque[0])
            magnet_torque_ft_contribution_dict[key_map['Tauy']].append(magnet_com_torque_from_torque[1])
            magnet_torque_ft_contribution_dict[key_map['Tauz']].append(magnet_com_torque_from_torque[2])


        if not use_local_frame_for_torques: # By default we calculate them in the local frame.
            actual_com_torque = R_VM @ actual_com_torque

        ## Depending on the torque evaluation frame and the current frame of the torques in the dataset.
        ## convert the desired torques. This was implemented because some experiments perform direct world
        ## frame torque control, while some perform body fixed frame torque control.
        if not dataset_torques_in_local_frame and use_local_frame_for_torques:
            # Also transform the desired torques from the world frame to the local frame.
            des_torques = desired_wrench.iloc[i][[key_map['Taux'], key_map['Tauy'], key_map['Tauz']]].to_numpy()
            des_torques = geometry_jit.rotate_vector_from_quaternion(
                geometry_jit.invert_quaternion(quaternion),
                des_torques
            )
            desired_wrench.loc[i, [key_map['Taux'], key_map['Tauy'], key_map['Tauz']]] = des_torques

        if dataset_torques_in_local_frame and not use_local_frame_for_torques:
             # Also transform the desired torques from the local frame to the world frame.
            des_torques = desired_wrench.iloc[i][[key_map['Taux'], key_map['Tauy'], key_map['Tauz']]].to_numpy()
            des_torques = geometry_jit.rotate_vector_from_quaternion(
                quaternion,
                des_torques
            )
            desired_wrench.loc[i, [key_map['Taux'], key_map['Tauy'], key_map['Tauz']]] = des_torques

        actual_wrench = np.concatenate((actual_com_torque, actual_com_force)) # This sequence should be correct. Please double check.

        for j, key in enumerate(list(actual_wrench_dict.keys())):
            actual_wrench_dict[key].append(actual_wrench[j])

    # Convert wrench dict to DataFrame
    actual_wrench_df = pd.DataFrame(actual_wrench_dict)

    if remove_gravity_compensation_force:
        actual_wrench_df[key_map['Fx']] -= fg_comp[0]
        actual_wrench_df[key_map['Fy']] -= fg_comp[1]
        actual_wrench_df[key_map['Fz']] -= fg_comp[2]
        desired_wrench[key_map['Fx']] -= fg_comp[0]
        desired_wrench[key_map['Fy']] -= fg_comp[1]
        desired_wrench[key_map['Fz']] -= fg_comp[2]

        for magnet_contribution in per_magnet_wrench_contributions:
            magnet_contribution[key_map['Fx']] -= fg_comp[0]
            magnet_contribution[key_map['Fy']] -= fg_comp[1]
            magnet_contribution[key_map['Fz']] -= fg_comp[2]

    # Plot settings
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    fig.suptitle('Actual Wrench (Non-Linear Model computed) v/s Desired Wrench')
    time_actual = actual_currents_df['time']
    time_des = desired_wrench['time']

    # Force subplots (columns 0, 1, 2), Forces converted to mN
    for i, force_component in enumerate(['Fx', 'Fy', 'Fz']):
        actual_ft_label = 'Actual FT' if i == 2 else None
        desired_ft_label = 'Desired FT' if i == 2 else None
        axes[0, i].plot(time_actual, actual_wrench_df[key_map[force_component]]*1000, label=actual_ft_label, color='tab:green', linewidth=1, alpha=1.0, zorder=9, **kwargs)
        axes[0, i].plot(time_des, desired_wrench[key_map[force_component]]*1000, label=desired_ft_label, color='tab:blue', linewidth=1, zorder=10, alpha=0.8, linestyle="--", **kwargs)
        if plot_for_each_magnet:
            for num, (magnet_wrench_contribution, (magnet_tf, _)) in enumerate(zip(per_magnet_wrench_contributions, dipole.magnet_stack)):
                _, force_color, _, alpha, mag_zorder = stack_properties[num]
                axes[0, i].plot(time_actual, np.array(magnet_wrench_contribution[key_map[force_component]])*1000,
                                color=force_color,
                                alpha=alpha,
                                linewidth=1,
                                zorder=mag_zorder,
                                **component_plot_kwargs)
        if plot_mean_values:
            mean_value = np.mean(actual_wrench_df[key_map[force_component]]*1000)
            axes[0, i].axhline(mean_value, color='tab:purple', linestyle='--', label=f'Mean actual force', zorder=11)
        axes[0, i].set_title(f'{force_component} - Force')
        axes[0, i].grid(True)
        if i == 0:
            axes[0, i].set_ylabel('Force (mN)')
        if i == 2:
            axes[0, i].legend(loc='lower right').set_zorder(12)

    # Torque subplots (columns 0, 1, 2), Torques converted to mN-m
    for i, (torque_component, torque_from_force_component) in enumerate(zip(['Taux', 'Tauy', 'Tauz'], ['Fx', 'Fy', 'Fz'])):
        axes[1, i].plot(time_actual, actual_wrench_df[key_map[torque_component]]*1e3, color='tab:green', linewidth=1, alpha=1.0, zorder=9, **kwargs)
        axes[1, i].plot(time_des, desired_wrench[key_map[torque_component]]*1e3, color='tab:blue', linewidth=1, zorder=10, alpha=0.8, linestyle="--", **kwargs)
        title = f'{torque_component} - Torque'
        if plot_torque_components_separately:
            title += ' \n(dotted: F contrib., dashed: Tau contrib.)'
        if plot_for_each_magnet:
            for num, (magnet_wrench_contribution, magnet_torque_components, (magnet_tf, _)) in enumerate(zip(per_magnet_wrench_contributions, per_magnet_torque_ft_contribution_components, dipole.magnet_stack)):
                label, _, torque_color, alpha, mag_zorder = stack_properties[num]
                if plot_overall_magnet_torque_component:
                    axes[1, i].plot(time_actual, np.array(magnet_wrench_contribution[key_map[torque_component]])*1000,
                                    label=label,
                                    color=torque_color,
                                    alpha=alpha,
                                    linewidth=1,
                                    zorder=mag_zorder,
                                    **component_plot_kwargs)
                if plot_torque_components_separately:
                    # Plotting the contribution from forces
                    if plot_overall_magnet_torque_component:
                        # No need to label
                        axes[1, i].plot(time_actual, np.array(magnet_torque_components[key_map[torque_from_force_component]])*1000,
                                        color=torque_color,
                                        alpha=alpha,
                                        zorder=mag_zorder,
                                        linestyle=":",
                                        **component_plot_kwargs)
                        axes[1, i].plot(time_actual, np.array(magnet_torque_components[key_map[torque_component]])*1000,
                                        color=torque_color,
                                        alpha=alpha,
                                        zorder=mag_zorder,
                                        linestyle="--",
                                        **component_plot_kwargs)
                    if not plot_overall_magnet_torque_component:
                        # Need to label one plot. I just label pure torques.
                        axes[1, i].plot(time_actual, np.array(magnet_torque_components[key_map[torque_from_force_component]])*1000,
                                        color=torque_color,
                                        alpha=alpha,
                                        zorder=mag_zorder,
                                        linestyle=":",
                                        **component_plot_kwargs)
                        axes[1, i].plot(time_actual, np.array(magnet_torque_components[key_map[torque_component]])*1000,
                                        color=torque_color,
                                        alpha=alpha,
                                        label=label,
                                        zorder=mag_zorder,
                                        linestyle="--",
                                        **component_plot_kwargs)
        if plot_mean_values:
            mean_value = np.mean(actual_wrench_df[key_map[torque_component]]*1e3)
            axes[1, i].axhline(mean_value, color='tab:purple', linestyle='--', label=f'Mean actual torque', zorder=11)
        axes[1, i].set_title(title)
        axes[1, i].grid(True)
        if i == 0:
            axes[1, i].set_ylabel('Torque (mN-m)')
        if i == 2:
            axes[1, i].legend(loc='upper right').set_zorder(12)

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

    fig.tight_layout()

    if save_as:
        save_plot(save_as, fig, inkscape_path=INKSCAPE_PATH)


    if not DISABLE_PLT_SHOW:
        fig.show()

    if return_actual_wrench:
        actual_wrench_df['time'] = time_actual
        return fig, axes, actual_wrench_df
    return fig, axes

def plot_estimated_velocities(dipole_center_pose_df: pd.DataFrame,
                              save_as: str = None,
                              save_as_emf: bool = False,
                              inkscape_path: str = INKSCAPE_PATH,
                              also_plot_pynumdiff: bool = True,
                              also_use_wrench: bool = False,
                              dipole_actual_wrench_df: pd.DataFrame = None,
                              rigid_body_dipole : mechanical.MultiDipoleRigidBody = None,
                              cutoff_frequency: float = 50,
                              local_frame_for_ang_vel: bool = True,
                              **kwargs) -> Tuple[Figure, np.ndarray]:

    """
    Plots the linear velocities and angular velocities (in local frame) over time for a dipole center,
    based on the given pose data.

    This function computes the linear and angular velocities using finite differences on the pose data.
    """

    # Time and pose data (Assuming pose is given in quaternion, position in xyz)
    time = dipole_center_pose_df['time'].to_numpy()
    position_columns = ['transform.translation.x', 'transform.translation.y', 'transform.translation.z']
    orientation_columns = ['transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w']  # Quaternion components

    # Compute linear velocities using finite differences
    positions = dipole_center_pose_df[position_columns].to_numpy()

    linear_velocities = np.diff(positions, axis=0) / np.diff(time)[:, None]
    linear_velocities = np.vstack([np.zeros(3), linear_velocities])  # Insert a zero at the start for the first time step
    linear_velocities_pynumdiff = None

    quaternions = dipole_center_pose_df[orientation_columns].to_numpy()

    # Compute angular velocities using finite differences on Euler angles
    euler_angles = np.array([geometry_jit.euler_xyz_from_quaternion(dipole_center_pose_df[orientation_columns].iloc[i].to_numpy())
                             for i in range(len(dipole_center_pose_df))])

    # Compute angular velocities using finite differences of Euler angles
    euler_rates_fd = np.diff(euler_angles, axis=0) / np.diff(time)[:, None]
    euler_rates_fd = np.vstack([np.zeros(3), euler_rates_fd])  # Insert a zero at the start for the first time step

    # Prepare figure and axes for subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colors = ['tab:blue', 'tab:red', 'tab:green']  # Velocity (actual, reference), Angular Velocity (actual, reference)

    # Plot angular velocities (wx, wy, wz) on second row
    angular_velocities_fd = np.array([geometry_jit.euler_xyz_rate_to_local_angular_velocity(euler_rates_fd[i], euler_angles[i])
                                      for i in range(len(dipole_center_pose_df))]) # Get local angular velocities from Euler angle derivatives

    angular_velocities_pynumdiff = None

    # Use pynumdiff if enabled
    if also_plot_pynumdiff:
        # Compute all linear velocities and angular rates using pynumdiff
        # Strategy is to use the optimizer on the first component and then use the optimal
        # parameters for the rest of the components too.
        dt = np.average(np.diff(time.flatten()))
        log_gamma = -1.6*np.log(cutoff_frequency) -0.71*np.log(dt) - 5.1
        tvgamma = np.exp(log_gamma)
        def optimal_diff_signal(signal: np.ndarray):
            signal_params, signal_val = pynumdiff.optimize.smooth_finite_difference.butterdiff(
                signal, dt, params=None, tvgamma=tvgamma, options={'iterate': True}
            )
            return pynumdiff.smooth_finite_difference.butterdiff(
                signal, dt, params=signal_params, options={'iterate': True}
            )

        linear_velocities_pynumdiff = np.zeros_like(linear_velocities)
        euler_rates_pynumdiff = np.zeros_like(euler_rates_fd)
        euler_angles_filtered = np.zeros_like(euler_angles)
        for i in range(3):
            signal = positions[:, i]
            signal_filtered, signal_dot = optimal_diff_signal(signal)
            linear_velocities_pynumdiff[:, i] = signal_dot

            ang_signal = euler_angles[:, i]
            ang_signal_filtered, ang_signal_dot = optimal_diff_signal(ang_signal)
            euler_rates_pynumdiff[:, i] = ang_signal_dot
            euler_angles_filtered[:, i] = ang_signal_filtered

        # Finally convert the euler rates to angular velocities
        angular_velocities_pynumdiff = np.array([geometry_jit.euler_xyz_rate_to_local_angular_velocity(euler_rates_pynumdiff[i], euler_angles_filtered[i])
                                                 for i in range(len(dipole_center_pose_df))]) # Get local angular velocities from Euler angle derivatives

    linear_velocities_wrench = None
    angular_velocities_wrench = None

    if also_use_wrench:
        dipole_actual_wrench_df["dt"] = dipole_actual_wrench_df["time"].diff().fillna(0)  # First entry has no previous sample
        linear_velocities_wrench = np.zeros_like(linear_velocities)
        angular_velocities_wrench = np.zeros_like(angular_velocities_fd)
        # Compute linear velocity (integrate forces)
        linear_velocities_wrench[:, 0] = ((dipole_actual_wrench_df["wrench.force.x"] / rigid_body_dipole.mass_properties.m) * dipole_actual_wrench_df["dt"]).cumsum()
        linear_velocities_wrench[:, 1] = ((dipole_actual_wrench_df["wrench.force.y"] / rigid_body_dipole.mass_properties.m) * dipole_actual_wrench_df["dt"]).cumsum()
        linear_velocities_wrench[:, 2] = ((dipole_actual_wrench_df["wrench.force.z"] / rigid_body_dipole.mass_properties.m) * dipole_actual_wrench_df["dt"]).cumsum()

        # Compute angular velocity (integrate torques)
        angular_velocities_wrench[:, 0] = ((dipole_actual_wrench_df["wrench.torque.x"] / rigid_body_dipole.mass_properties.I_bf[0, 0]) * dipole_actual_wrench_df["dt"]).cumsum()
        angular_velocities_wrench[:, 1] = ((dipole_actual_wrench_df["wrench.torque.y"] / rigid_body_dipole.mass_properties.I_bf[1, 1]) * dipole_actual_wrench_df["dt"]).cumsum()
        angular_velocities_wrench[:, 2] = ((dipole_actual_wrench_df["wrench.torque.z"] / rigid_body_dipole.mass_properties.I_bf[2, 2]) * dipole_actual_wrench_df["dt"]).cumsum()

    if not local_frame_for_ang_vel:
        # Convert angular velocities to the global frame.
        # The estimated angular velocities are directly in the local frame always.
        def rotate_array_from_quat_array(qarr: np.ndarray, varr: np.ndarray):
            return np.array([geometry_jit.rotate_vector_from_quaternion(qarr[i], varr[i])
                                          for i in range(len(qarr))])
        angular_velocities_fd = rotate_array_from_quat_array(quaternions, angular_velocities_fd) # Get local angular velocities from Euler angle derivatives
        if angular_velocities_pynumdiff is not None:
            angular_velocities_pynumdiff = rotate_array_from_quat_array(quaternions, angular_velocities_pynumdiff)

    # Plot linear velocities (vx, vy, vz) on first row
    for i, component in enumerate(position_columns):
        axes[0, i].plot(time, linear_velocities[:, i] * 1e3, label=f'{component} (mm/s)', color=colors[0], zorder=1, **kwargs)
        if also_plot_pynumdiff:
            axes[0, i].plot(time, linear_velocities_pynumdiff[:, i] * 1e3, label=f'Pynumdiff {component} (mm/s)', color=colors[1], zorder=2, **kwargs)
        if also_use_wrench:
            axes[0, i].plot(time, linear_velocities_wrench[:, i] * 1e3, label=f'Torque integrated {component} (mm/s)', color=colors[2], zorder=3, **kwargs)

        axes[0, i].set_title(f'{component} - Linear Velocity')
        axes[0, i].minorticks_on()
        axes[0, i].grid(which='major', color=mcolors.CSS4_COLORS['lightslategray'], linewidth=0.8)
        axes[0, i].grid(which='minor', color=mcolors.CSS4_COLORS['lightslategray'], linestyle=':', linewidth=0.5)
        if i == 0:
            axes[0, i].set_ylabel('Velocity (mm/s)')
        if i == 2:
            axes[0, i].legend(loc='upper right')

    for i, component in enumerate(['wx', 'wy', 'wz']):
        axes[1, i].plot(time, np.rad2deg(angular_velocities_fd[:, i]), label=f'{component} (deg/s)', color=colors[0], zorder=1, **kwargs)
        if also_plot_pynumdiff:
            axes[1, i].plot(time, np.rad2deg(angular_velocities_pynumdiff[:, i]), label=f'Pynumdiff {component} (deg/s)', color=colors[1], zorder=2, **kwargs)
        if also_use_wrench:
            axes[1, i].plot(time, np.rad2deg(angular_velocities_wrench[:, i]), label=f'Torque integrated {component} (deg/s)', color=colors[2], zorder=3, **kwargs)
        axes[1, i].set_title(f'{component} - Angular Velocity')
        axes[1, i].minorticks_on()
        axes[1, i].grid(which='major', color=mcolors.CSS4_COLORS['lightslategray'], linewidth=0.8)
        axes[1, i].grid(which='minor', color=mcolors.CSS4_COLORS['lightslategray'], linestyle=':', linewidth=0.5)
        if i == 0:
            axes[1, i].set_ylabel('Angular Velocity (deg/s)')
        if i == 2:
            axes[1, i].legend(loc='upper right')

    fig.suptitle(f"Estimated velocities using numerical differentiation. Local frame for angular velocities: {local_frame_for_ang_vel}")

    # Shared X-axis for all subplots
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

    fig.tight_layout()

    # Saving plot if requested
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()

    return fig, axes

######################################
# MAGNETIC ACTUATION ANALYSIS PLOTS
######################################

def plot_jma_condition_number(jma_cond_df: pd.DataFrame,
                              save_as: str = None,
                              save_as_emf: bool = False,
                              inkscape_path: str = INKSCAPE_PATH,
                              **kwargs):

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(jma_cond_df['time'], jma_cond_df['vector_0'], color='#0343df', label='Allocation condition', **kwargs)  # Blue
    ax.set_xlabel('Time')
    ax.set_ylabel('Vector 0')
    ax.set_title('Allocation Condition Plot')
    ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()

    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)


    if not DISABLE_PLT_SHOW:
        fig.show()

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
    actual_euler = np.array([geometry_jit.euler_xyz_from_quaternion(q) for q in actual_orientations])

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
    axs[6].plot(cond_df['time'], cond_df['vector_0'], color=colors[6], label="Allocation Condition Number", **kwargs)
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()
    return fig, axs

######################################
# UTILITY PLOTS LIKE COMPUTATION TIMES
######################################

def plot_computation_times(comptime_df: pd.DataFrame,
                           save_as: str = None,
                           save_as_emf: bool = False,
                           inkscape_path: str = INKSCAPE_PATH,
                           **kwargs) -> Tuple[Figure, plt.Axes]:
    """
    Plots computation times from a DataFrame.

    Parameters:
        comptime_df (pd.DataFrame): DataFrame containing computation times.
        save_as (str, optional): Path to save the plot as an SVG or PNG file (without extension).
        save_as_emf (bool, optional): If True, saves the plot as an EMF file. Requires 'inkscape_path' to be provided.
        inkscape_path (str, optional): Path to the Inkscape executable for converting to EMF format.
        **kwargs: Additional keyword arguments passed to plt.plot().
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(comptime_df['time'], comptime_df['vector_0'], color='tab:blue', label='Computation Time', **kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Computation Time Plot')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()

    if save_as and save_as.endswith('.svg'):
        fig.savefig(save_as, format='svg')
        if save_as_emf:
            emf_file = save_as.replace('.svg', '.emf')
            export_to_emf(save_as, emf_file, inkscape_path=inkscape_path)

    if not DISABLE_PLT_SHOW:
        fig.show()

    return fig, ax
