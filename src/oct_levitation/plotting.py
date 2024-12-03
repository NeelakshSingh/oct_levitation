import numpy as np
import matplotlib.pyplot as plt
import oct_levitation.geometry as geometry

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
                          xscale=1, yscale=1, zscale=1):
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
    axis.plot3D(X[:,0],X[:,1],X[:,2], 'r-', linewidth=linewidth)
    axis.plot3D(Y[:,0],Y[:,1],Y[:,2], 'g-', linewidth=linewidth)
    axis.plot3D(Z[:,0],Z[:,1],Z[:,2], 'b-', linewidth=linewidth)

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