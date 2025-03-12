#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import TransformStamped

import oct_levitation.geometry as geometry
import oct_levitation.rigid_bodies as rigid_bodies

class LiveVectorPlot:
    def __init__(self):
        rospy.init_node("reduced_attitude_visualizer", anonymous=True)
        
        self.pose_topic = rigid_bodies.Onyx80x22DiscCenterRingDipole.pose_frame
        self.pose_sub = rospy.Subscriber(self.pose_topic, TransformStamped, self.pose_callback)

        # Matplotlib setup
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Fixed arrows
        self.ez = np.array([0, 0, 1])
        self.Gamma_sp = geometry.inertial_reduced_attitude_from_quaternion(geometry.IDENTITY_QUATERNION, self.ez)
        
        # Path storage
        self.path = []

        self.pose_received = False
        self.init_plot()

    def init_plot(self):
        """Draws fixed elements: sphere, initial, and final arrow."""
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        self.ax.plot_wireframe(x, y, z, color='c', alpha=0.3)  # Sphere

        # Initial and final arrow
        self.ax.quiver(0, 0, 0, *self.Gamma_sp, color='g', label="Final (Green)")

    def pose_callback(self, msg: TransformStamped):
        """Processes ROS pose message and updates plot dynamically."""
        quaternion = [
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w
        ]

        Gamma = geometry.inertial_reduced_attitude_from_quaternion(quaternion, self.ez)
        if not self.pose_received:
            self.Gamma_initial = Gamma
            self.ax.quiver(0, 0, 0, *self.Gamma_initial, color='r', label="Initial (Red)")

        self.path.append(Gamma)  # Store for path tracing

        # Live update
        self.update_plot(Gamma)
        self.pose_received = True

    def update_plot(self, vec):
        """Redraws live elements: current arrow and path."""
        self.ax.clear()
        self.init_plot()

        # Plot traced path
        path_array = np.array(self.path)
        if len(self.path) > 1:
            self.ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'b', label="Path (Blue)")

        # Current position arrow
        self.ax.quiver(0, 0, 0, *vec, color='b', label="Current (Blue)")

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)  # Keeps the plot responsive

    def run(self):
        """Loop to keep ROS and Matplotlib running."""
        while not rospy.is_shutdown():
            if not self.pose_received:
                rospy.loginfo_once("Waiting for pose messages...")
            plt.pause(0.1)

if __name__ == "__main__":
    plotter = LiveVectorPlot()
    plotter.run()
