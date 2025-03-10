import numpy as np
import rospy
import tf2_ros
import tf.transformations as tr

from geometry_msgs.msg import TransformStamped, WrenchStamped
from tnb_mns_driver.msg import DesCurrentsReg

import oct_levitation.common as common
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry

from scipy.linalg import expm

def ft_array_from_wrench(wrench: WrenchStamped):
    return (np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z]),
            np.array([wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z]))


class DynamicsSimulator:

    def __init__(self):
        rospy.init_node('dynamics_simulator', anonymous=True)
        self.Ts = 1e-2
        self.__tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.__tf_msg = TransformStamped()
        self.rigid_body = rigid_bodies.Onyx80x22DiscCenterRingDipole
        self.vicon_frame = self.rigid_body.pose_frame
        self.world_frame = "vicon/world"
        self.vicon_pub = rospy.Publisher(self.vicon_frame, TransformStamped, queue_size=10)

        self.last_command_recv = 0
        self.last_commmand_timeout = 0.2
        self.__last_command_recv_time = 0.0
        self.__first_command = False
        self.__last_command_warning_sent = False
        self.__first_sim_step = True
        self.last_time = rospy.Time.now().to_sec()

        ### Initial Conditions
        gravity_on = rospy.get_param("~gravity_on", False)
        self.p = np.array(rospy.get_param("~initial_position", [0.0, 0.0, 0.0])) # World frame
        self.v = np.array(rospy.get_param("~initial_velocity", [0.0, 0.0, 0.0])) # World frame
        self.q = np.array(rospy.get_param("~initial_orientation", [0.0, 0.0, 0.0])) # Local frame orientation w.r.t world frame
        self.R = geometry.rotation_matrix_from_quaternion(self.q) # Same as above.
        self.omega = np.array(rospy.get_param("~initial_angular_velocity", [0.0, 0.0, 0.0])) # w.r.t world frame resolved in local frame.

        if gravity_on:
            self.F_amb = np.array([0, 0, -common.Constants.g])
        else:
            self.F_amb = np.array([0, 0, 0])
        
        self.__tf_msg.header.frame_id = self.world_frame
        self.__tf_msg.child_frame_id = self.vicon_frame

        self.last_recvd_wrench = WrenchStamped()

        self.wrench_sub = rospy.Subscriber(self.rigid_body.com_wrench_topic, WrenchStamped, self.wrench_callback, queue_size=1)

        self.last_sim_time = rospy.Time.now().to_sec()
        self.I_bf = self.rigid_body.mass_properties.I_bf
        self.m = self.rigid_body.mass_properties.m
        self.I_bf_inv = np.linalg.inv(self.I_bf)

        self.simulation_timer = rospy.Timer(rospy.Duration(self.Ts), self.simulation_loop)
    
    def simulation_loop(self, event):

        # This simulation loop will assume ZOH, therefore the last command is just kept on being repeated
        # we won't explicitly set it to zero here. The forces are assumed to be in the world frame while
        # the torques are assumed to be in the body fixed frame.
        if self.__first_sim_step:
            self.__first_sim_step = False
            self.last_sim_time = rospy.Time.now().to_sec()
            return

        if rospy.Time.now().to_sec() - self.__last_command_recv_time > self.last_commmand_timeout:
            if self.__first_command:
                if not self.__last_command_warning_sent:
                    self.__last_command_warning_sent = True
                    rospy.logwarn("No command received in the last %.2f seconds." % self.last_commmand_timeout)

        dt = rospy.Time.now().to_sec() - self.last_sim_time
        self.last_sim_time = rospy.Time.now().to_sec()

        # Computing the velocity and position
        F, Tau = ft_array_from_wrench(self.last_recvd_wrench)
        self.v += (F + self.F_amb)/self.m * dt
        self.p += self.v * dt

        # Numerically integration the orientation through the lie group exponential map of angular velocity.
        # The angular velocity is resolved in the local frame.
        omega_dot = self.I_bf_inv @ (Tau - np.cross(self.omega, self.I_bf @ self.omega))
        self.omega += omega_dot * dt
        # In the following step, we make use of the fact that the angular velocity represents the instantaneous
        # axis of rotation and use the skew symmetric tangent space representation of the lie group SO(3) to
        # compute the change rotation matrix.
        R_dot = expm(geometry.skew_symmetric(self.omega) * dt)
        self.R = self.R @ R_dot

        self.q = geometry.quaternion_from_

        self.__tf_msg.header.stamp = rospy.Time.now()
        self.__tf_msg.transform.translation.z = state[0]
        self.__tf_broadcaster.sendTransform(self.__tf_msg)
        self.vicon_pub.publish(self.__tf_msg)

    def wrench_callback(self, com_wrench: WrenchStamped):
        if not self.__first_command:
            self.__first_command = True
        self.last_recvd_wrench = com_wrench
        self.__last_command_recv_time = rospy.Time.now().to_sec()
        if self.__last_command_warning_sent:
            rospy.loginfo("Command received again.")
            self.__last_command_warning_sent = False
    
    def run(self):
        self.timer = rospy.Timer(rospy.Duration(self.Ts), self.timer_callback)

if __name__ == "__main__":
    dynamics_simulator = DynamicsSimulator()
    dynamics_simulator.run()
    rospy.spin()