import numpy as np
import rospy
import tf2_ros
import tf.transformations as tr

from geometry_msgs.msg import TransformStamped, WrenchStamped, Vector3, Quaternion
from tnb_mns_driver.msg import DesCurrentsReg

import oct_levitation.common as common
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry
import oct_levitation.numerical as numerical

from scipy.integrate import solve_ivp

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
        initial_rpy = np.array(rospy.get_param("~initial_rpy", [0.0, 0.0, 0.0])) # World frame
        self.q = geometry.quaternion_from_euler_xyz(np.rad2deg(initial_rpy)) # World frame
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
        self.p, self.v = numerical.integrate_linear_dynamics_constant_force(self.p, self.v, F, self.m, dt)

        # Numerically integration the orientation through the lie group exponential map of angular velocity.
        # The angular velocity is resolved in the local frame.
        self.R, self.omega = numerical.integrate_R_omega_constant_torque(self.R, self.omega, Tau, self.I_bf, dt)

        self.q = geometry.quaternion_from_rotation_matrix(self.R)

        self.__tf_msg.header.stamp = rospy.Time.now()
        self.__tf_msg.transform.translation = Vector3(*self.p)
        self.__tf_msg.transform.rotation = Quaternion(*self.q / np.linalg.norm(self.q))
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