import numpy as np
import rospy
import tf2_ros
import os
import tf.transformations as tr
import time

from geometry_msgs.msg import TransformStamped, WrenchStamped, Vector3, Quaternion
from std_msgs.msg import Bool
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
        self.Ts = 1/rospy.get_param("~sim_freq", 3000)
        rospy.loginfo(f"[free_rigid_body_simulator] Requested frequency: {1/self.Ts}")
        # self.Ts = 1/3000
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

        ### Calibration model to use nonlinear model.
        self.calibration = common.OctomagCalibratedModel(calibration_type="legacy_yaml",
                                                         calibration_file="mc3ao8s_md200_handp.yaml")

        ### Initial Conditions
        gravity_on = rospy.get_param("~gravity_on", False)
        self.p = np.array(rospy.get_param("~initial_position", [0.0, 0.0, 0.0])) # World frame
        self.v = np.array(rospy.get_param("~initial_velocity", [0.0, 0.0, 0.0])) # World frame
        initial_rpy = np.array(rospy.get_param("~initial_rpy", [0.0, 0.0, 0.0])) # World frame
        rospy.loginfo(f"[free_body_sim] initial_rpy: {initial_rpy}")
        self.q = geometry.quaternion_from_euler_xyz(np.deg2rad(initial_rpy)) # World frame
        self.R = geometry.rotation_matrix_from_quaternion(self.q) # Same as above.
        self.omega = np.array(rospy.get_param("~initial_angular_velocity", [0.0, 0.0, 0.0])) # w.r.t world frame resolved in local frame.
        self.omega = np.deg2rad(self.omega)
        self.use_wrench = rospy.get_param("~use_wrench", False)
        self.publish_status = rospy.get_param("~pub_status", False)
        self.print_ft = rospy.get_param("~print_ft", False)
        self.current_noise_covariance = rospy.get_param("~current_noise_covariance", 0.01)

        self.vicon_pub_time_ns = 1e9/rospy.get_param("~vicon_pub_freq", 100) #
        if self.publish_status:
            self.sim_status_pub = rospy.Publisher("oct_levitation/free_rigid_body_sim/status", Bool, queue_size=1) # This is just to measure the simulator run freq
        self.__last_vicon_pub_time_ns = -np.inf
        
        self.__tf_msg.header.frame_id = self.world_frame
        self.__tf_msg.child_frame_id = self.vicon_frame

        self.last_recvd_wrench = WrenchStamped()

        self.wrench_sub = None
        self.currents_sub = None

        if self.use_wrench:
            self.wrench_sub = rospy.Subscriber(self.rigid_body.com_wrench_topic, WrenchStamped, self.wrench_callback, queue_size=1)
        else:
            self.currents_sub = rospy.Subscriber("/tnb_mns_driver/des_currents_reg/delayed_sim", DesCurrentsReg, self.currents_callback, queue_size=1)

        self.last_sim_time = rospy.Time.now().to_sec()
        self.I_bf = self.rigid_body.mass_properties.I_bf
        ### Uncomment the lines below to use the principal moments of inertia instead.
        # com_inertia = self.rigid_body.mass_properties.com_inertia_properties
        # self.I_bf = np.diag([com_inertia.Px, com_inertia.Py, com_inertia.Pz])
        self.m = self.rigid_body.mass_properties.m
        self.I_bf_inv = np.linalg.inv(self.I_bf)

        if gravity_on:
            self.F_amb = np.array([0, 0, -self.m*common.Constants.g])
        else:
            self.F_amb = np.array([0, 0, 0])
    
    def simulation_loop(self, event):

        # This simulation loop will assume ZOH, therefore the last command is just kept on being repeated
        # we won't explicitly set it to zero here. The forces are assumed to be in the world frame while
        # the torques are assumed to be in the body fixed frame.
        t_comp_start_ns = rospy.Time.now().to_nsec()
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
        if self.print_ft:
            rospy.loginfo(f"Applying F: {F}, Tau: {Tau}")
        F = F + self.F_amb # Adding gravity and other constant forces

        self.p, self.v = numerical.integrate_linear_dynamics_constant_force_undamped(self.p, self.v, F, self.m, dt)

        # Numerically integration the orientation through the lie group exponential map of angular velocity.
        # The angular velocity is resolved in the local frame.
        self.R, self.omega = numerical.integrate_R_omega_constant_torque(self.R, self.omega, Tau, self.I_bf, dt)

        self.q = geometry.quaternion_from_rotation_matrix(self.R)

        self.__tf_msg.header.stamp = rospy.Time.now()
        self.__tf_msg.transform.translation = Vector3(*self.p)
        self.__tf_msg.transform.rotation = Quaternion(*self.q / np.linalg.norm(self.q))
        self.__tf_broadcaster.sendTransform(self.__tf_msg)

        if self.publish_status:
            self.sim_status_pub.publish(Bool(True))

        t_comp_end_ns = rospy.Time.now().to_nsec()
        if (rospy.Time.now().to_nsec() + t_comp_end_ns - t_comp_start_ns - self.__last_vicon_pub_time_ns) >= self.vicon_pub_time_ns:
            self.__last_vicon_pub_time_ns = rospy.Time.now().to_nsec()
            self.vicon_pub.publish(self.__tf_msg)

    def wrench_callback(self, com_wrench: WrenchStamped):
        if not self.__first_command:
            self.__first_command = True
        self.last_recvd_wrench = com_wrench
        self.__last_command_recv_time = rospy.Time.now().to_sec()
        if self.__last_command_warning_sent:
            rospy.loginfo("Command received again.")
            self.__last_command_warning_sent = False
    
    def currents_callback(self, des_currents_msg: DesCurrentsReg):
        # The idea is to use the latest available pose and the forward model
        # to calculate the actual wrench at dipole center.
        wrench = WrenchStamped()
        dipole_quat, dipole_pos = geometry.numpy_arrays_from_tf_msg(self.__tf_msg.transform)
        Mq = geometry.magnetic_interaction_matrix_from_quaternion(dipole_quat,
                                                                  dipole_strength=self.rigid_body.dipole_list[0].strength,
                                                                  full_mat=True,
                                                                  torque_first=True,
                                                                  dipole_axis=self.rigid_body.dipole_list[0].axis)
        currents = np.asarray(des_currents_msg.des_currents_reg)
        field_grad = self.calibration.get_exact_field_grad5_from_currents(dipole_pos, currents)
        actual_Tau_force = (Mq @ field_grad).flatten() # This will be in the world frame.
        
        wrench.header.stamp = rospy.Time.now()
        wrench.wrench.torque = Vector3(*actual_Tau_force[:3])
        wrench.wrench.force = Vector3(*actual_Tau_force[3:])

        if not self.__first_command:
            self.__first_command = True
        self.last_recvd_wrench = wrench
        self.__last_command_recv_time = rospy.Time.now().to_sec()
        if self.__last_command_warning_sent:
            rospy.loginfo("Command received again.")
            self.__last_command_warning_sent = False

    def run(self):
        self.simulation_timer = rospy.Timer(rospy.Duration(self.Ts), self.simulation_loop)
        rospy.spin()


if __name__ == "__main__":
    dynamics_simulator = DynamicsSimulator()
    dynamics_simulator.run()