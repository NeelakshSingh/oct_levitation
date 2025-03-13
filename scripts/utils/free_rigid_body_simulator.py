import numpy as np
import rospy
import tf2_ros
import os
import tf.transformations as tr

from geometry_msgs.msg import TransformStamped, WrenchStamped, Vector3, Quaternion, Vector3Stamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tnb_mns_driver.msg import DesCurrentsReg
from collections import deque

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
        self.Ts = 0.5e-2
        self.__tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.__tf_msg = TransformStamped()
        self.rigid_body = rigid_bodies.Onyx80x22DiscCenterRingDipole
        self.vicon_frame = self.rigid_body.pose_frame
        self.world_frame = "vicon/world"
        self.vicon_pub = rospy.Publisher(self.vicon_frame, TransformStamped, queue_size=10)

        self.last_commmand_timeout = 0.2
        self.__last_command_recv_time = 0.0
        self.__first_command = False
        self.__last_command_warning_sent = False
        self.__first_sim_step = True
        self.last_time = rospy.Time.now().to_sec()

        ### Calibration model to use nonlinear model.
        self.calibration = common.OctomagCalibratedModel(calibration_type="legacy_yaml",
                                                         calibration_file="mc3ao8s_md200_handp.yaml")

        ### Initial Conditions
        gravity_on = rospy.get_param("~gravity_on", False)
        self.p = np.array(rospy.get_param("~initial_position", [0.0, 0.0, 0.0])) # World frame
        self.v = np.array(rospy.get_param("~initial_velocity", [0.0, 0.0, 0.0])) # World frame
        initial_rpy = np.array(rospy.get_param("~initial_rpy", [15.0, 15.0, 0.0])) # World frame
        self.q = geometry.quaternion_from_euler_xyz(np.deg2rad(initial_rpy)) # World frame
        self.R = geometry.rotation_matrix_from_quaternion(self.q) # Same as above.
        self.omega = np.array(rospy.get_param("~initial_angular_velocity", [0.0, 0.0, 0.0])) # w.r.t world frame resolved in local frame.
        self.omega = np.deg2rad(self.omega)
        self.use_wrench = rospy.get_param("~use_wrench", False)
        self.control_timer_frequency = rospy.get_param("~control_timer_freq", 1000) # in Hz
        self.simulated_control_delay = rospy.get_param("~simulated_control_delay", 0.000) # in seconds
        self.delayed_sample_queue_size = rospy.get_param("~control_delay_queue_size", 100)

        if gravity_on:
            self.F_amb = np.array([0, 0, -common.Constants.g])
        else:
            self.F_amb = np.array([0, 0, 0])
        
        self.__tf_msg.header.frame_id = self.world_frame
        self.__tf_msg.child_frame_id = self.vicon_frame
        self.__tf_msg.transform.rotation = Quaternion(0, 0, 0, 1)

        self.last_recvd_wrench = WrenchStamped()
        self.delayed_currents = DesCurrentsReg()
        self.delayed_wrench = WrenchStamped()
        self.last_recvd_currents = DesCurrentsReg()
        self.__last_input_update_time = 0.0
        self.delayed_currents_queue = deque(maxlen=self.delayed_sample_queue_size)
        self.delayed_wrench_queue = deque(maxlen=self.delayed_sample_queue_size)

        self.wrench_sub = None
        self.currents_sub = None

        if self.use_wrench:
            self.wrench_sub = rospy.Subscriber(self.rigid_body.com_wrench_topic, WrenchStamped, self.wrench_callback, queue_size=1)
        else:
            self.currents_sub = rospy.Subscriber("/tnb_mns_driver/des_currents_reg", DesCurrentsReg, self.currents_callback, queue_size=1)

        self.last_sim_time = rospy.Time.now().to_sec()
        self.I_bf = self.rigid_body.mass_properties.I_bf
        ### Uncomment the lines below to use the principal moments of inertia instead.
        # com_inertia = self.rigid_body.mass_properties.com_inertia_properties
        # self.I_bf = np.diag([com_inertia.Px, com_inertia.Py, com_inertia.Pz])
        self.m = self.rigid_body.mass_properties.m
        self.I_bf_inv = np.linalg.inv(self.I_bf)
    
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
        if self.use_wrench:
            if not self.__first_command:
                self.__first_command = True
                self.__last_input_update_time = rospy.Time.now().to_sec()
                self.delayed_wrench = com_wrench
            else:
                self.delayed_wrench_queue.appendleft(com_wrench)

            self.__last_command_recv_time = rospy.Time.now().to_sec()

            if self.__last_command_warning_sent:
                rospy.loginfo("Command received again.")
                self.__last_command_warning_sent = False
        
    
    def currents_callback(self, des_currents_msg: DesCurrentsReg):
        if not self.use_wrench:
            if not self.__first_command:
                self.__first_command = True
                self.__last_input_update_time = rospy.Time.now().to_sec()
                self.delayed_currents = des_currents_msg
            else:
                self.delayed_currents_queue.appendleft(des_currents_msg)

            self.__last_command_recv_time = rospy.Time.now().to_sec()

            if self.__last_command_warning_sent:
                rospy.loginfo("Command received again.")
                self.__last_command_warning_sent = False

    
    def control_timer_loop(self, event):
        # This is where we will implement delays and allow the control input to properly
        # experience delay application.
        current_time = rospy.Time.now().to_sec()

        wrench = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        if (current_time - self.__last_input_update_time) >= self.simulated_control_delay:
            if not self.use_wrench:
                if len(self.delayed_currents_queue) > 0:
                    self.delayed_currents = self.delayed_currents_queue.pop()
                dipole_quat, dipole_pos = geometry.numpy_arrays_from_tf_msg(self.__tf_msg)
                Mq = geometry.magnetic_interaction_matrix_from_quaternion(dipole_quat,
                                                                        dipole_strength=self.rigid_body.dipole_list[0].strength,
                                                                        full_mat=True,
                                                                        torque_first=True,
                                                                        dipole_axis=self.rigid_body.dipole_list[0].axis)
                field_grad = self.calibration.get_exact_field_grad5_from_currents(dipole_pos, np.asarray(self.delayed_currents.des_currents_reg))
                actual_Tau_force = (Mq @ field_grad).flatten()
                
                wrench.wrench.torque = Vector3(*actual_Tau_force[:3])
                wrench.wrench.force = Vector3(*actual_Tau_force[3:])
            else:
                if len(self.delayed_wrench_queue) > 0:
                    self.delayed_wrench = self.delayed_wrench_queue.pop()
                wrench = self.delayed_wrench
            self.__last_input_update_time = rospy.Time.now().to_sec()

        self.last_recvd_wrench = wrench

    def run(self):
        self.control_timer = rospy.Timer(rospy.Duration(1/self.control_timer_frequency), self.control_timer_loop)
        self.simulation_timer = rospy.Timer(rospy.Duration(self.Ts), self.simulation_loop)
        rospy.spin()


if __name__ == "__main__":
    dynamics_simulator = DynamicsSimulator()
    dynamics_simulator.run()