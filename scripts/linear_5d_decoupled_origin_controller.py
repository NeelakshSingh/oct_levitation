import numpy as np
import rospy
import tf2_ros
import control as ct
import scipy.signal as signal


import oct_levitation.common as common
import oct_levitation.controllers as controllers
import oct_levitation.dynamics as dynamics

import oct_levitation.geometry as geometry

from time import perf_counter
from copy import deepcopy

from tnb_mns_driver.msg import DesCurrentsReg
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
from control_utils.general.utilities_jecb import init_hardware_and_shutdown_handler
from control_utils.general.filters import SingleChannelLiveFilter, MultiChannelLiveFilter
from oct_levitation.mechanical import NarrowRingMagnetS1

HARDWARE_CONNECTED = True
INTEGRATOR_WINDUP_LIMIT = 10
CLEGG_INTEGRATOR = True
CURRENT_MAX = 3.0 # A

DIPOLE_BODY = NarrowRingMagnetS1() # Initialize in order to use methods.
g_vector = np.array([0, 0, -common.Constants.g])

# Controller Design
f_controller = 100 # Hz
T_controller = 1/f_controller

A = np.array([[0, 1], [0, 0]])
B_forces = np.array([[0], [1/DIPOLE_BODY.mass_properties.m]])
Q_linear = np.diag([1, 1])
R_linear = np.diag([0.1])

B_phi = np.array([[0], [1/DIPOLE_BODY.mass_properties.I_bf[0, 0]]])
B_theta = np.array([[0], [1/DIPOLE_BODY.mass_properties.I_bf[1, 1]]])
Q_rot = np.diag([1, 1])*1
R_rot = np.diag([0.1]) # Have large penalty on torque magnitude to encourage very small torques.
Rot_windup_limit = 0.5 # rad

C_all = np.array([[1, 0]]) # In all the 5 decoupled controller we only measure the position/angle terms.
Ki_x = np.array([[0.1]])
Ki_y = np.array([[0.1]])
Ki_z = np.array([[1]])
Ki_phi = np.array([[1e-4]])
Ki_theta = np.array([[1e-4]])

class Linear1DPositionPIDController:

    def __init__(self):
        self.model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_77pt_avg_bias_corrected.yaml")
        rospy.init_node('linear_5d_controller', anonymous=True)
        self.current_pub = rospy.Publisher("/tnb_mns_driver/des_currents_reg", DesCurrentsReg, queue_size=10)
        self.current_msg = DesCurrentsReg()
        rospy.logwarn("HARDWARE_CONNECTED is set to {}".format(HARDWARE_CONNECTED))
        init_hardware_and_shutdown_handler(HARDWARE_CONNECTED)
        self.vicon_frame = DIPOLE_BODY.tracking_data.pose_frame
        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)
        self.dipole_strength = DIPOLE_BODY.dipole_strength
        self.dipole_axis = DIPOLE_BODY.dipole_axis
        rospy.sleep(0.1)
        # Using the full rigid body dynamics model linearized around the origin at rest.
        # First we need to get the initial position and orientation of the dipole.
        initial_dipole_tf = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())        
        self.full_state_estimator = controllers.Vicon6DOFEulerXYZStateEstimator(initial_dipole_tf)
        # In this version we only consider the linearized dynamics near the origin and use that for control.
        initial_state = self.full_state_estimator.get_latest_state_estimate()
        self.x_controller = controllers.IntegralSeriesLQR(A, B_forces, C_all, Q_linear, R_linear, Ki_x, dt=T_controller, discretize=True, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=True)
        self.y_controller = controllers.IntegralSeriesLQR(A, B_forces, C_all, Q_linear, R_linear, Ki_y, dt=T_controller, discretize=True, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=True)
        self.z_controller = controllers.IntegralSeriesLQR(A, B_forces, C_all, Q_linear, R_linear, Ki_z, dt=T_controller, discretize=True, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=True)
                            
        self.phi_controller = controllers.IntegralSeriesLQR(A, B_phi, C_all, Q_rot, R_rot, Ki_phi, dt=T_controller, discretize=True, windup_lim=Rot_windup_limit, clegg_integrator=True)
        self.theta_controller = controllers.IntegralSeriesLQR(A, B_theta, C_all, Q_rot, R_rot, Ki_theta, dt=T_controller, discretize=True, windup_lim=Rot_windup_limit, clegg_integrator=True)
        
        rospy.loginfo("Initial state: {}".format(initial_state))
        self.home_z = 0.02 # OctoMag origin
        self.desired_x = np.array([[0, 0]]).T
        self.desired_y = np.array([[0, 0]]).T
        self.desired_z = np.array([[self.home_z, 0]]).T
        self.desired_phi = np.array([[0, 0]]).T
        self.desired_theta = np.array([[0, 0]]).T
        self.control_input_pub = rospy.Publisher("/oct_levitation/linear_1d_controller/control_input", Float64MultiArray, queue_size=10)

        last_x = initial_dipole_tf.transform.translation.x
        last_y = initial_dipole_tf.transform.translation.y
        last_z = initial_dipole_tf.transform.translation.z
        eulers = geometry.euler_xyz_from_quaternion(np.array([initial_dipole_tf.transform.rotation.x,
                                                        initial_dipole_tf.transform.rotation.y,
                                                        initial_dipole_tf.transform.rotation.z,
                                                        initial_dipole_tf.transform.rotation.w]))
        last_phi = eulers[0]
        last_theta = eulers[1]
        self.last_s = np.array([last_x, last_y, last_z, last_phi, last_theta])

        self.last_time = rospy.Time.now().to_sec()
        self.current_timer = rospy.Timer(rospy.Duration(T_controller), self.current_callback)

    def current_callback(self, event):
        # This is where the control loop runs.
        dipole_tf = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())
        dt = T_controller
        # self.full_state_estimator.update(dipole_tf, dt)
        # y = self.full_state_estimator.get_latest_yaw_removed_state_estimate()
        
        # Manually Estimating the states.
        x = dipole_tf.transform.translation.x
        y = dipole_tf.transform.translation.y
        z = dipole_tf.transform.translation.z
        eulers = geometry.euler_xyz_from_quaternion(np.array([dipole_tf.transform.rotation.x,
                                                        dipole_tf.transform.rotation.y,
                                                        dipole_tf.transform.rotation.z,
                                                        dipole_tf.transform.rotation.w]))
        phi = eulers[0]
        theta = eulers[1]
        s = np.array([x, y, z, phi, theta])
        s_dot = (s - self.last_s) / dt
        self.last_s = s

        # For some reason u is a matrix object we need to use ravel to flatten it into an array.
        u = np.zeros(5)
        u[0] = self.x_controller.update(self.desired_x, np.array([[s[0], s_dot[0]]]).T, dt).flatten()
        u[1] = self.y_controller.update(self.desired_y, np.array([[s[1], s_dot[1]]]).T, dt).flatten()
        u[2] = self.z_controller.update(self.desired_z, np.array([[s[2], s_dot[2]]]).T, dt).flatten()
        u[3] = self.phi_controller.update(self.desired_phi, np.array([[s[3], s_dot[3]]]).T, dt).flatten()
        u[4] = self.theta_controller.update(self.desired_theta, np.array([[s[4], s_dot[4]]]).T, dt).flatten()
        self.last_time = rospy.Time.now().to_sec()
        M = common.get_magnetic_interaction_matrix(dipole_tf, 
                                                   self.dipole_strength,
                                                   torque_first=False,
                                                   dipole_axis=self.dipole_axis)
        dipole_position = np.array([dipole_tf.transform.translation.x,
                                    dipole_tf.transform.translation.y,
                                    dipole_tf.transform.translation.z])
        A = self.model.get_actuation_matrix(dipole_position)
        alloc_mat = np.linalg.pinv(M @ A)
        # Now so far the torque has been in the body frame. We need to convert it to the world frame.
        quaternion = np.array([dipole_tf.transform.rotation.x,
                               dipole_tf.transform.rotation.y,
                               dipole_tf.transform.rotation.z,
                               dipole_tf.transform.rotation.w])
        torque = np.array([u[3], u[4], 0]) # Zero Tau_z in body frame
        torque = geometry.rotate_vector_from_quaternion(quaternion, torque)
        desired_wrench = np.concatenate((u[:3], torque)) # desired forces and torques in world frame
        # Performing gravity compensation
        desired_wrench -= DIPOLE_BODY.get_gravitational_wrench(q=quaternion, g=g_vector)
        self.control_input_pub.publish(Float64MultiArray(data=desired_wrench.flatten().tolist()))
        desired_currents = (alloc_mat @ desired_wrench).flatten()
        # desired_currents = self.currents_filter(desired_currents)
        desired_currents = np.clip(desired_currents, -CURRENT_MAX, CURRENT_MAX)
        self.current_msg.des_currents_reg = desired_currents.tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)


if __name__ == "__main__":
    controller = Linear1DPositionPIDController()
    rospy.spin()