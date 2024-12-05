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

HARDWARE_CONNECTED = False
INTEGRATOR_WINDUP_LIMIT = 100
CLEGG_INTEGRATOR = False
CURRENT_MAX = 3.0 # A

DIPOLE_BODY = NarrowRingMagnetS1() # Initialize in order to use methods.
g_vector = np.array([0, 0, -common.Constants.g])

# Controller Design
f_controller = 30 # Hz
T_controller = 1/f_controller

A = np.array([[0, 1], [0, 0]])
B_forces = np.array([[0], [1/DIPOLE_BODY.mass_properties.m]])
Q_linear = np.diag([1, 1])*1e-1
R_linear = np.diag([1])*1e-1
Qi_linear = np.diag([1, 1])*1e-1

B_phi = np.array([[0], [1/DIPOLE_BODY.mass_properties.I_bf[0, 0]]])
B_theta = np.array([[0], [1/DIPOLE_BODY.mass_properties.I_bf[1, 1]]])
Q_rot = np.diag([1, 1])*1
Qi_rot = np.diag([1, 1])*1
R_rot = np.diag([1])*1e7 # Have large penalty on torque magnitude to encourage very small torques.
Rot_windup_limit = 0.5 # rad

class Linear1DPositionPIDController:

    def __init__(self):
        self.model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_77pt_corrected_minimag_init.yaml")
        rospy.init_node('linear_1d_controller', anonymous=True)
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
        self.x_controller = controllers.IntegralLQR(A, B_forces, Q_linear, R_linear, Qi_linear, dt=T_controller, discretize=True, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=True)
        self.y_controller = controllers.IntegralLQR(A, B_forces, Q_linear, R_linear, Qi_linear, dt=T_controller, discretize=True, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=True)
        self.z_controller = controllers.IntegralLQR(A, B_forces, Q_linear, R_linear, Qi_linear, dt=T_controller, discretize=True, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=True)
                            
        self.phi_controller = controllers.IntegralLQR(A, B_phi, Q_rot, R_rot, Qi_rot, dt=T_controller, discretize=True, windup_lim=Rot_windup_limit, clegg_integrator=True)
        self.theta_controller = controllers.IntegralLQR(A, B_theta, Q_rot, R_rot, Qi_rot, dt=T_controller, discretize=True, windup_lim=Rot_windup_limit, clegg_integrator=True)
        
        rospy.loginfo("Initial state: {}".format(initial_state))
        self.home_z = 0.02 # OctoMag origin
        self.desired_x = np.array([[0, 0]]).T
        self.desired_y = np.array([[0, 0]]).T
        self.desired_z = np.array([[self.home_z, 0]]).T
        self.desired_phi = np.array([[0, 0]]).T
        self.desired_theta = np.array([[0, 0]]).T
        self.control_input_pub = rospy.Publisher("/oct_levitation/linear_1d_controller/control_input", Float64MultiArray, queue_size=10)

        self.last_time = rospy.Time.now().to_sec()
        self.current_timer = rospy.Timer(rospy.Duration(T_controller), self.current_callback)

    def current_callback(self, event):
        # This is where the control loop runs.
        dipole_tf = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())
        dt = T_controller
        self.full_state_estimator.update(dipole_tf, dt)
        y = self.full_state_estimator.get_latest_yaw_removed_state_estimate()
        # For some reason u is a matrix object we need to use ravel to flatten it into an array.
        u = np.zeros(5)
        u[1] = self.x_controller.update(np.array([[y[0]. y[3]]]).T, self.desired_x, dt).flatten()
        u[2] = self.y_controller.update(np.array([[y[1]. y[4]]]).T, self.desired_y, dt).flatten()
        u[3] = self.z_controller.update(np.array([[y[2]. y[5]]]).T, self.desired_z, dt).flatten()
        u[4] = self.phi_controller.update(np.array([[y[6]. y[9]]]).T, self.desired_phi, dt).flatten()
        u[5] = self.theta_controller.update(np.array([[y[7]. y[10]]]).T, self.desired_theta, dt).flatten()
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
        # torque = geometry.rotate_vector_from_quaternion(quaternion, torque)
        desired_wrench = np.concatenate((u[:3], torque)) # desired forces and torques in world frame
        # Performing gravity compensation
        desired_wrench -= DIPOLE_BODY.get_gravitational_wrench(q=quaternion, g=g_vector)
        desired_currents = (alloc_mat @ desired_wrench).flatten()
        # desired_currents = self.currents_filter(desired_currents)
        desired_currents = np.clip(desired_currents, -CURRENT_MAX, CURRENT_MAX)
        self.current_msg.des_currents_reg = desired_currents.tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)


if __name__ == "__main__":
    controller = Linear1DPositionPIDController()
    rospy.spin()