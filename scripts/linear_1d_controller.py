import numpy as np
import rospy
import tf2_ros
import control as ct
import scipy.signal as signal


import oct_levitation.common as common
import oct_levitation.controllers as controllers
import oct_levitation.dynamics as dynamics

import control_utils.general.geometry as geometry

from time import perf_counter
from copy import deepcopy

from tnb_mns_driver.msg import DesCurrentsReg
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
from control_utils.general.utilities_jecb import init_hardware_and_shutdown_handler
from control_utils.general.filters import SingleChannelLiveFilter, MultiChannelLiveFilter

HARDWARE_CONNECTED = False
INTEGRATOR_WINDUP_LIMIT = 100
CLEGG_INTEGRATOR = False
MAGNET_TYPE = "small_ring" # options: "wide_ring", "small_ring"
VICON_FRAME_TYPE = "carbon_fiber_3_arm_1"
MAGNET_STACK_SIZE = 1
DIPOLE_STRENGTH_DICT = {"wide_ring": 0.1, "small_ring": common.NarrowRingMagnet.dps} # per stack unit [si]
DIPOLE_AXIS = np.array([0, 0, 1])
TOL = 1e-3

CURRENT_MAX = 3.0 # [A]
CURRENT_POLARITY_FLIPPED = False

# Controller Design
B_friction = 0.0 # friction coefficient
m = common.NarrowRingMagnet.m * MAGNET_STACK_SIZE + common.NarrowRingMagnet.mframe
f_controller = 100 # Hz
T_controller = 1/f_controller

# A = np.array([[0, 1], [0, -B_friction/m]])
# B = np.array([[0, 1/m]]).T
# Q = np.diag([1, 1])
# rho = 0.1
# R = 1*rho
# Qi = np.diag([1, 1]) # Weights for error integral

class Linear1DPositionPIDController:

    def __init__(self):
        self.model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_77pt_avg_bias_corrected.yaml")
        rospy.init_node('linear_1d_controller', anonymous=True)
        self.current_pub = rospy.Publisher("/orig_currents", DesCurrentsReg, queue_size=10)
        self.current_msg = DesCurrentsReg()
        rospy.logwarn("HARDWARE_CONNECTED is set to {}".format(HARDWARE_CONNECTED))
        init_hardware_and_shutdown_handler(HARDWARE_CONNECTED)  
        self.vicon_frame = f"vicon/{MAGNET_TYPE}_S{MAGNET_STACK_SIZE}/Origin"
        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)
        self.home_z = 0.02 # OctoMag origin
        self.dipole_strength = DIPOLE_STRENGTH_DICT[MAGNET_TYPE] * MAGNET_STACK_SIZE
        self.dipole_axis = DIPOLE_AXIS
        
        self.dipole_mass = m
 
        rospy.sleep(0.1)
        # Using the full rigid body dynamics model linearized around the origin at rest.
        # First we need to get the initial position and orientation of the dipole.
        initial_dipole_tf = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())        
        self.full_state_estimator = controllers.Vicon6DOFEulerXYZStateEstimator(initial_dipole_tf)
        # We will initialize the dynamics and linearize the system around this operating point.
        self.rigid_body_dynamics = dynamics.WrenchInput6DOFEulerXYZDynamics(d=self.dipole_strength,
                                                                            m=self.dipole_mass,
                                                                            I_m=common.NarrowRingMagnet.inertia_matrix_S1,
                                                                            g=common.Constants.g)

        # In this version we only consider the linearized dynamics near the origin and use that for control.
        initial_state = self.full_state_estimator.get_latest_state_estimate()
        # u_ss doesn't contain gravity compensation, so we need to add that manually later.
        initial_control_input = np.array([0, 0, 0, 0, 0]) # 3 forces in world frame, 2 torques in body frame
        A_op, B_op = self.rigid_body_dynamics.get_linearized_dynamics(initial_state, initial_control_input)
        Q_op = 2*np.eye(12)
        R_op = 0.5*np.eye(5)
        Qi_op = 0.1*np.eye(12)
        self.desired_state = np.array([0, 0, self.home_z, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.d_filter = None
        self.e_filter = None
        self.currents_filter = None

        self.controller = controllers.IntegralLQR(A_op, B_op, Q_op, R_op, Qi_op, dt=T_controller, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=CLEGG_INTEGRATOR)
        self.control_input_pub = rospy.Publisher("/oct_levitation/linear_1d_controller/control_input", Float64MultiArray, queue_size=10)

        self.last_time = rospy.Time.now().to_sec()
        self.current_timer = rospy.Timer(rospy.Duration(T_controller), self.current_callback)

    def current_callback(self, event):
        # This is where the control loop runs.
        dipole_tf = self.__tf_buffer.lookup_transform(self.vicon_frame, "vicon/world", rospy.Time())
        dt = T_controller
        y = self.full_state_estimator.update(dipole_tf, dt)
        u = self.controller.update(self.desired_state, y, dt).flatten()
        u[3] += common.Constants.g * self.dipole_mass # compensate for gravity
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
        torque = u[3:]
        torque = geometry.rotate_vector_from_quaternion(quaternion, torque)
        desired_wrench = np.concatenate((u[:3], torque))
        desired_currents = alloc_mat @ desired_wrench
        # desired_currents = self.currents_filter(desired_currents)
        desired_currents = np.clip(desired_currents, -CURRENT_MAX, CURRENT_MAX)
        if CURRENT_POLARITY_FLIPPED:
            desired_currents = -desired_currents
        self.current_msg.des_currents_reg = desired_currents.flatten().tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)


if __name__ == "__main__":
    controller = Linear1DPositionPIDController()
    rospy.spin()