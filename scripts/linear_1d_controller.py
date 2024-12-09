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
import oct_levitation.mechanical as mechanical

HARDWARE_CONNECTED = True
INTEGRATOR_WINDUP_LIMIT = 100
CLEGG_INTEGRATOR = False
CURRENT_MAX = 10.0 # A

# DIPOLE_BODY = mechanical.NarrowRingMagnetSymmetricSquareS1() # Initialize in order to use methods.
DIPOLE_BODY = mechanical.NarrowRingMagnetS1() # Initialize in order to use methods.
g_vector = np.array([0, 0, -common.Constants.g])

# Controller Design
B_friction = 0.0 # friction coefficient
f_controller = 30 # Hz
T_controller = 1/f_controller


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
        # We will initialize the dynamics and linearize the system around this operating point.
        self.rigid_body_dynamics = dynamics.WrenchInput6DOFDipoleEulerXYZDynamics(m=DIPOLE_BODY.mass_properties.m,
                                                                            I_m=DIPOLE_BODY.mass_properties.I_bf)

        # In this version we only consider the linearized dynamics near the origin and use that for control.
        initial_state = self.full_state_estimator.get_latest_state_estimate()
        rospy.loginfo("Initial state: {}".format(initial_state))
        # u_ss doesn't contain gravity compensation, so we need to add that manually later.
        initial_control_input = np.array([0, 0, 0, 0, 0]) # 3 forces in world frame, 2 torques in body frame
        A_op, B_op = self.rigid_body_dynamics.get_linearized_dynamics(initial_state, initial_control_input)
        # Because of the uncontrollability of the yaw and yaw rate at origin, we will remove them from our state space model.
        A_op, B_op = self.rigid_body_dynamics.remove_yaw_dynamics(A_op, B_op)
        Q_op = 10*np.eye(10)
        R_op = 0.5*np.eye(5)
        C_op = np.block([[np.eye(3), np.zeros((3, 7))],
                         [np.zeros((2, 6)), np.eye(2), np.zeros((2, 2))]])
        Ki = np.eye(5)*0.1
        Qi_op = 1*np.eye(10)
        self.home_z = 0.02 # OctoMag origin
        self.desired_state = np.array([0, 0, self.home_z, 0, 0, 0, 0, 0, 0, 0]) # [x, y, z, vx, vy, vz, phi, theta, wx, wy]
        self.desired_state[6:8] = initial_state[6:8] # We will try to maintain the initial orientation.
        # self.desired_state[:8] = initial_state[:8] # We will try to maintain the initial state
        # self.desired_state[8:] = initial_state[9:11]
        rospy.loginfo("Desired state: {}".format(self.desired_state))
        self.d_filter = None
        self.e_filter = None
        self.currents_filter = MultiChannelLiveFilter(
            channels=8,
            N=10,
            Wn=10,
            rs=100,
            btype="lowpass",
            ftype="cheby2",
            use_sos=True,
            fs=f_controller
        )

        # self.controller = controllers.IntegralLQR(A_op, B_op, Q_op, R_op, Qi_op, dt=T_controller, windup_lim=INTEGRATOR_WINDUP_LIMIT, clegg_integrator=CLEGG_INTEGRATOR)
        # self.controller = controllers.LQR(A_op, B_op, Q_op, R_op, dt=T_controller, discretize=True)
        self.controller = controllers.IntegralSeriesLQR(A_op, B_op, C_op, Q_op, R_op, Ki, dt=T_controller, discretize=True)
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
        u = np.array(self.controller.update(self.desired_state, y, dt)).flatten()
        self.control_input_pub.publish(Float64MultiArray(data=u.flatten().tolist()))
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
        desired_currents = (alloc_mat @ desired_wrench).flatten()
        desired_currents = self.currents_filter(desired_currents)
        desired_currents = np.clip(desired_currents, -CURRENT_MAX, CURRENT_MAX)
        self.current_msg.des_currents_reg = desired_currents.tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)


if __name__ == "__main__":
    controller = Linear1DPositionPIDController()
    rospy.spin()