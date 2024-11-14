import numpy as np
import rospy
import tf2_ros
import control as ct
import scipy.signal as signal
import oct_levitation.controllers as controllers

from time import perf_counter
from copy import deepcopy

import oct_levitation.common as common
from oct_levitation.filters import SingleChannelLiveFilter, MultiChannelLiveFilter

from control_utils.msg import VectorStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64
from control_utils.general.utilities_jecb import init_hardware_and_shutdown_handler

HARDWARE_CONNECTED = True
INTEGRATOR_WINDUP_LIMIT = 100
CLEGG_INTEGRATOR = False
MAGNET_TYPE = "small_ring" # options: "wide_ring", "small_ring"
VICON_FRAME_TYPE = "carbon_fiber_3_arm_1"
MAGNET_STACK_SIZE = 1
DIPOLE_STRENGTH_DICT = {"wide_ring": 0.1, "small_ring": common.NarrowRingMagnet.dps} # per stack unit [si]
DIPOLE_AXIS = np.array([0, 0, 1])
TOL = 1e-3

CURRENT_MAX = 3.0 # [A]
CURRENT_POLARITY_FLIPPED = True

# Controller Design
B_friction = 0.0 # friction coefficient
m = common.NarrowRingMagnet.m * MAGNET_STACK_SIZE + common.NarrowRingMagnet.mframe
f_controller = 100 # Hz
T_controller = 1/f_controller

A = np.array([[0, 1], [0, -B_friction/m]])
B = np.array([[0, 1/m]]).T
Q = np.diag([1, 1])
rho = 0.1
R = 1*rho
Qi = np.diag([1, 1]) # Weights for error integral

class Linear1DPositionPIDController:

    def __init__(self):
        self.model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_5point.yaml")
        rospy.init_node('linear_1d_controller', anonymous=True)
        self.current_pub = rospy.Publisher("/orig_currents", VectorStamped, queue_size=10)
        self.current_msg = VectorStamped()
        rospy.logwarn("HARDWARE_CONNECTED is set to {}".format(HARDWARE_CONNECTED))
        init_hardware_and_shutdown_handler(HARDWARE_CONNECTED)  
        self.vicon_frame = f"vicon/{MAGNET_TYPE}_S{MAGNET_STACK_SIZE}/Origin"
        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)
        self.home_z = 0.0 # OctoMag origin
        self.dipole_strength = DIPOLE_STRENGTH_DICT[MAGNET_TYPE] * MAGNET_STACK_SIZE
        self.dipole_axis = DIPOLE_AXIS
        
        self.dipole_mass = m
        
        rospy.sleep(0.1)
        self.d_filter = None
        self.e_filter = None
        self.currents_filter = None

        self.controller = controllers.IntegralLQR(A, B, Q, R, Qi, dt=T_controller)
        self.control_input_pub = rospy.Publisher("/oct_levitation/linear_1d_controller/control_input", Float64, queue_size=10)

        self.last_time = rospy.Time.now().to_sec()
        self.__first_time = True
        self.last_z = 0.0
        self.current_timer = rospy.Timer(rospy.Duration(T_controller), self.current_callback)

    def current_callback(self, event):
        if self.__first_time:
            self.__first_time = False
            dipole_tf = self.__tf_buffer.lookup_transform(self.vicon_frame, "vicon/world", rospy.Time())
            self.last_z = dipole_tf.transform.translation.z
            self.last_time = rospy.Time.now().to_sec()
            return
        # This is where the control loop runs.
        dipole_tf = self.__tf_buffer.lookup_transform(self.vicon_frame, "vicon/world", rospy.Time())
        dipole_position = np.array([dipole_tf.transform.translation.x,
                                    dipole_tf.transform.translation.y,
                                    dipole_tf.transform.translation.z])
        # dt = rospy.Time.now().to_sec() - self.last_time 
        dt = T_controller
        z_dot = (dipole_position[2] - self.last_z) / dt
        self.last_z = dipole_position[2]
        y = np.array([[dipole_position[2], z_dot]]).T
        r = np.array([[self.home_z, 0]]).T
        fz = self.controller.update(r, y, dt)
        fz = fz[0,0]
        fz += common.Constants.g * self.dipole_mass # compensate for gravity
        self.control_input_pub.publish(Float64(data=fz))
        rospy.loginfo("Control Input: {}".format(fz))
        self.last_time = rospy.Time.now().to_sec()
        M = common.get_magnetic_interaction_matrix(dipole_tf, 
                                                   self.dipole_strength,
                                                   self.dipole_axis)
        A = self.model.get_actuation_matrix(dipole_position)
        alloc_mat = np.linalg.pinv(M @ A)
        desired_wrench = np.array([[0, 0, 0, 0, 0, fz]]).T
        desired_currents = alloc_mat @ desired_wrench
        # desired_currents = self.currents_filter(desired_currents)
        desired_currents = np.clip(desired_currents, -CURRENT_MAX, CURRENT_MAX)
        if CURRENT_POLARITY_FLIPPED:
            desired_currents = -desired_currents
        self.current_msg.vector = desired_currents.flatten().tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)


if __name__ == "__main__":
    controller = Linear1DPositionPIDController()
    rospy.spin()