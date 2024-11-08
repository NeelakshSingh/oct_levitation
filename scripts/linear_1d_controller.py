import numpy as np
import rospy
import tf2_ros

from time import perf_counter
from copy import deepcopy

import oct_levitation.common as common
from oct_levitation.controllers import PID1D
from oct_levitation.filters import SingleChannelLiveFilter, MultiChannelLiveFilter

from control_utils.msg import VectorStamped
from geometry_msgs.msg import TransformStamped
from control_utils.general.utilities_jecb import init_hardware_and_shutdown_handler

HARDWARE_CONNECTED = False
KP = 1.0
KI = 0.0
KD = 1.00513681 # from LQR
INTEGRATOR_WINDUP_LIMIT = 100
CLEGG_INTEGRATOR = False
MAGNET_TYPE = "small_ring" # options: "wide_ring", "small_ring"
VICON_FRAME_TYPE = "carbon_fiber_3_arm_1"
MAGNET_STACK_SIZE = 1
DIPOLE_STRENGTH_DICT = {"wide_ring": 0.1, "small_ring": common.NarrowRingMagnet.dps} # per stack unit [si]
DIPOLE_AXIS = np.array([0, 0, 1])
TOL = 1e-3

CURRENT_MAX = 3.0 # [A]

class Linear1DPositionPIDController:

    def __init__(self):
        self.model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_5point.yaml")
        rospy.init_node('linear_1d_controller', anonymous=True)
        self.current_pub = rospy.Publisher("/orig_currents", VectorStamped, queue_size=10)
        self.current_msg = VectorStamped()
        self.Ts = 1/200
        rospy.logwarn("HARDWARE_CONNECTED is set to {}".format(HARDWARE_CONNECTED))
        init_hardware_and_shutdown_handler(HARDWARE_CONNECTED)  
        self.vicon_frame = f"vicon/{MAGNET_TYPE}_S{MAGNET_STACK_SIZE}/Origin"
        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)
        self.home_z = 0.02 # OctoMag origin
        self.dipole_strength = DIPOLE_STRENGTH_DICT[MAGNET_TYPE] * MAGNET_STACK_SIZE
        self.dipole_axis = DIPOLE_AXIS
        
        self.dipole_mass = common.NarrowRingMagnet.m * MAGNET_STACK_SIZE + \
                           common.NarrowRingMagnet.mframe
        
        rospy.sleep(0.1) # wait for the tf listener to get the first transform


        self.d_filter = SingleChannelLiveFilter(
            N = 3, Wn = 45,
            btype='low',
            analog=False,
            ftype='butter',
            fs = (1/self.Ts)
        )
        self.e_filter = deepcopy(self.d_filter)
        self.currents_filter = MultiChannelLiveFilter(
            channels=8,
            N = 3, 
            Wn = 60,
            btype='low',
            analog=False,
            ftype='butter',
            fs = (1/self.Ts)
        )
        self.pid = PID1D(KP, KI, KD, INTEGRATOR_WINDUP_LIMIT, CLEGG_INTEGRATOR)
        self.last_time = rospy.Time.now().to_sec()
        self.__first_time = True
        self.current_timer = rospy.Timer(rospy.Duration(self.Ts), self.current_callback)

    def current_callback(self, event):
        if self.__first_time:
            self.__first_time = False
            self.last_time = rospy.Time.now().to_sec()
        dt = rospy.Time.now().to_sec() - self.last_time 
        self.last_time = rospy.Time.now().to_sec()
        # This is where the control loop runs.
        dipole_tf = self.__tf_buffer.lookup_transform(self.vicon_frame, "vicon/world", rospy.Time())
        dipole_position = np.array([dipole_tf.transform.translation.x,
                                    dipole_tf.transform.translation.y,
                                    dipole_tf.transform.translation.z])
        fz = self.pid.update(self.home_z, dipole_position[2], dt)
        fz += common.Constants.g * self.dipole_mass # compensate for gravity
        M = common.get_magnetic_interaction_matrix(dipole_tf, 
                                                   self.dipole_strength,
                                                   self.dipole_axis)
        A = self.model.get_actuation_matrix(dipole_position)
        alloc_mat = np.linalg.pinv(M @ A)
        desired_wrench = np.array([[0, 0, 0, 0, 0, fz]]).T
        desired_currents = alloc_mat @ desired_wrench
        # desired_currents = self.currents_filter(desired_currents)
        desired_currents = np.clip(desired_currents, -CURRENT_MAX, CURRENT_MAX)
        self.current_msg.vector = desired_currents.flatten().tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)


if __name__ == "__main__":
    controller = Linear1DPositionPIDController()
    rospy.spin()