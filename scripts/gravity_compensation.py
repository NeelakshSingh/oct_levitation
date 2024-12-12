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
from control_utils.general.filters import SingleChannelLiveFilter
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
from control_utils.general.utilities_jecb import init_hardware_and_shutdown_handler
from control_utils.general.filters import AveragingLowPassFilter
from control_utils.general.utilities import SmoothIterativeUpdate
import oct_levitation.mechanical as mechanical
import oct_levitation.numerical as numerical

HARDWARE_CONNECTED = True
# Controller Design
F_control = 100 # Hz

class GravityCompensation:

    def __init__(self):
        rospy.init_node("oct_lev_gravity_compensation", anonymous=True)
        
        # OctoMag calibration and dipole properties
        self.calibration_model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_5point.yaml")
        self.dipole_object = mechanical.NarrowRingMagnetDisc7mmFrameS1()
        
        # For transforms and state estimation
        self.vicon_frame = self.dipole_object.tracking_data.pose_frame
        self.pose_diff = numerical.FirstOrderDifferentiator(alpha=0.3)

        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)

        # Initializing OctoMag
        rospy.logwarn("HARDWARE_CONNECTED is set to {}".format(HARDWARE_CONNECTED))
        init_hardware_and_shutdown_handler(HARDWARE_CONNECTED)
        self.current_pub = rospy.Publisher("/tnb_mns_driver/des_currents_reg", DesCurrentsReg, queue_size=10)
        self.current_msg = DesCurrentsReg()
        
        # Start controller routine
        rospy.sleep(0.1)
        self.controller_timer = rospy.Timer(rospy.Duration(1/F_control), self.control_loop)
    
    def control_loop(self, event):
        dipole_tf: TransformStamped = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())
        dipole_position = np.array([dipole_tf.transform.translation.x,
                                    dipole_tf.transform.translation.y,
                                    dipole_tf.transform.translation.z])
        gravity_compensation_force = -self.dipole_object.get_gravitational_force()
        desired_wrench = np.concatenate((gravity_compensation_force, np.zeros(3)))
        M = geometry.get_magnetic_interaction_matrix(dipole_tf, self.dipole_object.dipole_strength, full_mat=True, torque_first=False)

        # Trying with a constant actuation matrix at the origin.
        A = self.calibration_model.get_actuation_matrix(np.zeros(3))

        currents = np.linalg.pinv(M @ A).dot(desired_wrench) # Allocation Verified in System Component Analysis

        self.current_msg.des_currents_reg = currents.tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)

if __name__=="__main__":
    _instance = GravityCompensation()
    rospy.spin()
