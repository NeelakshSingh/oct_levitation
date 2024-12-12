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

HARDWARE_CONNECTED = False
# Controller Design
F_control = 100 # Hz

class ComponentWiseDiagnosis:

    def __init__(self):
        rospy.init_node("oct_lev_component_wise_diagnosis", anonymous=True)
        self.dipole_object = mechanical.NarrowRingMagnetSymmetricXFrameS1()

        # For diagnosing transforms and state estimation
        self.vicon_frame = self.dipole_object.tracking_data.pose_frame
        self.pose_diff = numerical.FirstOrderDifferentiator(alpha=0.3)

        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)

        rospy.sleep(0.1)
        # Controller routine
        self.controller_timer = rospy.Timer(rospy.Duration(1/F_control), self.control_loop)
    
    def control_loop(self, __event):
        # Diagnosing pose, transforms, euler angles, and state rate calculation.
        dipole_tf: TransformStamped = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())
        dipole_quaternion = np.array([dipole_tf.transform.rotation.x,
                                        dipole_tf.transform.rotation.y,
                                        dipole_tf.transform.rotation.z,
                                        dipole_tf.transform.rotation.w])
        dipole_exyz = geometry.euler_xyz_from_quaternion(dipole_quaternion)

        X = np.array([dipole_tf.transform.translation.x,
                      dipole_tf.transform.translation.y,
                      dipole_tf.transform.translation.z,
                      dipole_exyz[0],
                      dipole_exyz[1],
                      dipole_exyz[2]])
        X_euler_dot = self.pose_diff(X, 1/F_control)
        euler_rates = X_euler_dot[3:]

        angular_velocities = geometry.euler_xyz_rate_to_local_angular_velocity_map_matrix(dipole_exyz) @ euler_rates

        X_dot = np.concatenate((X_euler_dot[:3], angular_velocities))

        # rospy.loginfo(f"X: {X} \n X_dot: {X_dot}")

        # Dipole Normal Vector Checks
        R_VM = geometry.rotation_matrix_from_quaternion(dipole_quaternion)
        dipole_normal_vec = R_VM.dot(self.dipole_object.dipole_axis)
        # rospy.loginfo(f"Dipole normal vector: {dipole_normal_vec}")
        
        ## CHECKED AND VERIFIED UNTIL THIS POINT

        # Checking for suspected.


if __name__=="__main__":
    _class = ComponentWiseDiagnosis()
    rospy.spin()