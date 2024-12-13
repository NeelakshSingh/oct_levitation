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
from control_utils.general.utilities import angles_from_normal_vector, quaternion_to_normal_vector
import oct_levitation.mechanical as mechanical
import oct_levitation.numerical as numerical

HARDWARE_CONNECTED = True
# Controller Design
F_control = 100 # Hz

class TorqueControl:

    def __init__(self):
        rospy.init_node("oct_lev_gravity_compensation", anonymous=True)
        
        # OctoMag calibration and dipole properties
        self.calibration_model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_5point.yaml")
        self.dipole_object = mechanical.NarrowRingMagnetSymmDiscD50T5FrameS3()
        Ix = self.dipole_object.mass_properties.I_bf[0,0]
        Iy = self.dipole_object.mass_properties.I_bf[1,1]

        A_roll = np.zeros((2,2))
        A_roll[0, 1] = 1.0
        A_roll[1, 1] = 1e-5 # Very small damping term
        C_roll = np.array([[1, 0]])
        C_pitch = C_roll
        A_pitch = A_roll
        B_roll = np.array([[0, 1/Ix]]).T
        B_pitch = np.array([[0, 1/Iy]]).T
        Q = np.eye(2)
        R = 1
        Ki = np.array([[0.00]])

        self.roll_lqri = controllers.IntegralSeriesLQR(A_roll, B_roll, C_roll, Q, R, Ki, dt=1/F_control)
        self.pitch_lqri = controllers.IntegralSeriesLQR(A_pitch, B_pitch, C_pitch, Q, R, Ki, dt=1/F_control)
        
        # For transforms and state estimation
        self.vicon_frame = self.dipole_object.tracking_data.pose_frame
        self.roll_diff = numerical.FirstOrderDifferentiator(alpha=0.1)
        self.pitch_diff = numerical.FirstOrderDifferentiator(alpha=0.1)

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
        # POSE TRANSFORMS
        dipole_tf: TransformStamped = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())
        dipole_position = np.array([dipole_tf.transform.translation.x,
                                    dipole_tf.transform.translation.y,
                                    dipole_tf.transform.translation.z])
        dipole_quaternion = np.array([dipole_tf.transform.rotation.x,
                                      dipole_tf.transform.rotation.y,
                                      dipole_tf.transform.rotation.z,
                                      dipole_tf.transform.rotation.w])
        dipole_euler = geometry.euler_xyz_from_quaternion(dipole_quaternion)
        angle_y, angle_x = angles_from_normal_vector(
            quaternion_to_normal_vector(dipole_quaternion)
        )

        # CONTROLLERS
        roll_error = geometry.angle_residual(0, angle_x)
        pitch_error = geometry.angle_residual(0, angle_y)
        roll_e_dot = self.roll_diff(roll_error, 1/F_control)
        pitch_e_dot = self.pitch_diff(pitch_error, 1/F_control)

        Tau_x = self.roll_lqri.update_e(np.array([[roll_error, roll_e_dot]]).T, 1/F_control)[0,0]
        Tau_y = self.pitch_lqri.update_e(np.array([[pitch_error, pitch_e_dot]]).T, 1/F_control)[0,0]

        # desired_wrench = np.array([0.0, 0.0, 0.0, Tau_x, Tau_y, 0.0]) # Small angle assumption
        rospy.loginfo(f"roll_error: {roll_error}, pitch_error: {pitch_error}")
        rospy.loginfo(f"Tau_x: {Tau_x}, Tau_y: {Tau_y}")
        desired_wrench = np.array([0.0, 0.0, 0.0, -Tau_x, -Tau_y, 0.0]) # Small angle assumption

        # GRAVITY COMPENSATION
        gravity_compensation_force = -self.dipole_object.get_gravitational_force()

        # CURRENT ALLOCATION FROM FORCES AND TORQUES
        M = geometry.get_magnetic_interaction_matrix(dipole_tf, self.dipole_object.dipole_strength, full_mat=True, torque_first=False)

        # Trying with a constant actuation matrix at the origin.
        A = self.calibration_model.get_actuation_matrix(np.zeros(3))

        currents = np.linalg.pinv(M @ A).dot(desired_wrench) # Allocation Verified in System Component Analysis

        self.current_msg.des_currents_reg = currents.tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)

if __name__=="__main__":
    _instance = TorqueControl()
    rospy.spin()
