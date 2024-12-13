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
from oct_levitation.msg import Float64MultiArrayStamped
from control_utils.general.utilities_jecb import init_hardware_and_shutdown_handler
from control_utils.general.utilities import angles_from_normal_vector, quaternion_to_normal_vector
import oct_levitation.mechanical as mechanical
import oct_levitation.numerical as numerical

HARDWARE_CONNECTED = True
# Controller Design
F_control = 300 # Hz

class XYZControl:

    def __init__(self):
        rospy.init_node("oct_lev_gravity_compensation", anonymous=True)
        
        # OctoMag calibration and dipole properties
        self.calibration_model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_5point.yaml")
        self.dipole_object = mechanical.NarrowRingMagnetSymmDiscD50T5FrameS3()
        self.__tf_buffer = tf2_ros.Buffer()
        self.__tf_listener = tf2_ros.TransformListener(self.__tf_buffer)

        A_z = np.zeros((2,2)) # The same double integrator
        A_z[0,1] = 1.0
        A_z[1,1] = 1e-5 # More damping because of the rod
        B_z = np.array([[0.0, 1/self.dipole_object.mass_properties.m]]).T
        Q_z = np.eye(2)*0.5
        R_z = 1

        A_x = A_z.copy()
        A_y = A_z.copy()
        B_x = B_z.copy()
        B_y = B_z.copy()
        Q_x = np.eye(2)*0.5
        R_x = 1
        Q_y = Q_x.copy()
        R_y = R_x

        Azd, Bzd, Czd, Dzd, dzt = signal.cont2discrete((A_z, B_z, np.eye(A_z.shape[0]), 0), dt=1/F_control, method='zoh')
        K_z, S_z, E_z = ct.dlqr(Azd, Bzd, Q_z, R_z)
        Axd, Bxd, Cxd, Dxd, dxt = signal.cont2discrete((A_x, B_x, np.eye(A_x.shape[0]), 0), dt=1/F_control, method='zoh')
        K_x, S_x, E_x = ct.dlqr(Axd, Bxd, Q_x, R_x)
        Ayd, Byd, Cyd, Dyd, dyt = signal.cont2discrete((A_y, B_y, np.eye(A_y.shape[0]), 0), dt=1/F_control, method='zoh')
        K_y, S_y, E_y = ct.dlqr(Ayd, Byd, Q_y, R_y)
        self.kpz = K_z[0,0]
        self.kdz = K_z[0,1]
        self.kiz = 0.8

        self.kpx = K_x[0,0]
        self.kdx = K_x[0,1]
        self.kpy = K_y[0,0]
        self.kdy = K_y[0,1]
        
        # For transforms and state estimation
        self.vicon_frame = self.dipole_object.tracking_data.pose_frame


        # Initializing OctoMag
        rospy.logwarn("HARDWARE_CONNECTED is set to {}".format(HARDWARE_CONNECTED))
        init_hardware_and_shutdown_handler(HARDWARE_CONNECTED)
        self.current_pub = rospy.Publisher("/tnb_mns_driver/des_currents_reg", DesCurrentsReg, queue_size=10)
        self.control_input_pub = rospy.Publisher("/oct_levitation/lqri_xyz/control_input", Float64MultiArrayStamped, queue_size=10)
        self.reference_pose_pub = rospy.Publisher("/oct_levitation/lqri_xyz/reference_pose", TransformStamped, queue_size=10)
        self.current_msg = DesCurrentsReg()
        
        # Start controller routine
        rospy.sleep(0.1)
        dipole_tf: TransformStamped = self.__tf_buffer.lookup_transform("vicon/world", self.vicon_frame, rospy.Time())

        self.home_z = dipole_tf.transform.translation.z
        self.home_x = dipole_tf.transform.translation.x
        self.home_y = dipole_tf.transform.translation.y

        self.desired_pose = TransformStamped()
        self.desired_pose.header.frame_id = "vicon/world"
        self.desired_pose.child_frame_id = self.dipole_object.tracking_data.pose_frame
        self.desired_pose.transform.translation.z = self.home_z
        self.desired_pose.transform.translation.x = self.home_x
        self.desired_pose.transform.translation.y = self.home_y

        self.prev_z_error = self.home_z - dipole_tf.transform.translation.z
        self.prev_x_error = self.home_x - dipole_tf.transform.translation.x
        self.prev_y_error = self.home_y - dipole_tf.transform.translation.y
        self.z_e_integral = 0.0
        self.x_e_integral = 0.0
        self.y_e_integral = 0.0
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
        # CONTROLLERS
        z_error = self.home_z - dipole_position[2]
        self.z_e_integral += z_error/F_control
        z_e_dot = (z_error - self.prev_z_error)*F_control
        self.prev_z_error = z_error
        F_z = self.kpz*z_error + self.kdz*z_e_dot + self.kiz*self.z_e_integral

        x_error = self.home_x - dipole_position[0]
        self.x_e_integral += x_error/F_control
        x_e_dot = (x_error - self.prev_x_error)*F_control
        self.prev_x_error = x_error
        F_x = self.kpx*x_error + self.kdx*x_e_dot

        y_error = self.home_y - dipole_position[1]
        self.y_e_integral += y_error/F_control
        y_e_dot = (y_error - self.prev_y_error)*F_control
        self.prev_y_error = y_error
        F_y = self.kpy*y_error + self.kdy*y_e_dot

        gravity_compensation_force = -self.dipole_object.get_gravitational_force()

        # GRAVITY COMPENSATION
        F_z += gravity_compensation_force[2]

        desired_wrench = np.array([F_x, F_y, F_z, 0.0, 0.0, 0.0]) # Small angle assumption


        # DATA PUBLISHERS
        control_input_msg = Float64MultiArrayStamped()
        control_input_msg.header.stamp = rospy.Time.now()
        control_input_msg.array = desired_wrench.tolist()
        self.control_input_pub.publish(control_input_msg)

        # CURRENT ALLOCATION FROM FORCES AND TORQUES
        M = geometry.get_magnetic_interaction_matrix(dipole_tf, self.dipole_object.dipole_strength, full_mat=True, torque_first=False)

        # Trying with a constant actuation matrix at the origin.
        A = self.calibration_model.get_actuation_matrix(np.zeros(3))

        currents = np.linalg.pinv(M @ A).dot(desired_wrench) # Allocation Verified in System Component Analysis

        self.current_msg.des_currents_reg = currents.tolist()
        self.current_msg.header.stamp = rospy.Time.now()
        self.current_pub.publish(self.current_msg)

if __name__=="__main__":
    _instance = XYZControl()
    rospy.spin()
