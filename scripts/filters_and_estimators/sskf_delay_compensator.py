import rospy
import numba
import numpy as np
import numpy.typing as np_t
import oct_levitation.geometry_jit as geometry
import oct_levitation.common as common
import scipy.signal as signal
import control as ct
import os
import sys
import time

from geometry_msgs.msg import TransformStamped, Point, Quaternion, Vector3
from oct_levitation.rigid_bodies import REGISTERED_BODIES
from oct_levitation.msg import RigidBodyStateEstimate
from tnb_mns_driver.msg import TnbMnsStatus
from control_utils.msg import VectorStamped
from scipy.linalg import block_diag
from mag_manip import mag_manip
from threading import Lock

class SSKFDelayCompensator:

    def __init__(self):
        rospy.init_node('sskf_delay_compensator', anonymous=False)
        self.Ts = 1/rospy.get_param('oct_levitation/control_freq')
        self.ALL_PARAMS = rospy.get_param('oct_levitation/delay_compensation_kf') # gets the whole dictionary of parameters
        self.__N_CONNECTED_DRIVERS = int(rospy.get_param('oct_levitation/n_drivers'))
        self.__ACTIVE_COILS = np.asarray(rospy.get_param('oct_levitation/active_coils'), dtype=int)
        self.__ACTIVE_DRIVERS = np.asarray(rospy.get_param('oct_levitation/active_drivers'), dtype=int)
        self.RT_PRIORITY_ENABLED = rospy.get_param('~rtprio_node', False)
        self.mpem_model = mag_manip.ForwardModelMPEM()
        self.calfile_base_path = rospy.get_param("~calfile_base_path", os.path.join(os.environ["HOME"], ".ros/cal"))
        self.calibration_file = rospy.get_param('oct_levitation/calibration_file')
        self.mpem_model.setCalibrationFile(os.path.join(self.calfile_base_path, self.calibration_file))

        self.publish_computation_time = self.ALL_PARAMS['publish_computation_time']
        self.computation_time_pub = rospy.Publisher('oct_levitation/sskf_delay_compensator/computation_time', VectorStamped, queue_size=1)
        self.system_state_vicon_delay_pub = rospy.Publisher('oct_levitation/sskf_delay_compensator/system_state_vicon_delay', VectorStamped, queue_size=1) # System state - vicon time difference publisher
        
        self.N = self.ALL_PARAMS['compensation_steps']
        try:
            self.rigid_body = REGISTERED_BODIES[rospy.get_param('oct_levitation/rigid_body')]
        except KeyError:
            rospy.logerr("Rigid body not found in registered bodies. Please check the parameter.")
            raise

        self.state_estimate_pub = rospy.Publisher(self.rigid_body.pose_frame + '/state_estimate', RigidBodyStateEstimate, queue_size=1)

        # Setting up the augmented dynamics
        mass = self.rigid_body.mass_properties.m
        principal_inertia = self.rigid_body.mass_properties.principal_inertia_properties
        Ixxyy = (principal_inertia.Px + principal_inertia.Py) / 2.0
        Ac = np.block([[np.zeros((5,5)), np.eye(5)],
                      [np.zeros((5, 10))]]) # 5 double integrators
        M_bar = np.diag([Ixxyy, Ixxyy, mass, mass, mass])
        Bc = np.row_stack([np.zeros((5,5)), np.linalg.inv(M_bar)])
        Cc = np.block([[np.eye(5), np.zeros((5,5))]])
        
        # ZOH discretization
        A, B, C, D, dt = signal.cont2discrete((Ac, Bc, Cc, np.zeros((5,5))), self.Ts)
        
        # 4 time steps delayed measurement augmented state space
        A_tilde = np.block([[A, np.zeros((10, self.N*5))],
                            [C, np.zeros((5, self.N*5))],
                            [np.zeros(((self.N-1) * 5, 10)), block_diag(*[np.eye(5) for i in range(self.N-1)]), np.zeros(((self.N-1) * 5, 5))]])
        
        B_tilde = np.row_stack([B, np.zeros((self.N*5, 5))])
        
        C_tilde = np.column_stack([np.zeros((5, self.N*5 + 5)), np.eye(5)])

        unit_dipole_rpxyz_wrench_noise_std = self.ALL_PARAMS['unit_dipole_wrench_noise_std_rpxyz']
        vicon_noise_rpxyz_std = self.ALL_PARAMS['vicon_noise_std_rpxyz']
        
        # Noise matrices for the nominal system and measurements
        Q = np.diag(np.square(unit_dipole_rpxyz_wrench_noise_std))
        V = np.diag(np.square(vicon_noise_rpxyz_std))

        dipole_strength = self.rigid_body.dipole_list[0].strength

        # Augmented state noise covariance matrices
        V_aug_list = [np.zeros((10,10))] + [V for _ in range(self.N)]
        W_tilde = np.square(dipole_strength) * B_tilde @ Q @ B_tilde.T + block_diag(*V_aug_list)
        V_tilde = V
        L, Sigma_tilde_lqe, ALC_eig = ct.dlqe(A_tilde, np.eye(10 + self.N*5), C_tilde, W_tilde, V_tilde)

        ILC = np.eye(10 + self.N*5) - L @ C_tilde
        ILC_A = ILC @ A_tilde
        ILC_B = ILC @ B_tilde

        @numba.njit
        def next_state_estimate(rpxyz_vicon_delayed, s_tilde_prev, tau_f_hat_prev):
            s_next = ILC_A @ s_tilde_prev + ILC_B @ tau_f_hat_prev + L @ rpxyz_vicon_delayed
            return s_next

        y_test_k_4 = np.concatenate((np.deg2rad(np.array([0.01, 0.01])), np.array([1e-4, 1e-4, 1e-3])) )
        next_state_estimate(y_test_k_4, np.concatenate((np.zeros(10 + (self.N-1)*5), y_test_k_4)), np.array([1e-4, 1e-4, 1e-2, 1e-2, 1e-1])) # Forcing compilation

        self.__SS_KF_IMPL = next_state_estimate
        self.Tau_xy_f_amb = np.array([0.0, 0.0, 0.0, 0.0, -mass * common.Constants.g])

        self.local_dipole_moment = self.rigid_body.dipole_list[0].local_dipole_moment

        self.__vicon_meas_lock = Lock() # Used to lock when updating the latest vicon pose
        self.__system_state_lock = Lock() # Used to lock when updating the system state

        # Initializing the estimate variables
        self.__initial_state_acquired = False
        self.__s_tilde : np.ndarray = None
        self.__position_estimate : np.ndarray = None # used for wrench calculation
        self.__latest_vicon_quaternion = None
        self.__quaternion_estimate : np.ndarray = None # used for wrench calculation
        self.__latest_rpxyz_vicon = None
        self.__latest_currents = None
        self.__latest_rpy_vicon = None
        self.warn_on_expected_delay_mismatch = self.ALL_PARAMS['warn_on_expected_delay_mismatch']
        self.delay_msg = VectorStamped()
        self.computation_time_msg = VectorStamped()
        self.estimated_state_msg = RigidBodyStateEstimate()
        self.latest_vicon_pose_time : float = None
        self.latest_system_state_time : float = None

        self.vicon_pose_sub = rospy.Subscriber(self.rigid_body.pose_frame, TransformStamped, self.vicon_pose_callback, queue_size=1)
        self.tnb_mns_status_sub = rospy.Subscriber('tnb_mns_driver/system_state', TnbMnsStatus, self.system_state_callback, queue_size=1)

    def check_shutdown_rt(self):
        """
        This functions is important to ensure a clean exit of the node when it is run with RT priority since it will never yield
        otherwise which will never cause the code to check or respond to shutdown signals and will eventually be SIGKILLed by
        roslaunch. This function really matters in order to call the shutdown hook of the driver and make sure that the ECB's stop
        service is called. Of course, shutting down the tnb_mns_driver should still do the job and any launches of the driver node
        should be left as is and not be modified without knowing exactly what you are doing and what the consequences could be.

        The resetting of the scheduler to SCHED_OTHER was important otherwise it seems like roslaunch will not register the exit
        from the node.
        """
        if self.RT_PRIORITY_ENABLED and rospy.is_shutdown_requested():
            rospy.loginfo("[CONTROLLER CALLBACK LOGIC] ROS shutdown requested")
            rospy.signal_shutdown("ROS shutdown requested")
            os.sched_setscheduler(os.getpid(), os.SCHED_OTHER, os.sched_param(0)) # Revert to default scheduler and give up RT priority to catch the shutdown signal
            os.sched_yield()
            time.sleep(0.1)
            sys.exit(0)
            return

    def vicon_pose_callback(self, body_pose: TransformStamped):
        self.__vicon_meas_lock.acquire()
        self.latest_vicon_pose_time = body_pose.header.stamp.to_sec()
        position = geometry.numpy_translation_from_tf_msg(body_pose)
        quaternion = geometry.numpy_quaternion_from_tf_msg(body_pose)
        self.__latest_rpy_vicon = geometry.euler_xyz_from_quaternion(quaternion)
        self.__latest_rpxyz_vicon = np.zeros(5)
        self.__latest_rpxyz_vicon[:2] = self.__latest_rpy_vicon[:2]
        self.__latest_rpxyz_vicon[2:] = position
        self.__latest_vicon_quaternion = quaternion
        if not self.__initial_state_acquired:
            self.__s_tilde = np.concatenate( [self.__latest_rpxyz_vicon, np.zeros(5)] + [self.__latest_rpxyz_vicon for _ in range(self.N)] )
            self.__position_estimate = position
            self.__quaternion_estimate = quaternion
            self.__initial_state_acquired = True
        self.__vicon_meas_lock.release()
        self.check_shutdown_rt()

        # Maybe run the KF depending on whether all information is available
        self.run_kf_once()

    def system_state_callback(self, system_state_msg: TnbMnsStatus):
        self.__system_state_lock.acquire()
        self.latest_system_state_time = system_state_msg.header.stamp.to_sec()
        currents = np.asarray(system_state_msg.currents_reg)
        self.__latest_currents = np.zeros(8)
        self.__latest_currents[self.__ACTIVE_COILS] = currents[self.__ACTIVE_DRIVERS]
        self.__system_state_lock.release()

        self.check_shutdown_rt()

        # Maybe run the KF depending on whether all information is available
        self.run_kf_once()
    
    def compute_single_dipole_torques_and_forces_from_currents(self, currents: np_t.NDArray[np.float64]) -> np_t.NDArray[np.float64]:
        """
        Computes the dipole torques and forces from the given currents. Assumes a single dipole.
        :param currents: The currents to compute the dipole torques and forces from.
        :return: The computed dipole torques and forces.
        """
        bg_V = self.mpem_model.computeFieldGradient5FromCurrents(self.__position_estimate, currents)
        M = geometry.magnetic_interaction_force_local_torque(self.local_dipole_moment, self.__quaternion_estimate, remove_z_torque=True)
        tauxy_fxyz = M @ bg_V
        return tauxy_fxyz
    
    def run_kf_once(self):
        # Checking the time difference between the vicon pose and the system state
        if self.latest_vicon_pose_time is not None and self.latest_system_state_time is not None:
            comp_start_time = time.perf_counter()

            time_diff = self.latest_system_state_time - self.latest_vicon_pose_time
            if time_diff > self.Ts * self.N or time_diff <= self.Ts * (self.N - 1):
                if self.warn_on_expected_delay_mismatch:
                    rospy.logwarn(f"Time difference between vicon pose and system state is {time_diff:.3f} s. Expected delay: {self.Ts * self.N:.3f} s.")
            self.delay_msg.header.stamp = rospy.Time.now()
            self.delay_msg.vector = [time_diff]
            self.computation_time_msg.header.stamp = rospy.Time.now()

            tau_f_hat_mpem = self.compute_single_dipole_torques_and_forces_from_currents(self.__latest_currents)
            tau_f_hat_mpem = tau_f_hat_mpem + self.Tau_xy_f_amb # V. IMP: Remove constant disturbance and feedforward terms
            self.__s_tilde = self.__SS_KF_IMPL(self.__latest_rpxyz_vicon, self.__s_tilde, tau_f_hat_mpem)

            rpy = np.zeros(3)
            rpy[:2] = self.__s_tilde[:2]
            rpy[2] = self.__latest_rpy_vicon[2] # Keep the yaw angle from the vicon pose
            self.__quaternion_estimate = geometry.quaternion_from_euler_xyz(rpy)
            self.__position_estimate = self.__s_tilde[2:5]
            estimated_angular_velocity = np.zeros(3)
            estimated_angular_velocity[:2] = self.__s_tilde[5:7]
            estimated_velocity = self.__s_tilde[7:10]

            self.estimated_state_msg.header.stamp = rospy.Time.now()
            self.estimated_state_msg.header.frame_id = 'vicon/world'
            self.estimated_state_msg.child_frame_id = self.rigid_body.pose_frame
            self.estimated_state_msg.pose.position = Point(*self.__position_estimate)
            self.estimated_state_msg.pose.orientation = Quaternion(*self.__quaternion_estimate)
            self.estimated_state_msg.twist.linear = Vector3(*estimated_velocity)
            self.estimated_state_msg.twist.angular = Vector3(*estimated_angular_velocity)
            self.estimated_state_msg.eXYZ_rpy = Vector3(*rpy)

            self.state_estimate_pub.publish(self.estimated_state_msg)
            self.latest_vicon_pose_time = None
            self.latest_system_state_time = None
            self.computation_time_msg.vector = [time.perf_counter() - comp_start_time]
            if self.publish_computation_time:
                self.computation_time_pub.publish(self.computation_time_msg)
            self.system_state_vicon_delay_pub.publish(self.delay_msg)