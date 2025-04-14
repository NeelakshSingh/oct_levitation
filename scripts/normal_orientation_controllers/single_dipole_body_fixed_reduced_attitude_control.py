import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry
import oct_levitation.filters as filters

from scipy.linalg import block_diag
from oct_levitation.control_node import ControlSessionNodeBase
from control_utils.msg import VectorStamped
from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3, Quaternion
from tnb_mns_driver.msg import DesCurrentsReg

class SingleDipoleNormalOrientationController(ControlSessionNodeBase):

    def post_init(self):
        self.HARDWARE_CONNECTED = False
        self.tfsub_callback_style_control_loop = True
        self.control_rate = 100 # Set it to the vicon frequency
        self.dt = 1/self.control_rate
        self.publish_desired_com_wrenches = True
        self.publish_desired_dipole_wrenches = False
        
        self.control_input_publisher = rospy.Publisher("/com_single_dipole_normal_orientation_control/control_input",
                                                       VectorStamped, queue_size=1)
        
        # Extra publishers which I wrote only in the post init and are not mandatory end with the shorthand pub.
        self.error_state_pub = rospy.Publisher("/com_single_dipole_normal_orientation_control/error_states",
                                                         VectorStamped, queue_size=1)
        
        self.ref_actual_pub = rospy.Publisher("/com_single_dipole_normal_orientation_control/ref_actual_values",
                                                         VectorStamped, queue_size=1)
        
        self.control_gain_publisher = rospy.Publisher("/com_single_dipole_normal_orientation_control/control_gains",
                                                      VectorStamped, queue_size=1, latch=True)
        
        self.publish_jma_condition = True
        self.south_pole_up = True
        self.warn_jma_condition = True

        if self.publish_jma_condition:
            self.jma_condition_pub = rospy.Publisher("/com_single_dipole_normal_orientation_control/jma_condition",
                                                     VectorStamped, queue_size=1)
            
        self.Iavg = 0.5*(self.rigid_body_dipole.mass_properties.I_bf[0,0] + self.rigid_body_dipole.mass_properties.I_bf[1,1])

        ###########################################
        ### Linerization and DLQR based design
        
        ## Continuous time state space model for no spin.
        A = np.array([
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        C = np.eye(4)
        D = 0


        nx_max = 1
        ny_max = 1
        omega_x_max = np.pi*2 # rad/sec
        omega_y_max = omega_x_max

        tau_x_max = 1
        tau_y_max = 1

        ## Normalizing the state space model.
        Tzx = np.diag([nx_max, ny_max, omega_x_max, omega_y_max])
        Tzu = np.diag([tau_x_max, tau_y_max])
        Tzy = Tzx

        Az_norm = np.linalg.inv(Tzx) @ A @ Tzx
        Bz_norm = np.linalg.inv(Tzx) @ B @ Tzu
        Cz_norm = np.linalg.inv(Tzy) @ C @ Tzx

        # Az_norm = A
        # Bz_norm = B
        # Cz_norm = C

        Az_d_norm, Bz_d_norm, Cz_d_norm, Dz_d_norm, dt = signal.cont2discrete((Az_norm, Bz_norm, Cz_norm, 0), dt=self.dt,
                                                  method='zoh')
        
        Qz = np.diag([10.0, 100.0, 1.0, 1.0])
        Rz = np.diag([1.0, 1.0])
        Kz_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qz, Rz)

        # Denormalize the control gains.
        self.K_orientation = np.asarray(Tzu @ Kz_norm @ np.linalg.inv(Tzx))

        ### Linerization and DLQR based design
        ###########################################

        rospy.loginfo(f"Control gains K: {self.K_orientation}")

        self.control_gains_message = VectorStamped()
        self.control_gains_message.header.stamp = rospy.Time.now()

        ## Using tustin's method to calculate a filtered derivative in discrete time.
        # The filter is a first order low pass filter.
        # f_filter = 40
        # Tf = 1/(2*np.pi*f_filter)
        # Tf_phi = Tf
        # Tf = 0.00039996 # From PDF MATLAB PID Tuner
        # Tf_phi = 0.002319
        # self.control_gains_message.vector = np.concatenate((self.K_phi.flatten(), self.K_theta.flatten(), np.array([Tf, Tf_phi])))
        # self.diff_alpha = 2*self.control_rate/(2*self.control_rate*Tf + 1)
        # self.diff_beta = (2*self.control_rate*Tf - 1)/(2*self.control_rate*Tf + 1)

        self.diff_alpha = self.control_rate
        self.diff_beta = 0
        self.control_gains_message.vector = self.K_orientation.flatten()
        self.last_R = np.eye(3)
        self.R_dot = np.zeros((3,3))

        self.__first_reading = True
        self.metadata_msg.data = f"""
        Experiment metadata.
        Experiment type: Reduced attitude stabilization experiment with a single dipole.
        Dipole: {self.rigid_body_dipole.name}
        Calibration file: {self.calibration_file}
        Hardware Connected: {self.HARDWARE_CONNECTED}
        Gains: K: {self.K_orientation}
        Calibration type: Legacy yaml file
        """

    def simplified_Tauxy_allocation(self, tf_msg: TransformStamped, Tau_x: float, Tau_y: float) -> np.ndarray:
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg.transform)
        dipole_position = geometry.numpy_translation_from_tf_msg(tf_msg.transform)
        s_d = self.rigid_body_dipole.dipole_list[0].strength
        dipole_moment = s_d*geometry.get_normal_vector_from_quaternion(dipole_quaternion)
        if self.south_pole_up:
            dipole_moment = -dipole_moment
        M_tau = geometry.magnetic_interaction_field_to_torque(dipole_moment)
        # M_tau = geometry.magnetic_interaction_field_to_torque(s_d*np.array([0.0, 0.0, -1.0]))
        # Rejecting the singular row since we are almost always nearly upright.
        M_tau = M_tau[:2]
        M_f = geometry.magnetic_interaction_grad5_to_force(dipole_moment)
        # A_field = self.mpem_model.getActuationMatrix(np.zeros(3))[:3]
        A = self.mpem_model.getActuationMatrix(dipole_position)
        # A = self.mpem_model.getActuationMatrix(np.zeros(3))
        M = block_diag(M_tau, M_f)

        Tau_des = np.array([[Tau_x, Tau_y]]).T
        F_des = np.zeros((3,1))

        W_des = np.vstack((Tau_des, F_des))

        JMA = M @ A
        des_currents = np.linalg.pinv(JMA) @ W_des

        jma_condition = np.linalg.cond(JMA)

        if self.warn_jma_condition:
            condition_check_tol = 9e3
            if jma_condition > condition_check_tol:
                np.set_printoptions(linewidth=np.inf)
                rospy.logwarn_once(f"""JMA condition number is too high: {jma_condition}, Current TF: {tf_msg}
                                    \n JMA pinv: \n {np.linalg.pinv(JMA)}
                                    \n JMA: \n {JMA}""")
                rospy.loginfo_once("[Condition Debug] Trying to pinpoint the source of rank loss.")

                rospy.loginfo_once(f"""[Condition Debug] M rank: {np.linalg.matrix_rank(M)},
                                    M: {M},
                                    M condition number: {np.linalg.cond(M)}""")
                
                rospy.loginfo_once(f"""[Condition Debug] A rank: {np.linalg.matrix_rank(A)},
                                    A: {A},
                                    A condition number: {np.linalg.cond(A)}""")

        if self.publish_jma_condition:
            jma_condition_msg = VectorStamped()
            jma_condition_msg.header.stamp = rospy.Time.now()
            # jma_condition = np.linalg.cond(M_tau @ A_field)
            jma_condition_msg.vector = [jma_condition]
            self.jma_condition_pub.publish(jma_condition_msg)

        return des_currents.flatten()
    
    def local_frame_torque_allocation(self, tf_msg: TransformStamped, Tau_x: float, Tau_y: float):
        dipole_quaternion, dipole_position = geometry.numpy_arrays_from_tf_msg(tf_msg.transform)
        dipole = self.rigid_body_dipole.dipole_list[0]
        dipole_moment_local = dipole.strength * dipole.axis
        Mtau_local = geometry.magnetic_interaction_field_to_torque(dipole_moment_local)[:2]
        Tau_des_local = np.array([[Tau_x, Tau_y]]).T
        b_des_local = np.linalg.pinv(Mtau_local) @ Tau_des_local

        # Now let's allocate for forces.
        # Since we don't want any forces we really just want zero gradients.
        g_des = np.zeros((5, 1))
        b_des = geometry.rotate_vector_from_quaternion(dipole_quaternion, b_des_local)
        B_des = np.vstack((b_des.reshape(-1,1), g_des))

        A = self.mpem_model.getActuationMatrix(dipole_position)

        des_currents = (np.linalg.pinv(A) @ B_des).flatten()

        JMA = A
        jma_condition = np.linalg.cond(JMA)

        if self.warn_jma_condition:
            condition_check_tol = 9e3
            if jma_condition > condition_check_tol:
                np.set_printoptions(linewidth=np.inf)
                rospy.logwarn_once(f"""JMA condition number is too high: {jma_condition}, Current TF: {tf_msg}
                                    \n JMA pinv: \n {np.linalg.pinv(JMA)}
                                    \n JMA: \n {JMA}""")
                rospy.loginfo_once("[Condition Debug] Trying to pinpoint the source of rank loss.")
                
                rospy.loginfo_once(f"""[Condition Debug] A rank: {np.linalg.matrix_rank(A)},
                                    A: {A},
                                    A condition number: {np.linalg.cond(A)}""")

        if self.publish_jma_condition:
            jma_condition_msg = VectorStamped()
            jma_condition_msg.header.stamp = rospy.Time.now()
            # jma_condition = np.linalg.cond(M_tau @ A_field)
            jma_condition_msg.vector = [jma_condition]
            self.jma_condition_pub.publish(jma_condition_msg)

        return des_currents
    
    def full_local_torque_inertial_force_allocation(self, tf_msg: TransformStamped, Tau_x: float, Tau_y:float):
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg.transform)
        dipole_position = geometry.numpy_translation_from_tf_msg(tf_msg.transform)
        dipole = self.rigid_body_dipole.dipole_list[0]
        dipole_vector = dipole.strength*geometry.rotate_vector_from_quaternion(dipole_quaternion, dipole.axis)
        Mf = geometry.magnetic_interaction_grad5_to_force(dipole_vector)
        Mt_local = geometry.magnetic_interaction_field_to_local_torque(dipole.strength,
                                                                       dipole.axis,
                                                                       dipole_quaternion)[:2] # Only first two rows will be nonzero
        A = self.mpem_model.getActuationMatrix(dipole_position)
        M = block_diag(Mt_local, Mf)

        W_des = np.array([Tau_x, Tau_y, 0.0, 0.0, 0.0])

        JMA = M @ A
        des_currents = np.linalg.pinv(JMA) @ W_des

        jma_condition = np.linalg.cond(JMA)

        if self.warn_jma_condition:
            condition_check_tol = 300
            if jma_condition > condition_check_tol:
                np.set_printoptions(linewidth=np.inf)
                rospy.logwarn_once(f"""JMA condition number is too high: {jma_condition}, CHECK_TOL: {condition_check_tol} 
                                       Current TF: {tf_msg}
                                    \n JMA pinv: \n {np.linalg.pinv(JMA)}
                                    \n JMA: \n {JMA}""")
                rospy.loginfo_once("[Condition Debug] Trying to pinpoint the source of rank loss.")

                rospy.loginfo_once(f"""[Condition Debug] M rank: {np.linalg.matrix_rank(M)},
                                    M: {M},
                                    M condition number: {np.linalg.cond(M)}""")
                
                rospy.loginfo_once(f"""[Condition Debug] A rank: {np.linalg.matrix_rank(A)},
                                    A: {A},
                                    A condition number: {np.linalg.cond(A)}""")

        if self.publish_jma_condition:
            jma_condition_msg = VectorStamped()
            jma_condition_msg.header.stamp = rospy.Time.now()
            jma_condition_msg.vector = [jma_condition]
            self.jma_condition_pub.publish(jma_condition_msg)

        return des_currents.flatten()

    def callback_control_logic(self, tf_msg: TransformStamped):
        self.desired_currents_msg = DesCurrentsReg() # Empty message
        self.control_input_message = VectorStamped() # Empty message
        self.com_wrench_msg = WrenchStamped() # Empty message
        self.error_state_msg = VectorStamped() # Empty state estimate message
        self.ref_actual_msg = VectorStamped() # Empty message with reference and actual rp values

        self.desired_currents_msg.header.stamp = rospy.Time.now()
        self.com_wrench_msg.header.stamp = rospy.Time.now()
        self.control_input_message.header.stamp = rospy.Time.now()
        self.error_state_msg.header.stamp = rospy.Time.now()
        self.ref_actual_msg.header.stamp = rospy.Time.now()

        # Getting the desired COM wrench.
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg.transform)
        R = geometry.rotation_matrix_from_quaternion(dipole_quaternion)
        e_xyz = geometry.euler_xyz_from_quaternion(dipole_quaternion)

        ### Reference for tracking
        desired_quaternion = geometry.numpy_quaternion_from_tf_msg(self.last_reference_tf_msg.transform)
        ref_e_xyz = geometry.euler_xyz_from_quaternion(desired_quaternion)

        ### Publishing the reference and actual values.
        phi = e_xyz[0]
        theta = e_xyz[1]
        phi_ref = ref_e_xyz[0]
        theta_ref = ref_e_xyz[1]
        self.ref_actual_msg.vector = np.rad2deg([phi, phi_ref, theta, theta_ref])
        self.ref_actual_pub.publish(self.ref_actual_msg)

        ## Calculating the reference reduced attitude.
        Gamma = R.T @ np.array([0, 0, 1]).T
        Gamma_x = Gamma[0]
        Gamma_y = Gamma[1]

        if self.__first_reading:
            self.last_R = R
            self.__first_reading = False

        self.R_dot = self.diff_alpha*(R - self.last_R) + self.diff_beta*self.R_dot
        self.last_R = R

        omega = geometry.angular_velocity_body_frame_from_rotation_matrix(R, self.R_dot)
        omega_x = omega[0]
        omega_y = omega[1]

        state_s = np.array([Gamma_x, Gamma_y, omega_x, omega_y])
        # Conventional reduced attitude stabilization control law
        u = -self.K_orientation @ state_s
        
        # Local frame torque allocation
        Tau_x = u[0]*self.Iavg
        Tau_y = u[1]*self.Iavg

        ## Filtering out spikes through a median filter
        # Tau_des = np.array([Tau_x, Tau_y])
        # Tau_des = self.SpikeFilter(Tau_des)
        # Tau_x = Tau_des[0]
        # Tau_y = Tau_des[1]

        self.error_state_msg.vector = np.concatenate((state_s[:2], np.rad2deg(state_s[2:])))
        self.error_state_pub.publish(self.error_state_msg)
        self.control_input_message.vector = [Tau_x, Tau_y]

        com_wrench_des = np.array([Tau_x, Tau_y, 0.0, 0.0, 0.0, 0.0])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])

        # Performing the simplified allocation for the two torques.
        # des_currents = self.simplified_Tauxy_allocation(tf_msg, Tau_x, Tau_y)

        # Let's try the field local frame allocation which should always yield the correct torque configuration
        des_currents =  self.full_local_torque_inertial_force_allocation(tf_msg, Tau_x, Tau_y)

        self.desired_currents_msg.des_currents_reg = des_currents


if __name__ == "__main__":
    controller = SingleDipoleNormalOrientationController()
    rospy.spin()