import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry
import oct_levitation.common as common
import oct_levitation.numerical as numerical

from scipy.linalg import block_diag
from oct_levitation.control_node import ControlSessionNodeBase
from control_utils.msg import VectorStamped
from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3, Quaternion
from tnb_mns_driver.msg import DesCurrentsReg

class SingleDipoleNormalOrientationController(ControlSessionNodeBase):

    def post_init(self):
        self.HARDWARE_CONNECTED = True
        self.tfsub_callback_style_control_loop = True
        self.control_rate = 100 # Set it to the vicon frequency
        self.rigid_body_dipole = rigid_bodies.Onyx80x22DiscCenterRingDipole
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
        # We just consider the average of the inertia for both x and y dimensions. Of course this will change in reality
        # thanks to the changing orientation since are about to directly compute torques in the inertial frame.
        # For better accuracy we will need to eventually design torque control in the body fixed frame, but for small
        # angles we are fine with such approximations.
        k_rot = 1e-2 # rotational damping term

        A = np.array([[0, 1], [0, -k_rot/self.Iavg]])
        B = np.array([[0, 1/self.Iavg]]).T
        C = np.array([[1, 0]])

        A_d, B_d, C_d, D_d, dt = signal.cont2discrete((A, B, C, 0), dt=1/self.control_rate, method='zoh')

        Q = np.diag([1e-2, 1e-1])
        R = 1
        # self.K, S, E = ct.dlqr(A_d, B_d, Q, R)
        # self.K_theta = np.array([[0.0045055, 0.00066943]]) # Tuned for overdamped PD response.
        # self.K_phi = np.array([[0.0063449, 0.0009356]]) # Tuned to include the external disc
        # self.K_theta = np.array([[0.00829789566492576,0.000924926621820855]]) # Jasan's Gains
        # self.K_phi = np.array([[0.0101054837272838,0.00124597367951398]]) # Jasan's Gains

        # self.K_theta = np.array([[0.00859, 0.0007965]]) # Tuned for overdamped PD response.

        self.K_theta = np.array([[0.009982, 0.0007758]]) # Tuned for overdamped PD response.
        self.K_phi = np.array([[0.02121, 0.001454]]) # Tuned to include the external disc

        # self.K_theta = np.array([[0.0299982, 0.0007758]]) # Tuned for overdamped PD response.
        # self.K_phi = np.array([[0.02121, 0.001454]]) # Tuned to include the external disc

        rospy.loginfo(f"Control gains for Tx: {self.K_phi}, Ty: {self.K_theta}")

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

        # self.diff_alpha_phi = 2*self.control_rate/(2*self.control_rate*Tf_phi + 1)
        # self.diff_beta_phi = (2*self.control_rate*Tf_phi - 1)/(2*self.control_rate*Tf_phi + 1)

        self.diff_alpha = self.control_rate
        self.diff_beta = 0
        self.diff_alpha_phi = self.control_rate
        self.diff_beta_phi = 0
        self.control_gains_message.vector = np.concatenate((self.K_phi.flatten(), self.K_theta.flatten()))

        self.phi_dot = 0.0
        self.theta_dot = 0.0

        self.home_phi = 0.0
        self.home_theta = 0.0

        self.last_phi = 0.0
        self.last_theta = 0.0

        self.__first_reading = True
        self.metadata_msg.data = f"""
        Experiment metadata.
        Experiment type: Regulation experiment for 0 pose with position varying allocation matrix.
        Calibration file: {self.calibration_file}
        Gains: {self.K_phi.flatten(), self.K_theta.flatten()}
        Calibration type: Legacy yaml file
        """

    def simplified_Tauxy_allocation(self, tf_msg: TransformStamped, Tau_x: float, Tau_y: float) -> np.ndarray:
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg)
        dipole_position = geometry.numpy_translation_from_tf_msg(tf_msg)
        s_d = self.rigid_body_dipole.dipole_list[0].strength
        dipole_moment = s_d*geometry.get_normal_vector_from_quaternion(dipole_quaternion)
        if self.south_pole_up:
            dipole_moment = -dipole_moment
        M_tau = geometry.magnetic_interaction_field_to_torque(dipole_moment)
        # Rejecting the singular row since we are almost always nearly upright.
        M_tau = M_tau[:2]
        M_f = geometry.magnetic_interaction_grad5_to_force(dipole_moment)
        # A_field = self.mpem_model.getActuationMatrix(np.zeros(3))[:3]
        A = self.mpem_model.getActuationMatrix(dipole_position)
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
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg)

        e_zyx = geometry.euler_zyx_from_quaternion(dipole_quaternion)
        e_xyz = geometry.euler_xyz_from_quaternion(dipole_quaternion)

        ### Reference for tracking
        desired_quaternion = geometry.numpy_quaternion_from_tf_msg(self.last_reference_tf_msg)
        ref_e_xyz = geometry.euler_xyz_from_quaternion(desired_quaternion)

        # phi = e_zyx[0]
        # theta = e_zyx[1]
        phi = e_xyz[0]
        theta = e_xyz[1]
        phi_ref = ref_e_xyz[0]
        theta_ref = ref_e_xyz[1]
        self.ref_actual_msg.vector = [phi, phi_ref, theta, theta_ref]
        self.ref_actual_pub.publish(self.ref_actual_msg)
        # theta, phi = geometry.get_normal_alpha_beta_from_quaternion(dipole_quaternion)

        if self.__first_reading:
            self.last_phi = phi
            self.last_theta = theta
            self.__first_reading = False

        self.phi_dot = self.diff_alpha_phi*(phi - self.last_phi) + self.diff_beta_phi*self.phi_dot
        self.theta_dot = self.diff_alpha*(theta - self.last_theta) + self.diff_beta*self.theta_dot

        self.last_phi = phi
        self.last_theta = theta

        x_phi = np.array([[phi, self.phi_dot]]).T
        x_theta = np.array([[theta, self.theta_dot]]).T

        r_phi = np.array([[phi_ref, 0.0]]).T
        r_theta = np.array([[theta_ref, 0.0]]).T

        phi_error = x_phi - r_phi
        theta_error = x_theta - r_theta

        u_phi = -self.K_phi @ phi_error
        u_theta = -self.K_theta @ theta_error

        Tau_x = u_phi[0, 0]
        Tau_y = u_theta[0, 0]

        self.error_state_msg.vector = np.concatenate((phi_error.flatten(), theta_error.flatten()))
        self.error_state_pub.publish(self.error_state_msg)
        self.control_input_message.vector = [Tau_x, Tau_y]

        com_wrench_des = np.array([Tau_x, Tau_y, 0.0, 0.0, 0.0, 0.0])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])

        # Performing the simplified allocation for the two torques.
        des_currents = self.simplified_Tauxy_allocation(tf_msg, Tau_x, Tau_y)

        self.desired_currents_msg.des_currents_reg = des_currents


if __name__ == "__main__":
    controller = SingleDipoleNormalOrientationController()
    rospy.spin()