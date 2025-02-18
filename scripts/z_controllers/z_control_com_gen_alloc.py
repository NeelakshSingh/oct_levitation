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

class DirectCOMWrenchYawController(ControlSessionNodeBase):

    def post_init(self):
        self.HARDWARE_CONNECTED = True
        self.tfsub_callback_style_control_loop = True
        self.control_rate = 100 # Set it to the vicon frequency
        self.rigid_body_dipole = rigid_bodies.TwoDipoleDisc80x15_6HKCM10x3
        self.publish_desired_com_wrenches = True
        self.control_input_publisher = rospy.Publisher("/com_wrench_z_control/control_input",
                                                       VectorStamped, queue_size=1)

        self.estimated_state_pub = rospy.Publisher("/com_wrench_z_control/estimated_position_and_velocity",
                                                   VectorStamped, queue_size=1)
        
        self.publish_jma_condition = True
        self.warn_jma_condition = False
        if self.publish_jma_condition:
            self.jma_condition_pub = rospy.Publisher("/com_wrench_z_control/jma_condition",
                                                        VectorStamped, queue_size=1)
        
        # Overestimating mass is quite bad and leads to strong overshoots due to gravity compensation.
        # So I remove a few grams from the estimate.
        self.mass = self.rigid_body_dipole.mass_properties.m - 0.01 # Subtracting 10 grams from the mass.
        self.k_lin_z = 1e-3 # Friction damping parameter, to be tuned. Originally because of the rod.

        self.south_pole_up = True
        
        ## Continuous time state space model.
        A = np.array([[0, 1], [0, -self.k_lin_z/self.mass]])
        B = np.array([[0, 1/self.mass]]).T
        C = np.array([[1, 0]])

        z_max = 4e-2 # 4 cm maximum z displacement.
        z_dot_max = 5*z_max
        u_max = 5*z_dot_max # Assume very small maximum torque.

        ## Normalizing the state space model.
        Tx = np.diag([z_max, z_dot_max])
        Tu = np.diag([u_max])
        Ty = np.diag([z_max])

        A_norm = np.linalg.inv(Tx) @ A @ Tx
        B_norm = np.linalg.inv(Tx) @ B @ Tu
        C_norm = np.linalg.inv(Ty) @ C @ Tx

        # A_norm = A
        # B_norm = B
        # C_norm = C

        ## Setting up the DLQR parameters for exact system emulation.
        A_d_norm, B_d_norm, C_d_norm, D_d_norm, dt = signal.cont2discrete((A_norm, B_norm, C_norm, 0), dt=1/self.control_rate,
                                                  method='zoh')
        Q = 1e-2*np.diag([10.0, 1.0])
        R = 1
        K_norm, S, E = ct.dlqr(A_d_norm, B_d_norm, Q, R)

        # Denormalize the control gains.
        self.K = Tu @ K_norm @ np.linalg.inv(Tx)
        # self.K = K_norm

        self.T_pos_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([30e-3, 0, 0]))
        self.T_neg_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([-30e-3, 0, 0]))
        
        self.last_z = 0.0
        self.dt = 1/self.control_rate

        ## Using tustin's method to calculate a filtered derivative in discrete time.
        # The filter is a first order low pass filter.
        f_filter = 20
        Tf = 1/(2*np.pi*f_filter)
        # self.diff_alpha = 2*self.control_rate/(2*self.control_rate*Tf + 1)
        # self.diff_beta = (2*self.control_rate*Tf - 1)/(2*self.control_rate*Tf + 1)
        self.z_dot = 0.0
        self.home_z = 0.02

        # For finite differences. Just use
        self.diff_alpha = 1/self.dt
        self.diff_beta = 0

    def jm_currents_to_wrench_gen_alloc(self, origin_tf: TransformStamped) -> np.ndarray:
        # Hardcoding the dipole offsets now.
        com_position = np.array([
            origin_tf.transform.translation.x,
            origin_tf.transform.translation.y,
            origin_tf.transform.translation.z
        ])
        com_quaternion : Quaternion = origin_tf.transform.rotation

        T_WCOM = geometry.transformation_matrix_from_quaternion(
            np.array([com_quaternion.x, com_quaternion.y, com_quaternion.z, com_quaternion.w]),
            com_position
        )

        # The dipole transformations in the world frame.
        T_W_pos_x = T_WCOM @ self.T_pos_x
        T_W_neg_x = T_WCOM @ self.T_neg_x

        # The dipole positions in the world frame.
        p_pos_x = T_W_pos_x[:3, 3]
        p_neg_x = T_W_neg_x[:3, 3]

        # Dipole normals
        p_pos_x_normal = T_W_pos_x[:3, 2]
        p_neg_x_normal = T_W_neg_x[:3, 2]

        # Getting the magnetic interaction matrices.
        # The negative sign is because of south pole up
        dipole_pos_strength = self.rigid_body_dipole.dipole_list[0].strength
        dipole_neg_strength = self.rigid_body_dipole.dipole_list[1].strength
        if self.south_pole_up:
            dipole_neg_strength = -dipole_neg_strength
            dipole_pos_strength = -dipole_pos_strength

        M_pos_x = geometry.magnetic_interaction_matrix_from_dipole_moment(
            dipole_pos_strength*p_pos_x_normal,
            full_mat=True, torque_first=True)
        
        M_neg_x = geometry.magnetic_interaction_matrix_from_dipole_moment(
            dipole_neg_strength*p_neg_x_normal,
            full_mat=True, torque_first=True)
        
        # Now we need to get the force-torque jacobian matrices from the dipole centers to the COM.
        def mechanical_jacobian(v_rp: np.ndarray, v_rb: np.ndarray) -> np.ndarray:
            return np.block([
                            [np.eye(3), geometry.get_skew_symmetric_matrix(v_rp - v_rb)],
                            [np.zeros((3, 3)), np.eye(3)]
                        ])
    
        J_pos_x = mechanical_jacobian(p_pos_x, com_position)
        J_neg_x = mechanical_jacobian(p_neg_x, com_position)
        
        # Getting the actuation matrices.
        A_pos_x = self.mpem_model.getActuationMatrix(p_pos_x)
        A_neg_x = self.mpem_model.getActuationMatrix(p_neg_x)

        # The final allocation matrix is just the sum of JMA

        JMA = J_pos_x @ M_pos_x @ A_pos_x + J_neg_x @ M_neg_x @ A_neg_x

        if self.publish_jma_condition:
            jma_condition_msg = VectorStamped()
            jma_condition_msg.header.stamp = rospy.Time.now()
            jma_condition = np.linalg.cond(JMA)
            jma_condition_msg.vector = [jma_condition]
            self.jma_condition_pub.publish(jma_condition_msg)

        if self.warn_jma_condition:
            condition_check_tol = 9e3
            if jma_condition > condition_check_tol:
                np.set_printoptions(linewidth=np.inf)
                ezyx = geometry.euler_zyx_from_quaternion(np.array([com_quaternion.x, com_quaternion.y, com_quaternion.z, com_quaternion.w]))
                rospy.logwarn_once(f"""JMA condition number is too high: {jma_condition}, Yaw: {ezyx[2]}, Z: {origin_tf.transform.translation.z}
                                    \n JMA pinv: \n {np.linalg.pinv(JMA)}
                                    \n JMA: \n {JMA}""")
                rospy.loginfo_once("[Condition Debug] Trying to pinpoint the source of rank loss.")
                J_stack = np.hstack([J_pos_x, J_neg_x])
                rospy.loginfo_once(f"""[Condition Debug] J Stack Rank: {np.linalg.matrix_rank(J_stack)},
                                Should ideally be 6 for full row rank. 
                                J Stack Condition Number: {np.linalg.cond(J_stack)}""")
                
                M_diag = block_diag(M_pos_x, M_neg_x)
                rospy.loginfo_once(f"""[Condition Debug] M Diag Rank: {np.linalg.matrix_rank(M_diag)},
                                    Should ideally be 10 for 12 rows and 2 uncontrollable directions.
                                    Condition number will be inf because its not full rank.""")
                
                JM_stack = np.hstack([J_pos_x @ M_pos_x, J_neg_x @ M_neg_x])
                rospy.loginfo_once(f"""[Condition Debug] JM Stack Rank: {np.linalg.matrix_rank(JM_stack)},
                                    Should ideally be 6 for full row rank.
                                    JM Stack Condition Number: {np.linalg.cond(JM_stack)}""")
                
                A_stack = np.vstack([A_pos_x, A_neg_x])
                rospy.loginfo_once(f"""[Condition Debug] A Stack Rank: {np.linalg.matrix_rank(A_stack)},
                                    Should ideally be 8 for full column rank.
                                    A Stack Condition Number: {np.linalg.cond(A_stack)}""")
                
                rospy.loginfo_once(f"""[Condition Debug] JMA Rank: {np.linalg.matrix_rank(JMA)},
                                    Should ideally be 6 for full row rank.
                                    JMA Condition Number: {np.linalg.cond(JMA)}""")
                
                rospy.loginfo_once(f"""[Condition Debug] Let's look at individual dipole allocations.""")
                JMA_pos = J_pos_x @ M_pos_x @ A_pos_x
                JMA_neg = J_neg_x @ M_neg_x @ A_neg_x
                rospy.loginfo_once(f"""[Condition Debug] JMA Pos Rank: {np.linalg.matrix_rank(JMA_pos)},
                                    JMA Neg Rank: {np.linalg.matrix_rank(JMA_neg)}
                                    Should ideally be 5 because they cannot control all 6 DOFs alone. Condition numbers will be inf.""")
        
        return JMA
    
    def jm_fieldgrad_to_wrench_zero_rp(self, yaw: float):
        JM = []
        dipole = self.rigid_body_dipole.dipole_list[0]
        m_tilde = -dipole.strength # Because south pole up
        def J_a(a: float):
            JM.append(
                np.array([
                    [0, -m_tilde, 0, -a*m_tilde*np.sin(yaw), 0, 0, -a*m_tilde*np.sin(yaw), 0],
                    [m_tilde, 0, 0, a*m_tilde*np.cos(yaw), 0, 0, a*m_tilde*np.cos(yaw), 0],
                    [0, 0, 0, 0, 0, -a*m_tilde*np.sin(yaw), 0, a*m_tilde*np.cos(yaw)],
                    [0, 0, 0, 0, 0, m_tilde, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, m_tilde],
                    [0, 0, 0, -m_tilde, 0, 0, -m_tilde, 0],
                ])
            )

        # Posx first then negx
        J_a(30e-3)
        J_a(-30e-3)

        return np.hstack(JM)
    
    def jm_currents_to_wrench_zero_rp(self, origin_tf: TransformStamped) -> np.ndarray:
        # Hardcoding the dipole offsets now.
        com_position = np.array([
            origin_tf.transform.translation.x,
            origin_tf.transform.translation.y,
            origin_tf.transform.translation.z
        ])
        com_quaternion : Quaternion = origin_tf.transform.rotation

        T_WCOM = geometry.transformation_matrix_from_quaternion(
            np.array([com_quaternion.x, com_quaternion.y, com_quaternion.z, com_quaternion.w]),
            com_position
        )

        # The dipole transformations in the world frame.
        T_W_pos_x = T_WCOM @ self.T_pos_x
        T_W_neg_x = T_WCOM @ self.T_neg_x

        # The dipole positions in the world frame.
        p_pos_x = T_W_pos_x[:3, 3]
        p_neg_x = T_W_neg_x[:3, 3]

        # Getting the actuation matrices.
        A_pos_x = self.mpem_model.getActuationMatrix(p_pos_x)
        A_neg_x = self.mpem_model.getActuationMatrix(p_neg_x)

        A = np.vstack([A_pos_x, A_neg_x])
        JM = self.jm_fieldgrad_to_wrench_zero_rp(0.0)

        return JM @ A

    
    def callback_control_logic(self, tf_msg: TransformStamped):
        self.desired_currents_msg = DesCurrentsReg() # Empty message
        self.control_input_message = VectorStamped() # Empty message
        self.com_wrench_msg = WrenchStamped() # Empty message
        self.estimated_state_msg = VectorStamped() # Empty state estimate message

        self.desired_currents_msg.header.stamp = rospy.Time.now()
        self.com_wrench_msg.header.stamp = rospy.Time.now()
        self.control_input_message.header.stamp = rospy.Time.now()
        self.estimated_state_msg.header.stamp = rospy.Time.now()

        # Hardcoding the dipole offsets now.
        JMA = self.jm_currents_to_wrench_gen_alloc(tf_msg)

        # Getting the desired COM wrench.
        z_com = tf_msg.transform.translation.z
        self.z_dot = self.diff_alpha*(z_com - self.last_z) + self.diff_beta*self.z_dot
        self.last_z = z_com
        # rospy.loginfo(f"Z: {z_com}, Z dot: {z_dot}")
        x = np.array([[z_com, self.z_dot]]).T
        x_z_ref = np.array([[self.home_z, 0.0]]).T
        z_error = x - x_z_ref
        # u = -self.K @ x
        u = -self.K @ z_error + self.mass*common.Constants.g
        self.control_input_message.vector = u.flatten()
        self.estimated_state_msg.vector = z_error.flatten()

        self.estimated_state_pub.publish(self.estimated_state_msg)

        F_z = u[0, 0]
        com_wrench_des = np.array([0, 0, 0, 0, 0, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])
        # Let's solve for the currents using ridge regression. This will help against
        # poor conditioning of the JMA matrix.
        des_currents = numerical.solve_tikhonov_regularization(JMA, com_wrench_des, 1e-3)
        self.desired_currents_msg.des_currents_reg = des_currents.flatten()

if __name__ == "__main__":
    controller = DirectCOMWrenchYawController()
    rospy.spin()