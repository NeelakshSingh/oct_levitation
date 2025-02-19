import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry
import oct_levitation.numerical as numerical

from scipy.linalg import block_diag
from oct_levitation.control_node import ControlSessionNodeBase
from control_utils.msg import VectorStamped
from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3, Quaternion
from tnb_mns_driver.msg import DesCurrentsReg

def quaternion_to_x_axis(q):
    """
    Returns the x-axis from a unit-quaternion (Hamilton - convention)

    Args:
        q (np.array): (4,) quaternion of form [x, y, z, w]
    returns:
        np.array: (3,) x-axis
    """

    x_1 = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    x_2 = 2 * (q[0] * q[1] + q[2] * q[3])
    x_3 = 2 * (q[0] * q[2] - q[1] * q[3])

    return np.array([x_1, x_2, x_3])

class LimitFreezeFilter:

    def __init__(self, max_lim: float, min_limit: float, lim_tol: float = 1e-3):
        """
        For this filter to work properly 
        """
        self.max_lim = max_lim
        self.min_lim = min_limit
        self.lim_tol = lim_tol
        self.last_max_lim_breached = False
        self.last_min_lim_breached = False
        self.last_max_lim_breach_value = None
        self.last_min_lim_breach_value = None
    
    def filter(self, value: np.ndarray) -> float:
        if self.last_max_lim_breach_value is None:
            self.last_max_lim_breach_value = value
        
        if self.last_min_lim_breach_value is None:
            self.last_min_lim_breach_value = value

        if np.any(np.abs(value - self.max_lim) < self.lim_tol):
            self.last_max_lim_breach_value = value

        if np.any(np.abs(value - self.min_lim) < self.lim_tol):
            self.last_min_lim_breach_value = value
        
        if np.any(value > self.max_lim) and not np.any(value < self.min_lim):
            self.last_max_lim_breached = True
            if self.last_min_lim_breached:
                self.last_min_lim_breached = False
            return self.last_max_lim_breach_value
        elif np.any(value < self.min_lim) and not np.any(value > self.max_lim):
            self.last_min_lim_breached = True
            if self.last_max_lim_breached:
                self.last_max_lim_breached = False
            return self.last_min_lim_breach_value
        elif np.any(value > self.max_lim) and np.any(value < self.min_lim):
            self.last_max_lim_breached = True
            self.last_min_lim_breached = True
            return self.last_max_lim_breach_value

        return value

    def __call__(self, *args, **kwds):
        return self.filter(*args, **kwds)


class DirectCOMWrenchYawController(ControlSessionNodeBase):

    def post_init(self):
        self.HARDWARE_CONNECTED = True
        rospy.loginfo("[Yaw Control General Allocation] Hardware connected: {}".format(self.HARDWARE_CONNECTED))
        self.tfsub_callback_style_control_loop = True
        self.control_rate = 100 # Set it to the vicon frequency\
        self.rigid_body_dipole = rigid_bodies.TwoDipoleDisc80x15_6HKCM10x3
        self.publish_desired_com_wrenches = True
        self.control_input_publisher = rospy.Publisher("/com_wrench_yaw_control/control_input",
                                                       VectorStamped, queue_size=1)

        self.estimated_state_pub = rospy.Publisher("/com_wrench_yaw_control/estimated_position_and_velocity",
                                                   VectorStamped, queue_size=1)
        self.publish_jma_condition = True
        self.warn_jma_condition = True
        if self.publish_jma_condition:
            self.jma_condition_pub = rospy.Publisher("/com_wrench_yaw_control/jma_condition",
                                                        VectorStamped, queue_size=1)
        self.Iz = self.rigid_body_dipole.mass_properties.I_bf[2, 2]
        self.k_z = 1e-1 # Damping parameter, to be tuned. Originally because of the rod.
        # self.Tz_deadzone = 0.25 # Deadzone for the torque control. # LEADS TO OSCILLATIONS

        self.south_pole_up = True
        
        ## Continuous time state space model.
        A = np.array([[0, 1], [0, -self.k_z/self.Iz]])
        B = np.array([[0, 1/self.Iz]]).T
        C = np.array([[1, 0]])

        yaw_max = np.deg2rad(30)
        yaw_dot_max = 5*yaw_max
        u_max = 1e-4 # Assume very small maximum torque.

        ## Normalizing the state space model.
        Tx = np.diag([yaw_max, yaw_dot_max])
        Tu = np.diag([u_max])
        Ty = np.diag([yaw_max])

        A_norm = np.linalg.inv(Tx) @ A @ Tx
        B_norm = np.linalg.inv(Tx) @ B @ Tu
        C_norm = np.linalg.inv(Ty) @ C @ Tx

        # A_norm = A
        # B_norm = B
        # C_norm = C

        ## Setting up the DLQR parameters for exact system emulation.
        A_d_norm, B_d_norm, C_d_norm, D_d_norm, dt = signal.cont2discrete((A_norm, B_norm, C_norm, 0), dt=1/self.control_rate,
                                                  method='zoh')
        Q = np.diag([10000, 100])
        R = 1
        K_norm, S, E = ct.dlqr(A_d_norm, B_d_norm, Q, R)

        # Denormalize the control gains.
        # self.K = Tu @ K_norm @ np.linalg.inv(Tx)
        self.K = np.array([[0.081811855327392,0.00637044424414062]])
        rospy.loginfo(f"Control gain for Tz: {self.K}")
        # self.K = K_norm

        self.T_pos_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([30e-3, 0, 0]))
        self.T_neg_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([-30e-3, 0, 0]))
        
        self.last_yaw = 0.0
        self.dt = 1/self.control_rate

        self.Tau_z_max = 0.018 # Maximum torque in Nm

        self.current_limit_freezer = LimitFreezeFilter(7.0, -7.0, 1.0)

        # self.calibration_file = "octomag_5point.yaml"

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
        self.A_pos_x = self.mpem_model.getActuationMatrix(np.array([30e-3, 0, 0]))
        self.A_neg_x = self.mpem_model.getActuationMatrix(np.array([-30e-3, 0, 0]))
        # self.A_pos_x = self.mpem_model.getActuationMatrix(p_pos_x)
        # self.A_neg_x = self.mpem_model.getActuationMatrix(p_neg_x)

        # The final allocation matrix is just the sum of JMA

        JMA = J_pos_x @ M_pos_x @ self.A_pos_x + J_neg_x @ M_neg_x @ self.A_neg_x

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
                
                A_stack = np.vstack([self.A_pos_x, self.A_neg_x])
                rospy.loginfo_once(f"""[Condition Debug] A Stack Rank: {np.linalg.matrix_rank(A_stack)},
                                    Should ideally be 8 for full column rank.
                                    A Stack Condition Number: {np.linalg.cond(A_stack)}""")
                
                rospy.loginfo_once(f"""[Condition Debug] JMA Rank: {np.linalg.matrix_rank(JMA)},
                                    Should ideally be 6 for full row rank.
                                    JMA Condition Number: {np.linalg.cond(JMA)}""")
                
                rospy.loginfo_once(f"""[Condition Debug] Let's look at individual dipole allocations.""")
                JMA_pos = J_pos_x @ M_pos_x @ self.A_pos_x
                JMA_neg = J_neg_x @ M_neg_x @ self.A_neg_x
                rospy.loginfo_once(f"""[Condition Debug] JMA Pos Rank: {np.linalg.matrix_rank(JMA_pos)},
                                    JMA Neg Rank: {np.linalg.matrix_rank(JMA_neg)}
                                    Should ideally be 5 because they cannot control all 6 DOFs alone. Condition numbers will be inf.""")

        return JMA
    
    def allocation_jasan(self, origin_tf: TransformStamped) -> np.ndarray:
        """
        This function follows the same allocation style used by Jasan in his implementation.
        """
        yaw = self.last_yaw
        z = 1
        if self.south_pole_up: z = -1
        dipole_strength = self.rigid_body_dipole.dipole_list[0].strength
        Mf_pos_x = geometry.magnetic_interaction_grad5_to_force(
            dipole_strength*np.array([0, 0, z]))
        
        Mf_neg_x = geometry.magnetic_interaction_grad5_to_force(
            dipole_strength*np.array([0, 0, z]))
        
        Mf = block_diag(Mf_pos_x, Mf_neg_x)
        J_pinv = np.array([-np.sin(self.last_yaw), np.cos(self.last_yaw), np.sin(self.last_yaw), -np.cos(self.last_yaw)])/(2*30e-3)
        self.A_pos_x = self.mpem_model.getActuationMatrix(np.array([30e-3, 0, 0]))[3:, :]
        self.A_neg_x = self.mpem_model.getActuationMatrix(np.array([-30e-3, 0, 0]))[3:, :]
        A_grad5 = np.vstack([self.A_pos_x, self.A_neg_x])

        return J_pinv, Mf, A_grad5
    
    def gen_alloc_jasan_alloc_comparison(self, origin_tf: TransformStamped, Tau_z: float) -> np.ndarray:
        """
        This function is used to compare the allocation matrices generated by the two methods.
        """
        Jasan_J_pinv, Jasan_Mf, Jasan_A_grad5_stack = self.allocation_jasan(origin_tf)

        # Jasan's allocation
        Jasan_planar_forces = Jasan_J_pinv * Tau_z
        Jasan_full_force = np.concatenate((Jasan_planar_forces[:2], np.zeros(1), Jasan_planar_forces[2:4], np.zeros(1)))
        Jasan_full_grad_task = np.linalg.pinv(Jasan_Mf) @ Jasan_full_force
        Jasan_des_currents = np.linalg.pinv(Jasan_A_grad5_stack) @ Jasan_full_grad_task

        # Now let's do our general allocation and compare the results with using Jasan's allocation.
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

        # A_pos_x = self.mpem_model.getActuationMatrix(np.array([30e-3, 0, 0]))
        # A_neg_x = self.mpem_model.getActuationMatrix(np.array([-30e-3, 0, 0])) # just reuse the same one from jasan's allocation for comparison.
        A_pos_x = self.mpem_model.getActuationMatrix(p_pos_x)
        A_neg_x = self.mpem_model.getActuationMatrix(p_neg_x) # just reuse the same one from jasan's allocation for comparison.

        Agrad5_pos_x = A_pos_x[3:, :]
        Agrad5_neg_x = A_neg_x[3:, :]
        A_grad5_stack = np.vstack([Agrad5_pos_x, Agrad5_neg_x])

        M_full = block_diag(M_pos_x, M_neg_x)
        J_full = np.hstack([J_pos_x, J_neg_x])
        A_full = np.vstack([A_pos_x, A_neg_x])

        # Now let's do the general allocation, first step by step.
        # Then all at once. And we will compare the results of all allocation strategies.
        # First step by step.
        desired_com_wrench = np.array([0, 0, Tau_z, 0, 0, 0])
        dipole_wrenches = np.linalg.pinv(J_full) @ desired_com_wrench
        dipole_field_grad = np.linalg.pinv(M_full) @ dipole_wrenches
        des_currents_step_by_step = np.linalg.pinv(A_full) @ dipole_field_grad

        des_currents_all_at_once = np.linalg.pinv(J_full @ M_full @ A_full) @ desired_com_wrench

        # The final goal should be to make the des_currents_all_at_once the same as Jasan's allocation.

        # Finally we also compare tikhonov regularization.
        des_currents_tikhonov = numerical.solve_tikhonov_regularization(J_full @ M_full @ A_full, desired_com_wrench, 1e-3) # Regularization parameter is arbitrary.

        # last method where we first allocate field grad and then allocate currents.
        dipole_jm_field_grad = np.linalg.pinv(J_full @ M_full) @ desired_com_wrench
        des_currents_jm_field_grad = np.linalg.pinv(A_full) @ dipole_jm_field_grad
        dipole_jm_grad5 = np.concatenate([dipole_jm_field_grad[3:8], dipole_jm_field_grad[11:]])
        des_currents_jm_grad5 = np.linalg.pinv(A_grad5_stack) @ dipole_jm_grad5

        # Let's see if some SVD based approximation leads to better results.
        # Alloc_mat = J_full @ M_full @ A_full
        # U, S, Vt = np.linalg.svd(Alloc_mat)

        __dud_line_for_breakpoint_hehehe = False
        return des_currents_jm_field_grad
    
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
        com_quaternion : Quaternion = tf_msg.transform.rotation
        com_euler_zyx = geometry.euler_zyx_from_quaternion(
            np.array([com_quaternion.x, com_quaternion.y, com_quaternion.z, com_quaternion.w])
        )

        yaw = com_euler_zyx[2]
        # quat_rot = com_quaternion
        # x_axis = quaternion_to_x_axis([quat_rot.x, quat_rot.y, quat_rot.z, quat_rot.w])
        # yaw = np.arctan2(x_axis[1], x_axis[0])

        # JMA = self.jm_currents_to_wrench_zero_rp(tf_msg)

        # Getting the desired COM wrench.
        yaw_dot = (yaw - self.last_yaw)/self.dt
        self.last_yaw = yaw
        # rospy.loginfo(f"Yaw: {np.rad2deg(yaw)}, Yaw dot: {np.rad2deg(yaw_dot)}")
        x = np.array([[yaw, yaw_dot]]).T
        u = -self.K @ x
        self.control_input_message.vector = u.flatten()
        self.estimated_state_msg.vector = x.flatten()

        self.estimated_state_pub.publish(self.estimated_state_msg)

        Tau_z = u[0, 0]
        Tau_z = np.clip(Tau_z, -self.Tau_z_max, self.Tau_z_max)
        # Let's apply inverted deadzone, since this is without gravity compensation.
        # LEADS TO OSCILLATIONS, SO COMMENTED OUT
        # if Tau_z > 0.0:
        #     Tau_z += self.Tz_deadzone
        # elif Tau_z < 0.0:
        #     Tau_z -= self.Tz_deadzone

        # com_wrench_des = np.array([0, 0, Tau_z, 0, 0, 0])
        # self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        # self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])
        # Use tikhonov regularization instead to get around poorly conditioned matrix near the origin
        # for some yaw values.
        # des_currents = np.linalg.pinv(JMA) @ com_wrench_des
        # des_currents = numerical.solve_tikhonov_regularization(JMA, com_wrench_des, 1e-3)

        des_currents = self.gen_alloc_jasan_alloc_comparison(tf_msg, Tau_z)

        self.desired_currents_msg.des_currents_reg = des_currents.flatten()

if __name__ == "__main__":
    controller = DirectCOMWrenchYawController()
    rospy.spin()