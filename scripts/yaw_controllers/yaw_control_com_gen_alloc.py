import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry

from oct_levitation.control_node import ControlSessionNodeBase
from control_utils.msg import VectorStamped
from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3, Quaternion
from tnb_mns_driver.msg import DesCurrentsReg

class DirectCOMWrenchYawController(ControlSessionNodeBase):

    def post_init(self):
        self.HARDWARE_CONNECTED = False
        self.tfsub_callback_style_control_loop = True
        self.control_rate = 100 # Set it to the vicon frequency\
        self.rigid_body_dipole = rigid_bodies.TwoDipoleDisc80x15_6HKCM10x3
        self.publish_desired_com_wrenches = True
        self.control_input_publisher = rospy.Publisher("/com_wrench_yaw_control/control_input",
                                                       VectorStamped, queue_size=1)

        self.estimated_state_pub = rospy.Publisher("/com_wrench_yaw_control/estimated_position_and_velocity",
                                                   VectorStamped, queue_size=1)
        
        self.Iz = self.rigid_body_dipole.mass_properties.I_bf[2, 2]
        self.k_z = 1e-4 # Damping parameter, to be tuned. Originally because of the rod.

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
        Q = np.diag([10.0, 1.0])
        R = 1
        K_norm, S, E = ct.dlqr(A_d_norm, B_d_norm, Q, R)

        # Denormalize the control gains.
        self.K = Tu @ K_norm @ np.linalg.inv(Tx)
        # self.K = K_norm

        self.T_pos_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([30e-3, 0, 0]))
        self.T_neg_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([-30e-3, 0, 0]))
        
        self.last_yaw = 0.0
        self.dt = 1/self.control_rate

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
        A_pos_x = self.mpem_model.getActuationMatrix(p_pos_x)
        A_neg_x = self.mpem_model.getActuationMatrix(p_neg_x)

        # The final allocation matrix is just the sum of JMA

        return J_pos_x @ M_pos_x @ A_pos_x + J_neg_x @ M_neg_x @ A_neg_x
    
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

        JMA = self.jm_currents_to_wrench_zero_rp(tf_msg)

        # Getting the desired COM wrench.
        yaw_dot = (yaw - self.last_yaw)/self.dt
        self.last_yaw = yaw
        # rospy.loginfo(f"Yaw: {np.rad2deg(yaw)}, Yaw dot: {np.rad2deg(yaw_dot)}")
        x = np.array([[yaw, yaw_dot]]).T
        u = -self.K @ x
        self.control_input_message.vector = u.flatten()
        self.estimated_state_msg.vector = x.flatten()

        Tau_z = u[0, 0]
        com_wrench_des = np.array([0, 0, Tau_z, 0, 0, 0])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])
        des_currents = np.linalg.pinv(JMA) @ com_wrench_des
        self.desired_currents_msg.des_currents_reg = des_currents.flatten()

if __name__ == "__main__":
    controller = DirectCOMWrenchYawController()
    rospy.spin()