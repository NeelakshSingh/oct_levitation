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
        com_position = tf_msg.transform.translation
        com_quaternion : Quaternion = tf_msg.transform.rotation
        com_euler_zyx = geometry.euler_zyx_from_quaternion(
            np.array([com_quaternion.x, com_quaternion.y, com_quaternion.z, com_quaternion.w])
        )

        T_WCOM = geometry.transformation_matrix_from_quaternion(
            np.array([com_quaternion.x, com_quaternion.y, com_quaternion.z, com_quaternion.w]),
            np.array([com_position.x, com_position.y, com_position.z])
        )

        # The dipole transformations in the world frame.
        T_W_pos_x = T_WCOM @ self.T_pos_x
        T_W_neg_x = T_WCOM @ self.T_neg_x

        # The dipole positions in the world frame.
        pos_x = T_W_pos_x[:3, 3]
        neg_x = T_W_neg_x[:3, 3]

        yaw = com_euler_zyx[2]

        # Getting the JM product.
        JM = self.jm_fieldgrad_to_wrench_zero_rp(yaw)

        # Getting the actuation matrices.
        A_pos_x = self.mpem_model.getActuationMatrix(pos_x)
        A_neg_x = self.mpem_model.getActuationMatrix(neg_x)

        A_field_grad = np.vstack([A_pos_x, A_neg_x])

        JMA = JM @ A_field_grad

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