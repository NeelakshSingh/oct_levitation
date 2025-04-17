import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry_jit as geometry
import oct_levitation.common as common
import oct_levitation.numerical as numerical

from scipy.linalg import block_diag
from oct_levitation.control_node import ControlSessionNodeBase
from control_utils.msg import VectorStamped
from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3, Quaternion
from std_msgs.msg import String
from tnb_mns_driver.msg import DesCurrentsReg

class DirectCOMWrenchYSingleDipoleController(ControlSessionNodeBase):

    def post_init(self):
        ## EXPERIMENT FLAGS
        self.INITIAL_DESIRED_POSITION = np.array([0.0, 0.0, 0.0])
        self.tfsub_callback_style_control_loop = True
        self.control_rate = self.CONTROL_RATE # Set it to the vicon frequency (from parameter file)
        self.publish_desired_com_wrenches = True
        self.control_input_publisher = rospy.Publisher("/com_wrench_z_control/control_input",
                                                       VectorStamped, queue_size=1)

        self.estimated_state_pub = rospy.Publisher("/com_wrench_z_control/estimated_position_and_velocity",
                                                   VectorStamped, queue_size=1)
        
        self.control_gain_publisher = rospy.Publisher("/com_wrench_z_control/control_gains",
                                                 VectorStamped, queue_size=1, latch=True)
        

        # Overestimating mass is quite bad and leads to strong overshoots due to gravity compensation.
        # So I remove a few grams from the estimate.
        mass_offset = 0
        self.mass = self.rigid_body_dipole.mass_properties.m + mass_offset # Subtracting 10 grams from the mass.
        self.k_lin_y = 0 # Friction damping parameter, to be tuned. Originally because of the rod.

        self.south_pole_up = True
        
        ## Continuous time state space model.
        A = np.array([[0, 1], [0, -self.k_lin_y/self.mass]])
        B = np.array([[0, 1/self.mass]]).T
        C = np.array([[1, 0]])

        x_max = 4e-2 # 4 cm maximum x displacement.
        x_dot_max = 5*x_max
        u_max = 5*self.mass*x_dot_max # Assume very small maximum torque.

        ## Normalizing the state space model.
        Tx = np.diag([x_max, x_dot_max])
        Tu = np.diag([u_max])
        Ty = np.diag([x_max])

        A_norm = np.linalg.inv(Tx) @ A @ Tx
        B_norm = np.linalg.inv(Tx) @ B @ Tu
        C_norm = np.linalg.inv(Ty) @ C @ Tx

        # A_norm = A
        # B_norm = B
        # C_norm = C

        ## Setting up the DLQR parameters for exact system emulation.
        A_d_norm, B_d_norm, C_d_norm, D_d_norm, dt = signal.cont2discrete((A_norm, B_norm, C_norm, 0), dt=1/self.control_rate,
                                                  method='zoh')
        Q = np.diag([100.0, 10.0])
        R = 1
        K_norm, S, E = ct.dlqr(A_d_norm, B_d_norm, Q, R)

        # Denormalize the control gains.
        self.K = np.asarray(Tu @ K_norm @ np.linalg.inv(Tx))
        # self.K = np.asarray(K_norm)
        # self.K = np.array([[12.1024, 2.5101]]) # For POM Disc's Overdamped tuning without friction damping
        # self.K = np.array([[9.0896, 1.3842]]) # For Onyx disc's Overdamped tuning without friction damping.
        # self.K = np.array([[9.0896, 1.3842]])
        

        self.control_gains_message = VectorStamped()
        self.control_gains_message.header.stamp = rospy.Time.now()
        # self.K = K_norm
        rospy.loginfo(f"[X Control Single Dipole Full Alloc], Control gain K:{self.K}")
        
        self.last_y = 0.0
        self.dt = 1/self.control_rate

        ## Using tustin's method to calculate a filtered derivative in discrete time.
        # The filter is a first order low pass filter.
        # f_filter = 100
        # Tf = 1/(2*np.pi*f_filter)
        # self.diff_alpha = 2*self.control_rate/(2*self.control_rate*Tf + 1)
        # self.diff_beta = (2*self.control_rate*Tf - 1)/(2*self.control_rate*Tf + 1)
        self.y_dot = 0.0

        self.__first_iteration = True

        # For finite differences. Just use
        self.diff_alpha = 1/self.dt
        self.diff_beta = 0

        # self.SoftStarter = numerical.SigmoidSoftStarter(0)

        # self.calibration_file = "octomag_5point.yaml"
        self.control_gains_message.vector = np.concatenate((self.K.flatten(), np.array([self.diff_alpha, self.diff_beta])))
        self.metadata_msg.metadata.data = f"""
        Experiment type: Y axis control experiment with full allocation while requesting 0 fields.
        Calibration file: {self.calibration_file}
        Gains: {self.K.flatten()}
        Calibration type: Legacy yaml file
        """
        self.set_path_metadata(__file__)

    def local_frame_torque_global_force_allocation(self, tf_msg: TransformStamped, F_x: float):
        quaternion = np.array([
            tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w
        ])
        normal = -geometry.get_normal_vector_from_quaternion(quaternion) # -ve because south pole up
        dipole = self.rigid_body_dipole.dipole_list[0]
        dipole_vector = dipole.strength * normal
        dipole_position = np.array([
            tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z
        ])
        Mf = geometry.magnetic_interaction_grad5_to_force(dipole_vector)
        Mt_local = geometry.magnetic_interaction_field_to_local_torque(dipole.strength,
                                                                       dipole.axis,
                                                                       quaternion)[:2] # Only first two rows will be nonzero
        w_des = np.array([0.0, 0.0, F_x, 0.0, 0.0]) # Tau_local_xy, F_v
        M = block_diag(Mt_local, Mf)
        A = self.mpem_model.getActuationMatrix(dipole_position)
        JMA = M @ A
        des_currents = np.linalg.pinv(JMA) @ w_des
        
        jma_condition = np.linalg.cond(JMA)

        if self.warn_jma_condition:
            condition_check_tol = 50
            if jma_condition > condition_check_tol:
                np.set_printoptions(linewidth=np.inf)
                rospy.logwarn_once(f"""JMA condition number is too high: {jma_condition}, Current TF: {tf_msg}
                                    \n JMA pinv: \n {np.linalg.pinv(JMA)}
                                    \n JMA: \n {JMA}""")
                rospy.loginfo_once("[Condition Debug] Trying to pinpoint the source of rank loss.")

                rospy.loginfo_once(f"""[Condition Debug] Mf Rank: {np.linalg.matrix_rank(Mf)},
                                    Should ideally be 3 for 3 rows and all 3 controllable forces.
                                    Condition number: {np.linalg.cond(Mf)}""")
                
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

    
    def callback_control_logic(self, tf_msg: TransformStamped):
        self.desired_currents_msg = DesCurrentsReg() # Empty message
        self.control_input_message = VectorStamped() # Empty message
        self.com_wrench_msg = WrenchStamped() # Empty message
        self.estimated_state_msg = VectorStamped() # Empty state estimate message

        self.desired_currents_msg.header.stamp = rospy.Time.now()
        self.com_wrench_msg.header.stamp = rospy.Time.now()
        self.control_input_message.header.stamp = rospy.Time.now()
        self.estimated_state_msg.header.stamp = rospy.Time.now()

        # Getting the desired COM wrench.
        y_com = tf_msg.transform.translation.y
        if self.__first_iteration: # In order to ensure a soft start for the y_dot term.
            self.last_y = y_com
            self.__first_iteration = False
        self.y_dot = self.diff_alpha*(y_com - self.last_y) + self.diff_beta*self.y_dot
        self.last_y = y_com
        # rospy.loginfo(f"y: {y_com}, y dot: {y_dot}")
        x = np.array([[y_com, self.y_dot]]).T # I refer to the full state as x. Even though it corresponds to just y.
        x_y_ref = np.array([[self.last_reference_tf_msg.transform.translation.y, 0.0]]).T
        # rospy.loginfo(f"x_y_ref: {x_y_ref}")
        x_error = x - x_y_ref
        u = -self.K @ x_error
        self.control_input_message.vector = u.flatten()
        self.estimated_state_msg.vector = x_error.flatten()

        self.estimated_state_pub.publish(self.estimated_state_msg)

        F_y = u[0, 0]
        # Adding a little bit of gravity compensation in order to reduce normal forces and friction a little bit.
        F_z = 0.90*self.mass*common.Constants.g
        # F_z = 0.0
        com_wrench_des = np.array([0, 0, 0, 0, F_y, F_z])
        com_wrench_5dof = np.array([0, 0, 0, F_y, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])
        
        # Performing simplified allocation to get the currents
        # des_currents = self.local_frame_torque_global_force_allocation(tf_msg, F_x)
        des_currents = self.five_dof_wrench_allocation_single_dipole(tf_msg, com_wrench_5dof)
        # self.desired_currents_msg.des_currents_reg = des_currents.flatten() * self.SoftStarter(self.dt)
        self.desired_currents_msg.des_currents_reg = des_currents.flatten()

if __name__ == "__main__":
    controller = DirectCOMWrenchYSingleDipoleController()
    rospy.spin()