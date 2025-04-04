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
from std_msgs.msg import String
from tnb_mns_driver.msg import DesCurrentsReg

class DirectCOMWrenchZSingleDipoleController(ControlSessionNodeBase):

    def post_init(self):
        ## EXPERIMENT FLAGS
        self.HARDWARE_CONNECTED = False
        self.ORIENTATION_VARYING_Mf = False
        self.POSITION_VARYING_Af = True
        # self.ACTIVE_COILS = np.array([2,3,4,5,7]) # Only use this set of coils for actuation and field allocation. Defaults to all 8 coils.
        self.ACTIVE_COILS = np.array([1, 3, 4, 5, 6]) # Only use this set of coils for actuation and field allocation. Defaults to all 8 coils.
        # self.ACTIVE_COILS = np.array([0, 1, 3, 5, 6]) # Only use this set of coils for actuation and field allocation. Defaults to all 8 coils.
        # self.ACTIVE_COILS = np.array([0, 3, 5, 6, 7]) # Only use this set of coils for actuation and field allocation. Defaults to all 8 coils.
        self.INITIAL_POSITION = np.array([0.0, 0.0, 0.0])
        self.MAX_CURRENT = 8.0 # Amps

        self.tfsub_callback_style_control_loop = True
        self.control_rate = 100 # Set it to the vicon frequency
        self.rigid_body_dipole = rigid_bodies.Onyx80x22DiscCenterRingDipole
        self.publish_desired_com_wrenches = True
        self.control_input_publisher = rospy.Publisher("/com_wrench_z_control/control_input",
                                                       VectorStamped, queue_size=1)

        self.estimated_state_pub = rospy.Publisher("/com_wrench_z_control/estimated_position_and_velocity",
                                                   VectorStamped, queue_size=1)
        
        self.control_gain_publisher = rospy.Publisher("/com_wrench_z_control/control_gains",
                                                 VectorStamped, queue_size=1, latch=True)
        
        self.publish_jma_condition = True
        if self.publish_jma_condition:
            self.jma_condition_pub = rospy.Publisher("/com_wrench_z_control/jma_condition",
                                                        VectorStamped, queue_size=1)
        
        # Overestimating mass is quite bad and leads to strong overshoots due to gravity compensation.
        # So I remove a few grams from the estimate.
        mass_offset = 0
        self.mass = self.rigid_body_dipole.mass_properties.m + mass_offset # Subtracting 10 grams from the mass.
        self.k_lin_z = 1 # Friction damping parameter, to be tuned. Originally because of the rod.

        self.south_pole_up = True
        
        ## Continuous time state space model.
        A = np.array([[0, 1], [0, -self.k_lin_z/self.mass]])
        B = np.array([[0, 1/self.mass]]).T
        C = np.array([[1, 0]])

        z_max = 4e-2 # 4 cm maximum z displacement.
        z_dot_max = 5*z_max
        u_max = 5*self.mass*z_dot_max # Assume very small maximum torque.

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
        rospy.loginfo(f"[Z Control Single Dipole Simplified], Control gain K:{self.K}")

        self.T_pos_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([30e-3, 0, 0]))
        self.T_neg_x = geometry.transformation_matrix_from_quaternion(geometry.IDENTITY_QUATERNION,
                                                                 np.array([-30e-3, 0, 0]))
        
        self.last_z = 0.0
        self.dt = 1/self.control_rate

        ## Using tustin's method to calculate a filtered derivative in discrete time.
        # The filter is a first order low pass filter.
        f_filter = 100
        # Tf = 1/(2*np.pi*f_filter)
        # self.diff_alpha = 2*self.control_rate/(2*self.control_rate*Tf + 1)
        # self.diff_beta = (2*self.control_rate*Tf - 1)/(2*self.control_rate*Tf + 1)
        self.z_dot = 0.0

        self.__first_iteration = True

        # For finite differences. Just use
        self.diff_alpha = 1/self.dt
        self.diff_beta = 0

        self.SoftStarter = numerical.SigmoidSoftStarter(1)

        # self.calibration_file = "octomag_5point.yaml"
        self.control_gains_message.vector = np.concatenate((self.K.flatten(), np.array([self.diff_alpha, self.diff_beta])))
        self.metadata_msg = String()
        self.metadata_msg.data = f"""
        Experiment metadata.
        Experiment type: Vicon at 800Hz, 2Hz Sinusoidal Reference with DLQR Gains. Using Tustin's filtered differentiator with 100Hz cutoff.
        Calibration file: {self.calibration_file}
        Hardware Connected: {self.HARDWARE_CONNECTED}
        Gains: {self.K.flatten()}
        Calibration type: Legacy yaml file
        """

    def simplified_Fz_allocation(self, tf_msg: TransformStamped, Fz_des: float):
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
        w_des = np.array([0.0, 0.0, 0.0, 0.0, Fz_des]) # Tau_local_xy, F_v
        M = block_diag(Mt_local, Mf)
        A = self.mpem_model.getActuationMatrix(dipole_position)
        des_currents = np.linalg.pinv(M @ A) @ w_des

        if self.publish_jma_condition:
            jma_condition_msg = VectorStamped()
            jma_condition_msg.header.stamp = rospy.Time.now()
            jma_condition = np.linalg.cond(M @ A)
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
        z_com = tf_msg.transform.translation.z
        if self.__first_iteration: # In order to ensure a soft start for the z_dot term.
            self.last_z = z_com
            self.__first_iteration = False
        self.z_dot = self.diff_alpha*(z_com - self.last_z) + self.diff_beta*self.z_dot
        self.last_z = z_com
        # rospy.loginfo(f"Z: {z_com}, Z dot: {z_dot}")
        x = np.array([[z_com, self.z_dot]]).T
        x_z_ref = np.array([[self.last_reference_tf_msg.transform.translation.z, 0.0]]).T
        # rospy.loginfo(f"x_z_ref: {x_z_ref}")
        z_error = x - x_z_ref
        u = -self.K @ z_error + self.mass*common.Constants.g
        # u = -self.K @ z_error
        self.control_input_message.vector = u.flatten()
        self.estimated_state_msg.vector = z_error.flatten()

        self.estimated_state_pub.publish(self.estimated_state_msg)

        F_z = u[0, 0]
        com_wrench_des = np.array([0, 0, 0, 0, 0, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])
        
        # Performing simplified allocation to get the currents
        w_des = np.array([0, 0, 0, 0, F_z])
        # des_currents = self.simplified_Fz_allocation(tf_msg, F_z)
        des_currents = self.five_dof_wrench_allocation_single_dipole(tf_msg, w_des)
        # self.desired_currents_msg.des_currents_reg = des_currents.flatten() * self.SoftStarter(self.dt)
        self.desired_currents_msg.des_currents_reg = des_currents.flatten()

if __name__ == "__main__":
    controller = DirectCOMWrenchZSingleDipoleController()
    rospy.spin()