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

class SimpleCOMWrenchSingleDipoleController(ControlSessionNodeBase):

    def post_init(self):
        self.HARDWARE_CONNECTED = True
        self.tfsub_callback_style_control_loop = True
        self.control_rate = 100 # Set it to the vicon frequency
        self.dt = 1/self.control_rate
        self.publish_desired_com_wrenches = True
        self.publish_desired_dipole_wrenches = False
        
        self.control_input_publisher = rospy.Publisher("/xyz_rp_control_single_dipole/control_input",
                                                       VectorStamped, queue_size=1)
        
        # Extra publishers which I wrote only in the post init and are not mandatory end with the shorthand pub.
        self.error_state_pub = rospy.Publisher("/xyz_rp_control_single_dipole/error_states",
                                                         VectorStamped, queue_size=1)
        
        self.ref_actual_pub = rospy.Publisher("/xyz_rp_control_single_dipole/ref_actual_values",
                                                         VectorStamped, queue_size=1)
        
        self.control_gain_publisher = rospy.Publisher("/xyz_rp_control_single_dipole/control_gains",
                                                      VectorStamped, queue_size=1, latch=True)
        
        self.publish_jma_condition = True
        self.south_pole_up = True
        self.warn_jma_condition = True

        if self.publish_jma_condition:
            self.jma_condition_pub = rospy.Publisher("/xyz_rp_control_single_dipole/jma_condition",
                                                     VectorStamped, queue_size=1)
            
        #############################
        ### Z CONTROL DLQR DESIGN ###
        self.mass = self.rigid_body_dipole.mass_properties.m
        self.k_lin_z = 1 # Friction damping parameter, to be tuned. Originally because of the rod.

        self.south_pole_up = True
        
        ## Continuous time state space model.
        Az = np.array([[0, 1], [0, -self.k_lin_z/self.mass]])
        Bz = np.array([[0, 1/self.mass]]).T
        Cz = np.array([[1, 0]])

        z_max = 4e-2 # 4 cm maximum z displacement.
        z_dot_max = 5*z_max
        Fz_max = 5*self.mass*z_dot_max # Assume very small maximum force.

        ## Normalizing the state space model.
        Tzx = np.diag([z_max, z_dot_max])
        Tzu = np.diag([Fz_max])
        Tzy = np.diag([z_max])

        Az_norm = np.linalg.inv(Tzx) @ Az @ Tzx
        Bz_norm = np.linalg.inv(Tzx) @ Bz @ Tzu
        Cz_norm = np.linalg.inv(Tzy) @ Cz @ Tzx

        # Az_norm = Az
        # Bz_norm = Bz
        # Cz_norm = Cz

        Az_d_norm, Bz_d_norm, Cz_d_norm, Dz_d_norm, dt = signal.cont2discrete((Az_norm, Bz_norm, Cz_norm, 0), dt=self.dt,
                                                  method='zoh')
        
        Qz = np.diag([100.0, 10.0])
        Rz = 1
        Kz_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qz, Rz)

        # Denormalize the control gains.
        self.K_z = np.asarray(Tzu @ Kz_norm @ np.linalg.inv(Tzx))
        # self.K_z = np.asarray(Kz_norm)

        # Since X and Y have the same dynamics, we use the same gains.
        self.K_x = np.copy(self.K_z)
        self.K_y = np.copy(self.K_z)

        # self.K_x = np.zeros((1,2))
        # self.K_y = np.zeros((1,2))

        ### Z CONTROL DLQR DESIGN ###
        #############################

        #############################
        ### RP CONTROL DLQR DESIGN ###

        # Technically I design the controller with average inertia of both roll and pitch because the object
        # is almost symmetric.
        k_rot = 1e-6 # rotational damping term
        self.Iavg = 0.5*(self.rigid_body_dipole.mass_properties.I_bf[0,0] + self.rigid_body_dipole.mass_properties.I_bf[1,1])

        Ar = np.array([[0, 1], [0, -k_rot/self.Iavg]])
        Br = np.array([[0, 1/self.Iavg]]).T
        Cr = np.array([[1, 0]])

        # Let's use the normalized system close to the upright equilibrium.
        r_max = np.deg2rad(30) # More than 30 degrees is not logical to consider.
        r_dot_max = 5*z_max
        Tx_max = 5*self.Iavg*r_dot_max # Assume very small maximum force.

        ## Normalizing the state space model.
        Trx = np.diag([r_max, r_dot_max])
        Tru = np.diag([Tx_max])
        Try = np.diag([r_max])

        Ar_norm = np.linalg.inv(Trx) @ Ar @ Trx
        Br_norm = np.linalg.inv(Trx) @ Br @ Tru
        Cr_norm = np.linalg.inv(Try) @ Cr @ Trx

        Ar_d_norm, Br_d_norm, Cr_d_norm, Dr_d_norm, dt = signal.cont2discrete((Ar_norm, Br_norm, Cr_norm, 0), dt=1/self.control_rate,
                                                  method='zoh')
        
        Qr = np.diag([100.0, 10.0])
        Rr = 1
        Kr_norm, S, E = ct.dlqr(Ar_d_norm, Br_d_norm, Qr, Rr)

        # Denormalize the control gains.
        self.K_phi = np.asarray(Tru @ Kr_norm @ np.linalg.inv(Trx))
        self.K_theta = np.copy(self.K_phi)

        ### RP CONTROL DLQR DESIGN ###
        #############################

        self.mass = self.rigid_body_dipole.mass_properties.m
        # self.K_theta = np.array([[0.002621, 0.0007889]]) # Tuned for overdamped PD response.
        # self.K_phi = np.array([[0.002621, 0.0007889]]) # Tuned to include the external disc
        # self.K_theta = np.array([[0.007157, 0.0006609]]) # Tuned for overdamped PD response.
        # self.K_phi = np.array([[0.008001, 0.0007387]]) # Tuned to include the external disc
        # self.K_z = np.array([[9.0896, 1.3842]])

        rospy.loginfo(f"""Control gains for Fx: {self.K_x}, 
                                            Fy: {self.K_y},
                                            Fz: {self.K_z}, 
                                            Tx: {self.K_phi}, 
                                            Ty: {self.K_theta}""")

        self.control_gains_message = VectorStamped()
        self.control_gains_message.header.stamp = rospy.Time.now()

        self.diff_alpha_theta = self.control_rate
        self.diff_beta_theta = 0
        self.diff_alpha_phi = self.control_rate
        self.diff_beta_phi = 0
        self.diff_alpha_z = 1/self.dt
        self.diff_beta_z = 0
        self.diff_alpha_x = self.control_rate
        self.diff_beta_x = 0
        self.diff_alpha_y = self.control_rate
        self.diff_beta_y = 0

        self.control_gains_message.vector = np.concatenate(
            (self.K_x.flatten(), 
             self.K_y.flatten(),
             self.K_z.flatten(),
             self.K_phi.flatten(), 
             self.K_theta.flatten(), 
             np.array([self.diff_alpha_x,
                       self.diff_beta_x,
                       self.diff_alpha_y,
                       self.diff_beta_y,
                       self.diff_alpha_z,
                       self.diff_beta_z,
                       self.diff_alpha_phi,
                       self.diff_beta_phi,
                       self.diff_alpha_theta,
                       self.diff_beta_theta]))
        )

        self.phi_dot = 0.0
        self.theta_dot = 0.0
        self.z_dot = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0

        self.last_phi = 0.0
        self.last_theta = 0.0
        self.last_z = 0.0
        self.last_x = 0.0
        self.last_y = 0.0

        self.__first_reading = True
        self.metadata_msg.data = f"""
        Experiment metadata.
        Experiment type: Regulation experiment for 0 pose with position varying allocation matrix.
        Controlled States: Z, Roll (Alpha), Pitch (Beta)
        Calibration file: {self.calibration_file}
        Gains: {self.K_z.flatten(), self.K_phi.flatten(), self.K_theta.flatten()}
        Calibration type: Legacy yaml file
        Extra information: This run uses the solid fiberglass rod with some oil to reduce friction.
        """

    def simplified_allocation(self, tf_msg: TransformStamped, Tau_x: float, Tau_y: float, F_x: float, F_y: float, F_z: float) -> np.ndarray:
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg)
        dipole_position = geometry.numpy_translation_from_tf_msg(tf_msg)
        s_d = self.rigid_body_dipole.dipole_list[0].strength
        # dipole_moment = s_d*geometry.get_normal_vector_from_quaternion(dipole_quaternion)
        dipole_moment = s_d*np.array([0.0, 0.0, 1.0])
        if self.south_pole_up:
            dipole_moment = -dipole_moment
        M_tau = geometry.magnetic_interaction_field_to_torque(dipole_moment)
        # Rejecting the the Tz row since we are almost always nearly upright.
        # We have to reject one row from the skew symmetric portion because it
        # as one linearly dependent row which leads to very ill conditioned matrix.
        M_tau = M_tau[:2]
        M_f = geometry.magnetic_interaction_grad5_to_force(dipole_moment)
        A = self.mpem_model.getActuationMatrix(dipole_position)
        M = block_diag(M_tau, M_f)

        Tau_des = np.array([[Tau_x, Tau_y]]).T
        F_des = np.array([[F_x, F_y, F_z]]).T

        W_des = np.vstack((Tau_des, F_des))

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

        # Getting the current orientation and positions
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg)

        e_zyx = geometry.euler_zyx_from_quaternion(dipole_quaternion)
        e_xyz = geometry.euler_xyz_from_quaternion(dipole_quaternion)
        z_com = tf_msg.transform.translation.z
        x_com = tf_msg.transform.translation.x
        y_com = tf_msg.transform.translation.y

        ### Reference for tracking
        desired_quaternion = geometry.numpy_quaternion_from_tf_msg(self.last_reference_tf_msg)
        ref_e_xyz = geometry.euler_xyz_from_quaternion(desired_quaternion)
        ref_z = self.last_reference_tf_msg.transform.translation.z
        ref_x = self.last_reference_tf_msg.transform.translation.x
        ref_y = self.last_reference_tf_msg.transform.translation.y

        phi = e_xyz[0]
        theta = e_xyz[1]
        phi_ref = ref_e_xyz[0]
        theta_ref = ref_e_xyz[1]
        self.ref_actual_msg.vector = np.concatenate((np.array([x_com, ref_x, y_com, ref_y, z_com, ref_z]),
                                                     np.rad2deg(np.array([phi, phi_ref, theta, theta_ref]))))
        self.ref_actual_pub.publish(self.ref_actual_msg)

        if self.__first_reading:
            self.last_phi = phi
            self.last_theta = theta
            self.last_z = z_com
            self.last_x = x_com
            self.last_y = y_com
            self.__first_reading = False

        self.phi_dot = self.diff_alpha_phi*(phi - self.last_phi) + self.diff_beta_phi*self.phi_dot
        self.theta_dot = self.diff_alpha_theta*(theta - self.last_theta) + self.diff_beta_theta*self.theta_dot
        self.z_dot = self.diff_alpha_z*(z_com - self.last_z) + self.diff_beta_z*self.z_dot
        self.x_dot = self.diff_alpha_x*(x_com - self.last_x) + self.diff_beta_x*self.x_dot
        self.y_dot = self.diff_alpha_y*(y_com - self.last_y) + self.diff_beta_y*self.y_dot

        self.last_phi = phi
        self.last_theta = theta
        self.last_z = z_com
        self.last_x = x_com
        self.last_y = y_com

        x_phi = np.array([[phi, self.phi_dot]]).T
        x_theta = np.array([[theta, self.theta_dot]]).T
        x_z = np.array([[z_com, self.z_dot]]).T
        x_x = np.array([[x_com, self.x_dot]]).T
        x_y = np.array([[y_com, self.y_dot]]).T

        r_phi = np.array([[phi_ref, 0.0]]).T
        r_theta = np.array([[theta_ref, 0.0]]).T
        r_z = np.array([[ref_z, 0.0]]).T
        r_x = np.array([[ref_x, 0.0]]).T
        r_y = np.array([[ref_y, 0.0]]).T

        phi_error = x_phi - r_phi
        theta_error = x_theta - r_theta
        z_error = x_z - r_z
        x_error = x_x - r_x
        y_error = x_y - r_y

        u_phi = -self.K_phi @ phi_error
        u_theta = -self.K_theta @ theta_error
        u_z = -self.K_z @ z_error + self.mass*common.Constants.g
        u_x = -self.K_x @ x_error
        u_y = -self.K_y @ y_error

        Tau_x = u_phi[0, 0]
        Tau_y = u_theta[0, 0]
        F_z = u_z[0, 0]
        F_x = u_x[0, 0]
        F_y = u_y[0, 0]

        self.error_state_msg.vector = np.concatenate((x_error.flatten(), 
                                                      y_error.flatten(), 
                                                      z_error.flatten(), 
                                                      np.rad2deg(phi_error.flatten()), 
                                                      np.rad2deg(theta_error.flatten())))
        self.error_state_pub.publish(self.error_state_msg)
        self.control_input_message.vector = [Tau_x, Tau_y, F_x, F_y, F_z]

        com_wrench_des = np.array([Tau_x, Tau_y, 0.0, F_x, F_y, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])

        # Performing the simplified allocation for the two torques.
        des_currents = self.simplified_allocation(tf_msg, Tau_x=Tau_x, Tau_y=Tau_y, F_x=F_x, F_y=F_x, F_z=F_z)

        self.desired_currents_msg.des_currents_reg = des_currents

if __name__=="__main__":
    controller = SimpleCOMWrenchSingleDipoleController()
    rospy.spin()