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
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3, Quaternion
from tnb_mns_driver.msg import DesCurrentsReg

class SimpleCOMWrenchSingleDipoleController(ControlSessionNodeBase):

    def post_init(self):
        self.tfsub_callback_style_control_loop = True
        self.INITIAL_DESIRED_POSITION = np.array([0.0, 0.0, 7.0e-3])
        self.INITIAL_DESIRED_ORIENTATION_EXYZ = np.deg2rad(np.array([0.0, -20.0, 0.0]))

        self.control_rate = self.CONTROL_RATE
        self.dt = 1/self.control_rate
        self.publish_desired_com_wrenches = True
        self.publish_desired_dipole_wrenches = False
        
        self.control_input_publisher = rospy.Publisher("/xyz_rp_control_single_dipole/control_input",
                                                       VectorStamped, queue_size=1, latch=True)
        
        # Extra publishers which I wrote only in the post init and are not mandatory end with the shorthand pub.
        self.error_state_pub = rospy.Publisher("/xyz_rp_control_single_dipole/error_states",
                                                         VectorStamped, queue_size=1)
        
        self.ref_actual_pub = rospy.Publisher("/xyz_rp_control_single_dipole/ref_actual_values",
                                                         VectorStamped, queue_size=1)
        
        self.control_gain_publisher = rospy.Publisher("/xyz_rp_control_single_dipole/control_gains",
                                                      String, queue_size=1, latch=True)
            
        #############################
        ### Z CONTROL DLQR DESIGN ###
        self.mass = self.rigid_body_dipole.mass_properties.m
        self.k_lin_z = 1 # Friction damping parameter, to be tuned. Originally because of the rod.
        
        ## Continuous time state space model.
        Az = np.array([[0, 1], [0, -self.k_lin_z/self.mass]])
        Bz = np.array([[0, 1/self.mass]]).T
        Cz = np.array([[1, 0]])

        z_max = 5e-3 # 5 mm maximum z displacement.
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
        
        Qz = np.diag([10.0, 1.0])
        Rz = 1
        Kz_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qz, Rz)

        # Denormalize the control gains.
        # self.K_z = np.asarray(Tzu @ Kz_norm @ np.linalg.inv(Tzx))
        # self.Ki_z = 1
        # self.K_z = np.asarray(Kz_norm)
        # self.K_z = np.array([[7.447, 1.006]])
        # self.K_z = np.array([[1.078, 0.3326]])
        # self.K_z = np.array([[0.03033, 0.06008]])

        # Since X and Y have the same dynamics, we use the same gains.
        # self.K_x = np.copy(self.K_z)
        # self.K_y = np.copy(self.K_z)
        self.K_x = np.zeros((1,2))
        self.K_y = np.zeros((1,2))

        # self.K_x = np.zeros((1,2))
        # self.K_y = np.zeros((1,2))

        ### Z CONTROL DLQR DESIGN ###
        #############################

        #############################
        ### REDUCED ATTITUDE CONTROL DESIGN ###
        
        self.Ixx = self.rigid_body_dipole.mass_properties.I_bf[0,0]
        self.Iyy = self.rigid_body_dipole.mass_properties.I_bf[1,1]
        self.k_ra_p = 1.0
        self.K_ra_d = np.diag([1.0, 1.0])*1.0
        # self.k_ra_p = 0.0
        # self.K_ra_d = np.diag([1.0, 1.0])*0.0

        ### REDUCED ATTITUDE CONTROL DESIGN ###
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
RA Kp: {self.k_ra_p}, 
RA Kd: {self.K_ra_d}""")

        self.control_gains_message : String = String()

        self.diff_alpha_RA = self.control_rate
        self.diff_beta_RA = 0
        self.diff_alpha_z = 1/self.dt
        self.diff_beta_z = 0
        self.diff_alpha_x = self.control_rate
        self.diff_beta_x = 0
        self.diff_alpha_y = self.control_rate
        self.diff_beta_y = 0

        self.control_gains_message.data = f"""Control gains for Fx: {self.K_x}, 
Fy: {self.K_y},
Fz: {self.K_z}, 
RA Kp: {self.k_ra_p}, 
RA Kd: {self.K_ra_d},
Diff alpha RA: {self.diff_alpha_RA},
Diff beta RA: {self.diff_beta_RA},
Diff alpha z: {self.diff_alpha_z},
Diff beta z: {self.diff_beta_z},
Diff alpha x: {self.diff_alpha_x},
Diff beta x: {self.diff_beta_x},
Diff alpha y: {self.diff_alpha_y},
Diff beta y: {self.diff_beta_y},
LQR Parameters:
Qz: {Qz},
Rz: {Rz},
Tzx: {Tzx},
Tzu: {Tzu},
Tzy: {Tzy}"""

        self.ez_integral = 0.0
        self.z_dot = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.last_R = np.eye(3)

        self.R_dot = np.zeros((3,3))
        self.last_z = 0.0
        self.last_x = 0.0
        self.last_y = 0.0

        self.__first_reading = True
        self.metadata_msg.metadata.data = f"""
Experiment metadata.
Experiment type: Regulation experiment for 0 pose with position varying allocation matrix.
Controlled States: Z, Reduced Attitude (Body fixed Z axis in world frame)
Calibration file: {self.calibration_file}
Gains: {self.control_gains_message.data}
Calibration type: Legacy yaml file
Dipole object used: {self.rigid_body_dipole}
"""
        self.set_path_metadata(__file__)

    def local_torque_inertial_force_allocation(self, tf_msg: TransformStamped, Tau_x: float, Tau_y: float, F_x: float, F_y: float, F_z: float) -> np.ndarray:
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

        W_des = np.array([Tau_x, Tau_y, F_x, F_y, F_z])

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

        # Getting the current orientation and positions
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg.transform)

        e_xyz = geometry.euler_xyz_from_quaternion(dipole_quaternion)
        z_com = tf_msg.transform.translation.z
        x_com = tf_msg.transform.translation.x
        y_com = tf_msg.transform.translation.y

        ### Reference for tracking
        # The following explicit type casting is required by numba jit versions. np.linalg.norm
        # will fail for int arguments, in some cases this argument can be an int.
        desired_quaternion = np.asarray(geometry.numpy_quaternion_from_tf_msg(self.last_reference_tf_msg.transform), dtype=np.float64)
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
        Lambda_d = geometry.inertial_reduced_attitude_from_quaternion(desired_quaternion, b=np.array([0.0, 0.0, 1.0]))
        R = geometry.rotation_matrix_from_quaternion(dipole_quaternion)

        if self.__first_reading:
            self.last_R = R
            self.last_z = z_com
            self.last_x = x_com
            self.last_y = y_com
            self.__first_reading = False

        self.R_dot = self.diff_alpha_RA*(R - self.last_R) + self.diff_beta_RA*self.R_dot
        self.z_dot = self.diff_alpha_z*(z_com - self.last_z) + self.diff_beta_z*self.z_dot
        self.x_dot = self.diff_alpha_x*(x_com - self.last_x) + self.diff_beta_x*self.x_dot
        self.y_dot = self.diff_alpha_y*(y_com - self.last_y) + self.diff_beta_y*self.y_dot

        self.last_z = z_com
        self.last_x = x_com
        self.last_y = y_com
        self.last_R = R

        x_z = np.array([[z_com, self.z_dot]]).T
        x_x = np.array([[x_com, self.x_dot]]).T
        x_y = np.array([[y_com, self.y_dot]]).T

        r_z = np.array([[ref_z, 0.0]]).T
        r_x = np.array([[ref_x, 0.0]]).T
        r_y = np.array([[ref_y, 0.0]]).T

        z_error = x_z - r_z # This error is in meters. We need it to be big enough in mm.
        x_error = x_x - r_x
        y_error = x_y - r_y
        self.ez_integral += z_error[0, 0]*self.dt

        u_z = -self.K_z @ z_error + self.mass*common.Constants.g # Gravity compensation
        # u_z = -self.K_z @ z_error - self.Ki_z * self.ez_integral + self.mass*common.Constants.g # Gravity compensation
        u_x = -self.K_x @ x_error
        u_y = -self.K_y @ y_error

        F_z = u_z[0, 0] - 0.100
        F_x = u_x[0, 0]
        F_y = u_y[0, 0]

        omega = geometry.angular_velocity_body_frame_from_rotation_matrix(R, self.R_dot)
        E = np.hstack((np.eye(2), np.zeros((2, 1)))) # Just selects x and y components from a 3x1 vector
        omega_tilde = E @ omega
        Lambda = geometry.inertial_reduced_attitude_from_rotation_matrix(R, b=np.array([0.0, 0.0, 1.0]))
        reduced_attitude_error = 1 - np.dot(Lambda_d, Lambda)
        u_RA = -self.K_ra_d @ omega_tilde + self.k_ra_p * E @ R.T @ np.cross(Lambda, Lambda_d)
        # Local frame torque allocation
        Tau_x = u_RA[0]*self.Ixx
        Tau_y = u_RA[1]*self.Iyy

        self.error_state_msg.vector = np.concatenate((x_error.flatten(), 
                                                      y_error.flatten(), 
                                                      z_error.flatten(), 
                                                      [reduced_attitude_error],
                                                      omega_tilde.flatten()))
        self.error_state_pub.publish(self.error_state_msg)
        w_des = np.array([Tau_x, Tau_y, F_x, F_y, F_z])
        self.control_input_message.vector = w_des

        com_wrench_des = np.array([Tau_x, Tau_y, 0.0, F_x, F_y, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])

        # Performing the simplified allocation for the two torques.
        des_currents = self.indiv_magnet_contribution_allocation(tf_msg, w_des)

        self.desired_currents_msg.des_currents_reg = des_currents

if __name__=="__main__":
    controller = SimpleCOMWrenchSingleDipoleController()
    rospy.spin()