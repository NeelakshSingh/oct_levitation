import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.geometry_jit as geometry
import oct_levitation.common as common

from oct_levitation.control_node import ControlSessionNodeBase
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped, Vector3
from tnb_mns_driver.msg import DesCurrentsReg

def remove_extra_spaces(string):
    """
    Remove extra spaces from a string.
    """
    return ' '.join(string.split())


class SimpleCOMWrenchSingleDipoleController(ControlSessionNodeBase):

    def post_init(self):
        self.tfsub_callback_style_control_loop = True
        self.INITIAL_DESIRED_POSITION = np.array([0.0, 0.0, 2e-3]) # for horizontal attachment
        self.INITIAL_DESIRED_ORIENTATION_EXYZ = np.deg2rad(np.array([0.0, 0.0, 0.0]))

        self.control_rate = self.CONTROL_RATE
        self.dt = 1/self.control_rate
        self.publish_desired_com_wrenches = True
        self.publish_desired_dipole_wrenches = False
        
        self.control_gain_publisher = rospy.Publisher("/xyz_rp_control_single_dipole/control_gains",
                                                      String, queue_size=1, latch=True)
            
        #############################
        ### Z CONTROL DLQR DESIGN ###
        self.mass = self.rigid_body_dipole.mass_properties.m
        self.k_lin_z = 0 # Friction damping parameter, to be tuned. Originally because of the rod.
        
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
        
        Qz = np.diag([1.0, 1.0])
        Rz = 0.1
        Kz_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qz, Rz)

        # Denormalize the control gains.
        self.K_z = np.asarray(Tzu @ Kz_norm @ np.linalg.inv(Tzx))
        # self.Ki_z = 1
        # self.K_z = np.array([[5422, 557.6]]) # I40 Tuned for input disturbance offset of 0.18mm for 100gms of force as a step disturbance and 60 deg PM at 28.1 rad (37ms delay tolerance). Assuming 24Hz ECB 3dB bandwidth.
        # self.K_z = np.array([[957.4, 201.8]]) # I40 Tuned for input disturbance offset of 1.2mm for 100gms of force as a step disturbance and 60 deg PM at 9.28 rad (37ms delay tolerance). Assuming 24Hz ECB 3dB bandwidth.
        # self.K_z = np.array([[125, 41.99]]) # I40 Tuned for input disturbance offset of 8mm for 100gms of force step disturbance, 43 deg phase margin at 2.84 rad. The goal is to bring the currents down to a manageable level.
        # self.K_z = np.array([[125, 4.199]]) # I40 Tuned for input disturbance offset of 8mm for 100gms of force step disturbance, 43 deg phase margin at 2.84 rad. The goal is to bring the currents down to a manageable level.
        # self.K_z = np.array([[69.23, 37.39]]) # I40 Tuned for input disturbance offset of 1.4mm for 10gms of force step disturbance, 4sec settling time for 1mm offset, 60 deg phase margin at 2.07 rad. 24Hz ECB 3dB bandwidth. 
        # self.K_z = np.array([[33.69, 31.02]]) # I40 Tuned for input disturbance offset of 3mm for 10gms of force step disturbance, 5.8sec settling time for 1mm step setpoint, 55 geg phase margin at 1.61 rad. 24Hz ECB 3dB bandwidth. 
        # self.K_z = np.array([[33.69, 3.975]]) # I40 Tuned for input disturbance offset of 3mm for 10gms of force step disturbance, 5.8sec settling time for 1mm step setpoint, 55 geg phase margin at 1.61 rad. 24Hz ECB 3dB bandwidth. 
        # self.K_z = np.array([[2.083, 7.957]]) # I40 N52 10x3, tuned for 60deg phase margin.
        # self.K_z = np.array([[4.359, 9.641]]) # I40 N52 10x3, tuned for 60deg phase margin.
        # self.K_z = np.array([[0.3438, 2.3]]) # I40 N52 10x3, tuned for less D gain to avoid noise amplification.
        # self.K_z = np.array([[7.229, 4.923]]) # I40 N52 10x3, tuned for high P gain to have fast enough response so as to not let D and the amplified noise drive the system.
        # self.K_z = np.array([[7.229, 4.923]]) # I40 N52 10x3, tuned for high P gain to have fast enough response so as to not let D and the amplified noise drive the system.
        # self.K_z = np.array([[7.229, 4.923]]) # I40 N52 10x3, tuned for high P gain to have fast enough response so as to not let D and the amplified noise drive the system.
        # self.K_z = np.array([[0.002613, 1.758]]) # I40 6 N52 10x3, tuned with delay of 10ms. 0.1 rad/s crossover.
        # self.K_z = np.array([[0.6281, 6.051]]) # I40 6 N52 10x3, tuned with delay of 10ms. 0.288 rad/s crossover.
        

        # Since X and Y have the same dynamics, we use the same gains.
        self.K_x = np.copy(self.K_z)
        self.K_y = np.copy(self.K_z)
        
        # self.K_x = np.zeros((1,2))
        # self.K_y = np.zeros((1,2))

        ### Z CONTROL DLQR DESIGN ###
        #############################

        #############################
        ### REDUCED ATTITUDE CONTROL DESIGN ###
        
        self.Ixx = self.rigid_body_dipole.mass_properties.principal_inertia_properties.Px
        self.Iyy = self.rigid_body_dipole.mass_properties.principal_inertia_properties.Py
        self.k_ra_p = 30
        self.K_ra_d = np.diag([1.0, 1.0])*15
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

        rospy.loginfo(remove_extra_spaces(f"""Control gains for Fx: {self.K_x}, 
        Fy: {self.K_y},
        Fz: {self.K_z}, 
        RA Kp: {self.k_ra_p}, 
        RA Kd: {self.K_ra_d}"""))

        self.control_gains_message : String = String()

        self.diff_alpha_RA = self.control_rate
        self.diff_beta_RA = 0
        self.diff_alpha_z = 1/self.dt
        self.diff_beta_z = 0
        self.diff_alpha_x = self.control_rate
        self.diff_beta_x = 0
        self.diff_alpha_y = self.control_rate
        self.diff_beta_y = 0

        self.control_gains_message.data = remove_extra_spaces(f"""Control gains for Fx: {self.K_x}, 
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
        Tzy: {Tzy}""")

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
        self.metadata_msg.metadata.data = remove_extra_spaces(f"""
        Experiment metadata.
        Experiment type: Regulation experiment for 0 pose with position varying allocation matrix.
        Controlled States: Z, Reduced Attitude (Body fixed Z axis in world frame)
        Calibration file: {self.calibration_file}
        Gains: {self.control_gains_message.data}
        Calibration type: Legacy yaml file
        Dipole object used: {self.rigid_body_dipole.name}
        """)
        self.set_path_metadata(__file__)

        self.E = np.hstack((np.eye(2), np.zeros((2, 1)))) # Just selects x and y components from a 3x1 vector
        
    def callback_control_logic(self, 
                               position : np.ndarray, 
                               quaternion: np.ndarray,
                               rpy: np.ndarray,
                               linear_velocity: np.ndarray = None, 
                               angular_velocity: np.ndarray = None, 
                               sft_coeff: float = 1.0):
        self.desired_currents_msg = DesCurrentsReg() # Empty message
        self.com_wrench_msg = WrenchStamped() # Empty message

        self.com_wrench_msg.header.stamp = rospy.Time.now()

        x_com = position[0]
        y_com = position[1]
        z_com = position[2]

        ### Reference for tracking
        # The following explicit type casting is required by numba jit versions. np.linalg.norm
        # will fail for int arguments, in some cases this argument can be an int.
        desired_quaternion = np.asarray(geometry.numpy_quaternion_from_tf_msg(self.last_reference_tf_msg.transform), dtype=np.float64)

        ref_z = self.last_reference_tf_msg.transform.translation.z
        ref_x = self.last_reference_tf_msg.transform.translation.x
        ref_y = self.last_reference_tf_msg.transform.translation.y

        Lambda_d = geometry.inertial_reduced_attitude_from_quaternion(desired_quaternion, b=np.array([0.0, 0.0, 1.0]))
        R = geometry.rotation_matrix_from_quaternion(quaternion)

        if linear_velocity is None or angular_velocity is None:
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
            omega = geometry.angular_velocity_body_frame_from_rotation_matrix(R, self.R_dot)

            self.last_z = z_com
            self.last_x = x_com
            self.last_y = y_com
            self.last_R = R
        else:
            self.x_dot, self.y_dot, self.z_dot = linear_velocity
            omega = angular_velocity

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
        u_x = -self.K_x @ x_error
        u_y = -self.K_y @ y_error

        F_z = u_z[0, 0] * sft_coeff
        F_x = u_x[0, 0]
        F_y = u_y[0, 0]
        
        omega_tilde = self.E @ omega
        Lambda = geometry.inertial_reduced_attitude_from_rotation_matrix(R, b=np.array([0.0, 0.0, 1.0]))
        reduced_attitude_error = 1 - np.dot(Lambda_d, Lambda)
        u_RA = -self.K_ra_d @ omega_tilde + self.k_ra_p * self.E @ R.T @ geometry.numba_cross(Lambda, Lambda_d)
        # Local frame torque allocation
        Tau_x = u_RA[0]*self.Ixx
        Tau_y = u_RA[1]*self.Iyy

        w_des = np.array([Tau_x, Tau_y, F_x, F_y, F_z])

        com_wrench_des = np.array([Tau_x, Tau_y, 0.0, F_x, F_y, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])

        # Performing the simplified allocation for the two torques.
        des_currents = self.five_dof_wrench_allocation_single_dipole(position, quaternion, w_des)

        self.desired_currents_msg.des_currents_reg = des_currents

if __name__=="__main__":
    controller = SimpleCOMWrenchSingleDipoleController()
    rospy.spin()