import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.geometry_jit as geometry
import oct_levitation.common as common
import oct_levitation.numerical as numerical

from oct_levitation.control_node import ControlSessionNodeBase
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped, Vector3, TwistStamped
from tnb_mns_driver.msg import DesCurrentsReg

def remove_extra_spaces(string):
    """
    Remove extra spaces from a string.
    """
    return ' '.join(string.split())


class SimpleCOMWrenchSingleDipoleController(ControlSessionNodeBase):

    def post_init(self):
        self.tfsub_callback_style_control_loop = True
        self.INITIAL_DESIRED_POSITION = np.array([0.0, 0.0, 0.0e-3]) # for horizontal attachment
        self.INITIAL_DESIRED_ORIENTATION_EXYZ = np.deg2rad(np.array([0.0, 0.0, 0.0]))

        self.control_rate = self.CONTROL_RATE
        self.dt = 1/self.control_rate
        self.publish_desired_com_wrenches = True
        self.publish_desired_dipole_wrenches = False
        
        self.control_gain_publisher = rospy.Publisher("/xyz_rp_control_single_dipole/control_gains",
                                                      String, queue_size=1, latch=True)
        
        self.estimated_velocity_publisher = rospy.Publisher("control_session/finite_difference_velocity", TwistStamped, queue_size=1)
            
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
        
        Qz = np.diag([10.0, 1.0])
        Rz = 0.1
        Kz_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qz, Rz)

        # Denormalize the control gains.
        self.K_z = np.asarray(Tzu @ Kz_norm @ np.linalg.inv(Tzx))      

        # Since X and Y have the same dynamics, we use the same gains.
        self.K_x = np.copy(self.K_z)
        self.K_y = np.copy(self.K_z)

        ### Z CONTROL DLQR DESIGN ###
        #############################

        #############################
        ### REDUCED ATTITUDE CONTROL DESIGN ###
        
        self.Ixx = self.rigid_body_dipole.mass_properties.principal_inertia_properties.Px
        self.Iyy = self.rigid_body_dipole.mass_properties.principal_inertia_properties.Py
        self.k_ra_p = 150
        self.K_ra_d = np.diag([1.0, 1.0])*70

        ### REDUCED ATTITUDE CONTROL DESIGN ###
        #############################

        ##############################
        ### INTEGRAL ACTION DESIGN TO COMPENSATE FOR SS ERRORS ###

        self.Ki_lin = 1.0
        self.Ki_ang = 60

        integrator_params = self.INTEGRATOR_PARAMS

        self.use_integrator = integrator_params["use_integrator"]
        self.switch_off_integrator_on_convergence = integrator_params["switch_off_on_convergence"]
        self.__integrator_enable = np.asarray(integrator_params['integrator_enable_rpxyz'], dtype=int)
        self.__convergence_time = np.zeros(5)
        self.__integrator_converged = False
        self.__integrator_convergence_check_time = integrator_params['convergence_check_time']
        self.integrator_start_time = integrator_params['start_time']
        self.integrator_end_time = integrator_params['end_time']
        self.__pos_error_tol = integrator_params['position_error_tol']
        self.__att_error_tol = integrator_params['reduced_attitude_error_tol']
        self.disturbance_rpxyz = np.zeros(5)
        self.experiment_start_time = None

        self.trajectory_start_time = self.TRAJECTORY_PARAMS['start_time']

        ### INTEGRAL ACTION DESIGN TO COMPENSATE FOR SS ERRORS ###
        ##############################


        rospy.loginfo(remove_extra_spaces(f"""Control gains for Fx: {self.K_x}, 
        Fy: {self.K_y},
        Fz: {self.K_z}, 
        RA Kp: {self.k_ra_p}, 
        RA Kd: {self.K_ra_d}"""))

        self.control_gains_message : String = String()

        self.diff_alpha_RA = 1/self.dt
        self.diff_beta_RA = 0
        self.diff_alpha_z = 1/self.dt
        self.diff_beta_z = 0
        self.diff_alpha_x = 1/self.dt
        self.diff_beta_x = 0
        self.diff_alpha_y = 1/self.dt
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
        self.velocity_msg = TwistStamped() # Empty message
        self.velocity_msg.header.stamp = rospy.Time.now()
        self.velocity_msg.header.frame_id = self.rigid_body_dipole.name
        if self.experiment_start_time is None:
            self.experiment_start_time = rospy.Time.now().to_sec()
        time_elapsed = rospy.Time.now().to_sec() - self.experiment_start_time

        self.com_wrench_msg.header.stamp = rospy.Time.now()

        x_com = position[0]
        y_com = position[1]
        z_com = position[2]

        ### Reference for tracking
        # The following explicit type casting is required by numba jit versions. np.linalg.norm
        # will fail for int arguments, in some cases this argument can be an int.
        desired_quaternion = np.asarray(geometry.numpy_quaternion_from_tf_msg(self.last_reference_tf_msg.transform), dtype=np.float64)

        # ref_z = self.last_reference_tf_msg.transform.translation.z
        # ref_x = self.last_reference_tf_msg.transform.translation.x
        # ref_y = self.last_reference_tf_msg.transform.translation.y

        if time_elapsed > self.trajectory_start_time:
            f = 0.2
            ref_z = (2.0*np.sin(2*np.pi*time_elapsed*f) + 4.0)*1e-3
            ref_z_dot = 2*np.pi*4.0*np.cos(2*np.pi*time_elapsed*f)*1e-3
        else:
            ref_z = self.INITIAL_DESIRED_POSITION[2]
            ref_z_dot = 0.0

        ref_x = 0.0
        ref_y = 0.0

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

            self.x_dot, self.y_dot, self.z_dot = numerical.numba_clip(np.array([self.x_dot, self.y_dot, self.z_dot]), -self.MAX_LINEAR_VELOCITY, self.MAX_LINEAR_VELOCITY)
            omega = numerical.numba_clip(omega, -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)

            self.last_z = z_com
            self.last_x = x_com
            self.last_y = y_com
            self.last_R = R
        else:
            self.x_dot, self.y_dot, self.z_dot = linear_velocity
            omega = angular_velocity

        self.velocity_msg.twist.linear = Vector3(self.x_dot, self.y_dot, self.z_dot)
        self.velocity_msg.twist.angular = Vector3(omega[0], omega[1], omega[2])
        self.estimated_velocity_publisher.publish(self.velocity_msg)

        x_z = np.array([[z_com, self.z_dot]]).T
        x_x = np.array([[x_com, self.x_dot]]).T
        x_y = np.array([[y_com, self.y_dot]]).T

        r_z = np.array([[ref_z, ref_z_dot]]).T
        r_x = np.array([[ref_x, 0.0]]).T
        r_y = np.array([[ref_y, 0.0]]).T

        z_error = r_z - x_z # This error is in meters. We need it to be big enough in mm.
        x_error = r_x - x_x
        y_error = r_y - x_y

        omega_tilde = self.E @ omega
        Lambda = geometry.inertial_reduced_attitude_from_rotation_matrix(R, b=np.array([0.0, 0.0, 1.0]))
        reduced_attitude_error = self.E @ R.T @ geometry.numba_cross(Lambda, Lambda_d)

        # Calculating the integral action.
        if self.use_integrator:
            if time_elapsed > self.integrator_start_time and time_elapsed < self.integrator_end_time:
                self.disturbance_rpxyz[4] += self.Ki_lin * z_error[0, 0] * self.dt * self.__integrator_enable[4]
                self.disturbance_rpxyz[3] += self.Ki_lin * y_error[0, 0] * self.dt * self.__integrator_enable[3]
                self.disturbance_rpxyz[2] += self.Ki_lin * x_error[0, 0] * self.dt * self.__integrator_enable[2]
                self.disturbance_rpxyz[0] += self.Ki_ang * reduced_attitude_error[0] * self.dt * self.__integrator_enable[0]
                self.disturbance_rpxyz[1] += self.Ki_ang * reduced_attitude_error[1] * self.dt * self.__integrator_enable[1]
                
                if self.switch_off_integrator_on_convergence and not self.__integrator_converged:
                    if abs(z_error[0, 0]) < self.__pos_error_tol and self.__integrator_enable[4]:
                        self.__convergence_time[4] += self.dt
                        if self.__convergence_time[4] > self.__integrator_convergence_check_time:
                            rospy.logwarn_once("Z convergence achieved, stopping Z integrator.")
                            self.__integrator_enable[4] = 0
                    if abs(x_error[0, 0]) < self.__pos_error_tol and self.__integrator_enable[2]:
                        self.__convergence_time[2] += self.dt
                        if self.__convergence_time[2] > self.__integrator_convergence_check_time:
                            rospy.logwarn_once("X convergence achieved, stopping X integrator.")
                            self.__integrator_enable[2] = 0
                    if abs(y_error[0, 0]) < self.__pos_error_tol and self.__integrator_enable[3]:
                        self.__convergence_time[3] += self.dt
                        if self.__convergence_time[3] > self.__integrator_convergence_check_time:
                            rospy.logwarn_once("Y convergence achieved, stopping Y integrator.")
                            self.__integrator_enable[3] = 0
                    if abs(reduced_attitude_error[0]) < self.__att_error_tol and self.__integrator_enable[0]:
                        self.__convergence_time[0] += self.dt
                        if self.__convergence_time[0] > self.__integrator_convergence_check_time:
                            rospy.logwarn_once("Reduced attitude Nx convergence achieved, stopping RA integrator.")
                            self.__integrator_enable[0] = 0
                    if abs(reduced_attitude_error[1]) < self.__att_error_tol and self.__integrator_enable[1]:
                        self.__convergence_time[1] += self.dt
                        if self.__convergence_time[1] > self.__integrator_convergence_check_time:
                            rospy.logwarn_once("Reduced attitude Ny convergence achieved, stopping RA integrator.")
                            self.__integrator_enable[1] = 0

                    if np.sum(self.__integrator_enable) == 0:
                        self.__integrator_converged = True
                        rospy.logwarn_once("All convergence achieved.")

        u_z = self.K_z @ z_error + self.mass*common.Constants.g + self.disturbance_rpxyz[4] # Gravity compensation
        u_x = self.K_x @ x_error + self.disturbance_rpxyz[3]
        u_y = self.K_y @ y_error + self.disturbance_rpxyz[2]

        F_z = u_z[0, 0] * sft_coeff
        F_x = u_x[0, 0]
        F_y = u_y[0, 0]
        
        u_RA = -self.K_ra_d @ omega_tilde + self.k_ra_p * reduced_attitude_error + self.disturbance_rpxyz[:2]
        # Local frame torque allocation
        Tau_x = u_RA[0]*self.Ixx
        Tau_y = u_RA[1]*self.Iyy

        w_des = np.array([Tau_x, Tau_y, F_x, F_y, F_z])

        com_wrench_des = np.array([Tau_x, Tau_y, 0.0, F_x, F_y, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])

        # Performing the simplified allocation instead of the summation of individual magnet contributions.
        des_currents = self.five_dof_wrench_allocation_single_dipole(position, quaternion, w_des)

        self.desired_currents_msg.des_currents_reg = des_currents

if __name__=="__main__":
    controller = SimpleCOMWrenchSingleDipoleController()
    rospy.spin()