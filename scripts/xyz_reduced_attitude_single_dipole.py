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
        self.INITIAL_DESIRED_POSITION = np.array([-5.0, 5.0, 10.0])*1.0e-3
        self.INITIAL_DESIRED_ORIENTATION_EXYZ = np.deg2rad(np.array([30.0, 30.0, 0.0]))

        self.control_rate = self.CONTROL_RATE
        self.dt = 1/self.control_rate
        self.publish_desired_com_wrenches = True
        self.publish_desired_dipole_wrenches = False
        
        self.control_gain_publisher = rospy.Publisher("/xyz_rp_control_single_dipole/control_gains",
                                                      String, queue_size=1, latch=True)
        
        self.estimated_velocity_publisher = rospy.Publisher("control_session/finite_difference_velocity", TwistStamped, queue_size=1)
            
        #############################
        ### Z CONTROL DLQR DESIGN (Same for X and Y) ###
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

        Az_d_norm, Bz_d_norm, Cz_d_norm, Dz_d_norm, dt = signal.cont2discrete((Az_norm, Bz_norm, Cz_norm, 0), dt=self.dt,
                                                method='zoh')

        #### Bronzefill 27gms with integrator compensation.
        Qz = np.diag([30.0, 10.0]) # This tuning can be used for X and Z axis, but slight noise amplification will be present.
        Qx = np.diag([22.0, 7.0]) # Different tuning for X axis because it seemed to have a different response due to some unmodelled effect.
        Qy = np.diag([15.0, 7.0]) # Different tuning for Y axis because it seemed to have a different response due to some unmodelled effect.
        # self.f_z_ff = 0.016871079683868213 # The extra feedforward force computed from the integrator.
        self.f_z_ff = 0.0 # The extra feedforward force computed from the integrator.

        #### Greentec Pro Do80 Di67
        # Qz = np.diag([30.0, 10.0]) # This tuning can be used for X and Z axis, but slight noise amplification will be present.
        # Qx = np.diag([25.0, 7.0]) # Different tuning for X axis because it seemed to have a different response due to some unmodelled effect.
        # Qy = np.diag([15.0, 7.0]) # Different tuning for Y axis because it seemed to have a different response due to some unmodelled effect.
        # self.f_z_ff = 0.0 # The extra feedforward force computed from the integrator.

        Rz = 0.1
        Ry = 0.1
        Rx = 0.1
        Kz_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qz, Rz)
        Ky_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qy, Ry)
        Kx_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qx, Rx)

        # Denormalize the control gains.
        self.K_z = np.asarray(Tzu @ Kz_norm @ np.linalg.inv(Tzx))      
        self.K_y = np.asarray(Tzu @ Ky_norm @ np.linalg.inv(Tzx))
        self.K_x = np.asarray(Tzu @ Kx_norm @ np.linalg.inv(Tzx))


        ### Z CONTROL DLQR DESIGN ###
        #############################

        #############################
        ### REDUCED ATTITUDE CONTROL DESIGN ###
        
        self.Ixx = self.rigid_body_dipole.mass_properties.principal_inertia_properties.Px
        self.Iyy = self.rigid_body_dipole.mass_properties.principal_inertia_properties.Py

        #### Bronzefill 27gms with integrator compensation.
        # scale = 1.65 # Almost starts noise amplification at this value.
        scale = 1.35
        self.k_ra_p = 350 * scale
        self.K_ra_d = np.diag([1.0, 1.0])*80 * scale

        #### Greentec Pro Do80 Di67
        # scale = 1.45
        # self.k_ra_p = 350 * scale
        # self.K_ra_d = np.diag([1.0, 1.0])*80 * scale

        ### REDUCED ATTITUDE CONTROL DESIGN ###
        #############################

        ##############################
        ### INTEGRAL ACTION DESIGN TO COMPENSATE FOR SS ERRORS ###

        ### Bronzefill 27gms
        self.Ki_lin_x = 10.0
        self.Ki_lin_y = 10.0
        self.Ki_lin_z = 10.0
        self.Ki_ang = 100.0

        ### Greentec Pro Do80 Di67
        # self.Ki_lin_x = 10.0
        # self.Ki_lin_y = 10.0
        # self.Ki_lin_z = 10.0
        # self.Ki_ang = 100.0

        integrator_params = self.INTEGRATOR_PARAMS

        self.use_integrator = integrator_params["use_integrator"]
        self.switch_off_integrator_on_convergence = integrator_params["switch_off_on_convergence"]
        self.__integrator_enable = np.asarray(integrator_params['integrator_enable_rpxyz'], dtype=int)
        self.__indiv_integrator_converge_state = np.ones(5, dtype=bool)
        self.__indiv_integrator_converge_state[self.__integrator_enable == 1] = False
        self.__convergence_time = np.zeros(5)
        self.__integrator_converged = False
        self.__integrator_convergence_check_time = integrator_params['convergence_check_time']
        self.integrator_start_time = integrator_params['start_time']
        self.integrator_end_time = integrator_params['end_time']
        self.__integrator_check_convergence = integrator_params['check_convergence']
        self.__pos_error_tol = integrator_params['position_error_tol']
        self.__att_error_tol = integrator_params['reduced_attitude_error_tol']
        self.disturbance_rpxyz = np.zeros(5)

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
        self.pause_trajectory_tracking = False
        
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
        

        self.com_wrench_msg.header.stamp = rospy.Time.now()

        x_com = position[0]
        y_com = position[1]
        z_com = position[2]

        ### Reference for tracking
        # The following explicit type casting is required by numba jit versions. np.linalg.norm
        # will fail for int arguments, in some cases this argument can be an int.
        desired_quaternion = self.LAST_REFERENCE_TRAJECTORY_POINT[2]
        ref_x, ref_y, ref_z = self.LAST_REFERENCE_TRAJECTORY_POINT[0]
        ref_x_dot, ref_y_dot, ref_z_dot = self.LAST_REFERENCE_TRAJECTORY_POINT[1]
        ref_omega_inertial = self.LAST_REFERENCE_TRAJECTORY_POINT[3] # Assumed to be in the body frame.

        Lambda_d = geometry.inertial_reduced_attitude_from_quaternion(desired_quaternion, b=np.array([0.0, 0.0, 1.0]))
        R = geometry.rotation_matrix_from_quaternion(quaternion)

        ref_omega_local = R @ ref_omega_inertial

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
        r_x = np.array([[ref_x, ref_x_dot]]).T
        r_y = np.array([[ref_y, ref_y_dot]]).T

        z_error = r_z - x_z # This error is in meters. We need it to be big enough in mm.
        x_error = r_x - x_x
        y_error = r_y - x_y

        omega_tilde = self.E @ omega
        ref_omega_tilde = self.E @ ref_omega_local
        Lambda = geometry.inertial_reduced_attitude_from_rotation_matrix(R, b=np.array([0.0, 0.0, 1.0]))
        reduced_attitude_error = self.E @ R.T @ geometry.numba_cross(Lambda, Lambda_d)

        # Calculating the integral action.
        if self.use_integrator:
            if self.time_elapsed > self.integrator_start_time and self.time_elapsed < self.integrator_end_time:
                self.disturbance_rpxyz[4] += self.Ki_lin_x * z_error[0, 0] * self.dt * self.__integrator_enable[4]
                self.disturbance_rpxyz[3] += self.Ki_lin_y * y_error[0, 0] * self.dt * self.__integrator_enable[3]
                self.disturbance_rpxyz[2] += self.Ki_lin_z * x_error[0, 0] * self.dt * self.__integrator_enable[2]
                self.disturbance_rpxyz[0] += self.Ki_ang * reduced_attitude_error[0] * self.dt * self.__integrator_enable[0]
                self.disturbance_rpxyz[1] += self.Ki_ang * reduced_attitude_error[1] * self.dt * self.__integrator_enable[1]
                
                if not self.__integrator_converged and self.__integrator_check_convergence:
                    if abs(z_error[0, 0]) < self.__pos_error_tol and self.__integrator_enable[4]:
                        self.__convergence_time[4] += self.dt
                        if self.__convergence_time[4] > self.__integrator_convergence_check_time:
                            rospy.logwarn_once(f"Z convergence achieved. Compensation force: {self.disturbance_rpxyz[4]}")
                            self.__indiv_integrator_converge_state[4] = True
                            if self.switch_off_integrator_on_convergence:
                                rospy.logwarn_once("Stopping Z integrator.")
                                self.__integrator_enable[4] = 0
                    if abs(x_error[0, 0]) < self.__pos_error_tol and self.__integrator_enable[2]:
                        self.__convergence_time[2] += self.dt
                        if self.__convergence_time[2] > self.__integrator_convergence_check_time:
                            self.__indiv_integrator_converge_state[2] = True
                            rospy.logwarn_once(f"X convergence achieved. Compensation force: {self.disturbance_rpxyz[2]}")
                            if self.switch_off_integrator_on_convergence:
                                rospy.logwarn_once("Stopping X integrator.")
                                self.__integrator_enable[2] = 0
                    if abs(y_error[0, 0]) < self.__pos_error_tol and self.__integrator_enable[3]:
                        self.__convergence_time[3] += self.dt
                        if self.__convergence_time[3] > self.__integrator_convergence_check_time:
                            self.__indiv_integrator_converge_state[3] = True
                            rospy.logwarn_once(f"Y convergence achieved. Compensation force: {self.disturbance_rpxyz[3]}")
                            if self.switch_off_integrator_on_convergence:
                                rospy.logwarn_once("Stopping Y integrator.")
                                self.__integrator_enable[3] = 0
                    if abs(reduced_attitude_error[0]) < self.__att_error_tol and self.__integrator_enable[0]:
                        self.__convergence_time[0] += self.dt
                        if self.__convergence_time[0] > self.__integrator_convergence_check_time:
                            self.__indiv_integrator_converge_state[0] = True
                            rospy.logwarn_once(f"Reduced attitude Nx convergence achieved. Compensation torque: {self.disturbance_rpxyz[0]}")
                            if self.switch_off_integrator_on_convergence:
                                rospy.logwarn_once("Stopping RA x integrator.")
                                self.__integrator_enable[0] = 0
                    if abs(reduced_attitude_error[1]) < self.__att_error_tol and self.__integrator_enable[1]:
                        self.__convergence_time[1] += self.dt
                        if self.__convergence_time[1] > self.__integrator_convergence_check_time:
                            self.__indiv_integrator_converge_state[1] = True
                            rospy.logwarn_once(f"Reduced attitude Ny convergence achieved. Compensation torque: {self.disturbance_rpxyz[1]}")
                            if self.switch_off_integrator_on_convergence:
                                rospy.logwarn_once("Stopping RA y integrator.")
                                self.__integrator_enable[1] = 0

                    if np.all(self.__indiv_integrator_converge_state):
                        self.__integrator_converged = True
                        rospy.logwarn_once("All convergence achieved.")

        u_x = self.K_x @ x_error + self.disturbance_rpxyz[2]
        u_y = self.K_y @ y_error + self.disturbance_rpxyz[3]
        u_z = self.K_z @ z_error + self.mass*common.Constants.g + self.disturbance_rpxyz[4] + self.f_z_ff # Gravity compensation
        u_RA = -self.K_ra_d @ omega_tilde + self.k_ra_p * reduced_attitude_error + self.disturbance_rpxyz[:2]

        F_z = u_z[0, 0] * sft_coeff
        F_x = u_x[0, 0]
        F_y = u_y[0, 0]
        
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