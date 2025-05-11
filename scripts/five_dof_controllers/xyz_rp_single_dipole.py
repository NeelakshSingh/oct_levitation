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
        ## Continuous time state space model.
        Az = np.array([[0, 1], [0, 0]])
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

        Az_d_norm, Bz_d_norm, _, _, _ = signal.cont2discrete((Az_norm, Bz_norm, Cz_norm, 0), dt=self.dt,
                                                method='zoh')
        
        Qz = np.diag([1.0, 1.0])
        Rz = 0.1
        Kz_norm, S, E = ct.dlqr(Az_d_norm, Bz_d_norm, Qz, Rz)

        # Denormalize the control gains.
        self.K_lin = np.asarray(Tzu @ Kz_norm @ np.linalg.inv(Tzx))
        
        ### Z CONTROL DLQR DESIGN ###
        #############################

        #############################
        ### ANGULAR DLQR DESIGN ###
        self.Ixxyy = (self.rigid_body_dipole.mass_properties.principal_inertia_properties.Px + 
                      self.rigid_body_dipole.mass_properties.principal_inertia_properties.Py)/2

        ## Continuous time state space model.
        A_ang = np.array([[0, 1], [0, 0]])
        B_ang = np.array([[0, 1/self.Ixxyy]]).T
        C_ang = np.array([[1, 0]])

        ## Normalization parameters
        pitch_max = np.deg2rad(20) # 20 deg maximum rotation
        pitch_dot_max = 5*pitch_max
        Ty_ang_max = 5*self.Ixxyy*pitch_dot_max # Assume very small maximum force.
        T_ang_x = np.diag([pitch_max, pitch_dot_max])
        T_ang_u = np.diag([Ty_ang_max])
        T_ang_y = np.diag([pitch_max])

        A_ang_norm = np.linalg.inv(T_ang_x) @ A_ang @ T_ang_x
        B_ang_norm = np.linalg.inv(T_ang_x) @ B_ang @ T_ang_u
        C_ang_norm = np.linalg.inv(T_ang_y) @ C_ang @ T_ang_x

        use_normalization = True

        if use_normalization:
            A_ang_d_norm, B_ang_d_norm, _, _, _ = signal.cont2discrete((A_ang_norm, B_ang_norm, C_ang_norm, 0), dt=self.dt,
                                                    method='zoh')

            Q_ang = np.diag([1, 1])*1e-3
            R_ang = 1
            K_ang_norm, S, E = ct.dlqr(A_ang_d_norm, B_ang_d_norm, Q_ang, R_ang)
            self.K_ang = np.asarray(T_ang_u @ K_ang_norm @ np.linalg.inv(T_ang_x))
        
        else:
            A_ang_d, B_ang_d, _, _, _ = signal.cont2discrete((A_ang, B_ang, C_ang, 0), dt=self.dt,
                                                    method='zoh')

            Q_ang = np.diag([0.01, 0.01])*1e-5
            R_ang = 1

            self.K_ang, _, _ = ct.dlqr(A_ang_d, B_ang_d, Q_ang, R_ang)        

        ### ANGULAR DLQR DESIGN ###
        #############################

        rospy.loginfo(remove_extra_spaces(f"""Control gains for linear: {self.K_lin},
        angular: {self.K_ang}"""))

        self.control_gains_message : String = String()

        self.control_gains_message.data = remove_extra_spaces(f"""Control gains for linear: {self.K_lin},
        angular: {self.K_ang},
        LQR Parameters: 
        Linear: Q = {Qz}, R = {Rz},
        Angular: Q_ang = {Q_ang}, R_ang = {R_ang},
        Normalization parameters:
        Linear: Tx = {Tzx}, Tu = {Tzu}, Ty = {Tzy},
        Angular: T_ang_x = {T_ang_x}, T_ang_u = {T_ang_u}, T_ang_y = {T_ang_y}. Angular normalization used: {use_normalization}""")

        self.z_dot = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.roll_dot = 0.0
        self.pitch_dot = 0.0
        self.last_R = np.eye(3)

        self.R_dot = np.zeros((3,3))
        self.last_z = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_roll = 0.0
        self.last_pitch = 0.0

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

        ref_z = 0.0
        ref_x = 0.0
        ref_y = 0.0

        ref_rpy = np.zeros(3)


        if linear_velocity is None or angular_velocity is None:
            R = geometry.rotation_matrix_from_quaternion(quaternion)
            if self.__first_reading:
                self.last_R = R
                self.last_z = z_com
                self.last_x = x_com
                self.last_y = y_com
                self.__first_reading = False

            self.R_dot = (R - self.last_R)/self.dt
            self.z_dot = (z_com - self.last_z)/self.dt
            self.x_dot = (x_com - self.last_x)/self.dt
            self.y_dot = (y_com - self.last_y)/self.dt
            omega = geometry.angular_velocity_body_frame_from_rotation_matrix(R, self.R_dot)
            self.roll_dot = omega[0]
            self.pitch_dot = omega[1]

            self.last_z = z_com
            self.last_x = x_com
            self.last_y = y_com
            self.last_R = R
        
        else:
            self.x_dot, self.y_dot, self.z_dot = linear_velocity
            self.roll_dot, self.pitch_dot, _ = angular_velocity

        x_z = np.array([[z_com, self.z_dot]]).T
        x_x = np.array([[x_com, self.x_dot]]).T
        x_y = np.array([[y_com, self.y_dot]]).T
        x_roll = np.array([[rpy[0], self.roll_dot]]).T
        x_pitch = np.array([[rpy[1], self.pitch_dot]]).T

        r_z = np.array([[ref_z, 0.0]]).T
        r_x = np.array([[ref_x, 0.0]]).T
        r_y = np.array([[ref_y, 0.0]]).T
        r_roll = np.array([[ref_rpy[0], 0.0]]).T
        r_pitch = np.array([[ref_rpy[1], 0.0]]).T

        z_error = r_z - x_z
        x_error = r_x - x_x
        y_error = r_y - x_y
        roll_error = r_roll - x_roll
        pitch_error = r_pitch - x_pitch

        # u_z = self.K_lin @ z_error + (self.mass + 8e-3)*common.Constants.g # Gravity compensation
        # u_z = self.K_lin @ z_error + (self.mass + 8e-3)*common.Constants.g # Gravity compensation
        u_z = self.K_lin @ z_error + self.mass*common.Constants.g # Gravity compensation
        u_x = self.K_lin @ x_error
        u_y = self.K_lin @ y_error
        u_roll = self.K_ang @ roll_error
        u_pitch = self.K_ang @ pitch_error

        F_z = u_z[0, 0] * sft_coeff
        # F_x = u_x[0, 0]
        F_x = 0.0
        # F_y = u_y[0, 0]
        F_y = 0.0
        # Tau_x = u_roll[0, 0]
        Tau_x = 0.0
        # Tau_y = u_pitch[0, 0]
        Tau_y = 0.0
           
        w_des = np.array([Tau_x, Tau_y, F_x, F_y, F_z])

        com_wrench_des = np.array([Tau_x, Tau_y, 0.0, F_x, F_y, F_z])
        self.com_wrench_msg.wrench.torque = Vector3(*com_wrench_des[:3])
        self.com_wrench_msg.wrench.force = Vector3(*com_wrench_des[3:])

        # Performing the simplified allocation for the two torques.
        des_currents = self.five_dof_wrench_allocation_single_dipole(position, quaternion, w_des)
        # des_currents = self.five_dof_2_step_torque_force_allocation(position, quaternion, w_des)

        self.desired_currents_msg.des_currents_reg = des_currents

if __name__=="__main__":
    controller = SimpleCOMWrenchSingleDipoleController()
    rospy.spin()