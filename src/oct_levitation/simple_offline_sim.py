import numpy as np
import scipy.signal as signal
import os
import pandas as pd
import rospkg
import control as ct
import alive_progress as ap

from collections import deque
from typing import Dict, Tuple, List, Optional, Any, Callable
import numpy.typing as np_t


import oct_levitation.common as common
from oct_levitation.rigid_bodies import REGISTERED_BODIES
import oct_levitation.geometry_jit as geometry
import oct_levitation.numerical as numerical
import oct_levitation.plotting as plotting

# Fixing seed
np.random.seed(0)

# Utility functions
def append_list_to_pd_df(df: pd.DataFrame, row: List[Any]) -> pd.DataFrame:
    new_row_df = pd.DataFrame([row], columns=df.columns)
    df = pd.concat([df, new_row_df], ignore_index=True)
    return df

# Utility typedefs
CurrentsType = np_t.NDArray[float]
TorqueForceVectorType = np_t.NDArray[float]
PositionType = np_t.NDArray[float]
QuaternionType = np_t.NDArray[float]
JMAConditionType = float

class SimControllerBase:
    
    def __init__(self):
        pass
    
    def run(PositionType, QuaternionType) -> Tuple[CurrentsType, TorqueForceVectorType, JMAConditionType]:
        raise NotImplementedError

class DynamicsSimulator:
    
    def __init__(self, 
                 SIM_PARAMS: Dict[str, Any], 
                 controller_class: SimControllerBase) -> None:
        self.SIM_PARAMS = SIM_PARAMS
        self.Ts = 1/self.SIM_PARAMS['sim_freq']
        self.rigid_body = REGISTERED_BODIES[self.SIM_PARAMS['rigid_body']]
        self.calibration = common.OctomagCalibratedModel(calibration_type="legacy_yaml",
                                                         calibration_file="mc3ao8s_md200_handp.yaml")
        
        # Pose storing parameters
        self.position = self.SIM_PARAMS['initial_position']
        self.p_limit = self.SIM_PARAMS['position_limit'] # NOT USED FOR NOW
        self.velocity = self.SIM_PARAMS['initial_velocity']
        self.rpy_limit = np.abs(np.deg2rad(self.SIM_PARAMS['rpy_limit_deg'])) # NOT USED FOR NOW
        self.quaternion = geometry.quaternion_from_euler_xyz(self.SIM_PARAMS['initial_rpy_deg'])
        self.R = geometry.rotation_matrix_from_quaternion(self.quaternion)
        self.omega = np.deg2rad(self.SIM_PARAMS['initial_angular_velocity_deg_ps'])
        self.current_noise_std = self.SIM_PARAMS['current_noise_std']
        self.vicon_noise_covariance_exyz = np.diag(np.square(self.SIM_PARAMS['vicon_noise_std_exyz']))
        
        # History and dataset storing variables
        self.actual_currents_df = pd.DataFrame(columns=['time', 'currents_reg_0', 'currents_reg_1', 'currents_reg_2', 'currents_reg_3', 'currents_reg_4', 'currents_reg_5', 'currents_reg_6', 'currents_reg_7'])
        self.des_currents_df = pd.DataFrame(columns=['time', 'des_currents_reg_0', 'des_currents_reg_1', 'des_currents_reg_2', 'des_currents_reg_3', 'des_currents_reg_4', 'des_currents_reg_5', 'des_currents_reg_6', 'des_currents_reg_7'])
        self.des_wrench_df = pd.DataFrame(columns=['time', 'wrench.torque.x', 'wrench.torque.y', 'wrench.torque.z', 'wrench.force.x', 'wrench.force.y', 'wrench.force.z'])
        self.jma_condition_df = pd.DataFrame(columns=['time', 'allocation_condition_number'])
        self.pose_df = pd.DataFrame(columns=['time', 'child_frame_id', 'transform.translation.x', 'transform.translation.y', 'transform.translation.z', 'transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w'])
        
        # Dataset storage metadata
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('oct_levitation')
        self.data_base_folder = os.path.join(pkg_path, 'data/offline_sim_nb')
        
        ### SIMULATION LOOP MANAGING VARIABLES
        self.current_time = 0.0
        self.SIM_DURATION = self.SIM_PARAMS['duration']
        self.actual_current = np.zeros(8) # Always starts at zero
        self.__first_control_command_received = False
        self.control_input_queue = deque() # For delaying the control input
        self.des_currents = np.zeros(8)
        self.des_wrench = np.zeros(8)
        self.jma_condition = 0.0
        self.feedback_position = self.position
        self.feedback_quaternion = self.quaternion
        self.active_dimension_mask = self.SIM_PARAMS['active_dimensions']
        ## Control related parameters
        self.control_input_delay = self.SIM_PARAMS['control_input_delay']
        self.controller_frequency = self.SIM_PARAMS['controller_frequency']
        self.CONTROLLER_CLASS = controller_class
        self.controller_max_current = self.SIM_PARAMS['max_controller_current']
        self.__last_control_recv_time = 0.0
        
        ## Dynamics related functions, 
        # I will just use linear dynamics as per Jasan's recommendation.
        # If the change is not made in this class as you read this, and you want the full non-linear
        # dynamics integration, feel free to read the implementation in free_body_sim.py
        # It only contains undamped dynamics though, but one can calcualate the solutions for damped versions too.
        
        # Linear Dynamics
        A_lin = np.array([[0.0, 1.0], [0.0, 0.0]])
        B_lin = np.array([[0.0, 1/self.rigid_body.mass_properties.m]]).T
        C_lin = np.eye(2)
        self.lin_dynamics_ss : ct.StateSpace = ct.ss(*signal.cont2discrete((A_lin, B_lin, C_lin, 0.0), dt=self.Ts, method='zoh'))
        
        # Angular dynamics in terms of euler angles (any representation works), small angle assumption implying simple double integrator
        A_ang = np.array([[0.0, 1.0], [0.0, 0.0]])
        I_avg = (self.rigid_body.mass_properties.principal_inertia_properties.Px + self.rigid_body.mass_properties.principal_inertia_properties.Py)/2
        B_ang = np.array([[0.0, 1/I_avg]]).T
        C_ang = np.eye(2)
        self.ang_dynamics_ss : ct.StateSpace = ct.ss(*signal.cont2discrete((A_ang, B_ang, C_ang, 0.0), dt=self.Ts, method='zoh'))
        
        # ECB dynamics
        self.ecb_bandwidth = self.SIM_PARAMS['ecb_bandwidth_hz'] * 2 * np.pi
        self.ecb_ss : ct.StateSpace = ct.ss(*signal.cont2discrete((np.array([[-self.ecb_bandwidth]]), np.array([[self.ecb_bandwidth]]), np.array([[1.0]]), 0.0), dt=self.Ts, method='zoh'))
        
        # Gravity and ambient force disturbances
        self.F_amb = np.zeros(3)
        
        if self.SIM_PARAMS['gravity_on']:
            self.F_amb = np.array([0.0, 0.0, -self.rigid_body.mass_properties.m * common.Constants.g])
        
        
    def calculate_com_wrench_indiv_magnets(self, currents: np.ndarray):
        com_quaternion = self.quaternion
        com_position = self.position

        ### Nomenclature details
        # V: World frame (vicon frame)
        # M: Body fixed frame (attached to COM usually, tracked using vicon)
        # D: Dipole frame (attached to the dipole)
        # G: Magnet frame (attached to the magnet)
        T_VM = geometry.transformation_matrix_from_quaternion(com_quaternion, com_position)
        R_VM = T_VM[:3, :3] # from body fixed frame to world frame

        com_force = np.zeros(3)
        com_torque = np.zeros(3)

        for dipole in self.rigid_body.dipole_list:
            dipole_quat = geometry.numpy_quaternion_from_tf_msg(dipole.transform)
            dipole_position = geometry.numpy_translation_from_tf_msg(dipole.transform)
            T_MD = geometry.transformation_matrix_from_quaternion(dipole_quat, dipole_position)
            for i, (magnet_tf, magnet) in enumerate(dipole.magnet_stack):
                mag_quaternion = geometry.numpy_quaternion_from_tf_msg(magnet_tf)
                mag_position = geometry.numpy_translation_from_tf_msg(magnet_tf)
                T_DG= geometry.transformation_matrix_from_quaternion(mag_quaternion, mag_position)
                T_MG = T_MD @ T_DG
                t_MG_M = T_MG[:3, 3] # relative position of the magnet w.r.t the body fixed frame expressed in the body fixed frame

                T_VG = T_VM @ T_MG
                R_VG = T_VG[:3, :3] # rotmat from magnet frame to world frame
                R_MG = T_MG[:3, :3] # rotmat from magnet frame to body fixed frame
                p_G_V = T_VG[:3, 3] # position of the magnet frame (magnet's dipole center) in world frame (calibration frame)

                bg_V = self.calibration.get_exact_field_grad5_from_currents(p_G_V, currents)
                b_V = bg_V[:3] # magnetic field in world frame
                g_V = bg_V[3:] # magnetic field gradient in world frame

                mag_dipole_G = magnet.magnetization_axis * magnet.get_dipole_strength()
                mag_dipole_V = R_VG @ mag_dipole_G # magnet's dipole moment expressed in world frame
                mag_dipole_M = R_MG @ mag_dipole_G # magnet's dipole moment expressed in body fixed frame

                Mf = geometry.magnetic_interaction_grad5_to_force(mag_dipole_V) # magnetic interaction from V frame gradients to V frame forces on the magnet center
                magnet_force_V = Mf @ g_V
                magnet_force_M = (R_VM.T @ magnet_force_V).flatten()

                Mbar_tau = geometry.magnetic_interaction_field_to_local_torque_from_rotmat(mag_dipole_M, R_VM) # This will map the V frame field to M frame torques
                
                magnet_force_world = magnet_force_V.flatten()
                magnet_com_torque_from_torque = (Mbar_tau @ b_V).flatten()

                com_force += magnet_force_world
                magnet_com_torque_from_force = np.cross(t_MG_M, magnet_force_M).flatten()
                magnet_torque_com_M = magnet_com_torque_from_force + magnet_com_torque_from_torque
                com_torque += R_VM @ magnet_torque_com_M # Because the applied torques in this simulator are in the intertial frame.

        torque_force = np.concatenate((com_torque, com_force))
        return torque_force
    
    def run_simulation(self):
        while self.current_time < self.SIM_DURATION:
            # run the controller
            if not self.__first_control_command_received or ((self.current_time - self.__last_control_recv_time) >= 1/self.controller_frequency):
                self.control_input_queue.append((self.current_time, self.CONTROLLER_CLASS.run(self.feedback_position, self.feedback_quaternion)))
                self.__last_control_recv_time = self.current_time

            # Extract the control input in a delayed fashion
            if not self.__first_control_command_received:
                # Do not apply the ECB's dynamics and just pass the currents through. Important for stable start.
                # No need to apply noise either
                _, (self.des_currents, self.des_wrench, self.jma_condition) = self.control_input_queue.popleft()
                self.actual_currents = np.clip(self.des_currents, -self.controller_max_current, self.controller_max_current)
                self.__first_control_command_received = True
            else:
                # Add pure input delay
                if self.control_input_queue and (self.current_time - self.control_input_queue[0][0]) >= self.control_input_delay:
                    _, (self.des_currents, self.des_wrench, self.jma_condition) = self.control_input_queue.popleft()
                # Add ECB dynamics
                self.actual_currents = self.ecb_ss.A[0, 0] * self.actual_currents + self.ecb_ss.B[0, 0] * self.des_currents
                # Clipping before noise is added to avoid clipping noise (doesn't make much difference though)
                self.actual_currents = np.clip(self.actual_currents, -self.controller_max_current, self.controller_max_current)
                # Add noise
                self.actual_currents = self.actual_currents + np.random.normal(loc=0.0, scale=self.current_noise_std, size=(8,))

            ## Fillin in the currents and control related data in the dataframes
            self.actual_currents_df = append_list_to_pd_df(self.actual_currents_df, [self.current_time] + list(self.actual_currents))
            self.des_currents_df = append_list_to_pd_df(self.des_currents_df, [self.current_time] + list(self.des_currents))
            self.des_wrench_df = append_list_to_pd_df(self.des_wrench_df, [self.current_time] + list(self.des_wrench))
            self.jma_condition_df = append_list_to_pd_df(self.jma_condition_df, [self.current_time, self.jma_condition])

            ## Calculating the actual torques and forces
            torque_force = self.calculate_com_wrench_indiv_magnets(self.actual_currents)
            # torque_force = self.des_wrench
            torque_force[np.logical_not(self.active_dimension_mask)] = 0.0
            torque = torque_force[:3]
            force = torque_force[3:] + self.F_amb

            ## Forward propogating the dynamics
            for i in range(3):
                x_lin = np.array([self.position[i], self.velocity[i]])
                u_lin = np.array([force[i]])
                x_new = self.lin_dynamics_ss.dynamics(0.0, x_lin, u_lin)
                self.position[i] = x_new[0]
                self.velocity[i] = x_new[1]

                rpy = geometry.euler_xyz_from_quaternion(self.quaternion)
                x_ang = np.array([rpy[i], self.omega[i]])
                u_ang = np.array([torque[i]])
                x_ang_new = self.ang_dynamics_ss.dynamics(0.0, x_ang, u_ang)
                rpy[i] = x_ang_new[0]
                self.omega[i] = x_ang_new[1]
                self.quaternion = geometry.quaternion_from_euler_xyz(rpy)

            # NOT CLIPPING THE VARIABLES TO BE WITHIN SPECIFIED LIMITS FOR NOW

            ## Add output noise to the feedback variables which are used by the controller
            pose_noise = np.random.multivariate_normal(mean=np.zeros(6), cov=self.vicon_noise_covariance_exyz)
            self.feedback_position = self.position + pose_noise[:3]
            rpy = geometry.euler_xyz_from_quaternion(self.quaternion)
            rpy += pose_noise[3:]
            self.feedback_quaternion = geometry.quaternion_from_euler_xyz(rpy)

            # Storing the pose df
            self.pose_df = append_list_to_pd_df(self.pose_df, [self.current_time, self.rigid_body.pose_frame] + list(self.feedback_position) + list(self.feedback_quaternion))

            # Moving the clock forward
            self.current_time += self.Ts