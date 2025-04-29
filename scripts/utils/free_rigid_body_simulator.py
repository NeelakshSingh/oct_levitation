import numpy as np
import rospy
import scipy.signal as signal
import tf2_ros
import os
import tf.transformations as tr
import time
import numba

from geometry_msgs.msg import TransformStamped, WrenchStamped, Vector3, Quaternion
from std_msgs.msg import Bool
from tnb_mns_driver.msg import DesCurrentsReg
from tnb_mns_driver.msg import TnbMnsStatus

import oct_levitation.common as common
from oct_levitation.rigid_bodies import REGISTERED_BODIES
import oct_levitation.geometry_jit as geometry
import oct_levitation.numerical as numerical

from scipy.integrate import solve_ivp

def ft_array_from_wrench(wrench: WrenchStamped):
    return (np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z]),
            np.array([wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z]))

@numba.njit
def vector_clip_update_mask(vec, vec_lim):
    vec_lim = np.abs(vec_lim)
    vel_update_mask = np.array([True, True, True])
    if vec[0] > vec_lim[0]:
        vec[0] = vec_lim[0]
        vel_update_mask[0] = False
    elif vec[0] < -vec_lim[0]:
        vec[0] = -vec_lim[0]
        vel_update_mask[0] = False
    if vec[1] > vec_lim[1]:
        vec[1] = vec_lim[1]
        vel_update_mask[1] = False
    elif vec[1] < -vec_lim[1]:
        vec[1] = -vec_lim[1]
        vel_update_mask[1] = False
    if vec[2] > vec_lim[2]:
        vec[2] = vec_lim[2]
        vel_update_mask[2] = False
    elif vec[2] < -vec_lim[2]:
        vec[2] = -vec_lim[2]
        vel_update_mask[2] = False
    return vec, vel_update_mask

@numba.njit
def clip_R_from_rpy_lim(R, rpy_limit):
    y = np.arcsin(R[0, 2])
    x = np.arcsin(-R[1, 2] / np.cos(y))
    z = np.arcsin(-R[0, 1] / np.cos(y))
    rpy = np.array([x, y, z])
    rpy, vel_update_mask = vector_clip_update_mask(rpy, rpy_limit)

    x, y, z = rpy
    R = np.array([
            [  np.cos(y) * np.cos(z),                                     -np.cos(y) * np.sin(z),                                       np.sin(y)                         ],
            [  np.cos(x) * np.sin(z) + np.sin(x) * np.sin(y) * np.cos(z),  np.cos(x) * np.cos(z) - np.sin(x) * np.sin(y) * np.sin(z),  -np.sin(x) * np.cos(y)  ],
            [  np.sin(x) * np.sin(z) - np.cos(x) * np.sin(y) * np.cos(z),  np.sin(x) * np.cos(z) + np.cos(x) * np.sin(y) * np.sin(z),   np.cos(x) * np.cos(y)  ]
        ])
    return R, vel_update_mask

class DynamicsSimulator:

    def __init__(self):
        rospy.init_node('dynamics_simulator', anonymous=True)
        self.Ts = 1/rospy.get_param("~sim_freq", 3000)
        rospy.loginfo(f"[free_rigid_body_simulator] Requested frequency: {1/self.Ts}")
        # self.Ts = 1/3000
        self.__tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.__tf_msg = TransformStamped()
        self.rigid_body = REGISTERED_BODIES[rospy.get_param("oct_levitation/rigid_body")]
        self.vicon_frame = self.rigid_body.pose_frame
        self.world_frame = "vicon/world"
        self.vicon_pub = rospy.Publisher(self.vicon_frame, TransformStamped, queue_size=10)

        self.last_command_recv = 0
        self.last_commmand_timeout = 0.2
        self.__last_command_recv_time = 0.0
        self.__first_command = False
        self.__last_command_warning_sent = False
        self.__first_sim_step = True

        ### Calibration model to use nonlinear model.
        self.calibration = common.OctomagCalibratedModel(calibration_type="legacy_yaml",
                                                         calibration_file="mc3ao8s_md200_handp.yaml")

        ### Initial Conditions and Operation Parameters
        gravity_on = rospy.get_param("free_body_sim/gravity_on")
        self.p = np.array(rospy.get_param("free_body_sim/initial_position")) # World frame
        self.p_limit = np.abs(np.array(rospy.get_param("free_body_sim/position_limit"))) # World frame
        self.v = np.array(rospy.get_param("free_body_sim/initial_velocity")) # World frame

        initial_rpy = np.array(rospy.get_param("free_body_sim/initial_rpy_deg")) # World frame
        self.rpy_limit = np.abs(np.deg2rad(np.array(rospy.get_param("free_body_sim/rpy_limit_deg")))) # World frame
        rospy.loginfo(f"[free_body_sim] initial_rpy: {initial_rpy}")
        self.q = geometry.quaternion_from_euler_xyz(np.deg2rad(initial_rpy)) # World frame
        self.R = geometry.rotation_matrix_from_quaternion(self.q) # Same as above.
        self.omega = np.array(rospy.get_param("free_body_sim/initial_angular_velocity_deg")) # w.r.t world frame resolved in local frame.
        self.omega = np.deg2rad(self.omega)
        self.use_wrench = rospy.get_param("free_body_sim/use_wrench", False)
        self.publish_status = rospy.get_param("free_body_sim/pub_status", False)
        self.print_ft = rospy.get_param("free_body_sim/print_ft", False)
        self.vicon_noise_covariance_exyz = np.diag(np.square(rospy.get_param("free_body_sim/vicon_noise_std_exyz"))) # World frame
        self.current_noise_std = rospy.get_param("free_body_sim/current_noise_std", 0.029008974233986275)
        self.ecb_bandwidth = rospy.get_param("free_body_sim/ecb_bandwidth_hz", 15) * 2 * np.pi # Conservative ecb bandwidth in rad/s
        self.vicon_pub_freq = rospy.get_param("free_body_sim/vicon_pub_freq", 100)

        ### Periodic Poke Disturbance Parameters
        self.enable_poke_disturbance = rospy.get_param("free_body_sim/enable_poke_disturbance", True)
        self.poke_period_ns = 1e9*rospy.get_param("free_body_sim/poke_period_sec", 2) # Give a force and torque poke to the object every poke_period_sec seconds.
        self.poke_position_change = np.array(rospy.get_param("free_body_sim/poke_position_change_mm", [0.0, 0.0, 0.0]))*1e-3 # World frame
        self.post_poke_velocity = np.array(rospy.get_param("free_body_sim/post_poke_velocity_mmps", [0.0, 0.0, 0.0]))*1e-3 # World frame
        poke_orientation_rpy_change = np.deg2rad(np.array(rospy.get_param("free_body_sim/poke_rpy_change_deg", [0.0, 0.0, 0.0]))) # World frame
        self.poke_rotmat = geometry.rotation_matrix_from_euler_xyz(poke_orientation_rpy_change)
        self.post_poke_angular_velocity = np.deg2rad(np.array(rospy.get_param("free_body_sim/post_poke_angular_velocity_degps", [0.0, 0.0, 0.0]))) # World frame
        self.poke_start_time_ns = 1e9*rospy.get_param("free_body_sim/poke_start_time_sec", 0.0) # The time elapsed since simulation start to start poking.
        
        self.vicon_pub_time_ns = 1e9/self.vicon_pub_freq
        # Closed form soln is known, I will still do it cause why not
        self.__ecbd_A, self.__ecbd_B, *_ = signal.cont2discrete((np.array([[-self.ecb_bandwidth]]), np.array([[self.ecb_bandwidth]]), 0, 0), dt=1/self.vicon_pub_freq, method='zoh')
        self.__ecbd_A = self.__ecbd_A[0,0]
        self.__ecbd_B = self.__ecbd_B[0,0]
        self.__last_output_currents = np.zeros(8)

        rospy.loginfo("[free_rigid_body_simulator] ecbd_A: %f, ecbd_B: %f" % (self.__ecbd_A, self.__ecbd_B))
        rospy.loginfo("[free_rigid_body_simulator] gravity_on: %s" % (gravity_on))

        if self.publish_status:
            self.sim_status_pub = rospy.Publisher("oct_levitation/free_rigid_body_sim/status", Bool, queue_size=1) # This is just to measure the simulator run freq
        self.__last_vicon_pub_time_ns = -np.inf
        self.__last_poke_time_ns = None
        self.__first_command_time_ns = None
        
        self.__tf_msg.header.frame_id = self.world_frame
        self.__tf_msg.child_frame_id = self.vicon_frame

        self.last_recvd_wrench = WrenchStamped()

        self.wrench_sub = None
        self.currents_sub = None

        if self.use_wrench:
            self.wrench_sub = rospy.Subscriber(self.rigid_body.com_wrench_topic, WrenchStamped, self.wrench_callback, queue_size=1)
        else:
            self.currents_sub = rospy.Subscriber("/tnb_mns_driver/des_currents_reg/delayed_sim", DesCurrentsReg, self.currents_callback, queue_size=1)

        self.last_sim_time = rospy.Time.now().to_sec()
        self.I_bf = self.rigid_body.mass_properties.I_bf
        ### Uncomment the lines below to use the principal moments of inertia instead.
        self.m = self.rigid_body.mass_properties.m
        self.I_bf_inv = np.linalg.inv(self.I_bf)

        if gravity_on:
            self.F_amb = np.array([0, 0, -self.m*common.Constants.g])
        else:
            self.F_amb = np.array([0, 0, 0])

    def simulation_loop(self, event):

        # This simulation loop will assume ZOH, therefore the last command is just kept on being repeated
        # we won't explicitly set it to zero here. The forces are assumed to be in the world frame while
        # the torques are assumed to be in the body fixed frame.
        t_comp_start_ns = rospy.Time.now().to_nsec()
        if self.__first_sim_step:
            self.__first_sim_step = False
            self.last_sim_time = rospy.Time.now().to_sec()
            return
        
        if rospy.Time.now().to_sec() - self.__last_command_recv_time > self.last_commmand_timeout:
            if self.__first_command:
                if not self.__last_command_warning_sent:
                    self.__last_command_warning_sent = True
                    rospy.logwarn("No command received in the last %.2f seconds." % self.last_commmand_timeout)

        dt = rospy.Time.now().to_sec() - self.last_sim_time
        self.last_sim_time = rospy.Time.now().to_sec()

        # Computing the velocity and position
        F, Tau = ft_array_from_wrench(self.last_recvd_wrench)
        if self.print_ft:
            rospy.loginfo(f"Applying F: {F}, Tau: {Tau}")

        # Initialize to the previous values, because depending on first command and poking these go through different changes.
        p_next = self.p
        v_next = self.v
        R_next = self.R
        omega_next = self.omega

        if self.__first_command:
            F = F + self.F_amb # Adding gravity and other constant forces, IF we have started receiving commands.
            time_now_ns = rospy.Time.now().to_nsec()
            t_comp_poke_ns = time_now_ns - t_comp_start_ns
            if self.__last_poke_time_ns is not None and (time_now_ns - self.__first_command_time_ns) > self.poke_start_time_ns \
                and (time_now_ns + t_comp_poke_ns - self.__last_poke_time_ns) >= self.poke_period_ns:
                # Apply a sudden perturbation to the object
                # If the controller can regulate this then we can safely assume that the controller will be able to do so in the real world
                # since this is an impulse disturbance.
                p_next = self.p + self.poke_position_change
                v_next = self.post_poke_velocity
                R_next = self.R @ self.poke_rotmat
                omega_next = self.post_poke_angular_velocity

                self.__last_poke_time_ns = time_now_ns
            else:
                # Take a normal simulation step.
                p_next, v_next = numerical.integrate_linear_dynamics_constant_force_undamped(self.p, self.v, F, self.m, dt)
                R_next, omega_next = numerical.integrate_R_omega_constant_torque(self.R, self.omega, Tau, self.I_bf, dt)

        # Clip the position and orientation to the limits.
        self.p, v_update_mask = vector_clip_update_mask(p_next, self.p_limit)
        self.v[v_update_mask] = v_next[v_update_mask]
        self.v[np.logical_not(v_update_mask)] = 0.0 # If we clipped the position, we set the velocity to zero.
        # self.p = p_next
        # self.v = v_next

        # Numerically integration the orientation through the lie group exponential map of angular velocity.
        # The angular velocity is resolved in the local frame.
        self.R, omega_update_mask = clip_R_from_rpy_lim(R_next, self.rpy_limit)
        self.omega[omega_update_mask] = omega_next[omega_update_mask]
        self.omega[np.logical_not(omega_update_mask)] = 0.0 # If we clipped the orientation, we set the angular velocity to zero.
        # self.R = R_next
        # self.omega = omega_next

        ## Adding Feedback Noise
        pose_noise = np.random.multivariate_normal(mean=np.zeros(6), cov=self.vicon_noise_covariance_exyz)
        p_vicon = self.p + pose_noise[:3]
        R_vicon = self.R @ geometry.rotation_matrix_from_euler_xyz(pose_noise[3:])

        q_vicon = geometry.quaternion_from_rotation_matrix(R_vicon)
        q_vicon /= np.linalg.norm(q_vicon)

        self.__tf_msg.header.stamp = rospy.Time.now()
        self.__tf_msg.transform.translation = Vector3(*p_vicon)
        self.__tf_msg.transform.rotation = Quaternion(*q_vicon)
        self.__tf_broadcaster.sendTransform(self.__tf_msg)

        if self.publish_status:
            self.sim_status_pub.publish(Bool(True))

        t_comp_end_ns = rospy.Time.now().to_nsec()
        if (rospy.Time.now().to_nsec() + t_comp_end_ns - t_comp_start_ns - self.__last_vicon_pub_time_ns) >= self.vicon_pub_time_ns:
            self.__last_vicon_pub_time_ns = rospy.Time.now().to_nsec()
            self.vicon_pub.publish(self.__tf_msg)

    def wrench_callback(self, com_wrench: WrenchStamped):
        if not self.__first_command:
            self.__first_command = True
        self.last_recvd_wrench = com_wrench
        self.__last_command_recv_time = rospy.Time.now().to_sec()
        if self.__last_command_warning_sent:
            rospy.loginfo("Command received again.")
            self.__last_command_warning_sent = False
    
    def calculate_com_wrench_indiv_magnets(self, currents: np.ndarray):
        com_quaternion = geometry.numpy_quaternion_from_tf_msg(self.__tf_msg.transform)
        com_position = geometry.numpy_translation_from_tf_msg(self.__tf_msg.transform)

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
        
        # self.last_recvd_wrench.header.stamp = rospy.Time.now()
        # self.last_recvd_wrench.wrench.force = Vector3(*com_force)
        # self.last_recvd_wrench.wrench.torque = Vector3(*com_torque)

        torque_force = np.concatenate((com_torque, com_force))
        return torque_force
                
    
    def currents_callback(self, des_currents_msg: DesCurrentsReg):
        # The idea is to use the latest available pose and the forward model
        # to calculate the actual wrench at dipole center.
        currents = np.asarray(des_currents_msg.des_currents_reg)
        # First order TF on currents to simulate current bandwidth.
        if not self.__first_command:
            # For the first time, bypass the bandwidth filter.
            self.__last_output_currents = currents
            self.__first_command = True
            self.__first_command_time_ns = rospy.Time.now().to_nsec()
            if self.enable_poke_disturbance:
                self.__last_poke_time_ns = -np.inf # After the first command, we start poking. This will force the first poke to happen at the disturbance start time.
        else:
            # Apply the first order filter on the currents.
            # The reason this line is inside the else statement is that the currents for gravity compensation are quite large in value
            # and they won't be immediately available leading to an immediate drop in z position the first few iterations. So the first command is immediately serviced.
            # This assumption is not wrong because of stops in the real world. One will ideally use smooth start anyways.
            self.__last_output_currents = self.__ecbd_A * self.__last_output_currents + self.__ecbd_B * currents
        wrench = WrenchStamped()
        # dipole_quat, dipole_pos = geometry.numpy_arrays_from_tf_msg(self.__tf_msg.transform)
        # Mq = geometry.magnetic_interaction_matrix_from_quaternion(dipole_quat,
        #                                                           dipole_strength=self.rigid_body.dipole_list[0].strength,
        #                                                           full_mat=True,
        #                                                           torque_first=True,
        #                                                           dipole_axis=self.rigid_body.dipole_list[0].axis)
        # noisy_currents = self.__last_output_currents + np.random.normal(loc=0.0, scale=self.current_noise_std, size=(8,))
        # field_grad = self.calibration.get_exact_field_grad5_from_currents(dipole_pos, noisy_currents) # Already a compiled function
        # actual_Tau_force = (Mq @ field_grad).flatten() # This will be in the world frame.
        actual_Tau_force = self.calculate_com_wrench_indiv_magnets(self.__last_output_currents)
        
        wrench.header.stamp = rospy.Time.now()
        wrench.wrench.torque = Vector3(*actual_Tau_force[:3])
        wrench.wrench.force = Vector3(*actual_Tau_force[3:])

        self.last_recvd_wrench = wrench
        self.__last_command_recv_time = rospy.Time.now().to_sec()
        if self.__last_command_warning_sent:
            rospy.loginfo("Command received again.")
            self.__last_command_warning_sent = False

    def run(self):
        self.simulation_timer = rospy.Timer(rospy.Duration(self.Ts), self.simulation_loop)
        rospy.spin()


if __name__ == "__main__":
    # Run jit functions first to force compilation
    clip_R_from_rpy_lim(np.eye(3), np.array([np.pi, np.pi, np.pi]))
    numerical.integrate_linear_dynamics_constant_force_undamped(np.zeros(3), np.zeros(3), np.zeros(3), 1.0, 1.0)
    numerical.integrate_R_omega_constant_torque(np.eye(3), np.zeros(3), np.zeros(3), np.eye(3), 1.0)

    # Run the simulator
    dynamics_simulator = DynamicsSimulator()
    dynamics_simulator.run()