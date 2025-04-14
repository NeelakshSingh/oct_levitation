import rospy
import numpy as np
import scipy.signal as signal
import control as ct

import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.common as common

from oct_levitation.control_node import ControlSessionNodeBase
from control_utils.msg import VectorStamped
from control_utils.general.utilities import SmoothIterativeUpdate
from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3
from tnb_mns_driver.msg import DesCurrentsReg

class DirectCOMWrenchZController(ControlSessionNodeBase):

    def post_init(self):
        self.control_rate = 100
        self.publish_desired_com_wrenches = True
        self.control_input_publisher = rospy.Publisher("/com_wrench_z_control/control_input",
                                                       VectorStamped, queue_size=1)

        self.estimated_state_pub = rospy.Publisher("/com_wrench_z_control/estimated_position_and_velocity",
                                                   VectorStamped, queue_size=1)

        mass_buffer = 0.01 # Amount to reduce the mass by in order to avoid violent z-control overshoots.
        self.mass = self.rigid_body_dipole.mass_properties.m - mass_buffer

        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0, 1/self.mass]]).T
        C = np.array([[1, 0]]) # We only observer position and not velocity

        ## Setting up the DLQR parameters for exact system emulation.
        self.A_d, self.B_d, self.C_d, D_d, dt = signal.cont2discrete((A, B, C, 0), dt=1/self.control_rate,
                                                  method='zoh')
        Q = np.diag([1.0, 1.0])
        R = 1
        self.K, S, E = ct.dlqr(self.A_d, self.B_d, Q, R)
        rospy.loginfo(f"Control gain for forces: {self.K}")

        self.F_com_des = np.zeros(3)

        # Ideally the same controllers work for all X, Y, Z
        self.Ki_z = 0 # works well for a mass mismatch of around 10g TUNED
        self.last_p_com = np.zeros(3)
        self.z_err_integral = np.zeros((2,1))

        # Parameters for first order filtered differentiator with Tustin.
        self.Tc = 1/self.control_rate
        self.omega_f = 2*np.pi*self.control_rate/2 # rad/sec
        self.Tf = 1/self.omega_f # filter time constant in continuous time

        self.beta_ff = 2/(2*self.Tf + self.Tc)
        self.beta_fb = (2*self.Tf - self.Tc)/(2*self.Tf + self.Tc)

        self.p_com_dot_first_order_tustin = np.zeros(3)

        self.com_home_position = np.array([0.0, 0.0, 0.02])
        self.HARDWARE_CONNECTED = False

        self.get_position_once = True
        self.get_allocation_once = True
        self.com_tF_dummy = None
        self.MA_dummy = None

        self.SMOOTH_UPDATER = SmoothIterativeUpdate(0.0, 1.0, 1, Ts = self.Tc)

        # Anti aliasing filters for vicon.
    
    def control_logic(self):
        self.desired_currents_msg = DesCurrentsReg() # Empty message
        self.control_input_message = VectorStamped() # Empty message
        self.com_wrench_msg = WrenchStamped() # Empty message
        self.estimated_state_msg = VectorStamped() # Empty state estimate message

        self.desired_currents_msg.header.stamp = rospy.Time.now()
        self.com_wrench_msg.header.stamp = rospy.Time.now()
        self.control_input_message.header.stamp = rospy.Time.now()
        self.estimated_state_msg.header.stamp = rospy.Time.now()

        if self.get_position_once:
            com_tf: TransformStamped = self.tf_buffer.lookup_transform(self.world_frame, self.rigid_body_dipole.pose_frame,
                                                 rospy.Time())
            self.com_tF_dummy = com_tf
            self.get_position_once = False
        
        com_tf = self.com_tF_dummy
        
        com_position = np.array([com_tf.transform.translation.x,
                                com_tf.transform.translation.y,
                                com_tf.transform.translation.z])
        
        ### NORMAL FINITE DIFFERENCES
        com_position_err = self.com_home_position - com_position
        self.z_err_integral += com_position_err[2]/self.control_rate

        self.p_com_dot_first_order_tustin = self.beta_ff*(com_position - self.last_p_com) + self.beta_fb*self.p_com_dot_first_order_tustin

        self.estimated_state_msg.vector = [com_position[0], com_position[1], com_position[2],
                                           self.p_com_dot_first_order_tustin[0],
                                           self.p_com_dot_first_order_tustin[1],
                                           self.p_com_dot_first_order_tustin[2]]

        self.last_p_com = com_position

        self.Fz = self.K @ np.array([[com_position_err[2], self.p_com_dot_first_order_tustin[2]]]).T + \
                    self.Ki_z*self.z_err_integral + self.mass*common.Constants.g
        
        self.Fx = self.K @ np.array([[com_position_err[0], self.p_com_dot_first_order_tustin[0]]]).T
        self.Fy = self.K @ np.array([[com_position_err[1], self.p_com_dot_first_order_tustin[1]]]).T
        ### NORMAL FINITE DIFFERENCES

        self.F_com_des = np.array([self.Fx[0,0], self.Fy[0,0], self.Fz[0,0]])

        # Allocating them among the dipoles. Note that for pure translational
        # allocation no transform is required and we can just split the forces
        # equally among the dipoles.
        self.F_dp = self.F_com_des/2
        self.F_dn = self.F_com_des/2

        ### ALLOCATION STRATEGY 1: CONSTRAIN DIPOLE TORQUES TO 0
        self.des_wrench_p = np.concatenate([self.F_dp.flatten(), np.zeros(3)])
        self.des_wrench_n = np.concatenate([self.F_dn.flatten(), np.zeros(3)])

        # Get the magnetization matrices.
        if self.get_allocation_once:
            MA_list = self.rigid_body_dipole.get_current_wrench_matrices_from_world_frame_mpem(
                self.tf_buffer, self.mpem_model, torque_first=False, full_mat=True
            ) # will be pos x first

            MA_constraint_stack = np.vstack(MA_list)
            self.MA_dummy = MA_constraint_stack
            self.get_allocation_once = False
        
        MA_constraint_stack = self.MA_dummy
        desired_wrenches = np.concatenate((self.des_wrench_p, self.des_wrench_n))

        # Calculating desired currents
        i_des = np.linalg.pinv(MA_constraint_stack) @ desired_wrenches * self.SMOOTH_UPDATER.update()

        # Filling all the important messages.
        self.com_wrench_msg.wrench.force = Vector3(self.F_com_des[0], self.F_com_des[1], self.F_com_des[2])
        
        self.desired_currents_msg.des_currents_reg = i_des
        self.control_input_message.vector = desired_wrenches

        ### [TODO] ALLOCATION STRATEGY 2: OPTIMAL ALLOCATION ACCORDING TO CANCELLING TORQUES
        # the idea is that equal and opposite torques along all the 3 axes at the dipole 
        # centers are valid solutions since they will cancel out thanks to the aligned axes.
        # This can ideally be posed as an optimal current allocation problem through a simple
        # QP with quadratic cost and linear constraints.

        ## Publishing all custom defined messages
        self.estimated_state_pub.publish(self.estimated_state_msg)

        

if __name__ == "__main__":
    controller = DirectCOMWrenchZController()
    rospy.spin()