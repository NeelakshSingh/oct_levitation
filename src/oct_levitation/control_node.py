import os
import rospy
import oct_levitation.mechanical as mechanical
import oct_levitation.geometry as geometry
import tf2_ros
import numpy as np
import time

from geometry_msgs.msg import WrenchStamped, TransformStamped, Quaternion, Vector3
from tnb_mns_driver.msg import DesCurrentsReg
from control_utils.msg import VectorStamped
from std_msgs.msg import String
from mag_manip import mag_manip
from typing import List
from scipy.linalg import block_diag

from control_utils.general.utilities import init_system

class ControlSessionNodeBase:
    """
    This class contains all the basic functionalities that a ROS Node implementing a levitation controller
    must contain. These functions mostly pertain to creating topics required for important data logging
    purposes in order to perform a detailed experimental analysis later.
    """
    def __init__(self):
        rospy.init_node("oct_levitation_controller_node")

        self.calfile_base_path = rospy.get_param("~calfile_base_path", os.path.join(os.environ["HOME"], ".ros/cal"))
        self.calibration_file = rospy.get_param('~mpem_cal_file', "mc3ao8s_md200_handp.yaml")
        self.control_rate = rospy.get_param("oct_levitation/control_freq") # Set it to the vicon frequency
        self.sim_mode = rospy.get_param("oct_levitation/sim_mode") # Mandatory param, wait for it to be set.
        self.__N_CONNECTED_DRIVERS = 6 # number of used drivers can be less
        self.__MAX_CURRENT = 4 # Amps
        if self.sim_mode:
            self.__MAX_CURRENT = 12.0 # Amps

        self.world_frame = rospy.get_param("~world_frame", "vicon/world")

        self.publish_computation_time = rospy.get_param("oct_levitation/publish_computation_time")
        self.computation_time_avg_samples = rospy.get_param("~computation_time_avg_samples", 100)
        self.computation_time_topic = rospy.get_param("~computation_time_topic", "control_session/computation_time")
        self.compute_time_pub = None
        self.__ACTIVE_COILS = [0, 1, 2, 3, 5] # This variable should be hidden from the derived classes since this is supposed to stay fixed for all experiments.
        self.__ACTIVE_DRIVERS = [0, 1, 2, 4, 5] # These are the exact driver numbers these coils are connected to.
        if len(self.__ACTIVE_DRIVERS) != len(self.__ACTIVE_COILS):
            msg = f"Active coils and drivers must be the same size. Active coils: {self.__ACTIVE_COILS}, Active drivers: {self.__ACTIVE_DRIVERS}"
            rospy.logerr(msg)
            raise ValueError(msg)
        self.INITIAL_DESIRED_POSITION = np.array([0.0, 0.0, 0.0])
        self.INITIAL_DESIRED_ORIENTATION_EXYZ = np.array([0.0, 0.0, 0.0])
        self.warn_jma_condition = True
        self.publish_jma_condition = True
        if self.publish_jma_condition:
            self.jma_condition_pub = rospy.Publisher("/control_session/jma_condition",
                                                     VectorStamped, queue_size=1)

        if self.publish_computation_time:
            self.computation_time_pub = rospy.Publisher(self.computation_time_topic, VectorStamped, queue_size=1)

        self.rigid_body_dipole: mechanical.MultiDipoleRigidBody = None # Set this in post init
        self.coils_to_enable = [True]*9
        self.coils_to_enable[6] = False # Coil 7 is not being used at the moment.
        self.HARDWARE_CONNECTED = False # to force explicit enablement in post init.

        self.publish_desired_dipole_wrenches = rospy.get_param("~log_desired_dipole_wrench", False)
        self.publish_desired_com_wrenches = rospy.get_param("~log_desired_com_wrench", False)
        self.metadata_topic = rospy.get_param("~metadata_pub_topic", "control_session/metadata")

        self.metadata_pub = rospy.Publisher(self.metadata_topic, String, latch=True, queue_size=1)
        self.metadata_msg : String = String()

        self.control_gain_publisher: rospy.Publisher = None

        self.control_input_publisher: rospy.Publisher = None # Need to set it in post init
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tfsub_callback_style_control_loop = True
        
        # Set empty messages to be set in the main control logic.
        self.com_wrench_msg : WrenchStamped = None
        self.dipole_wrench_messages: List[WrenchStamped] = None
        self.control_input_message: VectorStamped = None
        self.control_gains_message: VectorStamped = None
        self.computation_time_msg = VectorStamped()
        self.computation_time_sample = 0
        self.current_computation_time = 0

        self.tracking_poses_on = rospy.get_param("oct_levitation/pose_tracking")

        ######## POST INIT CALL ########
        self.post_init()
        ######## POST INIT CALL ########

        rospy.logwarn(f"[Control Node] HARDWARE_CONNECTED: {self.HARDWARE_CONNECTED}")
        rospy.logwarn(f"[Control Node] Active Coils: {self.__ACTIVE_COILS}")
        rospy.logwarn(f"[Control Node] Active Driver Coils: {self.__ACTIVE_DRIVERS}")
        
        self.mpem_model = mag_manip.ForwardModelMPEM()
        self.mpem_model.setCalibrationFile(os.path.join(self.calfile_base_path, self.calibration_file))
        # Assuming that the dipole object has been set at this point. We will then start all the topics
        # and important subscribers.
        self.tf_sub_topic = self.rigid_body_dipole.pose_frame
        self.tf_reference_sub_topic = self.tf_sub_topic + "_reference"        

        if self.publish_desired_com_wrenches:
            self.com_wrench_publisher = rospy.Publisher(self.rigid_body_dipole.com_wrench_topic,
                                                        WrenchStamped,
                                                        queue_size=1)
        
        if self.publish_desired_dipole_wrenches:
            self.dipole_wrench_publishers = [rospy.Publisher(topic_name, WrenchStamped, queue_size= 1)
                                             for topic_name in self.rigid_body_dipole.dipole_wrench_topic_list]
        
        ######## HARDWARE ACTIVATION START ########
        # if in simulation mode, override the hardware connected flag
        if self.sim_mode:
            self.HARDWARE_CONNECTED = False
        self.desired_currents_msg, self.currents_publisher, self.publish_currents_impl = init_system("JECB", self.HARDWARE_CONNECTED, coil_nrs=self.__ACTIVE_DRIVERS)
        ######## HARDWARE ACTIVATION END ########

        # Start the timer
        timer_start_delay = rospy.get_param("~timer_start_delay", 1) # seconds
        if not self.tfsub_callback_style_control_loop:
            rospy.sleep(timer_start_delay) # This delay is used to stop the timer until important topics like the transforms
                                        # have been advertised.
            self.main_timer = rospy.Timer(rospy.Duration(1/self.control_rate), self.main_timer_loop)
        else:
            self.tf_sub = rospy.Subscriber(self.tf_sub_topic, TransformStamped, self.tfsub_callback,
                                           queue_size=1)
        
        self.tf_reference_sub = None
        self.last_reference_tf_msg = TransformStamped() # Empty message with zeros
        # Type casting to float to make sure JIT functions work with initial values.
        self.INITIAL_DESIRED_POSITION = np.asarray(self.INITIAL_DESIRED_POSITION, dtype=np.float64)
        self.INITIAL_DESIRED_ORIENTATION_EXYZ = np.asarray(self.INITIAL_DESIRED_ORIENTATION_EXYZ, dtype=np.float64)
        self.last_reference_tf_msg.transform.translation = Vector3(*self.INITIAL_DESIRED_POSITION)
        self.last_reference_tf_msg.transform.rotation = Quaternion(*geometry.quaternion_from_euler_xyz(self.INITIAL_DESIRED_ORIENTATION_EXYZ))
        if self.tracking_poses_on:
            self.tf_reference_sub = rospy.Subscriber(self.tf_reference_sub_topic, TransformStamped,
                                                     self.tf_reference_sub_callback, queue_size=1)
        
        self.control_gain_publisher.publish(self.control_gains_message)
        self.metadata_pub.publish(self.metadata_msg)

    def five_dof_wrench_allocation_single_dipole(self, tf_msg: TransformStamped, w_com: np.ndarray):
        """
        This function assumes that the dipole moment in the local frame aligns with the z axis and thus clips the 3rd row
        of the torque allocation map.
        tf_msg: The transform feedback from any state feedback sensor. Vicon usually.
        w_com: The desired COM/dipole center wrench. Forces are specified in the inertial frame while torques are specified in body fixed frame.
        """
        dipole_quaternion = geometry.numpy_quaternion_from_tf_msg(tf_msg.transform)
        dipole_position = geometry.numpy_translation_from_tf_msg(tf_msg.transform)
        dipole = self.rigid_body_dipole.dipole_list[0]
        dipole_vector = dipole.strength*geometry.rotate_vector_from_quaternion(dipole_quaternion, dipole.axis)
        Mf = geometry.magnetic_interaction_grad5_to_force(dipole_vector)
        Mt_local = geometry.magnetic_interaction_field_to_local_torque(dipole.strength,
                                                                       dipole.axis,
                                                                       dipole_quaternion)[:2] # Only first two rows will be nonzero
        A = self.mpem_model.getActuationMatrix(dipole_position)
        A = A[:, self.__ACTIVE_COILS] # only use active coils to compute currents.
        M = block_diag(Mt_local, Mf)

        JMA = M @ A
        computed_currents = np.linalg.pinv(JMA) @ w_com
        if self.sim_mode:
            des_currents = np.zeros(8)
            des_currents[self.__ACTIVE_COILS] = computed_currents
        else:
            # Due to real world coils being connected to a different limited set of drivers.
            des_currents = np.zeros(self.__N_CONNECTED_DRIVERS)
            des_currents[self.__ACTIVE_DRIVERS] = computed_currents # active coils are connected to these active drivers

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

    def post_init(self):
        """
        This function is always called at the end of the init function in the base class. Make sure to
        initialize all your variables like the dipole object properties for example in this function.
        Post init should preferably not be a blocking function. Make sure to read the base class
        if you are planning to add a blocking function call.

        The following mandatory attributed must be set:
        1. self.rigid_body_dipole: mechanical.MultiDipoleRigidBody (Set the rigid body)
        2. self.control_input_publisher: rospy.Publisher (Set the control input publisher)
        """
        raise NotImplementedError("The post init function has not been implemented yet.")
    
    def callback_control_logic(self, tf_msg: TransformStamped):
        """
        Implement all the important calculations and controller logic in this function.
        Set all the empty messages which are supposed to be published. The following
        mandatory attributed must be set:
            1. self.desired_currents_msg : DesCurrentsReg
            2. self.control_input_message

        The following optional attributed must be set.
            1. self.com_wrench_msg : WrenchStamped (if publish_desired_com_wrenches is True)
            2. self.dipole_wrench_messages: List[WrenchStamped] (if publish_desired_dipole_wrenches is True)
        """
        raise NotImplementedError("Control logic must be implemented")
    
    def tf_reference_sub_callback(self, tf_ref_msg: TransformStamped):
        self.last_reference_tf_msg = tf_ref_msg

    def publish_topics(self):
        # Publishing all the mandatory messages. They are all
        # set by the control_logic if it is implemented acc to
        # the specifications.
        self.control_input_publisher.publish(self.control_input_message)
        des_currents = np.asarray(self.desired_currents_msg.des_currents_reg)
        des_currents = np.clip(des_currents, -self.__MAX_CURRENT, self.__MAX_CURRENT)
        if np.any(np.abs(des_currents) == self.__MAX_CURRENT):
            rospy.logwarn_once(f"CURRENT LIMIT OF {self.__MAX_CURRENT}A HIT!")
        
        ### PUBLISH CURRENTS ACCORDING TO JASAN'S PROTOCOL ###
        self.publish_currents_impl(des_currents, self.desired_currents_msg, self.currents_publisher)

        if self.publish_desired_com_wrenches:
            self.com_wrench_publisher.publish(self.com_wrench_msg)
        
        if self.publish_desired_dipole_wrenches:
            for publisher, msg in zip(self.dipole_wrench_publishers, self.dipole_wrench_messages):
                publisher.publish(msg)
        
    def tfsub_callback(self, tf_msg: TransformStamped):
        start_time = time.perf_counter()
        self.callback_control_logic(tf_msg)
        self.publish_topics()
        stop_time = time.perf_counter()
        if self.publish_computation_time:
            self.current_computation_time += (stop_time - start_time)
            self.computation_time_sample += 1
            if self.computation_time_sample % self.computation_time_avg_samples == 0:
                self.computation_time_msg.header.stamp = rospy.Time.now()
                self.computation_time_msg.vector = [self.current_computation_time/self.computation_time_avg_samples]
                self.computation_time_pub.publish(self.computation_time_msg)
                self.computation_time_sample = 0
                self.current_computation_time = 0

    def main_timer_loop(self, event: rospy.timer.TimerEvent):
        start_time = time.perf_counter()
        dipole_tf = self.tf_buffer.lookup_transform("vicon/world", self.rigid_body_dipole.pose_frame, rospy.Time())
        self.callback_control_logic(dipole_tf)
        self.publish_topics()
        stop_time = time.perf_counter()
        if self.publish_computation_time:
            self.current_computation_time += (stop_time - start_time)
            self.computation_time_sample += 1
            if self.computation_time_sample % self.computation_time_avg_samples == 0:
                self.computation_time_msg.header.stamp = rospy.Time.now()
                self.computation_time_msg.vector = [self.current_computation_time/self.computation_time_avg_samples]
                self.computation_time_pub.publish(self.computation_time_msg)
                self.computation_time_sample = 0
                self.current_computation_time = 0

        