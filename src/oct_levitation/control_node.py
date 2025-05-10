import os
import rospy
import oct_levitation.mechanical as mechanical
import oct_levitation.geometry_jit as geometry
import oct_levitation.numerical as numerical
from oct_levitation.rigid_bodies import REGISTERED_BODIES
import tf2_ros
import numpy as np
import time
import sys

from geometry_msgs.msg import WrenchStamped, TransformStamped, Quaternion, Vector3
from tnb_mns_driver.msg import DesCurrentsReg
from control_utils.msg import VectorStamped
from std_msgs.msg import String
from mag_manip import mag_manip
from typing import List
from scipy.linalg import block_diag

from control_utils.general.utilities import init_system
from oct_levitation.msg import ControllerDetails

### Uncomment the following lines and associated code in the callback_control_logic to enable profiling
# import cProfile
# import atexit
# import pstats

# Profiler = cProfile.Profile()
# PROFILE_FREQUENCY = 4 # Hz
# LAST_PROFILE_TIME = 0

# import rospkg
# rospack = rospkg.RosPack()
# pkg_path = rospack.get_path("oct_levitation")
# profiler_file_stats_path = os.path.join(pkg_path, "profiler_stats")

# def dump_profiler_stats():
#     """
#     Dumps the profiler stats to a file.
#     """
#     stats = pstats.Stats(Profiler).sort_stats("cumulative")
#     if not os.path.exists(profiler_file_stats_path):
#         os.makedirs(profiler_file_stats_path)
#     profiler_file = os.path.join(profiler_file_stats_path, "profiler_stats.prof")
#     stats.dump_stats(profiler_file)
#     rospy.loginfo(f"Profiler stats dumped to {profiler_file}")

# atexit.register(dump_profiler_stats)

class ControlSessionNodeBase:
    """
    This class contains all the basic functionalities that a ROS Node implementing a levitation controller
    must contain. These functions mostly pertain to creating topics required for important data logging
    purposes in order to perform a detailed experimental analysis later.
    """
    def __init__(self):
        rospy.init_node("oct_levitation_controller_node")

        self.calfile_base_path = rospy.get_param("~calfile_base_path", os.path.join(os.environ["HOME"], ".ros/cal"))
        self.calibration_file = rospy.get_param('oct_levitation/calibration_file')
        self.RT_PRIORITY_ENABLED = rospy.get_param("~rtprio_controller", False)
        self.CONTROL_RATE = rospy.get_param("oct_levitation/control_freq") # Set it to the vicon frequency
        self.rigid_body_dipole: mechanical.MultiDipoleRigidBody = REGISTERED_BODIES[rospy.get_param("oct_levitation/rigid_body")]
        rospy.loginfo(f"Rigid body: {self.rigid_body_dipole.name}")
        self.sim_mode = rospy.get_param("~sim_mode") # Mandatory param, wait for it to be set.
        self.__N_CONNECTED_DRIVERS = int(rospy.get_param("oct_levitation/n_drivers")) # number of used drivers can be less
        self.__MAX_CURRENT = 4.0 # Amps
        self.__SOFT_START = rospy.get_param("oct_levitation/soft_start", True)
        self.soft_starter = None
        if self.__SOFT_START:
            self.soft_starter = numerical.LinearSoftStarter(0.5, 1.0)
        self.__HARDWARE_CONNECTED = rospy.get_param("~hardware_connected", default=False) # to force explicit enablement in post init.

        self.world_frame = rospy.get_param("~world_frame", "vicon/world")

        self.publish_computation_time = rospy.get_param("oct_levitation/publish_computation_time")
        self.computation_time_topic = "control_session/computation_time"
        self.compute_time_pub = None
        self.__ACTIVE_COILS = np.asarray(rospy.get_param('oct_levitation/active_coils'), dtype=int) # This variable should be hidden from the derived classes since this is supposed to stay fixed for all experiments.
        self.__ACTIVE_DRIVERS = np.asarray(rospy.get_param('oct_levitation/active_drivers'), dtype=int) # These are the exact driver numbers these coils are connected to.
        if len(self.__ACTIVE_DRIVERS) != len(self.__ACTIVE_COILS):
            msg = f"Active coils and drivers must be the same size. Active coils: {self.__ACTIVE_COILS}, Active drivers: {self.__ACTIVE_DRIVERS}"
            rospy.logerr(msg)
            raise ValueError(msg)
        self.INITIAL_DESIRED_POSITION = np.array([0.0, 0.0, 0.0])
        self.INITIAL_DESIRED_ORIENTATION_EXYZ = np.array([0.0, 0.0, 0.0])

        if self.publish_computation_time:
            self.computation_time_pub = rospy.Publisher(self.computation_time_topic, VectorStamped, queue_size=1)


        self.publish_desired_com_wrenches = rospy.get_param("~log_desired_com_wrench", False)
        self.metadata_topic = rospy.get_param("~metadata_pub_topic", "control_session/metadata")

        self.metadata_pub = rospy.Publisher(self.metadata_topic, ControllerDetails, latch=True, queue_size=1)
        self.metadata_msg : ControllerDetails = ControllerDetails()

        self.control_gain_publisher: rospy.Publisher = None

        self.control_input_publisher: rospy.Publisher = None # Need to set it in post init
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tfsub_callback_style_control_loop = True
        
        # Set empty messages to be set in the main control logic.
        self.com_wrench_msg : WrenchStamped = None
        self.dipole_wrench_messages: List[WrenchStamped] = None
        self.control_gains_message: VectorStamped = None
        self.computation_time_msg = VectorStamped()
        self.computation_time_sample = 0
        self.current_computation_time = 0

        self.tracking_poses_on = rospy.get_param("oct_levitation/pose_tracking")
        self.LAST_PROFILE_TIME = rospy.Time(0)

        ######## POST INIT CALL ########
        self.post_init()
        ######## POST INIT CALL ########

        rospy.logwarn(f"[Control Node] HARDWARE_CONNECTED: {self.__HARDWARE_CONNECTED}")
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
            self.__SOFT_START = False
            self.__HARDWARE_CONNECTED = False
            self.__MAX_CURRENT = 12.0 # Amps
        self.desired_currents_msg, self.currents_publisher, self.publish_currents_impl, shutdown_hook = init_system("JECB", self.__HARDWARE_CONNECTED, coil_nrs=self.__ACTIVE_DRIVERS)
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

        ###### Metadata message filling ######

        # Check if path metadata was set
        if self.metadata_msg.controller_path.data == "" or self.metadata_msg.controller_name.data == "":
            rospy.logerr("set_path_metadata() was not called in post_init() of derived class. Please add the call for proper logging. Aborting.")
            raise ValueError("set_path_metadata() was not called in post_init() of derived class. Please add the call for proper logging. Aborting.")

        self.metadata_msg.header.stamp = rospy.Time.now()
        self.metadata_msg.data_recording_sub_folder.data = rospy.get_param("~data_subfolder")
        self.metadata_msg.experiment_description.data = rospy.get_param("~experiment_description")
        self.metadata_msg.full_controller_class_state.data = str(self.__dict__)
        self.metadata_msg.metadata.data += f"\n HARDWARE_CONNECTED: {self.__HARDWARE_CONNECTED} \n"

        self.metadata_pub.publish(self.metadata_msg)


    def set_path_metadata(self, file):
        file_path = os.path.abspath(file)
        self.metadata_msg.controller_name.data = os.path.basename(file_path)
        self.metadata_msg.controller_path.data = file_path

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
        A = self.mpem_model.getActuationMatrix(dipole_position)
        A = A[:, self.__ACTIVE_COILS] # only use active coils to compute currents.
        M = geometry.magnetic_interaction_force_local_torque(dipole.local_dipole_moment, dipole_quaternion, remove_z_torque=True)
        JMA = M @ A

        computed_currents = numerical.numba_pinv(JMA) @ w_com
        if self.sim_mode:
            des_currents = np.zeros(8)
            des_currents[self.__ACTIVE_COILS] = computed_currents
        else:
            # Due to real world coils being connected to a different limited set of drivers.
            des_currents = np.zeros(self.__N_CONNECTED_DRIVERS)
            des_currents[self.__ACTIVE_DRIVERS] = computed_currents # active coils are connected to these active drivers

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
    
    def callback_control_logic(self, tf_msg: TransformStamped, sft_coeff: float = 1.0):
        """
        Implement all the important calculations and controller logic in this function.
        Set all the empty messages which are supposed to be published. The following
        mandatory attributed must be set:
            1. self.desired_currents_msg : DesCurrentsReg
            2. self.control_input_message

        The following optional attributed must be set.
            1. self.com_wrench_msg : WrenchStamped (if publish_desired_com_wrenches is True)
        """
        raise NotImplementedError("Control logic must be implemented")
    
    def tf_reference_sub_callback(self, tf_ref_msg: TransformStamped):
        self.last_reference_tf_msg = tf_ref_msg

    def publish_topics(self):
        # Publishing all the mandatory messages. They are all
        # set by the control_logic if it is implemented acc to
        # the specifications.
        des_currents = np.asarray(self.desired_currents_msg.des_currents_reg)
        des_currents = np.clip(des_currents, -self.__MAX_CURRENT, self.__MAX_CURRENT)
        if np.any(np.abs(des_currents) == self.__MAX_CURRENT):
            rospy.logwarn_once(f"CURRENT LIMIT OF {self.__MAX_CURRENT}A HIT!")
        
        ### PUBLISH CURRENTS ACCORDING TO JASAN'S PROTOCOL ###
        self.desired_currents_msg.header.stamp = rospy.Time.now()
        self.publish_currents_impl(des_currents, self.desired_currents_msg, self.currents_publisher)

        if self.publish_desired_com_wrenches:
            self.com_wrench_publisher.publish(self.com_wrench_msg)
    
    def check_shutdown_rt(self):
        """
        This functions is important to ensure a clean exit of the node when it is run with RT priority since it will never yield
        otherwise which will never cause the code to check or respond to shutdown signals and will eventually be SIGKILLed by
        roslaunch. This function really matters in order to call the shutdown hook of the driver and make sure that the ECB's stop
        service is called. Of course, shutting down the tnb_mns_driver should still do the job and any launches of the driver node
        should be left as is and not be modified without knowing exactly what you are doing and what the consequences could be.

        The resetting of the scheduler to SCHED_OTHER was important otherwise it seems like roslaunch will not register the exit
        from the node.
        """
        if self.RT_PRIORITY_ENABLED and rospy.is_shutdown_requested():
            rospy.loginfo("[CONTROLLER CALLBACK LOGIC] ROS shutdown requested")
            rospy.signal_shutdown("ROS shutdown requested")
            os.sched_setscheduler(os.getpid(), os.SCHED_OTHER, os.sched_param(0)) # Revert to default scheduler and give up RT priority to catch the shutdown signal
            os.sched_yield()
            time.sleep(0.1)
            sys.exit(0)
            return
        
    def tfsub_callback(self, tf_msg: TransformStamped):
        start_time = time.perf_counter()
        # now = rospy.Time.now()
        # PROFILER_ENABLED = False
        # if (now - self.LAST_PROFILE_TIME).to_sec() > 1.0 / PROFILE_FREQUENCY:
        #     Profiler.enable()
        #     PROFILER_ENABLED = True
        #     self.LAST_PROFILE_TIME = now
        ## TODO: Maybe it makes more sense to smooth start Fz
        self.check_shutdown_rt()
        coeff = 1
        if self.__SOFT_START:
            coeff = self.soft_starter(1/self.CONTROL_RATE)
            if np.allclose(coeff, 1.0):
                coeff = 1.0
                self.__SOFT_START = False # Disable it from this point onwards
        self.callback_control_logic(tf_msg, coeff)
        self.publish_topics()
        # if PROFILER_ENABLED:
        #     Profiler.disable()
        stop_time = time.perf_counter()
        if self.publish_computation_time:
            self.current_computation_time = (stop_time - start_time)
            self.computation_time_msg.header.stamp = rospy.Time.now()
            self.computation_time_msg.vector = [self.current_computation_time]
            self.computation_time_pub.publish(self.computation_time_msg)

    def main_timer_loop(self, event: rospy.timer.TimerEvent):
        start_time = time.perf_counter()
        self.check_shutdown_rt()
        dipole_tf = self.tf_buffer.lookup_transform("vicon/world", self.rigid_body_dipole.pose_frame, rospy.Time())
        self.callback_control_logic(dipole_tf)
        self.publish_topics()
        stop_time = time.perf_counter()
        if self.publish_computation_time:
            self.current_computation_time = (stop_time - start_time)
            self.computation_time_msg.header.stamp = rospy.Time.now()
            self.computation_time_msg.vector = [self.current_computation_time]
            self.computation_time_pub.publish(self.computation_time_msg)