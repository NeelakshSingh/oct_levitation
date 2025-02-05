import os
import rospy
import oct_levitation.mechanical as mechanical
import tf2_ros

from geometry_msgs.msg import WrenchStamped
from tnb_mns_driver.msg import DesCurrentsReg
from mag_manip import mag_manip
from typing import List

from control_utils.general.utilities_jecb import init_hardware_and_shutdown_handler

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
        self.mpem_model = mag_manip.ForwardModelMPEM()
        self.mpem_model.setCalibrationFile(os.path.join(self.calfile_base_path, self.calibration_file))

        self.control_rate = rospy.get_param("~control_rate", 100)

        self.world_frame = rospy.get_param("~world_frame", "vicon/world")

        self.rigid_body_dipole: mechanical.MultiDipoleRigidBody = None # Set this in post init
        self.HARDWARE_CONNECTED = False;

        self.publish_desired_dipole_wrenches = rospy.get_param("~log_desired_dipole_wrench", False)
        self.publish_desired_com_wrenches = rospy.get_param("~log_desired_com_wrench", False)

        self.control_input_publisher: rospy.Publisher = None # Need to set it in post init
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.post_init()
        # Assuming that the dipole object has been set at this point. We will then start all the topics
        # and important subscribers.

        init_hardware_and_shutdown_handler(self.HARDWARE_CONNECTED)

        if self.publish_desired_com_wrenches:
            self.com_wrench_publisher = rospy.Publisher(self.rigid_body_dipole.com_wrench_topic,
                                                        WrenchStamped,
                                                        queue_size=1)
        
        if self.publish_desired_dipole_wrenches:
            self.dipole_wrench_publishers = [rospy.Publisher(topic_name, WrenchStamped, queue_size= 1)
                                             for topic_name in self.rigid_body_dipole.dipole_wrench_topic_list]
        
        self.currents_publisher = rospy.Publisher("tnb_mns_driver/des_currents_reg", DesCurrentsReg, queue_size=1)

        # Set empty messages to be set in the main control logic.
        self.desired_currents_msg : DesCurrentsReg = None
        self.com_wrench_msg : WrenchStamped = None
        self.dipole_wrench_messages: List[WrenchStamped] = None
        self.control_input_message = None

        # Start the timer
        timer_start_delay = rospy.get_param("~timer_start_delay", 1) # seconds
        rospy.sleep(timer_start_delay) # This delay is used to stop the timer until important topics like the transforms
                                       # have been advertised.
        self.main_timer = rospy.Timer(rospy.Duration(1/self.control_rate), self.main_timer_loop)

    
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
    
    def control_logic(self):
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
    
    def main_timer_loop(self, event):

        self.control_logic()

        # Publishing all the mandatory messages. They are all
        # set by the control_logic if it is implemented acc to
        # the specifications.
        self.control_input_publisher.publish(self.control_input_message)
        self.currents_publisher.publish(self.desired_currents_msg)

        if self.publish_desired_com_wrenches:
            self.com_wrench_publisher.publish(self.com_wrench_msg)
        
        if self.publish_desired_dipole_wrenches:
            for publisher, msg in zip(self.dipole_wrench_publishers, self.dipole_wrench_messages):
                publisher.publish(msg)