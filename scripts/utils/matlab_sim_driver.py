import rospy
import os
import numpy as np
import tf2_ros
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry

from geometry_msgs.msg import WrenchStamped, TransformStamped, Vector3
from enum import Enum
from functools import partial

from tnb_mns_driver.msg import DesCurrentsReg

from mag_manip import mag_manip

"""
This file exposes a general interface to work with simulators like MATLAB and Gazebo. The idea is that instead of
building a simulated eMNS through physically based simulation we just used the forward field computation models
generally obtained through calibration procedures to calculate the actual (estimated) wrench on a body and then
we advertise this wrench over a topic to which simulators can subscribe.

Always use bodies initialized with the Multidipole rigid body interface in order to use this script.
"""

RigidBody = rigid_bodies.TwoDipoleDisc100x15_6HKCM10x3

class OperationModes(Enum):
    DESIRED_CURRENTS = 0
    DISABLE_WRENCH_PUB = 1

def get_operation_mode(value: int) -> OperationModes:
    """Returns the corresponding OperationModes enum for a given integer value."""
    try:
        return OperationModes(value)
    except ValueError:
        raise ValueError(f"Invalid operation mode: {value}. Must be one of {list(OperationModes)}")
    
class ControlSimDriver:
    
    def __init__(self) -> None:
        rospy.init_node("sim_driver_main", anonymous=False)

        self.wrench_topic = rospy.get_param("~wrench_topic", "actual_wrench")
        self.world_frame = rospy.get_param("~world_frame", "vicon/world")
        self.calfile_base_path = rospy.get_param("~calfile_base_path", os.path.join(os.environ["HOME"], ".ros/cal"))
        self.calibration_file = rospy.get_param("~mpem_cal_file", "mc3ao8s_md200_handp.yaml")
        self.operation_mode = get_operation_mode(rospy.get_param("~operation_mode_int", 0))

        # Initializing the forward model.
        self.mpem_model = mag_manip.ForwardModelMPEM()
        self.mpem_model.setCalibrationFile(os.path.join(self.calfile_base_path, self.calibration_file))

        self.operating_frequency = rospy.get_param("~frequency", 1000) # 1kHz publishing rate

        # The currents subscriber will be defined in a different process to make sure that the only 
        # thread running in this process can attain up to 1kHz operating frequencies.
        self.last_currents_received = np.zeros(8)
        self.currents_recv_timeout = rospy.get_param("~current_msg_timeout", 3)

        self.pub_topic_base_name = rospy.get_param("~wrench_topic_base_name", "/simscape/wrench")
        dipole_names = [os.path.basename(dipole.frame_name) for dipole in RigidBody.dipole_list]
        topic_names = [os.path.join(self.pub_topic_base_name, name) for name in dipole_names]

        # Operating mode specific topics
        self.currents_sub = rospy.Subscriber("tnb_mns_driver/des_currents_reg", DesCurrentsReg, self.currents_callback) # Service only the latest recvd msg.
        self.currents = np.zeros(8)
        self.__current_timeout_triggered = False
        self.__last_current_recv_time = rospy.Time.now().to_sec()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.wrench_pub_list = [rospy.Publisher(topic_name, WrenchStamped, queue_size=100) for topic_name in topic_names]

        # Due to issues with handling string message fields in simulink the pose topics are currently published
        # without the child frame. We will republish them with the child frame added.
        vicon_pose_pub = rospy.Publisher(RigidBody.pose_frame, TransformStamped, queue_size=1)
        self.matlab_vicon_pose_sub = rospy.Subscriber(RigidBody.pose_frame + "_no_frame",
                                          TransformStamped,
                                          partial(self.republish_transform_with_child_frame, 
                                                  child_frame=RigidBody.pose_frame,
                                                  republisher=vicon_pose_pub))
        
        self.__vicon_topics_republished = False
        
        self.matlab_dipole_pose_subs = []

        for dipole in RigidBody.dipole_list:
            self.matlab_dipole_pose_subs.append(
                rospy.Subscriber(
                    dipole.frame_name + "_no_frame",
                    TransformStamped,
                    partial(self.republish_transform_with_child_frame,
                            child_frame=dipole.frame_name,
                            republisher=rospy.Publisher(dipole.frame_name, TransformStamped, queue_size=1))
                )
            )

        # Starting the matlab wrench publishing routine
        if self.operation_mode != OperationModes.DISABLE_WRENCH_PUB:
            rospy.sleep(0.1)
            self.main_timer = rospy.Timer(rospy.Duration(1/self.operating_frequency), self.main_communication_loop)
    
    def currents_callback(self, msg: DesCurrentsReg) -> None:
        if self.__current_timeout_triggered:
            rospy.loginfo("[Sim Driver] Current msg stream online again.")
            self.__current_timeout_triggered = False

        self.currents = np.asarray(msg.des_currents_reg)
        if self.currents.shape[0] != 8:
            rospy.logwarn(f"[Sim Driver] Invalid currents received. Expected shape (8,) received: {self.currents.shape}")
        self.__last_current_recv_time = rospy.Time.now().to_sec()

    def republish_transform_with_child_frame(self, msg:TransformStamped, child_frame: str, republisher: rospy.Publisher):
        self.__vicon_topics_republished = True
        msg.child_frame_id = child_frame
        msg.header.frame_id = "vicon/world"
        msg.header.stamp = rospy.Time.now() # Restamping the data otherwise for some reason tf throws redundant data warning
        self.tf_broadcaster.sendTransform(msg)
        republisher.publish(msg)
    
    def main_communication_loop(self, event) -> None:
        # In every instant we will compute the wrench at the actual dipole frames based on 
        # the world frame and then publish it to the topics.
        if not self.__vicon_topics_republished:
            rospy.loginfo_once("Waiting for Vicon topics to be republished. Delaying main loop by 1 sec.")
            return
        time = rospy.Time.now().to_sec()
        if time - self.__last_current_recv_time > self.currents_recv_timeout:
            if not self.__current_timeout_triggered:
                rospy.logwarn("[Sim Driver] Current stream timeout. Sending 0A.")
                self.currents = np.zeros(8)
                self.__current_timeout_triggered = True
            
        # Calculating the actual force at each dipole frame position based on the state estimation
        # data published to the transform server. Ideally one should use Vicon Nexus to do so in 
        # the real system. For this case it will be Matlab.
        
        for idx, dipole in enumerate(RigidBody.dipole_list):
            dipole_tf: TransformStamped = self.tf_buffer.lookup_transform(self.world_frame,
                                                                          dipole.frame_name,
                                                                          rospy.Time())
            dipole_position = np.array([dipole_tf.transform.translation.x,
                                        dipole_tf.transform.translation.y,
                                        dipole_tf.transform.translation.z])
            
            actual_field_and_grad: np.ndarray = self.mpem_model.computeFieldGradient5FromCurrents(dipole_position, self.currents)

            M_dipole: np.ndarray = geometry.get_magnetic_interaction_matrix(dipole_tf=dipole_tf,
                                                                            dipole_strength=dipole.strength,
                                                                            dipole_axis=dipole.axis,
                                                                            full_mat=True,
                                                                            torque_first=True)
            
            actual_wrench = (M_dipole @ actual_field_and_grad).flatten()

            wrench_msg = WrenchStamped()

            wrench_msg.header.frame_id = dipole.frame_name
            wrench_msg.header.stamp = rospy.Time.now()
            
            wrench_msg.wrench.torque = Vector3(actual_wrench[0], actual_wrench[1], actual_wrench[2])
            wrench_msg.wrench.force = Vector3(actual_wrench[3], actual_wrench[4], actual_wrench[5])

            self.wrench_pub_list[idx].publish(wrench_msg)

if __name__ == "__main__":
    driver = ControlSimDriver()
    rospy.spin()