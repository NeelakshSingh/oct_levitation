import numpy as np
import oct_levitation.geometry as geometry
import rospy

from geometry_msgs.msg import TransformStamped
from oct_levitation.msg import Float64MultiArrayStamped

class EulerPoseConverter:

    def __init__(self):
        rospy.init_node("euler_pose_converter")
        self.pose_topic = rospy.get_param("/pose_topic", "vicon/small_ring_S1/Origin")
        self.euler_pose_topic = self.pose_topic + "_euler"
        self.pose_sub = rospy.Subscriber(self.pose_topic, TransformStamped, self.pose_callback)
        self.euler_pose_pub = rospy.Publisher(self.euler_pose_topic, Float64MultiArrayStamped, queue_size=10)
    
    def pose_callback(self, msg: TransformStamped):
        quaternion = np.array([msg.transform.rotation.x,
                               msg.transform.rotation.y,
                               msg.transform.rotation.z,
                               msg.transform.rotation.w])
        euler = geometry.euler_xyz_from_quaternion(quaternion)
        euler_msg = Float64MultiArrayStamped()
        euler_msg.header = msg.header
        euler_msg.array.data = [msg.transform.translation.x,
                                msg.transform.translation.y,
                                msg.transform.translation.z,
                                euler[0],
                                euler[1],
                                euler[2]]
        self.euler_pose_pub.publish(euler_msg)

if __name__ == "__main__":
    converter = EulerPoseConverter()
    rospy.spin()