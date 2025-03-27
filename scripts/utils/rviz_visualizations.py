import rospy
import numpy as np
import os

from geometry_msgs.msg import TransformStamped, WrenchStamped, Vector3, Quaternion, Vector3Stamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry


class OctLevitationRvizVisualizations:

    def __init__(self) -> None:
        rospy.init_node('oct_levitation_rviz_visualizations', anonymous=True)
        # Parameters
        self.rigid_body = rigid_bodies.Onyx80x22DiscCenterRingDipole
        self.world_frame = rospy.get_param("~world_frame", "vicon/world")
        self.reduced_attitude_params = {
            "fixed_vector": np.array(rospy.get_param("~reduced_attitude_fixed_vector", [0.0, 0.0, 1.0])),
            "rgba": rospy.get_param("~reduced_attitude_rgba", [0.0, 0.0, 1.0, 1.0]),
            "scale": rospy.get_param("~reduced_attitude_scale", [0.05, 0.1, 0.1])
        }
        self.topics_base_name = rospy.get_param("~topics_base_name", "single_dipole_rviz_visualizations")

        self.reduced_attitude_params["fixed_vector"] = self.reduced_attitude_params["fixed_vector"] / np.linalg.norm(self.reduced_attitude_params["fixed_vector"])

        self.vicon_frame = self.rigid_body.pose_frame
        self.rpy_topic = os.path.join(self.vicon_frame, "rpy")

        self.vicon_sub = rospy.Subscriber(self.vicon_frame, TransformStamped, self.vicon_callback)
        self.last_received_vicon_msg = None
        self.rpy_pub = rospy.Publisher(self.rpy_topic, Vector3Stamped, queue_size=10)
        self.reduced_attitude_topic = os.path.join(self.topics_base_name, "reduced_attitude")
        self.reduced_attitude_marker_topic = os.path.join(self.topics_base_name, "reduced_attitude/marker")
        self.reduced_attitude_pub = rospy.Publisher(self.reduced_attitude_topic, Vector3Stamped, queue_size=10)
        self.reduced_attitude_marker_pub = rospy.Publisher(self.reduced_attitude_marker_topic, Marker, queue_size=10)

        self.__rpy_msg = Vector3Stamped()
        self.__rpy_msg.header.frame_id = self.world_frame
        self.__reduced_attitude_msg = Vector3Stamped()
        self.__reduced_attitude_msg.header.frame_id = self.world_frame
        
        ## Reduced attitude marker
        self.__reduced_attitude_marker_msg = Marker()
        self.__reduced_attitude_marker_msg.header.frame_id = self.world_frame # Because reduced attitude vector is resolved in the world frame.
        self.__reduced_attitude_marker_msg.ns = "reduced_attitude"
        self.__reduced_attitude_marker_msg.id = 0
        self.__reduced_attitude_marker_msg.type = Marker.ARROW
        self.__reduced_attitude_marker_msg.action = Marker.MODIFY
        self.__reduced_attitude_marker_msg.pose.orientation = Quaternion(0, 0, 0, 1)  # No rotation
        # Arrow properties
        self.__reduced_attitude_marker_msg.scale = Vector3(*self.reduced_attitude_params["scale"])  # Shaft diameter, head diameter, head length
        self.__reduced_attitude_marker_msg.color = ColorRGBA(*self.reduced_attitude_params["rgba"])  # Blue color (RGBA)
        self.__reduced_attitude_marker_msg.lifetime = rospy.Duration(0)  # Keep arrow visible indefinitely

        self.__reduced_attitude_ref_vec = np.array(rospy.get_param("~reduced_attitude_ref_vec", [0.0, 0.0, 1.0]))
        self.__reduced_attitude_ref_vec = self.__reduced_attitude_ref_vec / np.linalg.norm(self.__reduced_attitude_ref_vec)

        ## For COM Wrench
        self.wrench_sub = rospy.Subscriber(self.rigid_body.com_wrench_topic, WrenchStamped, self.wrench_callback, queue_size=1)
        self.wrench_pub = rospy.Publisher(os.path.join(self.topics_base_name, "dipole_frame_com_wrench"), WrenchStamped, queue_size=10)
    
    def vicon_callback(self, msg: TransformStamped) -> None:
        ## Publishing reduced attitude and its marker
        self.__reduced_attitude_msg.header.stamp = rospy.Time.now()
        Lambda_vec = Vector3(*geometry.inertial_reduced_attitude_from_quaternion(geometry.numpy_quaternion_from_tf_msg(msg.transform), 
                                                                                 self.reduced_attitude_params["fixed_vector"]))
        self.__reduced_attitude_msg.vector = Lambda_vec
        self.reduced_attitude_pub.publish(self.__reduced_attitude_msg)

        start_point = Point()
        end_point = Point(Lambda_vec.x, Lambda_vec.y, Lambda_vec.z)
        self.__reduced_attitude_marker_msg.points = [start_point, end_point]
        self.__reduced_attitude_marker_msg.header.stamp = rospy.Time.now()

        self.reduced_attitude_marker_pub.publish(self.__reduced_attitude_marker_msg)

        ## Publishing RPY
        rpy = geometry.euler_xyz_from_quaternion(geometry.numpy_quaternion_from_tf_msg(msg.transform))
        self.__rpy_msg.header.stamp = rospy.Time.now()
        self.__rpy_msg.vector = Vector3(*np.rad2deg(rpy))
        self.rpy_pub.publish(self.__rpy_msg)

        self.last_received_vicon_msg = msg
    
    def wrench_callback(self, msg: WrenchStamped) -> None:
        """
        The purpose of this callback is to visualize the global frame forces on the object.
        And the local/global frame torques on the object.
        """
        if self.last_received_vicon_msg:
            # Forces are published in world frame, so we will convert them to the object frame.
            # in order to visualize them at the object origin in rviz.
            wrench_msg = msg
            wrench_msg.header.stamp = rospy.Time.now()
            # Also we will publish torques in mN-m and forces in N
            wrench_msg.header.frame_id = self.rigid_body.pose_frame
            local_force = geometry.rotate_vector_from_quaternion(
                geometry.invert_quaternion(geometry.numpy_quaternion_from_tf_msg(self.last_received_vicon_msg.transform)),
                np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]))
            wrench_msg.wrench.force = Vector3(*local_force)
            wrench_msg.wrench.torque = Vector3(*np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])*1e3)
            # Torques are assumed to be published in the object frame.
            self.wrench_pub.publish(wrench_msg)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    visualizer = OctLevitationRvizVisualizations()
    visualizer.run()