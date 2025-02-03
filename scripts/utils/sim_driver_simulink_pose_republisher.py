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
Due to issues with handling string message fields in simulink the pose topics are currently published
without the child frame. This will republish them with the child frame added so that they can be added
to the tf tree.

Always use bodies initialized with the Multidipole rigid body interface in order to use this script.
"""

RigidBody = rigid_bodies.TwoDipoleDisc100x15_6HKCM10x3

class SimulinkPoseRepublisher:
    
    def __init__(self) -> None:
        rospy.init_node("sim_driver_main", anonymous=False)

        self.world_frame = rospy.get_param("~world_frame", "vicon/world")

        # Due to issues with handling string message fields in simulink the pose topics are currently published
        # without the child frame. We will republish them with the child frame added.
        vicon_pose_pub = rospy.Publisher(RigidBody.pose_frame, TransformStamped, queue_size=100)
        self.matlab_vicon_pose_sub = rospy.Subscriber(RigidBody.pose_frame + "_no_frame",
                                          TransformStamped,
                                          partial(self.republish_transform_with_child_frame, 
                                                  child_frame=RigidBody.pose_frame,
                                                  republisher=vicon_pose_pub))
                
        self.matlab_dipole_pose_subs = []

        for dipole in RigidBody.dipole_list:
            self.matlab_dipole_pose_subs.append(
                rospy.Subscriber(
                    dipole.frame_name + "_no_frame",
                    TransformStamped,
                    partial(self.republish_transform_with_child_frame,
                            child_frame=dipole.frame_name,
                            republisher=rospy.Publisher(dipole.frame_name, TransformStamped, queue_size=100))
                )
            )

    def republish_transform_with_child_frame(self, msg:TransformStamped, child_frame: str, republisher: rospy.Publisher):
        msg.child_frame_id = child_frame
        msg.header.frame_id = self.world_frame
        republisher.publish(msg)


if __name__ == "__main__":
    driver = SimulinkPoseRepublisher()
    rospy.spin()