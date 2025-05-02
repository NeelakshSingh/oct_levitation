#!/usr/bin/env python

import rospy
from control_utils.msg import VectorStamped
from tnb_mns_driver.msg import DesCurrentsReg
from geometry_msgs.msg import TransformStamped
import os

class MessageDelayNode:
    def __init__(self, topic_name):
        if "vicon" in topic_name:
            self.sub = rospy.Subscriber(topic_name, TransformStamped, self.callback)
        elif "tnb_mns_driver/des_currents_reg" in topic_name:
            self.sub = rospy.Subscriber(topic_name, DesCurrentsReg, self.callback)
        else:
            raise ValueError("Unsupported topic type: " + topic_name)
        pub_topic = os.path.join(topic_name, "delay")
        self.pub = rospy.Publisher(pub_topic, VectorStamped, queue_size=10)
        rospy.loginfo(f"Subscribed to topic: {topic_name}")

    def callback(self, msg):
        now = rospy.Time.now()
        delay = (now - msg.header.stamp).to_sec()
        to_send = VectorStamped()
        to_send.header.stamp = now
        to_send.vector = [delay]
        self.pub.publish(to_send)
        rospy.logdebug(f"Message delay: {delay:.6f} seconds")

if __name__ == "__main__":
    rospy.init_node("message_delay_meas_node", anonymous=True)

    topic = rospy.get_param("~topic")
    node = MessageDelayNode(topic)
    rospy.spin()