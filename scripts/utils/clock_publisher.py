import rospy
import time

from rosgraph_msgs.msg import Clock

"""
Simple helper node in order to publish system time over the clock topic.
Useful for synchronizing MATLAB simulators on different systems with this system's
timestamp.
"""

if __name__ == "__main__":
    rospy.init_node("oct_levitation_clock_publisher")
    clock_rate = rospy.get_param("~clock_rate", 1000)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)

    rospy.loginfo(f"[Clock Publisher] Running at {clock_rate} Hz.")

    def publish_clock_timer(event):
        clock_msg = Clock()
        clock_msg.clock = rospy.Time.from_sec(time.time())
        clock_pub.publish(clock_msg)

    clock_timer = rospy.Timer(rospy.Duration(1/clock_rate), publish_clock_timer)
    rospy.spin()