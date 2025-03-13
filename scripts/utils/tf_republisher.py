import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import rosgraph.masterapi

class DynamicTFRepublisher:
    def __init__(self):
        rospy.init_node('dynamic_tf_republisher', anonymous=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.subscribers = {}  # Store active subscribers

        # Periodically update the list of TransformStamped topics
        rospy.Timer(rospy.Duration(1.0), self.update_subscriptions)

    def tf_callback(self, msg: TransformStamped):
        """Broadcasts received TransformStamped messages to TF"""
        # Update the timestamp to the current time to allow for looping
        # bags without triggering Extrapolation errors from tf2.
        msg.header.stamp = rospy.Time.now()
        self.tf_broadcaster.sendTransform(msg)

    def update_subscriptions(self, event):
        """Checks active topics and subscribes to any new TransformStamped topics"""
        master = rosgraph.masterapi.Master('/rostopic')
        topic_list = master.getPublishedTopics('/')
        
        for topic_name, topic_type in topic_list:
            if topic_type == "geometry_msgs/TransformStamped" and topic_name not in self.subscribers:
                rospy.loginfo(f"Subscribing to new TransformStamped topic: {topic_name}")
                self.subscribers[topic_name] = rospy.Subscriber(topic_name, TransformStamped, self.tf_callback)

if __name__ == "__main__":
    DynamicTFRepublisher()
    rospy.spin()
