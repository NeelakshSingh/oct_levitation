import rospy
import numpy as np
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry

from geometry_msgs.msg import TransformStamped, Quaternion

class FakePosePublisher:
    def __init__(self):
        rospy.init_node("fake_pose_publisher")
        self.tf_topic = rigid_bodies.TwoDipoleDisc80x15_6HKCM10x3.pose_frame
        self.T = 10 # cycle period in seconds
        self.omega = 2*np.pi/self.T
        self.amplitude = np.pi/4
        self.start_time = rospy.Time.now().to_sec()

        self.f_pub = 100
        self.tf_pub = rospy.Publisher(self.tf_topic, TransformStamped, queue_size=1)
        self.tf_timer = rospy.Timer(rospy.Duration(1/self.f_pub), self.tf_timer)
    
    def tf_timer(self, Event):
        t = rospy.Time.now().to_sec() - self.start_time
        pose = TransformStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "vicon/world"
        pose.child_frame_id = self.tf_topic
        pose.transform.translation.x = 0.0
        pose.transform.translation.y = 0.0
        pose.transform.translation.z = 0.0
        
        yaw = self.amplitude*np.sin(self.omega*t)
        quaternion = geometry.quaternion_from_euler_zyx(np.array([0, 0, yaw]))
        pose.transform.rotation = Quaternion(*quaternion)
        self.tf_pub.publish(pose)
    
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    fpp = FakePosePublisher()
    fpp.run()