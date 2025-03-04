import rospy
import numpy as np
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry

from geometry_msgs.msg import TransformStamped, Quaternion

class FakePosePublisher:
    def __init__(self):
        rospy.init_node("fake_pose_publisher")
        self.tf_topic = rigid_bodies.Onyx80x22DiscCenterRingDipole.pose_frame
        self.T = 2 # cycle period in seconds
        self.omega = 2*np.pi/self.T
        # self.yaw_amplitude = 2*np.pi/3
        self.yaw_amplitude = 0.0
        self.roll_amplitude = np.deg2rad(15)
        self.pitch_amplitude = np.deg2rad(15)
        self.f_roll = 0.1
        self.omega_roll = 2*np.pi*self.f_roll
        self.f_pitch = 0.1
        self.omega_pitch = 2*np.pi*self.f_pitch
        self.z_amplitude = 0.5e-2
        # self.amplitude = 0.0007252745435843604 # This one for z will lead to poor conditioning of JMA matrix.
        self.start_time = rospy.Time.now().to_sec()

        self.f_pub = 1000
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
        # pose.transform.translation.z = 0.0129
        
        pose.transform.rotation = Quaternion(0, 0, 0, 1)
        # yaw = self.yaw_amplitude*np.sin(self.omega*t)        

        # roll = self.roll_amplitude*np.sin(self.omega_roll*t)
        # pitch = self.pitch_amplitude*np.sin(self.omega_pitch*t)
        roll = np.deg2rad(15)
        pitch = np.deg2rad(15)
        yaw = 0

        quaternion = geometry.quaternion_from_euler_zyx(np.array([roll, pitch, yaw]))
        pose.transform.rotation = Quaternion(*quaternion)

        pose.transform.translation.z = self.z_amplitude*np.sin(self.omega*t) + 0.01

        self.tf_pub.publish(pose)
    
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    fpp = FakePosePublisher()
    fpp.run()