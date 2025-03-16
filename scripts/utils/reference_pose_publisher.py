import rospy
import numpy as np
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry

from geometry_msgs.msg import TransformStamped, Quaternion
from control_utils.traj_gen.lissajous_figures import return_lissajous_traj

class ReferencePosePublisher:
    def __init__(self):
        rospy.init_node("reference_pose_publisher")
        self.tf_topic = rigid_bodies.Onyx80x22DiscCenterRingDipole.pose_frame + "_reference"
        self.f_pub = 100

        ### Sinusoidal yaw
        # self.sine_yaw_amplitude = 2*np.pi/3

        ### Sinusoidal rp trajectory params
        # self.yaw_amplitude = 0.0
        # self.sine_roll_amplitude = np.deg2rad(15)
        # self.sine_pitch_amplitude = np.deg2rad(15)
        # self.f_roll = 0.1
        # self.omega_roll = 2*np.pi*self.f_roll
        # self.f_pitch = 0.1
        # self.omega_pitch = 2*np.pi*self.f_pitch

        ### Lissajous rp trajectory
        # self.lissajous_angle = np.deg2rad(30)
        # self.T_lissajous = 15
        # self.alpha_beta_traj = return_lissajous_traj('circle', self.lissajous_angle, 1/self.f_pub, False, 
        #                                         self.T_lissajous)
        # self.lissajous_traj_N = self.alpha_beta_traj.shape[0]
        # self.lissajous_step = 0
        # self.lissajous_counter = 0

        ### Sinusoidal Z
        self.z_amplitude = 1e-2
        self.Tz = 2
        self.z_omega = 2*np.pi/self.Tz


        self.start_time = rospy.Time.now().to_sec()


        self.tf_pub = rospy.Publisher(self.tf_topic, TransformStamped, queue_size=1)
        self.tf_timer = rospy.Timer(rospy.Duration(1/self.f_pub), self.tf_timer)
    
    def tf_timer(self, Event):
        t = rospy.Time.now().to_sec() - self.start_time
        pose = TransformStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "vicon/world"
        pose.child_frame_id = self.tf_topic

        # Origin position
        pose.transform.translation.x = 0.0
        pose.transform.translation.y = 0.0
        pose.transform.translation.z = 0.015
        
        # Zero RPY
        pose.transform.rotation = Quaternion(0, 0, 0, 1)

        ### Sinusoidal rpy
        # yaw = self.sine_yaw_amplitude*np.sin(self.omega*t)        
        # roll = self.sine_roll_amplitude*np.sin(self.omega_roll*t)
        # pitch = self.sine_pitch_amplitude*np.sin(self.omega_pitch*t)

        ### Constant rpy
        # roll = np.deg2rad(15)
        # pitch = np.deg2rad(15)

        ### Lissajous trajectory for rp
        # yaw = 0
        # self.lissajous_step = self.lissajous_counter % self.lissajous_traj_N
        # self.lissajous_counter += 1
        # roll = self.alpha_beta_traj[self.lissajous_step, 0]
        # pitch = self.alpha_beta_traj[self.lissajous_step, 1]

        # quaternion = geometry.quaternion_from_euler_xyz(np.array([roll, pitch, yaw]))
        # pose.transform.rotation = Quaternion(*quaternion)
        
        ### Sinusoidal Z
        pose.transform.translation.z = self.z_amplitude*np.sin(self.z_omega*t) + 0.01

        self.tf_pub.publish(pose)
    
    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rpp = ReferencePosePublisher()
    rpp.run()