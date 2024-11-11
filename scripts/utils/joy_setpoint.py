import rospy
import numpy as np
import tf2_ros
import tf.transformations as tr

from sensor_msgs.msg import Joy, JoyFeedback
from geometry_msgs.msg import TransformStamped, Quaternion
from dataclasses import dataclass

from oct_levitation.common import vector_skew_symmetric_matrix

# Button Mappings for Xbox One Controller with medusalix/xone
# https://github.com/medusalix/xone

@dataclass
class XboxOneAxes:
    LEFT_STICK_X: int = 0
    LEFT_STICK_Y: int = 1
    LEFT_TRIGGER: int = 5
    RIGHT_TRIGGER: int = 4
    RIGHT_STICK_X: int = 2
    RIGHT_STICK_Y: int = 3
    DPAD_UD: int = 7
    DPAD_LR: int = 6
    

    stick_range: tuple = (-1.0, 1.0)
    trigger_range: tuple = (-1.0, 1.0)
    stick_free_signal: float = 0.0
    trigger_free_signal: float = -1.0
    dpad_range: tuple = (-1.0, 1.0)
    dpad_left_signal: float = -1.0
    dpad_right_signal: float = 1.0
    dpad_up_signal: float = 1.0
    dpad_down_signal: float = -1.0

@dataclass
class XboxOneButtons:
    A: int = 0
    B: int = 1
    X: int = 4
    Y: int = 3
    LB: int = 6
    RB: int = 7
    LS: int = 13
    RS: int = 14
    MENU: int = 11
    SCREENSHOT: int = 10
    XBOX: int = 12

@dataclass
class XboxOneController:
    Axes: XboxOneAxes = XboxOneAxes()
    Buttons: XboxOneButtons = XboxOneButtons()


class JoySetpointSetter:

    def __init__(self):
        rospy.init_node('joy_setpoint_setter', anonymous=True)
        ## CONFIGURATION PARAMETERS
        # Other settings for setpoint frame broadcasting
        self.magnet_name = rospy.get_param("~magnet_name", "small_ring")
        self.magnet_stack_size = rospy.get_param("~magnet_stack_size", 1)
        self.home_frame = "vicon/{}_S{}/Origin".format(self.magnet_name, self.magnet_stack_size)
        self.setpoint_frame = "{}_S{}/DeltaSetpoint".format(self.magnet_name, self.magnet_stack_size)
        self.setpoint_rate = rospy.get_param("~setpoint_rate", 50) # [Hz]
        # Linear Velocity Limits
        self.vz_max = rospy.get_param("~vz_max", 0.01) # [m/s]
        self.vx_max = rospy.get_param("~vx_max", 0.01)
        self.vy_max = rospy.get_param("~vy_max", 0.01)
        # Angular Velocity Limits
        self.wz_max = rospy.get_param("~wz_max", np.deg2rad(10)) # [rad/s]
        self.wx_max = rospy.get_param("~wx_max", np.deg2rad(10))
        self.wy_max = rospy.get_param("~wy_max", np.deg2rad(10))

        ## INITIALIZATION
        self.__tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.__tf_msg = TransformStamped()
        self.__tf_msg.header.frame_id = self.home_frame
        self.__tf_msg.child_frame_id = self.setpoint_frame

        ## Time bookkeeping
        self.__last_time = rospy.Time.now().to_sec()
        self.__first_time = True
        self.__last_joy_msg = Joy()
        self.__last_joy_msg.axes = [0.0]*8
        self.__last_joy_msg.buttons = [0]*15

        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=1)
        self.setpoint_timer = rospy.Timer(rospy.Duration(1/self.setpoint_rate), self.setpoint_timer_callback)
    
    def joy_callback(self, msg):
        self.__last_joy_msg = msg
    
    def setpoint_timer_callback(self, event):
        if self.__first_time:
            self.__first_time = False
            self.__last_time = rospy.Time.now().to_sec()
        dt = rospy.Time.now().to_sec() - self.__last_time
        # Linear Setpoint Changes
        self.__tf_msg.transform.translation.x = dt * self.vx_max * self.__last_joy_msg.axes[XboxOneController.Axes.LEFT_STICK_X]
        self.__tf_msg.transform.translation.y = dt * self.vy_max * self.__last_joy_msg.axes[XboxOneController.Axes.LEFT_STICK_Y]
        self.__tf_msg.transform.translation.z = dt * self.vz_max * (self.__last_joy_msg.axes[XboxOneController.Axes.LEFT_TRIGGER] - self.__last_joy_msg.axes[XboxOneController.Axes.RIGHT_TRIGGER])
        # Angular Setpoint Changes
        wx = self.wx_max * self.__last_joy_msg.axes[XboxOneController.Axes.RIGHT_STICK_X]
        wy = self.wy_max * self.__last_joy_msg.axes[XboxOneController.Axes.RIGHT_STICK_Y]
        wz = self.wz_max * self.__last_joy_msg.axes[XboxOneController.Axes.DPAD_LR]
        # We will define these rotations with respect to the world frame of vicon. As if they happen to an object whose frame origin
        # is at the vicon world frame and then we rotate the object frame. theta_x is about the x-axis of the world frame, and so on.
        # We will use the quaternion representation of these rotations.
        R = np.eye(4)
        R[:3, :3] = vector_skew_symmetric_matrix(np.array([wx, wy, wz])) * dt
        q = tr.quaternion_from_matrix(R)
        q /= np.linalg.norm(q, 2)
        self.__tf_msg.transform.rotation = Quaternion(*q)
        self.__tf_msg.header.stamp = rospy.Time.now()
        self.__tf_broadcaster.sendTransform(self.__tf_msg)
        return True

if __name__ == "__main__":
    joy_setpoint_setter = JoySetpointSetter()
    rospy.spin()