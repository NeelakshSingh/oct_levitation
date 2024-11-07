import numpy as np
import rospy
import tf2_ros
import tf.transformations as tr

from geometry_msgs.msg import TransformStamped
from tnb_mns_driver.msg import DesCurrentsReg

from oct_levitation.dynamics import ZLevitatingMassSystem
import oct_levitation.common as common

MAGNET_STACK_SIZE = 1
# MAGNET_TYPE = "wide_ring" # options: "wide_ring", "narrow_ring"
NORTH_DOWN = True
M_FRAME = 2.9e-3 # kg
M_SMALL_MAGNET = 2.25e-3 # kg
B = 0.0 # damping coefficient (friction)
X0 = np.array([0.03, 0]) # initial state, m
DIPOLE_AXIS = np.array([0, 0, 1])
L_BOUNDS = np.array([-0.05, -np.inf])
U_BOUNDS = np.array([0.05, np.inf])

class DynamicsSimulator:

    def __init__(self):
        rospy.init_node('dynamics_simulator', anonymous=True)
        self.Ts = 1e-3
        self.__tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.__tf_msg = TransformStamped()
        self.vicon_frame = f"vicon/narrow_ring_S{MAGNET_STACK_SIZE}/Origin"
        self.world_frame = "vicon/world"
        self.vicon_pub = rospy.Publisher(self.vicon_frame, TransformStamped, queue_size=10)

        self.last_command_recv = 0
        self.last_time = rospy.Time.now().to_sec()
        self.__first_call = True
        self.dt = 0

        self.__tf_msg.header.frame_id = self.world_frame
        self.__tf_msg.child_frame_id = self.vicon_frame
        m = M_FRAME + MAGNET_STACK_SIZE * M_SMALL_MAGNET
        self.sys = ZLevitatingMassSystem(m, B, X0)

        self.currents_sub = rospy.Subscriber("/tnb_mns_driver/des_currents_reg", DesCurrentsReg, self.currents_callback, queue_size=1)
        self.mag_model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                   calibration_file="octomag_5point.yaml")
        
        self.orientation_quat = None
        if NORTH_DOWN:
            self.orientation_quat = tr.quaternion_from_euler(np.pi, 0, 0)
        else:
            self.orientation_quat = tr.quaternion_from_euler(0, 0, 0)
        self.__tf_msg.transform.rotation.x = self.orientation_quat[0]
        self.__tf_msg.transform.rotation.y = self.orientation_quat[1]
        self.__tf_msg.transform.rotation.z = self.orientation_quat[2]
        self.__tf_msg.transform.rotation.w = self.orientation_quat[3]
        self.__tf_msg.transform.translation.x = 0
        self.__tf_msg.transform.translation.y = 0
        self.__tf_msg.transform.translation.z = X0[0]
        self.dipole_position = np.array([0, 0, X0[0]])
    
    def timer_callback(self, event):
        if self.__first_call:
            self.__first_call = False
            self.last_time = rospy.Time.now().to_sec()
        self.dt = rospy.Time.now().to_sec() - self.last_time
        self.last_time = rospy.Time.now().to_sec()
        self.sys.update(self.last_command_recv, self.dt)
        state = self.sys.get_state()
        self.dipole_position = np.array([0, 0, state[0]])
        self.__tf_msg.transform.translation.z = state[0]
        self.__tf_broadcaster.sendTransform(self.__tf_msg)
        self.vicon_pub.publish(self.__tf_msg)

    def currents_callback(self, msg: DesCurrentsReg):
        M = common.get_magnetic_interaction_matrix(self.__tf_msg, 
                                                   common.NarrowRingMagnet.dps,
                                                   DIPOLE_AXIS)
        A = self.mag_model.get_actuation_matrix(self.dipole_position)
        wrench = M @ A @ np.array(msg.des_currents_reg)
        fz = wrench[5]
        self.last_command_recv = fz
    
    def run(self):
        self.timer = rospy.Timer(rospy.Duration(self.Ts), self.timer_callback)

if __name__ == "__main__":
    dynamics_simulator = DynamicsSimulator()
    dynamics_simulator.run()
    rospy.spin()