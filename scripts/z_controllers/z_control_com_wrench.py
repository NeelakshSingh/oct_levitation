import rospy

import oct_levitation.rigid_bodies as rigid_bodies

from oct_levitation.control_node import ControlSessionNodeBase
from control_utils.msg import VectorStamped
from geometry_msgs.msg import WrenchStamped
from tnb_mns_driver.msg import DesCurrentsReg

class DirectCOMWrenchZController(ControlSessionNodeBase):

    def post_init(self):
        self.rigid_body_dipole = rigid_bodies.TwoDipoleDisc100x15_6HKCM10x3
        self.publish_desired_com_wrenches = True
        self.control_input_publisher = rospy.Publisher("/com_wrench_z_control/control_input",
                                                       VectorStamped, queue_size=1)
    
    def control_logic(self):
        self.desired_currents_msg = DesCurrentsReg() # Empty message
        self.control_input_message = VectorStamped() # Empty message
        self.com_wrench_msg = WrenchStamped() # Empty message

if __name__ == "__main__":
    controller = DirectCOMWrenchZController()
    rospy.spin()