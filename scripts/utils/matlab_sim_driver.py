import rospy
import numpy as np
import multiprocessing as mp
import oct_levitation.mechanical as mechanical
from std_msgs.msg import String

"""
This file exposes a general interface to work with simulators like MATLAB and Gazebo. The idea is that instead of
building a simulated eMNS through physically based simulation we just used the forward field computation models
generally obtained through calibration procedures to calculate the actual (estimated) wrench on a body and then
we advertise this wrench over a topic to which simulators can subscribe.
"""

class ControlSimDriver:
    
    def __init__(self):
        rospy.init_node("matlab_sim_driver_main", anonymous=False)

        self.wrench_topic = rospy.get_param("~config/wrench_topic", "actual_wrench")


if __name__ == "__main__":
    # Rest of the logic goes here
    driver = ControlSimDriver()