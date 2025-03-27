import numpy as np
import tf2_ros
import os
import tf.transformations as tr
import time

from geometry_msgs.msg import TransformStamped, WrenchStamped, Vector3, Quaternion
from std_msgs.msg import Bool
from tnb_mns_driver.msg import DesCurrentsReg

import oct_levitation.common as common
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.geometry as geometry
import oct_levitation.numerical as numerical

from scipy.integrate import solve_ivp
from argparse import ArgumentParser

## Actually this barebones offline simulation can be helpful in trying advanced
## controllers too. But one must be very aware of the differences between this 
## simulation and the actual world.

class OfflineDynamicsSimulator:

    def __init__(self):
        ### Parse arguments
        # Simulation duration
        # Control frequency
        # Simulation frequency
        # Noise strength
        # Initial conditions
        # Which noise components to use

        self.controller_init()
        pass

    def run_simulation_loop(self):
        # Based on the parameters received in the init function.
        # Run the simulation loop here with proper numerical
        # integration using solve_ivp. For rotation we stick to
        # my own implementation.
        pass

    def controller_init(self, ):
        # Define all the important parameters for the controller here.
        pass

    def controller(self, dipole_tf: TransformStamped):
        # Add simple controller logic here
        pass

    def process_data(self):
        # Package variables into formats recognized by the plotting library here.
        pass
    
    def plot_data(self):
        # Finally just plot the said data just like we do it for the experiment analysis.
        # And save it in the relevant directory.
        pass

if __name__ == "__main__":
    simple_offline_sim = OfflineDynamicsSimulator()