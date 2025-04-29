import rospy
import rospkg
import os
import datetime
import numpy as np
import pandas as pd

from oct_levitation.rigid_bodies import REGISTERED_BODIES
from mag_manip import mag_manip
from oct_levitation.mechanical import MultiDipoleRigidBody
import oct_levitation.plotting as plotting
import oct_levitation.common as common
import oct_levitation.geometry_jit as geometry



rospack = rospkg.RosPack()
pkg_path = rospack.get_path('oct_levitation')

data_base_folder = os.path.join(pkg_path, 'data', 'offline_sim')

data_folder = os.path.join(data_base_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
experiments_subfolders = os.path.join(data_folder, 'individual_conditions')

### IMPORTANT UTILITIES
MPEM_MODEL = mag_manip.ForwardModelMPEM()
MPEM_MODEL.setCalibrationFile(os.path.join(os.environ["HOME"], ".ros/cal", "mc3ao8s_md200_handp.yaml"))
# for plotting
CALIBRAITON_MODEL = common.OctomagCalibratedModel()

SIM_PARAMS = {
    "rigid_body": "onyx_disc_80x15_I40_N52",
    "duration": 2.0,
    "sim_freq": 1000,
    "gravity_on": True,
    "initial_position": np.array([0.0, 0.0, 0.005]),
    "position_limit": np.array([0.01, 0.01, 0.01]),
    "initial_velocity": np.array([0.0, 0.0, 0.0]),
    "initial_rpy_deg": np.array([0.0, 0.0, 0.0]),
    "rpy_limit_deg": np.array([50.0, 50.0, 50.0]),
    "initial_angular_velocity_deg_ps": np.array([0.0, 0.0, 0.0]),
    "vicon_noise_std_exyz": np.array([1.01646014e-05, 
                                      4.16398263e-06, 
                                      6.27907061e-06, 
                                      2.37471348e-04, 
                                      8.75457449e-04, 
                                      7.40946226e-05
                                     ]),
    "current_noise_std": 0.029008974233986275,
    "ecb_bandwidth_hz": 15,
    "controller_frequency": 400,
    "control_input_delay": 0.002,
    "max_controller_current": 4.0, # A
    "active_dimensions": np.array([False, False, False, False, False, True]), # roll, pitch, yaw, x, y, z
}

RIGID_BODY = REGISTERED_BODIES[SIM_PARAMS['rigid_body']]

