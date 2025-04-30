import rospy
import rospkg
import os
import datetime
import numpy as np
import pandas as pd
import yaml
import signal
import alive_progress as ap

from oct_levitation.rigid_bodies import REGISTERED_BODIES
from mag_manip import mag_manip
from oct_levitation.mechanical import MultiDipoleRigidBody
import oct_levitation.plotting as plotting
import oct_levitation.common as common
import oct_levitation.geometry_jit as geometry
import shutil

from oct_levitation.simple_offline_sim import DynamicsSimulator, SimControllerBase

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('oct_levitation')

data_base_folder = os.path.join(pkg_path, 'data', 'offline_sim', 'mass_delay_sweep')

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
    "active_dimensions": np.array([False, True, False, False, True, True]), # roll, pitch, yaw, x, y, z
    "mass": 0.1, # kg
    "inertia_xx_yy": 1.5e-5, # kg*m^2
}

### Only the magnetic properties of the rigid body are used. Everything else has been parameterized separately
RIGID_BODY = REGISTERED_BODIES[SIM_PARAMS['rigid_body']]

MASS_RANGE = np.arange(0.005, 0.2, 0.010) # kg
DELAY_RANGE = np.arange(0.0, 0.05, 0.0005) # s increments in 0.5 ms

## mass and inertia computation
REFERENCE_DISC_PROPERTIES = {
    "radius": 40e-3 , # m
    "height": 15e-3, # m
}

REFERENCE_MAGNET_PROPERTIES = {
    "radius": 5e-3, # m
    "height": 5e-3, # m
    "density" : 7469.67199, # kg
    "number_of_magnets": 6 
}

def get_simple_inertia_from_mass(mass):
    r_disc = REFERENCE_DISC_PROPERTIES['radius']
    h_disc = REFERENCE_DISC_PROPERTIES['height']
    r_mag = REFERENCE_MAGNET_PROPERTIES['radius']
    h_mag = REFERENCE_MAGNET_PROPERTIES['height'] * REFERENCE_MAGNET_PROPERTIES['number_of_magnets']
    mag_rho = REFERENCE_MAGNET_PROPERTIES['density']
    mag_mass = np.pi * r_mag**2 * h_mag * mag_rho
    I_xx_yy_disc = (1/12) * mass * (3 * r_disc**2 + h_disc**2)
    I_xx_yy_mag = (1/12) * mag_mass * (3 * r_mag**2 + h_mag**2)
    I_xx_yy = I_xx_yy_disc + I_xx_yy_mag
    return I_xx_yy, I_xx_yy_disc, I_xx_yy_mag

LIST_OF_FOLDERS = []
MASSES_TRIED = []
DELAY_TRIED = []
EVALUATION_PROGRESS = 0
TOTAL_POINTS = len(MASS_RANGE) * len(DELAY_RANGE)

# Copy this script and the dynamics base class scripts to the experiment folder for backup
script_path = os.path.abspath(__file__)
sim_base_path = os.path.join(pkg_path, 'src', 'oct_levitation', 'simple_offline_sim.py')

# Copying into the new folder
os.makedirs(data_folder, exist_ok=True)

with ap.alive_bar(TOTAL_POINTS) as bar:
    for mass in MASS_RANGE:
        for delay in DELAY_RANGE:
            MASSES_TRIED.append(mass)
            DELAY_TRIED.append(delay)
            SIM_PARAMS['mass'] = mass
            SIM_PARAMS['control_input_delay'] = delay

            # inertia will decrease proportionally to the mass 
            SIM_PARAMS['inertia_xx_yy'] = get_simple_inertia_from_mass(mass)

            # Create a new folder for each experiment
            folder_name = f"mass_{mass:.3f}_delay_{delay:.3f}"
            folder_path = os.path.join(experiments_subfolders, folder_name)
            plot_path = os.path.join(folder_path, 'plots')
            os.makedirs(folder_path, exist_ok=True)
            LIST_OF_FOLDERS.append(folder_path)
            # Save the parameters to a yaml file
            yaml_file_path = os.path.join(folder_path, 'sim_params.yaml')
            with open(yaml_file_path, 'w') as yaml_file:
                yaml.dump(SIM_PARAMS, yaml_file)
            # Save the rigid body parameters to a yaml file

            bar()
