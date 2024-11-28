import numpy as np
import numpy.typing as np_t
import scipy as sci
import os
import wand_calibration as wand

import tf.transformations as tr

from control_utils.general.utilities import get_actuation_matrix
from mag_manip import mag_manip
from geometry_msgs.msg import TransformStamped

from dataclasses import dataclass
from typing import Dict


OCTOMAG_ALL_COILS_ENABLED = [True, True, True, True, True, True, True, True]

@dataclass
class Constants:
    g: float = 9.81 # m/s^2
    mu_0: float = 4*np.pi*1e-7 # T*m/A
    Br_neodymium: float = 1.2 # T

@dataclass
class NarrowRingMagnet:
    Br: float = 1.36 # T
    rho: float = 7.5e3 # kg/m^3
    t: float = 4.96e-3 # m
    ri: float = (5.11e-3)/2 # m
    ro: float = (9.95e-3)/2 # m
    V: float = np.pi*(ro**2 - ri**2)*t # m^3
    m: float = rho*V # kg
    dps: float = Br*V/Constants.mu_0 # kg*m^2/s
    mframe: float = 2.9e-3
    inertia_matrix_S1 : np_t.NDArray = np.array([[492.29, -74.08, -9.38],
                                                 [-74.08, 807.43, -5.19],
                                                 [-9.38, -5.19, 1251.91]]) * 1e-9 # kg*m^2

class OctomagCalibratedModel:

    def __init__(self,
                 calibration_type: str = "legacy_yaml",
                 calibration_file: str = "octomag_5point.yaml",
                 **kwargs):
        
        if calibration_type == "legacy_yaml":
            model_path = os.path.join(os.environ['HOME'], '.ros', 'cal', calibration_file)
            model = mag_manip.ForwardModelMPEM()
            model.setCalibrationFile(model_path)
            self.calibration = model
            self.__actuation_matrix_method = self.calibration.getActuationMatrix
        elif calibration_type == "wand_calibration":
            self.calibration = wand.CalibratedField(calibration_folder=calibration_file, **kwargs)
            self.__actuation_matrix_method = self.calibration.get_actuation_matrix
        else:
            raise ValueError(f"Unknown calibration type: {calibration_type}")
        
    def get_actuation_matrix(self, position: np.ndarray) -> np.ndarray:
        return self.__actuation_matrix_method(position)

def get_magnetic_interaction_matrix(dipole_tf: TransformStamped,
                                    dipole_strength:float,
                                    torque_first: bool,
                                    dipole_axis: np.ndarray = np.array([0, 0, 1])):
    """
    This function returns the magnetic interaction matrix of a dipole.
    This is purely defined by the orientation of the dipole and its strength.

    Args:
        dipole_tf (TransformStamped): The transform of the dipole in the world frame.
        dipole_strength (float): The strength of the dipole.
        torque_first (bool): Whether to return the torque block first or the force block first.
                             If True, then [[M_Tau], [M_F]] is returned and vice versa.
        dipole_axis (np.ndarray): The axis of the dipole according to vicon in home position. 
                                  Defaults to [0, 0, 1].
    
    Returns:
        np.ndarray: The magnetic interaction matrix of the dipole
    """
    dipole_axis = dipole_axis/np.linalg.norm(dipole_axis, 2)
    dipole_quaternion = np.array([dipole_tf.transform.rotation.x,
                                  dipole_tf.transform.rotation.y,
                                  dipole_tf.transform.rotation.z,
                                  dipole_tf.transform.rotation.w])
    R_OH = tr.quaternion_matrix(dipole_quaternion)[:3, :3]
    dipole_axis = R_OH.dot(dipole_axis)
    dipole_axis = dipole_axis/np.linalg.norm(dipole_axis, 2)
    dipole_moment = dipole_strength*dipole_axis

    M_F = np.array([
                [ 0.0,               0.0,               0.0,               dipole_moment[0],  dipole_moment[1], dipole_moment[2], 0.0,              0.0 ],
                [ 0.0,               0.0,               0.0,               0.0,              dipole_moment[0],  0.0,              dipole_moment[1], dipole_moment[2]],
                [ 0.0,               0.0,               0.0,              -dipole_moment[2],  0.0,              dipole_moment[0], -dipole_moment[2], dipole_moment[1]]
            ])
    M_Tau = np.array([
                [ 0.0,              -dipole_moment[2],  dipole_moment[1],   0.0,              0.0,              0.0,              0.0,              0.0 ],
                [ dipole_moment[2],  0.0,              -dipole_moment[0],   0.0,              0.0,              0.0,              0.0,              0.0 ],
                [-dipole_moment[1],  dipole_moment[0],  0.0,                0.0,              0.0,              0.0,              0.0,              0.0 ],
            ])
    if torque_first:
        return np.vstack((M_Tau, M_F))
    else:
        return np.vstack((M_F, M_Tau))