import numpy as np
import os
import wand_calibration as wand

import tf.transformations as tr

from mag_manip import mag_manip
from geometry_msgs.msg import TransformStamped
from dataclasses import dataclass


OCTOMAG_ALL_COILS_ENABLED = [True, True, True, True, True, True, True, True]

@dataclass
class Constants:
    """
    Standard physical constants used in the code.
    """
    g: float = 9.80665 # m/s^2
    mu_0: float = 4*np.pi*1e-7 # T*m/A

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
    
def angle_residual(a: float, b: float):
    """
    Computes the smaller arc's angle residual between a and b by converting it to the 
    range [-pi, pi].
    """
    residual = a - b
    residual = residual % (2*np.pi) # First force to the range [0, 2*pi]
    if residual > np.pi:
        residual -= 2*np.pi
    return residual