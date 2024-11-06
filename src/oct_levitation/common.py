import numpy as np
import scipy as sci
import os
import wand_calibration as wand

import tf.transformations as tr

from control_utils.general.utilities import get_actuation_matrix
from mag_manip import mag_manip
from geometry_msgs.msg import TransformStamped


OCTOMAG_ALL_COILS_ENABLED = [True, True, True, True, True, True, True, True]

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
                                    dipole_axis: np.ndarray = np.array([0, 0, 1])):
    """
    This function returns the magnetic interaction matrix of a dipole.
    This is purely defined by the orientation of the dipole and its strength.

    Args:
        dipole_tf (TransformStamped): The transform of the dipole in the world frame.
        dipole_strength (float): The strength of the dipole.
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

    M = np.array([
                [ 0.0,              -dipole_moment[2],  dipole_moment[1],   0.0,              0.0,              0.0,              0.0,              0.0 ],
                [ dipole_moment[2],  0.0,              -dipole_moment[0],   0.0,              0.0,              0.0,              0.0,              0.0 ],
                [-dipole_moment[1],  dipole_moment[0],  0.0,                0.0,              0.0,              0.0,              0.0,              0.0 ],
                [ 0.0,               0.0,               0.0,               dipole_moment[0],  dipole_moment[1], dipole_moment[2], 0.0,              0.0 ],
                [ 0.0,               0.0,               0.0,               0.0,              dipole_moment[0],  0.0,              dipole_moment[1], dipole_moment[2]],
                [ 0.0,               0.0,               0.0,              -dipole_moment[2],  0.0,              dipole_moment[0], -dipole_moment[2], dipole_moment[1]]
            ])
    return M

    