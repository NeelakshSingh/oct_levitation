import numpy as np
import os
import wand_calibration as wand

import tf.transformations as tr

from mag_manip import mag_manip
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