import numpy as np
import os
try:
    import wand_calibration as wand
except ImportError:
    print("wand_calibration module not found, wand calibration will not be available.")

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
                 calibration_file: str = "mc3ao8s_md200_handp.yaml",
                 **kwargs):
        
        if calibration_type == "legacy_yaml":
            model_path = os.path.join(os.environ['HOME'], '.ros', 'cal', calibration_file)
            model = mag_manip.ForwardModelMPEM()
            model.setCalibrationFile(model_path)
            self.calibration = model
            self.__actuation_matrix_method = self.calibration.getActuationMatrix
            self.__exact_field_map = self.calibration.computeFieldGradient5FromCurrents
        elif calibration_type == "wand_calibration":
            self.calibration = wand.CalibratedField(calibration_folder=calibration_file, **kwargs)
            self.__actuation_matrix_method = self.calibration.get_actuation_matrix
        else:
            raise ValueError(f"Unknown calibration type: {calibration_type}")
        
    def get_actuation_matrix(self, position: np.ndarray) -> np.ndarray:
        return self.__actuation_matrix_method(position)
    
    def get_exact_field_grad5_from_currents(self, position: np.ndarray, currents: np.ndarray) -> np.ndarray:
        return self.__exact_field_map(position, currents)