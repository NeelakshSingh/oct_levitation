import numpy as np
import pynumdiff
import pandas as pd

from collections import deque

class SingleChannelCausalDifferentiator:

    available_paddings = ['repeated_initial', 'zeros']
    def __init__(self, window_size=100, padding='same'):
        """
        This class is used to differentiate a 1D signal online using the different causal 
        differentiators available in the pynumdiff library. Any non-causal options will throw an error.
        Args:
            window_size: int
            padding: str
        """
        self.window_size = window_size
        if padding not in self.available_paddings:
            raise ValueError(f"Padding option {padding} not available. Choose from {self.available_paddings}")
        self.padding = padding
        self.data = deque(maxlen=window_size)
        self.__first_sample = True
        
    def differentiate(self, x: float) -> float:
        """
        Args:
            x: current time step value to differentiate on
        """
        if self.__first_sample:
            if self.padding == 'repeated_initial':
                self.data.extend([x]*self.window_size)
            elif self.padding == 'zeros':
                self.data.extend([0]*self.window_size)
            else:
                # This should never happen
                raise ValueError(f"Padding option {self.padding} not available. Choose from {self.available_paddings}")
            self.__first_sample = False
        
        self.data.extend(x)
        x = np.array(self.data)
        raise NotImplementedError("This function is not implemented yet.")