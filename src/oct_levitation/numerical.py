import numpy as np
import numpy.typing as np_t
from typing import Union, Iterable

class FirstOrderDifferentiator:
    def __init__(self, alpha: float):
        """
        This is a simple first order forward difference differentiator.

        Parameters:
            alpha (float): The smoothing factor, 0 < alpha < 1. A higher alpha will result in a smoother output.
        """
        self.alpha = alpha
        self.prev_val = None
    
    def __call__(self, val: np_t.ArrayLike, dt: float) -> np_t.ArrayLike:
        if self.prev_val is None:
            self.prev_val = val
            return 0
        smooth_val = self.alpha*self.prev_val + (1-self.alpha)*val
        diff = (smooth_val - self.prev_val)/dt
        self.prev_val = smooth_val
        return diff

class MultiChannelFirstOrderDifferentiator:
    def __init__(self, channels: int, alpha: Union[float, Iterable[float]]):
        """
        This is a simple first order forward difference differentiator. Shouldn't be needed
        usually first order class can handle arrays.

        Parameters:
            channels (int): The number of channels to differentiate.
            alpha (Union[float, Iterable[float]]): The smoothing factor, 0 < alpha < 1. A higher alpha will result in a smoother output.
        """
        if isinstance(alpha, float):
            alpha = [alpha]*channels
        assert len(alpha) == channels, "Number of alphas must match number of channels."
        self.alpha = alpha
        self.diff_objs = [FirstOrderDifferentiator(a) for a in alpha]

    def __call__(self, vals: Iterable[float], dt: float) -> np_t.ArrayLike:
        return np.array([diff(val, dt) for diff, val in zip(self.diff_objs, vals)])

class FirstOrderIntegrator:
    def __init__(self,
                 windup_lim: float = np.inf,
                 clegg_integrator: bool = False):
        """
        This is a simple first order integrator.

        Parameters:
            windup_lim (float): The windup limit for the integrator.
            clegg_integrator (bool): Whether to use the Clegg integrator or not.
        """
        self.windup_lim = windup_lim
        self.integral = 0
        self.prev_val = None
        self.clegg_integrator = clegg_integrator

    
    def __call__(self, val: np_t.ArrayLike, dt: float) -> np_t.ArrayLike:
        if self.clegg_integrator and self.prev_val is not None:
            # Check which terms changed sign and reset the integral term.
            reset_idx = np.where(np.sign(val) != np.sign(self.prev_val))[0]
            if reset_idx.size > 0:
                self.integral[reset_idx] = 0
        self.integral += val*dt
        self.integral = np.clip(self.integral, -self.windup_lim, self.windup_lim)
        self.prev_val = val
        return self.integral