# !/usr/bin/env python3

import multiprocessing

import numpy as np
import scipy.signal as signal
import numpy.typing as np_t

from collections import deque
from typing import Union, List, Iterable, Optional


## Live Filter Base Classes ##
## Credits: Samuel Pröll | https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/
class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x: float):
        # do not process NaNs
        if np.isnan(x).any():
            return x

        return self._process(x)

    def __call__(self, x: float):
        return self.process(x)

    def _process(self, x: float):
        raise NotImplementedError("Derived class must implement _process")


class LiveLFilter(LiveFilter):
    """Live implementation of digital filter using difference equations.

    The following is almost equivalent to calling scipy.lfilter(b, a, xs):
    >>> lfilter = LiveLFilter(b, a)
    >>> [lfilter(x) for x in xs]
    """
    def __init__(self, b, a):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy
                filter design.
            a (array-like): denominator coefficients obtained from scipy
                filter design.
        """
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)

    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y


class LiveSosFilter(LiveFilter):
    """Live implementation of digital filter with second-order sections.
    Algorithm logic is based on the Direct Form II realization of the 
    discrete time filters.

    The following is equivalent to calling scipy.sosfilt(sos, xs):
    >>> sosfilter = LiveSosFilter(sos)
    >>> [sosfilter(x) for x in xs]
    """
    def __init__(self, sos):
        """Initialize live second-order sections filter.

        Args:
            sos (array-like): second-order sections obtained from scipy
                filter design (with output="sos").
        """
        self.sos = sos

        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))

    def _process(self, x: float):
        """Filter incoming data with cascaded second-order sections.
        """
        for s in range(self.n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = self.sos[s, :]

            # compute difference equations of transposed direct form II
            y = b0*x + self.state[s, 0]
            self.state[s, 0] = b1*x - a1*y + self.state[s, 1]
            self.state[s, 1] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.

        return y


## Modified classes for FT filtering ##
# Adopting multi-threaded design for fast filtering of incoming
# data from the FT sensor.

class AveragingLowPassFilter:

    def __init__(self, alpha: float):
        """
        A simple low pass filter with first order feedback inspired from the version used in
        moveit_servo.

        Parameters:
            alpha (float): The smoothing factor, 0 < alpha < 1. A higher alpha will result in stronger filtering but more lag.
        """
        self.scaling = 1/(1 + alpha)
        self.feedback = 1 - alpha
        self.x_prev = 0
        self.x = 0
        self.x_hat_prev = 0
    
    def reset(self, x0):
        self.x_prev = x0
        self.x = x0
        self.x_hat_prev = x0
    
    def __call__(self, x):
        self.x_hat_prev = self.x
        self.x = x
        self.x_hat = self.scaling*(self.x_prev + self.x - self.feedback*self.x_hat_prev)
        self.x_hat_prev = self.x_hat
        return self.x_hat

# Normal modified Class wrappers

class SingleChannelLiveFilter(LiveFilter):
    """Live implementation of digital filter using difference equations.
    Specially tailored for FT sensor filtering. Derives from `LiveLFilter`
    as the base class. Check the base class for more information. This one
    just wraps over the scipy.signal.iirfilter function to generate the
    filter coefficients. Will be useful for multi-threaded filtering.
    """

    def __init__(self, N: int, Wn: float,
                 rp: Optional[float] = None,
                 rs: Optional[float] = None,
                 btype='lowpass', 
                 analog=False, 
                 ftype='butter',
                 fs: Optional[float] = None,
                 use_sos=False):
        """
        Initialize live filter based on difference equation.

        Parameters:
        ----------
            N (int): Order of the filter.
            Wn (float): Cutoff frequency of the filter.
            rp (float): Maximum ripple allowed below unity gain in the passband.
                Specified in decibels, as a positive number.
            rs (float): Minimum attenuation required in the stopband.
                Specified in decibels, as a positive number.
            btype (str): Type of the filter. Can be 'lowpass', 'highpass',
                'bandpass', or 'bandstop'.
            analog (bool): If True, Wn is an angular frequency (in radians)
                otherwise it is a frequency in Hz.
            ftype (str): Type of the digital filter. Can be 'butter', 'cheby1',
                'cheby2', 'ellip', 'bessel', or 'butter'.
            fs (float): Sampling frequency of the data. Only used if use_sos is
            use_sos (bool): If True, return a second-order section representation
                for the digital filter. If False, return a numerator-denominator
                representation.
        """
        self.live_filter = None

        if use_sos:
            self.live_filter = LiveSosFilter(signal.iirfilter(N, Wn,
                                                              rp=rp,
                                                              rs=rs,
                                                              btype=btype, 
                                                              analog=analog, 
                                                              ftype=ftype, 
                                                              output='sos',
                                                              fs=fs))
        else:
            b, a = signal.iirfilter(N, Wn, 
                                    rp=rp,
                                    rs=rs,
                                    btype=btype, 
                                    analog=analog, 
                                    ftype=ftype,
                                    fs=fs)
            self.live_filter = LiveLFilter(b, a)
        
        self.last_output = None # Should help in getting a multi-threaded version if needed.

    def _process(self, x: float):
        """Filter incoming data with standard difference equations."""
        self.last_output = self.live_filter(x)
        return self.last_output
    
    def __call__(self, x):
        return self._process(x)


class SingleChannelMovingAverageFilter(LiveFilter):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.last_output = None

    def _process(self, x: float):
        self.window.append(x)
        self.last_output = np.mean(self.window)
        return self.last_output
    
    def __call__(self, x):
        return self._process(x)

class MultiChannelLiveFilter(LiveFilter):
    """Multi channel version of the single channel live filter class.
    """

    def __init__(self, 
                 channels: int,
                 N: int, 
                 Wn: float,
                 rp: Optional[float] = None,
                 rs: Optional[float] = None,
                 btype='lowpass', 
                 analog=False, 
                 ftype='butter',
                 fs: Optional[float] = None,
                 use_sos=False):
        """Initialize multiple live filters based on difference equation.

        Parameters:
        ----------
            channels (int): Number of channels to be filtered.
            N (int): Order of the filter.
            Wn (float): Cutoff frequency of the filter.
            btype (str): Type of the filter. Can be 'lowpass', 'highpass',
                'bandpass', or 'bandstop'.
            analog (bool): If True, Wn is an angular frequency (in radians)
                otherwise it is a frequency in Hz.
            ftype (str): Type of the digital filter. Can be 'butter', 'cheby1',
                'cheby2', 'ellip', 'bessel', or 'butter'.
            use_sos (bool): If True, return a second-order section representation
                for the digital filter. If False, return a numerator-denominator
                representation.
        """
        self.channels = channels
        self.live_filters = [
            SingleChannelLiveFilter(N, Wn,
                                    rp=rp,
                                    rs=rs,
                                    btype=btype,
                                    analog=analog,
                                    ftype=ftype,
                                    fs=fs,
                                    use_sos=use_sos) for _ in range(channels)
        ]
        
    def _process(self, x: np.ndarray):
        """Filter incoming data with standard difference equations."""
        if x.shape[0] != self.channels:
            raise ValueError("The the input data array does not match the number of channels in the filter.")
        return np.array([self.live_filters[i](x[i]) for i in range(self.channels)])
    
    def __call__(self, x: np.ndarray):
        return self._process(x)

class MultiChannelAveragingLPF:

    def __init__(self, channels: int, alpha: Union[float, Iterable[float]]):
        self.channels = channels
        if isinstance(alpha, float):
            self.filters = [AveragingLowPassFilter(alpha) for _ in range(channels)]
        else:
            if len(alpha) != channels:
                raise ValueError("The length of the alpha array does not match the number of channels.")
            self.filters = [AveragingLowPassFilter(alpha[i]) for i in range(channels)]
        
    def __call__(self, x: np.ndarray):
        if x.shape[0] != self.channels:
            raise ValueError("The the input data array does not match the number of channels in the filter.")
        return np.array([self.filters[i](x[i]) for i in range(self.channels)])
    
class MedianFilter(LiveFilter):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.mid_sample = self.window_size // 2
        self.sample_queue : deque = None
        self.__first_sample = True

    def _process(self, x: np_t.NDArray):
        assert(len(x.shape) == 1)
        if self.__first_sample:
            self.sample_queue = deque([np.zeros_like(x).reshape(-1,1)] * self.window_size, maxlen=self.window_size)
            self.__first_sample = False

        self.sample_queue.append(x.reshape(-1, 1))
        sample_mat = np.hstack(self.sample_queue)
        return sample_mat[:, self.mid_sample]
