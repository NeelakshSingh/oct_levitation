# !/usr/bin/env python3

import multiprocessing

import numpy as np
import scipy.signal as signal

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