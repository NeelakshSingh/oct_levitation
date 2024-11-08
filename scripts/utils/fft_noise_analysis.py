# !/usr/bin/env python3

import rospkg
import pandas as pd
import numpy as np
import pyfftw
import multiprocessing
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Name of the file to be analyzed")
    parser.add_argument("--tsuffix", help="Title suffix for the frequency plot")
    parser.add_argument("--rdc", help="Remove DC offset by mean shift", action="store_true")
    parser.add_argument("--positive", help="Only plot positive frequencies", action="store_true")
    parser.add_argument("--nrej", help="Number of samples to reject if --positive is selected. Useful for ignoring DC frequencies.", type=int, default=0)
    parser.add_argument("--plot_options", help="Plot options for matplotlib, except linewidth, label, color.", type=str, default="-")
    args = parser.parse_args()
    fname = args.fname

    if args.tsuffix:
        tsuffix = args.tsuffix
    else:
        tsuffix = ""

    rdc = args.rdc
    positive = args.positive
    nrej = args.nrej
    plot_options = args.plot_options

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ati_daq_interface')
    fpath = pkg_path + "/rosbags/" + fname

    df = pd.read_csv(fpath, sep=",")

    # Fetch time vector
    t_vec = df["field.header.stamp"].to_numpy()
    t_vec = t_vec - t_vec[0] # start at 0
    t_vec = t_vec / 1e9 # convert to seconds

    # Fetch force data
    Fx = df["field.wrench.force.x"].to_numpy()
    Fy = df["field.wrench.force.y"].to_numpy()
    Fz = df["field.wrench.force.z"].to_numpy()

    # The DC offset bits are commented since you should be able to
    # simply ignore them in the frequency domain.

    # Fetch torque data
    Tx = df["field.wrench.torque.x"].to_numpy()
    Ty = df["field.wrench.torque.y"].to_numpy()
    Tz = df["field.wrench.torque.z"].to_numpy()
    
    # Remove DC offset
    if rdc:
        Fx = Fx - np.mean(Fx)
        Fy = Fy - np.mean(Fy)
        Fz = Fz - np.mean(Fz)
        Tx = Tx - np.mean(Tx)
        Ty = Ty - np.mean(Ty)
        Tz = Tz - np.mean(Tz)
    
    F = np.array([Fx, Fy, Fz])
    T = np.array([Tx, Ty, Tz])

    # Compute FFT through FFTW. Will be slower at first.
    # But will be faster for subsequent calls by caching.

    dtype_str = str(Fx.dtype)
    
    n_threads = multiprocessing.cpu_count()
    pyfftw.config.NUM_THREADS = n_threads # Using all threads
    pyfftw.interfaces.cache.enable() # Enable caching

    # Getting the empty arrays for the FFT
    Fx_fft_arr = pyfftw.empty_aligned(Fx.shape, dtype=dtype_str)
    Fy_fft_arr = pyfftw.empty_aligned(Fy.shape, dtype=dtype_str)
    Fz_fft_arr = pyfftw.empty_aligned(Fz.shape, dtype=dtype_str)
    Tx_fft_arr = pyfftw.empty_aligned(Tx.shape, dtype=dtype_str)
    Ty_fft_arr = pyfftw.empty_aligned(Ty.shape, dtype=dtype_str)
    Tz_fft_arr = pyfftw.empty_aligned(Tz.shape, dtype=dtype_str)

    Fx_fft_arr[:] = Fx
    Fy_fft_arr[:] = Fy
    Fz_fft_arr[:] = Fz
    Tx_fft_arr[:] = Tx
    Ty_fft_arr[:] = Ty
    Tz_fft_arr[:] = Tz

    # Computing the FFT
    Fx_fft = pyfftw.interfaces.numpy_fft.fft(Fx_fft_arr)
    Fy_fft = pyfftw.interfaces.numpy_fft.fft(Fy_fft_arr)
    Fz_fft = pyfftw.interfaces.numpy_fft.fft(Fz_fft_arr)
    Tx_fft = pyfftw.interfaces.numpy_fft.fft(Tx_fft_arr)
    Ty_fft = pyfftw.interfaces.numpy_fft.fft(Ty_fft_arr)
    Tz_fft = pyfftw.interfaces.numpy_fft.fft(Tz_fft_arr)

    # Computing the frequency vector
    freq_vec = np.fft.fftfreq(t_vec.shape[-1], d=t_vec[1]-t_vec[0])

    pf_idx = int(freq_vec.shape[0]/2) # Positive frequencies index

    if positive:
        freq_vec = freq_vec[nrej:pf_idx]
        Fx_fft = Fx_fft[nrej:pf_idx]
        Fy_fft = Fy_fft[nrej:pf_idx]
        Fz_fft = Fz_fft[nrej:pf_idx]
        Tx_fft = Tx_fft[nrej:pf_idx]
        Ty_fft = Ty_fft[nrej:pf_idx]
        Tz_fft = Tz_fft[nrej:pf_idx]

    # Storing magnitudes
    Fx_mag = np.abs(Fx_fft)
    Fy_mag = np.abs(Fy_fft)
    Fz_mag = np.abs(Fz_fft)
    Tx_mag = np.abs(Tx_fft)
    Ty_mag = np.abs(Ty_fft)
    Tz_mag = np.abs(Tz_fft)

    # Plotting the results
    plt.figure()
    plt.suptitle("Frequency domain analysis of FT data: " + tsuffix)
    plt.subplot(3, 2, 1)
    plt.plot(freq_vec, Fx_mag, plot_options, label="Fx", linewidth=1, color="red")
    plt.title("Fx")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.subplot(3, 2, 3)
    plt.plot(freq_vec, Fy_mag, plot_options, label="Fy", linewidth=1, color="green")
    plt.title("Fy")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.subplot(3, 2, 5)
    plt.plot(freq_vec, Fz_mag, plot_options, label="Fz", linewidth=1, color="blue")
    plt.title("Fz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.subplot(3, 2, 2)
    plt.plot(freq_vec, Tx_mag, plot_options, label="Tx", linewidth=1, color="red")
    plt.title("Tx")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.subplot(3, 2, 4)
    plt.plot(freq_vec, Ty_mag, plot_options, label="Ty", linewidth=1, color="green")
    plt.title("Ty")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.subplot(3, 2, 6)
    plt.plot(freq_vec, Tz_mag, plot_options, label="Tz", linewidth=1, color="blue")
    plt.title("Tz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()