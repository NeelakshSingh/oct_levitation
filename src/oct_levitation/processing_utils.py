import os
import warnings
import numpy as np
import time
import pandas as pd
import oct_levitation.ext.bagreader as bagpy

import scipy.fft as scifft

from typing import Dict, Tuple, List, Union, Callable, Any

###############################################
# bagpyext based topic data extraction function.
###############################################

def extract_topic_data_csv_bagpy(dwd: Union[str, os.PathLike], logfunc: Callable[[str], Any] = print):
    """
    Extracts topic data from a bag file using bagpyext and saves it as CSV files. Adapted
    to work with bagpy's extraction code.
    """
    listOfBagFiles = [f for f in os.listdir(dwd) if f[-4:] == ".bag"]	#get list of only bag files in current dir.
    numberOfFiles = len(listOfBagFiles)
    if numberOfFiles > 1:
        logfunc("More than one bag file found in the directory. Ensure that only one bag file is present.")
        return None
    elif numberOfFiles == 0:
        logfunc("No bag files found in the directory. Ensure that the directory contains exactly one bag file.")
        return None
    else:
        ts = time.time()
        count = 0
        for bagFile in listOfBagFiles:
            count += 1
            print("reading file " + str(count) + " of " + str(numberOfFiles) + ": " + bagFile)
            bag = bagpy.bagreader(os.path.join(dwd, bagFile))
            for t in bag.topics:
                temp_csv = bag.message_by_topic(t)
            print("finished reading CSVs saved" + bagFile + "\n")
        print("total time: " + str(time.time()-ts) + " seconds")

###############################################
# Legacy utility functions from Jasan's
# control_utils adapted for use with bagpyext.
###############################################

def read_scalar_stamped(topicname, dwd):
    # keep for backwards compatibility
    df=pd.read_csv(dwd + topicname + '.csv', sep=',',header=0) # header = num. of rows to skip  
    data= df.values
    time  = data[:,0] # [secs]
    scalar   = data[:,1] # [rad]
    return time, scalar

def read_data_stamped(topicname, dwd):
    df=pd.read_csv(dwd + topicname + '.csv', sep=',',header=0) # header = num. of rows to skip  
    data= df.values
    time  = data[:,0] # [secs]
    return time, data[:,1:]


def shift_time_and_convert_to_sec(time, offset):
    time = time - offset
    return time*1e-9 # [secs]

def interp0(x, xp, yp):
    """Zeroth order hold interpolation w/ same
    (base)   signature  as numpy.interp."""

    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while x0 > xp[k]:
            k += 1
        return yp[k-1]
    
    if isinstance(x,float):
        return func(x)
    elif isinstance(x, list):
        return [func(x) for x in x]
    elif isinstance(x, np.ndarray):
        return np.asarray([func(x) for x in x])
    else:
        raise TypeError('argument must be float, list, or ndarray')
    
def read_data_multidof(dwd, topics):
    ## Read multidof data
    # Based on the fact that u_alpha and u_beta have the same length of non zero elements,
    # even though they have different time stamps, treat them as they are having the same
    # time stamps, i.e. simply put the non-zero element of u_beta on top of the time stamp
    # of u_alpha where u_alpha is non-zero. Besides that, interpolate data, e.g. alpha, beta
    # on the u_alpha time stamps.
    # This method also bases on the fact that the lengths of u_alpha and u_beta are the same
    # If they are not, offset is needed for u_beta 
    print("------------------READ DATA MULTIDOF INTERPOLATE ON U_ALPHA---------------------")

    time, data = {}, {}
    for topic in topics:
        time[topic], data[topic] = read_scalar_stamped(topic, dwd)
        # print(len(time[topic]))

    # Shift all the smallest starting time to 0 and convert it to s
    offset = min([time[topic][0] for topic in topics])
    for topic in topics:
        time[topic] = shift_time_and_convert_to_sec(time[topic], offset)

    # Crop all the signals to the interval of u_alpha
    min_time = time['_u_alpha'][0]
    max_time = time['_u_alpha'][-1]

    for topic in topics:
        # # Do not crop the u_beta series
        if topic != '_u_beta':  
            tmp_1 = (time[topic] >= min_time)
            tmp_2 = (time[topic] <= max_time)
            time[topic] = time[topic][tmp_1 & tmp_2]
            data[topic] = data[topic][tmp_1 & tmp_2]

    for topic in topics:
        # print(topic)
        # print(len(time[topic]))
        time_sum = 0
        for i in range(len(time[topic])-1):
            time_sum += (time[topic][i+1] - time[topic][i])
        # print('time step: ', time_sum/len(time[topic]))

    time_u = time['_u_alpha']

    variables = {}
    # When saving data, remove the '_' at the begining of the topic name
    # Interpolate on u_alpha, put u_beta on top of u_alpha's time stamp
    for topic in topics:
        # Do not interpolate u_beta assuming it has the same time stamps as u_alpha
        if topic != '_u_beta':
            variables[topic[1:]] = np.interp(time_u, time[topic], data[topic])
        else:
            variables[topic[1:]] = data[topic]

    # Use the input time series as the time stamp
    # time = time_u - time_u[0]
    time = time_u
    variables['time'] = time

    return variables

    
def read_data_interpolate_on_u(dwd, topics):
    print("------------------READ DATA SINGLEDOF INTERPOLATE ON U---------------------")

    time, data = {}, {}
    for topic in topics:
        time[topic], data[topic] = read_scalar_stamped(topic, dwd)
        # print(len(time[topic]))

    # Shift all the smallest starting time to 0 and convert it to s
    offset = min([time[topic][0] for topic in topics])
    for topic in topics:
        time[topic] = shift_time_and_convert_to_sec(time[topic], offset)

    # Crop all the signals to the interval when all signals presents
    min_time = max([time[topic][0] for topic in topics])
    max_time = min([time[topic][-1] for topic in topics])

    for topic in topics:
        # Do not crop the u_beta series
        if topic != ('_u_alpha' or '_u_beta'):  
            tmp_1 = (time[topic] >= min_time)
            tmp_2 = (time[topic] <= max_time)
            time[topic] = time[topic][tmp_1 & tmp_2]
            data[topic] = data[topic][tmp_1 & tmp_2]

    for topic in topics:
        # print(len(time[topic]))
        time_sum = 0
        for i in range(len(time[topic])-1):
            time_sum += (time[topic][i+1] - time[topic][i])
        # print('time step: ', time_sum/len(time[topic]))

    time_u = time['_u_alpha']

    variables = {}
    # When saving data, remove the '_' at the begining of the topic name
    # Interpolate on the input u
    for topic in topics:
        # Do not interpolate u_beta assuming it has the same time stamps as u_alpha
        if topic != '_u_beta':
            variables[topic[1:]] = np.interp(time_u, time[topic], data[topic])
        else:
            variables[topic[1:]] = data[topic]

    # Use the input time series as the time stamp
    time = time_u - time_u[0]
    variables['time'] = time

    return variables 

    


def read_data(dwd, topics, interpolate_topic, input_type=None):
    #%% Import:
    print("------------------READ DATA---------------------")

    time, data = {}, {}
    for topic in topics:
        time[topic], data[topic] = read_data_stamped(topic, dwd)
        

    # shift no really necessary, but makes numbers easier to read in debugging
    offset = min([time[topic][0] for topic in topics])
    for topic in topics:
        time[topic] = shift_time_and_convert_to_sec(time[topic], offset)

    min_time = max([time[topic][0] for topic in topics])
    max_time = min([time[topic][-1] for topic in topics])

    # crop all the data to the same time interval min_time to max_time
    for topic in topics:
        tmp_1 = (time[topic] >= min_time)
        tmp_2 = (time[topic] <= max_time)
        time[topic] = time[topic][tmp_1 & tmp_2]
        data[topic] = data[topic][tmp_1 & tmp_2]

    time_a = time[interpolate_topic]

    variables = {}
    for topic in topics:
        current_data = data[topic]
        interpolation_func = np.interp
        interpolated_data = [interpolation_func(time_a, time[topic], col) for col in current_data.T] # interp and interp0 require 1D arrays
        variables[topic.lstrip("_")] = np.asarray(interpolated_data).T
        
    time = time_a - time_a[0]
    variables['time'] = time

    return variables

def read_data_pandas(dwd, topics, interpolate_topic):
    #%% Import:
    print("------------------READ DATA---------------------")

    dfs = {}
    for topic in topics:
        dfs[topic] = pd.read_csv(dwd + topic + '.csv', sep=',',header=0) # header = num. of rows to skip  
        
    # shift no really necessary, but makes numbers easier to read in debugging
    offset = min([dfs[topic].time[0] for topic in topics])
    for topic in topics:
        dfs[topic].time = (dfs[topic].time - offset)/1e9

    min_time = max([dfs[topic].time.values[0] for topic in topics])
    max_time = min([dfs[topic].time.values[-1] for topic in topics])

    # crop all the data to the same time interval min_time to max_time
    for topic in topics:
        tmp_1 = (dfs[topic].time >= min_time)
        tmp_2 = (dfs[topic].time <= max_time)
        dfs[topic] = dfs[topic][tmp_1 & tmp_2]

    time_a = dfs[interpolate_topic].time.values
    time = time_a - time_a[0]

    interp_dfs = {}

    for topic in topics:
        current_df: pd.DataFrame = dfs[topic]
        i = 0
        interp_df = pd.DataFrame()
        interp_df["time"] = time
        for col in current_df.columns:
            if col == "time":
                i += 1
                continue
                # Check if the column's dtype is non-numeric
            if not np.issubdtype(current_df[col].dtype, np.number):
                i += 1
                continue
            interp_df[col] = np.interp(time_a, current_df.time.values, current_df.iloc[:, i])
            i += 1
        interp_dfs[topic] = interp_df

    return time, interp_dfs

def read_data_pandas_all(dwd: Union[str, os.PathLike], interpolate_topic: str, topic_exclude_list: List[str] = [],
                         exclude_known_latched_topics: bool = True) -> Dict[str, pd.DataFrame]:
    print(f"Reading data from directory: {dwd}")
    latched_topic_suffixes = ["_control_gains", "_control_session_metadata"]
    def latched_exclusion(topic: str) -> bool:
        return np.any(np.array([topic.endswith(suffix) for suffix in latched_topic_suffixes]))
    csv_list = [f[:-4] for f in os.listdir(dwd) if f.endswith(".csv")]
    latched_exclusion_topics = [f for f in csv_list if latched_exclusion(f)]

    final_exclusion_list = latched_exclusion_topics + topic_exclude_list

    csv_list_filtered = [f for f in csv_list if f not in final_exclusion_list]
    print(f"Found {len(csv_list_filtered)} CSV files: {csv_list_filtered}")
    if len(csv_list_filtered) == 0:
        warnings.warn("No CSV files were found. If you are accessing the rosbag folder, make\
                      sure that you run bagpyext first. Check the exclusion list too.")
    return read_data_pandas(dwd, csv_list_filtered, interpolate_topic)

###############################################
# DATA CONVERSION AND PROCESSING FUNCTIONS
###############################################

# All functions assume that dataframes were obtained through bagpyext CSV files
# and the above listed time synced interpolated dataset.
def filter_dataframe_by_time_range(dataframe: pd.DataFrame,  
                                   t_start: float, 
                                   t_end: float, 
                                   renormalize_time: bool = False,
                                   time_column: str = "time") -> pd.DataFrame:
    """
    Filters a dataframe to include only rows within the specified time range.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe with a time column.
        time_column (str): The name of the time column in the dataframe.
        t_start (float): The start of the time range (inclusive).
        t_end (float): The end of the time range (inclusive).
        renormalize_time (bool, optional): If True, renormalizes the time column to start from 0. Default is False.

    Returns:
        pd.DataFrame: A filtered dataframe with rows within the specified time range.
    """
    # Filter rows based on the time range
    filtered_df = dataframe[(dataframe[time_column] >= t_start) & (dataframe[time_column] <= t_end)].copy()

    # Renormalize the time column if specified
    if renormalize_time:
        filtered_df[time_column] -= t_start

    return filtered_df

def filter_dataset_by_time_range(dataset: Dict[str, pd.DataFrame],
                                 time_vec: np.ndarray,
                                 t_start: float, 
                                 t_end: float, 
                                 renormalize_time: bool = False,
                                 time_column: str = "time") -> Tuple[np.ndarray, Dict[str, pd.DataFrame]]:
    print(f"Filtering dataset with {len(dataset)} keys within time range {t_start} to {t_end}")
    filtered_dataset = {}
    time_vec_filtered = time_vec[np.logical_and(time_vec > t_start, time_vec < t_end)]
    print(f"Original time vector length: {len(time_vec)}, Filtered time vector length: {len(time_vec_filtered)}")
    filtered_dataset = {}
    time_vec_filtered = time_vec[np.logical_and(time_vec > t_start, time_vec < t_end)]
    for key, item in dataset.items():
        filtered_dataset[key] = filter_dataframe_by_time_range(item, t_start, t_end, renormalize_time, time_column)
    
    return time_vec_filtered, filtered_dataset

def downsample_dataframe(dataframe: pd.DataFrame,
                         fs_new: float,
                         time_column: str = "time") -> pd.DataFrame:
    
    return 
    

def field_position_dataframe_from_des_currents_reg(pose_df: pd.DataFrame, current_df: pd.DataFrame, calibration_fn) -> pd.DataFrame:
    """
    Generates a DataFrame containing field and gradient values at pose points based on 
    coil currents and calibration matrix.

    Parameters:
        pose_df (pd.DataFrame): DataFrame containing pose information with columns:
                                ['time', 'transform.translation.x', 'transform.translation.y', 
                                 'transform.translation.z']
        current_df (pd.DataFrame): DataFrame containing current information with columns:
                                   ['time', 'des_currents_reg_0', ..., 'des_currents_reg_7']
        calibration_fn (Callable): A callable function that takes a 3D position [x, y, z] and 
                                   returns an 8x8 actuation matrix A.
    
    Returns:
        pd.DataFrame: A DataFrame with columns:
                      ['Px', 'Py', 'Pz', 'Bx', 'By', 'Bz', 'dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']
    """
    # Ensure the two DataFrames are aligned on time
    combined_df = pd.merge_asof(pose_df, current_df, on='time')
    
    # Prepare lists for the output data
    output_data = {
        'time': [],
        'Px': [], 'Py': [], 'Pz': [],
        'Bx': [], 'By': [], 'Bz': [],
        'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [],
        'dBy/dy': [], 'dBy/dz': []
    }
    
    # Iterate over each row in the combined DataFrame
    for _, row in combined_df.iterrows():
        # Extract position and currents
        position = [row['transform.translation.x'], row['transform.translation.y'], row['transform.translation.z']]
        currents = np.array([
            row['des_currents_reg_0'], row['des_currents_reg_1'], row['des_currents_reg_2'],
            row['des_currents_reg_3'], row['des_currents_reg_4'], row['des_currents_reg_5'],
            row['des_currents_reg_6'], row['des_currents_reg_7']
        ])
        
        # Get the actuation matrix A using the calibration function
        A = calibration_fn(position)  # Shape: (8, 8)
        
        # Compute the field vector b (b = A @ currents)
        b = A @ currents  # Shape: (8,)
        
        # Append data to output
        output_data['time'].append(row['time'])
        output_data['Px'].append(position[0])
        output_data['Py'].append(position[1])
        output_data['Pz'].append(position[2])
        output_data['Bx'].append(b[0])
        output_data['By'].append(b[1])
        output_data['Bz'].append(b[2])
        output_data['dBx/dx'].append(b[3])
        output_data['dBx/dy'].append(b[4])
        output_data['dBx/dz'].append(b[5])
        output_data['dBy/dy'].append(b[6])
        output_data['dBy/dz'].append(b[7])
    
    # Convert the output dictionary to a DataFrame
    output_df = pd.DataFrame(output_data)
    return output_df

def field_position_dataframe_from_system_state(pose_df: pd.DataFrame, current_df: pd.DataFrame, calibration_fn) -> pd.DataFrame:
    """
    Generates a DataFrame containing field and gradient values at pose points based on 
    coil currents and calibration matrix.

    Parameters:
        pose_df (pd.DataFrame): DataFrame containing pose information with columns:
                                ['time', 'transform.translation.x', 'transform.translation.y', 
                                 'transform.translation.z']
        current_df (pd.DataFrame): DataFrame containing current information with columns:
                                   ['time', 'currents_reg_0', ..., 'currents_reg_7']
        calibration_fn (Callable): A callable function that takes a 3D position [x, y, z] and 
                                   returns an 8x8 actuation matrix A.
    
    Returns:
        pd.DataFrame: A DataFrame with columns:
                      ['Px', 'Py', 'Pz', 'Bx', 'By', 'Bz', 'dBx/dx', 'dBx/dy', 'dBx/dz', 'dBy/dy', 'dBy/dz']
    """
    # Ensure the two DataFrames are aligned on time
    combined_df = pd.merge_asof(pose_df, current_df, on='time')
    
    # Prepare lists for the output data
    output_data = {
        'time': [],
        'Px': [], 'Py': [], 'Pz': [],
        'Bx': [], 'By': [], 'Bz': [],
        'dBx/dx': [], 'dBx/dy': [], 'dBx/dz': [],
        'dBy/dy': [], 'dBy/dz': []
    }
    
    # Iterate over each row in the combined DataFrame
    for _, row in combined_df.iterrows():
        # Extract position and currents
        position = [row['transform.translation.x'], row['transform.translation.y'], row['transform.translation.z']]
        currents = np.array([
            row['currents_reg_0'], row['currents_reg_1'], row['currents_reg_2'],
            row['currents_reg_3'], row['currents_reg_4'], row['currents_reg_5'],
            row['currents_reg_6'], row['currents_reg_7']
        ])
        
        # Get the actuation matrix A using the calibration function
        A = calibration_fn(position)  # Shape: (8, 8)
        
        # Compute the field vector b (b = A @ currents)
        b = A @ currents  # Shape: (8,)
        
        # Append data to output
        output_data['time'].append(row['time'])
        output_data['Px'].append(position[0])
        output_data['Py'].append(position[1])
        output_data['Pz'].append(position[2])
        output_data['Bx'].append(b[0])
        output_data['By'].append(b[1])
        output_data['Bz'].append(b[2])
        output_data['dBx/dx'].append(b[3])
        output_data['dBx/dy'].append(b[4])
        output_data['dBx/dz'].append(b[5])
        output_data['dBy/dy'].append(b[6])
        output_data['dBy/dz'].append(b[7])
    
    # Convert the output dictionary to a DataFrame
    output_df = pd.DataFrame(output_data)
    return output_df

###############################################
# Signal to noise ratio calculation functions.
###############################################
def get_signal_fft(signal, dt,
                   remove_dc: bool = False,
                   positive_freqs_only: bool = False):
    n = len(signal)
    fft_values = scifft.fft(signal)
    fft_frequencies = scifft.fftfreq(n, dt)

    if remove_dc:
        fft_values[0] = 0
    
    fft_values = np.abs(np.roll(fft_values, len(fft_values)//2))
    fft_frequencies = np.roll(fft_frequencies, len(fft_frequencies)//2)

    # Only keep positive frequencies
    if positive_freqs_only:
        pos_mask = fft_frequencies > 0
        fft_frequencies = fft_frequencies[pos_mask]
        fft_values = fft_values[pos_mask]
    
    return fft_values, fft_frequencies

def get_signal_variance(signal: np.ndarray):
    N = len(signal)
    return (1/(N-1))*np.sum(np.square(signal - np.mean(signal)))

def get_signal_std_deviation(signal: np.ndarray):
    return np.sqrt(get_signal_variance(signal))