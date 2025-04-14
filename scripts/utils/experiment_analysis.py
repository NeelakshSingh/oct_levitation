import os
import numpy as np
import rospkg
import rospy
import pandas as pd
import subprocess

import oct_levitation.plotting as plotting
import oct_levitation.processing_utils as utils
import oct_levitation.common as common
import oct_levitation.geometry as geometry

from oct_levitation.rigid_bodies import REGISTERED_BODIES

import matplotlib.pyplot as plt

from datetime import datetime


rospy.init_node('experiment_analysis_node', anonymous=True)

def node_loginfo(msg):
    rospy.loginfo(f"[oct_levitation/experiment_analysis] {msg}")

def node_logerr(msg):
    rospy.logerr(f"[oct_levitation/experiment_analysis] {msg}")

def node_logwarn(msg):
    rospy.logwarn(f"[oct_levitation/experiment_analysis] {msg}")

def topic_name_to_bagpyext_name(topic_name: str) -> str:
    """
    Convert a ROS topic name to a CSV file name.
    """
    # Remove leading and trailing slashes
    if topic_name.startswith("/"):
        topic_name = topic_name[1:]
    if topic_name.endswith("/"):
        topic_name = topic_name[:-1]
    
    # Replace slashes with underscores
    return "_" + topic_name.replace("/", "_")

rospkg = rospkg.RosPack()
pkg_path = rospkg.get_path('oct_levitation')
data_base_folder = rospy.get_param('experiment_analysis/base_folder', "")
if data_base_folder == "":
    node_loginfo("No base folder specified. Using default: oct_levitation/data/experiment_data.")
    data_base_folder = os.path.join(pkg_path, 'data', 'experiment_data')


sim = rospy.get_param('~sim') # Mandatory parameter to specify the correct folder according to experiment_recorder.py

if sim:
    data_base_folder = os.path.join(data_base_folder, 'sim')

experiment_subfolder = rospy.get_param('~data_subfolder') # mandatory parameter
data_base_folder = os.path.join(data_base_folder, experiment_subfolder)

# For now this will use the param. But ideally we should get the rigid body name from the data stored by the
# experiment recorder.
dipole_body = REGISTERED_BODIES[rospy.get_param("oct_levitation/rigid_body")]

node_loginfo(f"Using data base folder: {data_base_folder}. Using dipole body: {dipole_body.name}")

def get_latest_dated_folder(base_folder):
    """
    Get the latest dated folder in the base folder.
    """
    date_format = "%Y-%m-%d_%H-%M-%S"
    valid_folders = []

    for f in os.listdir(base_folder):
        full_path = os.path.join(base_folder, f)
        if os.path.isdir(full_path):
            try:
                timestamp = datetime.strptime(f, date_format)
                valid_folders.append((timestamp, full_path))
            except ValueError:
                continue 

    if not valid_folders:
        return None
    
    latest_folder = max(valid_folders, key=lambda x: x[0])[1]
    return latest_folder

experiment_folder = rospy.get_param('~experiment_folder', '')
expt_dir = None
if experiment_folder != "": # Hope no one is insane enough to name a folder '' *tests using mkdir after writing this*
    expt_dir = os.path.join(data_base_folder, experiment_folder)
    if not os.path.exists(expt_dir):
        raise FileNotFoundError(f"Experiment folder {expt_dir} does not exist.")
else:
    expt_dir = get_latest_dated_folder(data_base_folder)

if expt_dir is not None:
    expt_dir = os.path.join(data_base_folder, expt_dir)
    node_loginfo(f"Using default experiment folder: {expt_dir}")
else:
    raise FileNotFoundError(f"No dated experiment folder found in {data_base_folder}. Please specify a folder name using the --experiment_folder argument if using a different name format than experiment_recorder.")

plot_folder = os.path.join(expt_dir, "plots")
os.makedirs(plot_folder, exist_ok=True)

### Bagpy extraction utility
if rospy.get_param("experiment_analysis/enable_topic_extraction", default=False):
    csv_list = [f for f in os.listdir(expt_dir) if f.endswith('.csv')]
    if len(csv_list) > 0:
        node_logwarn("Extraction requested but CSV files found in the experiment folder. If the script fails, consider removing them and re-running, check that the topics needed by the plots are extracted properly.")
    else:
        node_loginfo("Extraction requested. Proceeding to extract topics from bag files.")
        utils.extract_topic_data_csv_bagpy(expt_dir, node_loginfo)

## Some important parameters
plotting.DISABLE_PLT_SHOW = not rospy.get_param("experiment_analysis/show_after_each_plot", default=False)
display_plots = rospy.get_param("experiment_analysis/display_plots_at_end", default=True)
if plotting.DISABLE_PLT_SHOW and display_plots:
    display_plots = True

INKSCAPE_PATH = rospy.get_param("experiment_analysis/inkscape_path", default="/usr/bin/inkscape")
SAVE_PLOTS = rospy.get_param("experiment_analysis/save_plots", default=True)
SAVE_PLOTS_AS_EMF = rospy.get_param("experiment_analysis/save_plots_as_emf", default=True)

ACTIVE_COILS = rospy.get_param("experiment_analysis/active_coils", default=[0, 1, 2, 3, 4, 5, 6, 7])
ACTIVE_DRIVERS = rospy.get_param("experiment_analysis/active_drivers", default=[0, 1, 2, 3, 4, 5, 6, 7])
if len(ACTIVE_DRIVERS) != len(ACTIVE_COILS):
    raise ValueError(f"Active drivers list: {ACTIVE_DRIVERS} does not match active coils list: {ACTIVE_COILS}")

if len(ACTIVE_COILS) != 8:
    rospy.loginfo(f"Subset of coils active. Active coils list: {ACTIVE_COILS}")
    rospy.loginfo(f"Subset connected to drivers: {ACTIVE_DRIVERS}")


time_varying_reference = rospy.get_param("experiment_analysis/time_varying_reference", default=False)

const_reference_pose = np.zeros(7)
const_reference_pose[0] = rospy.get_param("experiment_analysis/const_reference_pose_mm_deg/x", default=0.0)*1e-3
const_reference_pose[1] = rospy.get_param("experiment_analysis/const_reference_pose_mm_deg/y", default=0.0)*1e-3
const_reference_pose[2] = rospy.get_param("experiment_analysis/const_reference_pose_mm_deg/z", default=0.0)*1e-3
const_reference_rpy = np.zeros(3)
const_reference_rpy[0] = np.deg2rad(rospy.get_param("experiment_analysis/const_reference_pose_mm_deg/roll", default=0.0))
const_reference_rpy[1] = np.deg2rad(rospy.get_param("experiment_analysis/const_reference_pose_mm_deg/pitch", default=0.0))
const_reference_rpy[2] = np.deg2rad(rospy.get_param("experiment_analysis/const_reference_pose_mm_deg/yaw", default=0.0))
const_reference_pose[3:] = geometry.quaternion_from_euler_xyz(const_reference_rpy)

if not time_varying_reference:
    node_loginfo(f"Constant reference pose: {const_reference_pose}")
else:
    node_loginfo(f"Using time varying reference pose.")

time_sync_topic = rospy.get_param("experiment_analysis/plot_time_sync_topic", default="tnb_mns_driver/des_currents_reg")
time_sync_topic = topic_name_to_bagpyext_name(time_sync_topic)

exclude_known_latched_topics = rospy.get_param("experiment_analysis/exclude_known_latched_topics", default=True)
topics_exclusion_list = rospy.get_param("experiment_analysis/topics_exclusion_list", default=[])
if len(topics_exclusion_list) > 0:
    for i, topic in enumerate(topics_exclusion_list):
        topics_exclusion_list[i] = topic_name_to_bagpyext_name(topic)

time, data = utils.read_data_pandas_all(expt_dir, interpolate_topic=time_sync_topic, 
                                        topic_exclude_list=topics_exclusion_list, 
                                        exclude_known_latched_topics=exclude_known_latched_topics)

time_filter_enable = rospy.get_param("experiment_analysis/filter_time_range/enable", default=False)
time_filter_start = rospy.get_param("experiment_analysis/filter_time_range/start", default=None)
time_filter_end = rospy.get_param("experiment_analysis/filter_time_range/end", default=None)

if time_filter_enable:
    if time_filter_start is None or time_filter_end is None:
        raise ValueError("Time filter start and end must be specified.")
    if time_filter_start > time_filter_end:
        raise ValueError(f"Start time {time_filter_start} is greater than end time {time_filter_end}.")

    time, data = utils.filter_dataset_by_time_range(data, time, time_filter_start, time_filter_end, renormalize_time=True)

if sim:
    # We need to fake the dataset for a few quantities
    tnb_mns_system_state_dict = {}
    tnb_mns_system_state_dict['time'] = data[time_sync_topic]['time'].to_numpy()
    for i in range(8):
        tnb_mns_system_state_dict[f"dclink_voltages_{i}"] = np.zeros(len(time))
        tnb_mns_system_state_dict[f"currents_reg_{i}"] = np.zeros_like(data['_tnb_mns_driver_des_currents_reg'][f'des_currents_reg_{i}'].to_numpy())

    data['_tnb_mns_driver_system_state'] = pd.DataFrame(tnb_mns_system_state_dict)
    node_loginfo("\033[96m Sim mode detected. Actual currents and wrench will be shown as zero \033[0m")

#################################
# Dataset pre-processing for real world experiments
if not sim:
    data['_tnb_mns_driver_des_currents_reg'], data['_tnb_mns_driver_system_state'] = utils.adjust_current_datasets_for_coil_subset(
        data['_tnb_mns_driver_des_currents_reg'], data['_tnb_mns_driver_system_state'], ACTIVE_COILS, ACTIVE_DRIVERS
    )

pose_topic = topic_name_to_bagpyext_name(dipole_body.pose_frame)
reference_pose_topic = pose_topic + "_reference"
com_wrench_topic = topic_name_to_bagpyext_name(dipole_body.com_wrench_topic)

#################################
### ECB RELATED PLOTS: CURRENTS, VOLTAGE, ETC.
current_plt_save = None
system_state_df = data['_tnb_mns_driver_system_state']
des_currents_df = data['_tnb_mns_driver_des_currents_reg']
if rospy.get_param("experiment_analysis/plot_raw_ecb_data", default=False):
    system_state_df = pd.read_csv(os.path.join(expt_dir, "_tnb_mns_driver_system_state.csv"))
    des_currents_df = pd.read_csv(os.path.join(expt_dir, "_tnb_mns_driver_des_currents_reg.csv"))
    t0 = system_state_df['time'].to_numpy()[0]
    system_state_df['time'] = system_state_df['time'].to_numpy() - t0
    des_currents_df['time'] = des_currents_df['time'].to_numpy() - t0
    des_currents_df, system_state_df = utils.adjust_current_datasets_for_coil_subset(
        des_currents_df, system_state_df, ACTIVE_COILS, ACTIVE_DRIVERS
    )

    
if SAVE_PLOTS:
    current_plt_save = os.path.join(plot_folder, "des_actual_currents.svg")
plotting.plot_currents_with_reference(system_state_df, des_currents_df=des_currents_df,
                                      save_as=current_plt_save, save_as_emf=SAVE_PLOTS_AS_EMF, inkscape_path=INKSCAPE_PATH)

dclink_voltages_plt_save = None
if SAVE_PLOTS:
    dclink_voltages_plt_save = os.path.join(plot_folder, "dclink_voltages.svg")
plotting.plot_dclink_voltages(system_state_df, save_as=dclink_voltages_plt_save, save_as_emf=SAVE_PLOTS_AS_EMF, inkscape_path=INKSCAPE_PATH)
### ECB RELATED PLOTS END
#################################

#################################
### POSE AND TRAJECTORY RELATED PLOTS
## Plotting pose
if rospy.get_param("experiment_analysis/enable_pose_plots"):
    pose_save = None
    if SAVE_PLOTS:
        pose_save = os.path.join(plot_folder, "des_actual_pose.svg")
    if time_varying_reference:
        plotting.plot_poses_variable_reference(data[pose_topic],
                                            data[reference_pose_topic],
                                            save_as=pose_save,
                                            save_as_emf=SAVE_PLOTS_AS_EMF,
                                            inkscape_path=INKSCAPE_PATH)
    else:
        plotting.plot_poses_constant_reference(data[pose_topic],
                                            const_reference_pose,
                                            save_as=pose_save,
                                            save_as_emf=SAVE_PLOTS_AS_EMF,
                                            inkscape_path=INKSCAPE_PATH)
### POSE AND TRAJECTORY RELATED PLOTS END
#################################

#################################
### FORCE AND TORQUE RELATED PLOTS
## Let's compare the desired and actual components between octomag5p and good calibration file

if rospy.get_param("experiment_analysis/enable_force_torque_plots"):
    calib_file = rospy.get_param("experiment_analysis/octomag_calibration_file", default="mc3ao8s_md200_handp.yaml")
    calibration_model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                    calibration_file=calib_file)

    # Actual wrench is based on each magnet, the actual currents, and the forward nonlinear MPEM model
    dipole = dipole_body.dipole_list[0]

    act_des_wrench_save = None
    if SAVE_PLOTS:
        act_des_wrench_save = os.path.join(plot_folder, "act_des_wrench.svg")
    
    ft_plot_params = rospy.get_param("experiment_analysis/ft_plot_params")
    plotting.plot_actual_wrench_on_dipole_center_from_each_magnet(data[pose_topic],
                                                                data['_tnb_mns_driver_system_state'],
                                                                data[com_wrench_topic],
                                                                calibration_model,
                                                                dipole,
                                                                use_local_frame_for_torques=ft_plot_params['use_local_frame_for_torques'],
                                                                dataset_torques_in_local_frame=ft_plot_params['dataset_torques_in_local_frame'],
                                                                plot_overall_magnet_torque_component=ft_plot_params['plot_overall_magnet_torque_component'],
                                                                plot_torque_components_separately=ft_plot_params['plot_torque_components_separately'],
                                                                save_as=act_des_wrench_save,
                                                                save_as_emf=SAVE_PLOTS_AS_EMF)
### FORCE AND TORQUE RELATED PLOTS END
#################################

#################################
### Z CONTROL SPECIFIC PLOTS
if rospy.get_param("experiment_analysis/enable_z_specific_plots", default=False):
    z_fz_plot_save = None
    if SAVE_PLOTS:
        z_fz_plot_save = os.path.join(plot_folder, "z_fz_plot.svg")

    if time_varying_reference:
        plotting.plot_z_position_Fz_variable_reference(
            data[pose_topic],
            data[reference_pose_topic],
            plotting.wrench_stamped_df_to_array_df(data[com_wrench_topic]),
            save_as=z_fz_plot_save,
            save_as_emf=SAVE_PLOTS_AS_EMF
        )
    else:
        plotting.plot_z_position_Fz_constant_reference(
            data[pose_topic],
            const_reference_pose[2],
            plotting.wrench_stamped_df_to_array_df(data[com_wrench_topic]),
            save_as=z_fz_plot_save,
            save_as_emf=SAVE_PLOTS_AS_EMF
        )

    z_plot_save = os.path.join(plot_folder, "z_position.svg")
    if time_varying_reference:
        fig, z_axes = plotting.plot_z_position_variable_reference(data[pose_topic],
                                                    data[reference_pose_topic],
                                                    save_as=z_plot_save, save_as_emf=SAVE_PLOTS_AS_EMF, inkscape_path=INKSCAPE_PATH)
    else:
        fig, z_axes = plotting.plot_z_position_constant_reference(data[pose_topic], const_reference_pose[2], save_as=z_plot_save, save_as_emf=SAVE_PLOTS_AS_EMF, inkscape_path=INKSCAPE_PATH)

### Z CONTROL SPECIFIC PLOTS END
#################################

#################################
### NORMAL SPECIFIC PLOTS
if rospy.get_param("experiment_analysis/enable_alpha_beta_plots", default=False):
    alpha_beta_save = None
    if SAVE_PLOTS:
        alpha_beta_save = os.path.join(plot_folder, "alpha_beta.svg")

    if time_varying_reference:
        fig, ab_axes = plotting.plot_alpha_beta_torques_variable_reference(data[pose_topic], 
                                                    data[reference_pose_topic],
                                                    data[com_wrench_topic],
                                                    save_as=alpha_beta_save, save_as_emf=SAVE_PLOTS_AS_EMF, inkscape_path=INKSCAPE_PATH)
    else:
        fig, ab_axes = plotting.plot_alpha_beta_torques_constant_reference(data[pose_topic], 
                                                    const_reference_rpy[:2],
                                                    data[com_wrench_topic],
                                                    save_as=alpha_beta_save, save_as_emf=SAVE_PLOTS_AS_EMF, inkscape_path=INKSCAPE_PATH)

### NORMAL SPECIFIC PLOTS END
#################################

#################################
### UTILITY CALLS - Note taking, and other features to come
observations_editor = rospy.get_param("experiment_analysis/observations_editor", default="nano")
def open_editor_for_notes(directory=".", filename_prefix="observation"):
    # Create the observations directory if needed
    os.makedirs(directory, exist_ok=True)

    # Create a timestamped filename, if no file with prefix specific exists
    list = os.listdir(directory)
    filename = None
    exists = False
    for file in list:
        if file.startswith(filename_prefix):
            filename = file
            exists = True
            break
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{filename_prefix}_{timestamp}.txt"
    filepath = os.path.join(directory, filename)

    # Create the file if it doesn't exist
    if not exists:
        with open(filepath, 'w') as f:
            f.write("# Enter your observations below. Timestamp in the file name refers to the time of the file's creation.\n")
            f.write(f"# FIRST Experiment analysis script run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        with open(filepath, 'a') as f:
            f.write(f"\n# Experiment analysis script run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Open the file in a text editor
    subprocess.Popen(f"{observations_editor} {filepath}", shell=True)

    print(f"Editor launched asynchronously. You can write notes at: {filepath}")

    return filepath

if rospy.get_param("~note_observations", default=False):
    open_editor_for_notes(os.path.join(expt_dir, "notes"), filename_prefix="observations")

### UTILITY CALLS END
#################################

if display_plots:
    plt.show()