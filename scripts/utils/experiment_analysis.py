import os
import numpy as np
import rospkg
import rospy

import oct_levitation.plotting as plotting
import oct_levitation.processing_utils as utils
import oct_levitation.common as common
import oct_levitation.rigid_bodies as rigid_bodies

import matplotlib.pyplot as plt

from datetime import datetime


rospy.init_node('experiment_analysis_node', anonymous=True)

rospkg = rospkg.RosPack()
pkg_path = rospkg.get_path('oct_levitation')
data_base_folder = rospy.get_param('experiment_analysis/base_folder', None)
if data_base_folder is None:
    rospy.loginfo("[oct_levitation/experiment_analysis] No base folder specified. Using default: oct_levitation/data/experiment_data.")
    data_base_folder = os.path.join(pkg_path, 'data', 'experiment_data')

experiment_folder = rospy.get_param('~experiment_folder', '')
sim = rospy.get_param('~sim', False)

if sim:
    data_base_folder = os.path.join(data_base_folder, 'sim')

rospy.loginfo(f"Using data base folder: {data_base_folder}")

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

expt_dir = None
if experiment_folder != "": # Hope no one is insane enough to name a folder '' *tests using mkdir after writing this*
    expt_dir = os.path.join(data_base_folder, experiment_folder)
    if not os.path.exists(expt_dir):
        raise FileNotFoundError(f"Experiment folder {expt_dir} does not exist.")
else:
    expt_dir = get_latest_dated_folder(data_base_folder)

if expt_dir is not None:
    expt_dir = os.path.join(data_base_folder, expt_dir)
    rospy.loginfo(f"Using default experiment folder: {expt_dir}")
else:
    raise FileNotFoundError(f"No dated experiment folder found in {data_base_folder}. Please specify a folder name using the --experiment_folder argument if using a different name format than experiment_recorder.")

plot_folder = os.path.join(expt_dir, "plots")
os.makedirs(plot_folder, exist_ok=True)

## Some important parameters
calib_file = rospy.get_param("experiment_analysis/octomag_calibration_file", default="mc3ao8s_md200_handp.yaml")
plotting.DISABLE_PLT_SHOW = not rospy.get_param("experiment_analysis/show_after_each_plot", default=False)
display_plots = rospy.get_param("experiment_analysis/display_plots_at_end", default=True)

INKSCAPE_PATH = rospy.get_param("experiment_analysis/inkscape_path", default="/usr/bin/inkscape")
SAVE_PLOTS = rospy.get_param("experiment_analysis/save_plots", default=True)
SAVE_PLOTS_AS_EMF = rospy.get_param("experiment_analysis/save_plots_as_emf", default=True)

time_varying_reference = False
constant_actuation_position = False

time, data = utils.read_data_pandas_all(expt_dir, interpolate_topic="_z_rp_control_single_dipole_control_input")

time, data = utils.filter_dataset_by_time_range(data, time, 8, 30, renormalize_time=True)

## Plotting Currents
current_plt_save = None
if SAVE_PLOTS:
    current_plt_save = os.path.join(plot_folder, "des_actual_currents.svg")
plotting.plot_currents_with_reference(data['_tnb_mns_driver_system_state'], des_currents_df=data['_tnb_mns_driver_des_currents_reg'],
                                      save_as=current_plt_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH)


## Ploting the desired forces and torques
ft_save = os.path.join(plot_folder, "des_ft.svg")
plotting.plot_forces_and_torques_from_wrench_stamped(data['_onyx_disc_80x22_com_wrench'], title="Desired Forces and Torques", save_as=ft_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH)

z_plot_save = os.path.join(plot_folder, "z_position.svg")
if time_varying_reference:
    fig, z_axes = plotting.plot_z_position_variable_reference(data['_vicon_onyx_disc_80x22_Origin'],
                                                data['_vicon_onyx_disc_80x22_Origin_reference'],
                                                save_as=z_plot_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH)
else:
    fig, z_axes = plotting.plot_z_position_constant_reference(data['_vicon_onyx_disc_80x22_Origin'], 0.02, save_as=z_plot_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH)


## Plotting alpha and beta angles
alpha_beta_save = os.path.join(plot_folder, "alpha_beta.svg")

if time_varying_reference:
    fig, ab_axes = plotting.plot_alpha_beta_torques_variable_reference(data['_vicon_onyx_disc_80x22_Origin'], 
                                                data['_vicon_onyx_disc_80x22_Origin_reference'],
                                                data['_onyx_disc_80x22_com_wrench'],
                                                save_as=alpha_beta_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH)
else:
    fig, ab_axes = plotting.plot_alpha_beta_torques_constant_reference(data['_vicon_onyx_disc_80x22_Origin'], 
                                                np.array([0.0, 0.0]),
                                                data['_onyx_disc_80x22_com_wrench'],
                                                save_as=alpha_beta_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH)

## Plotting alpha beta errors and alpha beta dot errors along with torques
alpha_beta_error_save = os.path.join(plot_folder, "alpha_beta_vel_errors_and_desired_torques.svg")

fig, ab_axes = plotting.plot_alpha_beta_vel_errors_torques(
                                            data['_z_rp_control_single_dipole_error_states'],
                                            data['_onyx_disc_80x22_com_wrench'],
                                            angle_error_cols=["vector_2", "vector_4"],
                                            velocity_error_cols=["vector_3", "vector_5"],
                                            save_as=alpha_beta_error_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH)


## Plotting pose
pose_save = os.path.join(plot_folder, "des_actual_pose.svg")
if time_varying_reference:
    plotting.plot_poses_variable_reference(data['_vicon_onyx_disc_80x22_Origin'],
                                           data['_vicon_onyx_disc_80x22_Origin_reference'],
                                           save_as=pose_save,
                                           save_as_emf=True,
                                           inkscape_path=INKSCAPE_PATH)
else:
    plotting.plot_poses_constant_reference(data['_vicon_onyx_disc_80x22_Origin'],
                                           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                                           save_as=pose_save,
                                           save_as_emf=True,
                                           inkscape_path=INKSCAPE_PATH)

z_fz_plot_save = os.path.join(plot_folder, "z_fz_plot.svg")

if time_varying_reference:
    plotting.plot_z_position_Fz_variable_reference(
        data['_vicon_onyx_disc_80x22_Origin'],
        data['_vicon_onyx_disc_80x22_Origin_reference'],
        plotting.wrench_stamped_df_to_array_df(data['_onyx_disc_80x22_com_wrench']),
        save_as=z_fz_plot_save,
        save_as_emf=True
    )
else:
    plotting.plot_z_position_Fz_constant_reference(
        data['_vicon_onyx_disc_80x22_Origin'],
        0.02,
        plotting.wrench_stamped_df_to_array_df(data['_onyx_disc_80x22_com_wrench']),
        save_as=z_fz_plot_save,
        save_as_emf=True
    )


## Let's compare the desired and actual components between octomag5p and good calibration file
calibration_model = common.OctomagCalibratedModel(calibration_type="legacy_yaml", 
                                                  calibration_file="mc3ao8s_md200_handp.yaml")


# ### These functions only make sense for constant actuation matrix
field_grad_save = os.path.join(plot_folder, "actual_fields_and_grads.svg")

plotting.plot_actual_field_and_gradients(
    data['_vicon_onyx_disc_80x22_Origin'],
    data['_tnb_mns_driver_system_state'],
    calibration_model,
    save_as=field_grad_save, save_as_emf=True, inkscape_path=INKSCAPE_PATH
)


# # Actual wrench is completely based on the currents and is therefore a good method
# # of comparing two calibration files.
dipole_body = rigid_bodies.Onyx80x22DiscCenterRingDipole
dipole = dipole_body.dipole_list[0]

act_des_wrench_save = os.path.join(plot_folder, "act_des_wrench.svg")
plotting.plot_actual_wrench_on_dipole_center_from_each_magnet(data['_vicon_onyx_disc_80x22_Origin'],
                                                              data['_tnb_mns_driver_system_state'],
                                                              data['_onyx_disc_80x22_com_wrench'],
                                                              calibration_model,
                                                              dipole,
                                                              use_local_frame_for_torques=True,
                                                              dataset_torques_in_local_frame=False,
                                                              save_as=act_des_wrench_save,
                                                              save_as_emf=False)

plt.show()