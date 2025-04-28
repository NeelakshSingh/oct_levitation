import os
import rospy
import rospkg
import itertools
import numpy as np
import pandas as pd

import oct_levitation.plotting as plotting
import oct_levitation.geometry as geometry
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.common as common
import alive_progress as ap

from argparse import ArgumentParser
from scipy.linalg import block_diag

from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction
from tvtk.api import tvtk

np.set_printoptions(linewidth=np.inf)

if __name__=="__main__":
    parser = ArgumentParser(
        prog='AllocationConditionNumberVolumeAnalysis',
        description='This program draws a volume plot of the allocation condition number for a dipole object.',
        epilog="""Edit the dipole object in the file directly to change its properties.
The arguments will be the orientation of the dipole in r, p, y extrinsic XYZ euler angles. This
is because the overall condition number can change because of M. For now only orientations close to
upright are assumed so the 3rd row for Tz is excluded from M. Plots are stored in oct_levitation/data/cond_plots/."""
    )

    parser.add_argument("num_coils", type=int, help="Number of coils to consider.")
    parser.add_argument("--cond_threshold", type=float, help="Max threshold of the good condition number, used for coloring the plots.", default=20)
    parser.add_argument("--cond_color_steps", type=float, help="Steps to use for coloring the condition number.", default=10)
    parser.add_argument("--num_samples", type=int, help="Number of points samples per dimension", default=50)
    parser.add_argument("--calib_file", type=str, help="Calibration file under $HOME/.ros/cal to use.", default="mc3ao8s_md200_handp.yaml")
    parser.add_argument("--x_lim", type=float, default=0.06)
    parser.add_argument("--y_lim", type=float, default=0.06)
    parser.add_argument("--z_lim", type=float, default=0.06)
    parser.add_argument("--eval_lim", type=float, default=0.01)
    parser.add_argument("--clip_cond", type=float, default=100)
    parser.add_argument("--slice_cond_max", type=float, default=100) # to make it consistent with the plotting.py plotter.
    parser.add_argument("--mode", type=int, default=0, help="0: Only use A matrix. 1: Uses MA matrix. 2: Uses Pendulum JMA matrix. For >0, one can specify dipole rpy and dipole strength.")
    parser.add_argument("--rpy", type=float, default=[0.0, 0.0, 0.0], nargs=3, help="[roll, pitch, yaw] intrinsic euler angles (in degrees) of local frame w.r.t inertial frame.")
    # These two parameters default to the Onyx disc's values.
    parser.add_argument("--dipole_strength", type=float, default=1.6453125, help="In tesla.")
    parser.add_argument("--dipole_axis", type=float, nargs=3, help="[x, y, z] local frame axis of the dipole.", default=[0.0, 0.0, -1.0])
    parser.add_argument("--store_cond_field_dataset", type=bool, default=False)

    args = parser.parse_args()
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('oct_levitation')
    rospy.loginfo(f"[Coil Subset Condition Calculator]: Received number of coils: {args.num_coils}")
    
    if args.mode == 0:
        base_folder = "A_matrix"
    elif args.mode == 1:
        base_folder = "MA_matrix"
    elif args.mode == 2:
        base_folder = "pend_JMA_matrix"
    elif args.mode == 3:
        base_folder = "normalized_MA_matrix"
    else:
        raise ValueError("Invalid value received for parameter 'mode'. Please select among 0 and 1.")

    save_path = os.path.join(pkg_path, "data/subset_allocation_cond_plots", f"{base_folder}/coil_count_{args.num_coils}")

    calibration_model = common.OctomagCalibratedModel(calibration_file=args.calib_file)

    all_coils = np.arange(8)

    cube_x_lim = np.array([-args.x_lim, args.x_lim])
    cube_y_lim = np.array([-args.y_lim, args.y_lim])
    cube_z_lim = np.array([-args.z_lim, args.z_lim])
    x_eval_lim = args.eval_lim
    y_eval_lim = args.eval_lim
    z_eval_lim = args.eval_lim
    print(f"x_eval_lim: {x_eval_lim} | y_eval_lim: {y_eval_lim} | z_eval_lim: {z_eval_lim}")
    x_ticks = np.linspace(cube_x_lim[0], cube_x_lim[1], args.num_samples)
    y_ticks = np.linspace(cube_y_lim[0], cube_y_lim[1], args.num_samples)
    z_ticks = np.linspace(cube_z_lim[0], cube_z_lim[1], args.num_samples)
    X, Y, Z = np.meshgrid(x_ticks, y_ticks, z_ticks)

    cond_threshold = args.cond_threshold
    cond_color_steps = args.cond_color_steps
    clip_cond = args.clip_cond

    plot_save_path = os.path.join(save_path, "plots")
    dataset_save_path = os.path.join(save_path, "dataset")

    os.makedirs(os.path.join(plot_save_path, "volumes"), exist_ok=True)
    os.makedirs(os.path.join(plot_save_path, "volume_slices"), exist_ok=True)
    os.makedirs(dataset_save_path, exist_ok=True)

    # Now we will make the plots for every coil combination and then save them. At the same time
    # we will determine which subset is the best with respect to the condition number and save a
    # report of the same.
    # Sampling points according to the desired plot style.

    ### Which measure to use for best condition number?
    ## For now, RMS of the condition number in the whole region. Problem? There are very high
    ## condition numbers in some regions. How about we just check it in a smaller subspace near
    ## the origin then?

    subspace_selection_mask = (np.abs(X) < x_eval_lim) & (np.abs(Y) < y_eval_lim) & (np.abs(Z) < z_eval_lim)
    
    def calc_rms(arr):
        return np.sqrt(np.mean(np.square(arr)))
    
    min_rms = np.inf
    min_config = None
    all_rms = []
    all_configs = np.asarray(list(itertools.combinations(all_coils, args.num_coils)))
    count = 0

    # Offscreen rendering for mayavi
    mlab.options.offscreen = True

    rpy = np.deg2rad(np.asarray(args.rpy))
    quat = geometry.quaternion_from_euler_xyz(rpy)
    dipole_strength = np.asarray(args.dipole_strength)
    R = geometry.rotation_matrix_from_euler_xyz(rpy)
    dipole_axis = np.asarray(args.dipole_axis)
    dipole_moment = (dipole_strength * R @ dipole_axis).flatten()

    N_ft = np.diag([5e-3, 5e-3, 0.1, 0.1, 1])
    S = np.linalg.inv(N_ft) # normalizing matrix for the torques and forces.

    def jacobian_torqueforce_to_torque(beta, alpha):
        # jacobian mapping torques and forces to control-torques
        l_mag = 38e-3
        force_torque_relation = np.array([[0,-l_mag*np.cos(beta)*np.cos(alpha) ,-np.sin(beta)*l_mag],
                                          [np.cos(beta)*np.cos(alpha)*l_mag, 0,-np.cos(beta)*np.sin(alpha)*l_mag],
                                          [np.sin(beta)*l_mag,l_mag*np.cos(beta)*np.sin(alpha) ,0]])
        
        return np.hstack((np.eye(3), force_torque_relation))
    
    J = jacobian_torqueforce_to_torque(0, 0)[:2] # Because we cannot control Tz.

    Mf = geometry.magnetic_interaction_grad5_to_force(dipole_moment)
    if args.mode in [1, 3]:
        M_tau_local = geometry.magnetic_interaction_field_to_local_torque(dipole_strength=dipole_strength,
                                                                        dipole_axis=dipole_axis,
                                                                        dipole_quaternion=quat) 
        M = block_diag(M_tau_local[:2], Mf) # We only consider the first 2 rows of M_tau. So its just best to let the dipole axis be [0, 0, 1].
    elif args.mode == 2:
        M_tau = geometry.magnetic_interaction_field_to_torque(dipole_moment)
        M = block_diag(M_tau, Mf) # We only consider the first 2 rows of M_tau. So its just best to let the dipole axis be [0, 0, 1].
        rospy.loginfo("[Coil Subset Condition Calculator]: Using pendulum's JMA matrix for analysis.")
    

    if args.mode > 0:
        rospy.loginfo(f"[Coil Subset Condition Calculator]: M matrix: {M} \n J matrix: {J}")
    
    rms_config_dict = {
        "rms_values": [],
        "configurations": []
    }

    for coil_set in ap.alive_it(all_configs):
        count += 1
        cond_eval_func = None
        rms_config_dict["configurations"].append(str(coil_set))
        @np.vectorize
        def get_A_condition_subset(x, y, z):
            A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
            A = A[:, coil_set]
            return np.linalg.cond(A)
        
        @np.vectorize
        def get_MA_condition_subset(x, y, z):
            A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
            A = A[:, coil_set]
            return np.linalg.cond(M @ A)
        
        @np.vectorize
        def get_SMA_condition_subset(x, y, z):
            A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
            A = A[:, coil_set]
            return np.linalg.cond(S @ M @ A)
        
        @np.vectorize
        def get_JMA_condition_subset(x, y, z):
            A = calibration_model.get_actuation_matrix(np.array([x, y, z]))
            A = A[:, coil_set]
            return np.linalg.cond(J @ M @ A)
        
        if args.mode == 0:
            cond_eval_func = get_A_condition_subset
        elif args.mode == 1:
            cond_eval_func = get_MA_condition_subset
        elif args.mode == 2:
            cond_eval_func = get_JMA_condition_subset
        elif args.mode == 3:
            cond_eval_func = get_SMA_condition_subset
        else:
            raise ValueError("Invalid mode value. Should be among [0, 1].")
        cond = cond_eval_func(X, Y, Z)
        if args.store_cond_field_dataset:
            dataset_df = pd.DataFrame({
                'X': X.flatten(),
                'Y': Y.flatten(),
                'Z': Z.flatten(),
                'cond': cond.flatten()
            })
            dataset_save = os.path.join(dataset_save_path, f"coils_{np.asarray(coil_set)}.csv")
            dataset_df.to_csv(dataset_save, index=False)

        # Let's evaluate the configuration's RMS value before clipping and select the best one.
        config_rms = calc_rms(cond[subspace_selection_mask])
        all_rms.append(config_rms)
        if config_rms < min_rms:
            min_rms = config_rms
            min_config = coil_set

        clipped_cond = np.clip(cond, 0.0, clip_cond)

        ### PLOTTING VOLUME
        cond_field = mlab.pipeline.scalar_field(clipped_cond)

        ctf = ColorTransferFunction()

        # # Add solid green color for very low values
        ctf.add_rgb_point(0,   0, 1, 0)  # Green at lowest value
        # ctf.add_rgb_point(cond_threshold - 0.1, 0, 1, 0)  # Green up to threshold

        # Add colormap for values ≥ 20 (e.g., blue to red gradient)
        ctf.add_rgb_point(cond_threshold,  0, 0, 1)  # Blue at threshold
        ctf.add_rgb_point(cond_threshold + cond_color_steps,  1, 1, 0)  # Yellow for arbitrarily high
        ctf.add_rgb_point(cond_threshold + 2*cond_color_steps,  1, 0, 0)  # Red until cond number steps from threshold


        # Create opacity transfer function (OTF) by manipulating the _volume_property
        opacity_function = tvtk.PiecewiseFunction()
        # Set opacity for values below the threshold to be opaque (1.0)
        opacity_function.add_point(1.0, 0.0)  # Fully opaque at lowest value
        opacity_function.add_point(cond_threshold, 0.2)  # Less opaque at threshold
        # Set opacity to 0.0 for higher values, making them fully transparent
        opacity_function.add_point(cond_threshold + cond_color_steps, 0.1)
        opacity_function.add_point(cond_threshold + 2*cond_color_steps, 0.01)

        cond_vol = mlab.pipeline.volume(cond_field, vmin=0, vmax=clip_cond)

        cond_vol._volume_property.set_color(ctf)  # Set custom color mapping
        cond_vol._volume_property.set_scalar_opacity(opacity_function)
        cond_vol.update_ctf = True  # Update color transfer function
        # Add axes to the plot
        axes = mlab.axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', ranges=np.array([cube_x_lim[0], 
                                                                    cube_x_lim[1], 
                                                                    cube_y_lim[0], 
                                                                    cube_y_lim[1],
                                                                    cube_z_lim[0],
                                                                    cube_z_lim[1]])*1e3)
        mlab.outline()

        save_volume_as = os.path.join(plot_save_path, f"volumes/coils_{np.asarray(coil_set)}.png")
        mlab.draw()
        mlab.process_ui_events()

        mlab.savefig(save_volume_as)
        
        ### PLOTTING VOLUME SLICES
        # Next we need to add slicing planes to this plot.
        mlab.close()
        x_plane_idx = args.num_samples // 2
        y_plane_idx = args.num_samples // 2
        z_plane_idx = args.num_samples // 2
        cond_range = np.array([0.0, args.slice_cond_max])
        slice_x = mlab.volume_slice(cond, plane_orientation='x_axes', slice_index=x_plane_idx, colormap="jet",
                                        vmin=cond_range[0], vmax=cond_range[1])
        slice_y = mlab.volume_slice(cond, plane_orientation='y_axes', slice_index=y_plane_idx, colormap="jet",
                                        vmin=cond_range[0], vmax=cond_range[1])
        slice_z = mlab.volume_slice(cond, plane_orientation='z_axes', slice_index=z_plane_idx, colormap="jet",
                                        vmin=cond_range[0], vmax=cond_range[1])
        
        axes = mlab.axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', ranges=np.array([cube_x_lim[0], 
                                                                cube_x_lim[1], 
                                                                cube_y_lim[0], 
                                                                cube_y_lim[1],
                                                                cube_z_lim[0],
                                                                cube_z_lim[1]])*1e3)
        colorbar = mlab.colorbar(orientation='vertical', nb_labels=5)
        mlab.outline()
        save_slices_as = os.path.join(plot_save_path, f"volume_slices/coils_{np.asarray(coil_set)}.png")
        mlab.draw()
        mlab.process_ui_events()
        mlab.savefig(save_slices_as)
        mlab.close()

    rms_config_dict["rms_values"] = all_rms

    rms_df = pd.DataFrame(rms_config_dict)

    ## Saving the results in a sort of a final report file.
    report = f"""
Arguments from the argument parser:
- Condition number threshold (cond_threshold): {args.cond_threshold}
- Condition number coloring steps (cond_color_steps): {args.cond_color_steps}
- Number of grid samples per dimension (num_samples): {args.num_samples}
- Calibration file (calib_file): {args.calib_file}
- X dimension limit (x_lim): {args.x_lim}
- Y dimension limit (y_lim): {args.y_lim}
- Z dimension limit (z_lim): {args.z_lim}
- X evaluation limit (x_eval_lim): {x_eval_lim}
- Y evaluation limit (y_eval_lim): {y_eval_lim}
- Z evaluation limit (z_eval_lim): {z_eval_lim}
- Condition number clipping (clip_cond): {args.clip_cond}
- Mode (mode): {args.mode}
- Dipole roll, pitch, yaw (rpy): {args.rpy}
- Dipole strength (dipole_strength): {args.dipole_strength}
- Dipole axis (dipole_axis): {args.dipole_axis}
- Normalization matrix (S): {S} [Used: {args.mode == 3}]
- Jacobian materix (J): {J} [Used: {args.mode == 2}]

Number of possible combinations: {count},
Minimum RMS allocation condition number encountered: {min_rms},
Minimum RMS allocation coil subset: {min_config},

Volume plots of condition numbers saved to: {os.path.join(plot_save_path, "volumes")},
Volume slice plots of condition numbers saved to: {os.path.join(plot_save_path, "volume_slices")},
Datasets saved to: {dataset_save_path},

"""
    
    report_save_file = os.path.join(save_path, "results.txt")
    rms_save_file = os.path.join(save_path, "rms_values.csv")
    with open(report_save_file, "w") as f:
        f.write(report)
    rms_df.to_csv(rms_save_file, index=False)
    

    rospy.loginfo("[Coil Subset Condition Calculator]: Saved the report. Calculations ended.")