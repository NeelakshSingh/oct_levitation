import numpy as np
import os

import rospkg
import oct_levitation.plotting as plotting
import oct_levitation.geometry as geometry
import oct_levitation.rigid_bodies as rigid_bodies
import oct_levitation.common as common

from argparse import ArgumentParser

if __name__=="__main__":
    parser = ArgumentParser(
        prog='AllocationConditionNumberVolumeAnalysis',
        description='This program draws a volume plot of the allocation condition number for a dipole object.',
        epilog="""Edit the dipole object in the file directly to change its properties.
The arguments will be the orientation of the dipole in r, p, y extrinsic XYZ euler angles. This
is because the overall condition number can change because of M. For now only orientations close to
upright are assumed so the 3rd row for Tz is excluded from M. Plots are stored in oct_levitation/data/cond_plots/."""
    )

    parser.add_argument("r", type=float, help="Roll angle in degrees")
    parser.add_argument("p", type=float, help="Pitch angle in degrees")
    parser.add_argument("y", type=float, help="Yaw angle in degrees")
    parser.add_argument("--num_samples", type=int, help="Number of points samples per dimension", default=50)
    parser.add_argument("--save_folder", type=str, help="Sub-folder to save the plot", default="six_10_4_5_N35_rings")
    parser.add_argument("--calib_file", type=str, help="Calibration file under $HOME/.ros/cal to use.", default="mc3ao8s_md200_handp.yaml")

    args = parser.parse_args()

    rpy = np.deg2rad([args.r, args.p, args.y])
    quat = geometry.quaternion_from_euler_xyz(rpy)

    dipole = rigid_bodies.Onyx80x22DiscCenterRingDipole.dipole_list[0]

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('oct_levitation')

    save_path = os.path.join(pkg_path, "data/cond_plots", args.save_folder)

    save_as = os.path.join(save_path, f"{rpy[0]}_{rpy[1]}_{rpy[2]}.vti")

    calibration_model = common.OctomagCalibratedModel(calibration_file=args.calib_file)

    # plotting.plot_volumetric_ma_condition_number_variation(dipole,
    #                                                        calibration_model,
    #                                                        quat,
    #                                                        save_as=save_as,
    #                                                        num_samples=args.num_samples)
    
    plotting.plot_slices_ma_condition_number_variation(dipole,
                                                       calibration_model,
                                                       quat,
                                                       save_as=save_as,
                                                       num_samples=args.num_samples)