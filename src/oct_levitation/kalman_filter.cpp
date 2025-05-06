#include <oct_levitation/kalman_filter.h>
#include <XmlRpcValue.h>
#include <filesystem>

namespace oct_levitation {
    KalmanFilter::KalmanFilter(ros::NodeHandle& nh) : nh_(nh) {
        try { // Parameter server queries
            RingMagnet magnet(
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/magnet/inner_radius", 0.0)),
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/magnet/outer_radius", 0.0)),
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/magnet/height", 0.0)),
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/magnet/density", 0.0)),
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/magnet/remanence", 0.0))
            );

            std::vector<double> magnet_z_positions_arr;
            nh_.getParam("oct_levitation/cpp_impl/z_values", magnet_z_positions_arr);

            RingMagnetsRingObject rigid_body_object(
                magnet,
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/disc/inner_radius", 0.0)),
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/disc/outer_radius", 0.0)),
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/disc/height", 0.0)),
                static_cast<double>(nh_.param("oct_levitation/cpp_impl/disc/density", 0.0)),
                static_cast<int>(nh_.param("oct_levitation/cpp_impl/num_magnets", 0)),
                static_cast<bool>(nh_.param("oct_levitation/cpp_impl/north_pole_down", false)),
                magnet_z_positions_arr
            );

            object_mass_ = rigid_body_object.mass;
            object_Ixxyy_ = rigid_body_object.Ixxyy;
            local_dipole_moment_ = rigid_body_object.getLocalDipoleMoment();

            // magnetic field related properties
            std::string calfile_name;
            nh_.getParam("oct_levitation/cpp_impl/calfile", calfile_name);
            std::filesystem::path calfile_path = std::filesystem::path(std::getenv("HOME")) / std::filesystem::path(".ros/cal") / std::filesystem::path(calfile_name);
            calibration_.setCalibrationFile(calfile_path);
        }
        catch (const std::exception& e) {
            ROS_ERROR("Error while loading object parameters: %s", e.what());
        }

        // Initialize state space matrices
        state_space_A_ = Eigen::MatrixXd::Identity(6, 6);
        state_space_B_ = Eigen::MatrixXd::Zero(6, 3);
    }

} // namespace oct_levitation